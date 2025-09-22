import re
import os
from typing import List, Dict, Tuple, Optional
from utils.pattern import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================== Configuration =====================
MIN_TOKENS = int(os.getenv("MIN_TOKENS", 300))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
USE_TIKTOKEN = os.getenv("USE_TIKTOKEN", "false").lower() == "true"
TAB_WIDTH = int(os.getenv("TAB_WIDTH", 4))
INDENT_STEP = int(os.getenv("INDENT_STEP", 4))

# ===================== Token counter =====================
_tokenc = None

def _approx_token_count(s: str) -> int:
    return max(1, len(s.split()))

def _tok_count(s: str) -> int:
    global _tokenc
    if USE_TIKTOKEN:
        try:
            import tiktoken
            if _tokenc is None:
                _tokenc = tiktoken.get_encoding("cl100k_base")
            return len(_tokenc.encode(s))
        except Exception:
            pass
    return _approx_token_count(s)

# ===================== TXT -> text (preserve line breaks) =====================
def _txt_to_text_with_lines(txt_path: str) -> str:
    """
    Read TXT file and return content with line breaks preserved
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        content = None
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file {txt_path} with any common encoding")
    
    return content

# ===================== Heading detection =====================
def _is_heading(line: str) -> Tuple[bool, int]:
    """
    Detect if a line is a heading and return its level (1=highest, 6=lowest)
    Returns: (is_heading, level)
    """
    line_clean = line.strip()
    if not line_clean or len(line_clean) < 3:
        return False, 0
    
    # Check numbered headings (1., 1.1., 1.1.1.)
    numbered_match = re.match(r"^\s*(\d+\.)+\s+(.+)$", line_clean)
    if numbered_match:
        dots = numbered_match.group(1).count('.')
        return True, min(dots, 6)
    
    # Check other patterns
    for i, pattern in enumerate(HEADING_PATTERNS):
        if pattern.match(line_clean):
            # First pattern (numbered) handled above
            if i == 0:
                continue
            # ALL CAPS = level 1, others = level 2-3
            level = 1 if i == 1 else (2 if len(line_clean) < 50 else 3)
            return True, level
    
    # Additional TXT-specific heading patterns
    # Lines that are standalone and short (likely titles/headers)
    if len(line_clean) < 80 and not line_clean.endswith('.') and not line_clean.endswith(','):
        # Check if line is in ALL CAPS or Title Case
        if line_clean.isupper() or line_clean.istitle():
            return True, 2
    
    # Lines surrounded by special characters (like === Title ===)
    if re.match(r"^\s*[=\-#\*]{3,}\s*.+\s*[=\-#\*]{3,}\s*$", line_clean):
        return True, 1
    
    # Lines starting with # (markdown-style)
    hash_match = re.match(r"^\s*(#{1,6})\s+(.+)$", line_clean)
    if hash_match:
        return True, len(hash_match.group(1))
    
    return False, 0

def _extract_headings_from_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Extract headings from blocks and create heading hierarchy
    Returns list of headings with their levels and positions
    """
    headings = []
    
    for i, block in enumerate(blocks):
        lines = block["text"].split('\n')
        for line in lines:
            is_head, level = _is_heading(line.strip())
            if is_head:
                headings.append({
                    "text": line.strip(),
                    "level": level,
                    "block_index": i,
                    "indent": block["indent"]
                })
                break  # Only take first heading from each block
    
    return headings

def _build_breadcrumbs(headings: List[Dict], current_block_index: int) -> List[str]:
    """
    Build breadcrumb path for a given block based on preceding headings
    Returns: ["Document Title", "Section", "Subsection"]
    """
    breadcrumbs = []
    current_levels = {}  # level -> heading text
    
    # Find headings that precede current block
    relevant_headings = [h for h in headings if h["block_index"] < current_block_index]
    
    for heading in relevant_headings:
        level = heading["level"]
        text = heading["text"]
        
        # Clear deeper levels when we encounter a higher level heading
        keys_to_remove = [k for k in current_levels.keys() if k >= level]
        for k in keys_to_remove:
            del current_levels[k]
        
        # Set current level
        current_levels[level] = text
    
    # Build breadcrumb path from level 1 to deepest
    if current_levels:
        for level in sorted(current_levels.keys()):
            breadcrumbs.append(current_levels[level])
    
    return breadcrumbs

# ===================== Line -> indent =====================
def _indent_width(line: str) -> int:
    # calculate indent in number of spaces (tab => TAB_WIDTH spaces)
    leading = len(line) - len(line.lstrip("\t "))
    # convert tabs and spaces: simple approach, replace tabs with TAB_WIDTH spaces then count
    expanded = line[:leading].replace("\t", " " * TAB_WIDTH)
    return len(expanded)

# ===================== Group into blocks by indent & blank lines =====================
def _lines_to_blocks(text: str) -> List[Dict]:
    """
    Convert text string with line breaks into list of blocks:
    - block boundary: blank line or indent change >= INDENT_STEP
    - Each block: {"indent": int, "text": str}
    """
    raw_lines = text.splitlines()
    blocks = []
    buf: List[str] = []
    cur_indent = None

    def flush():
        nonlocal buf, cur_indent
        if buf:
            # join preserving internal line breaks within block
            blk_text = "\n".join(buf).strip("\n")
            blocks.append({"indent": cur_indent or 0, "text": blk_text})
            buf = []

    for ln in raw_lines:
        if WHITESPACE_ONLY_PATTERN.match(ln):
            # blank line => end current block
            flush()
            cur_indent = None
            continue

        ind = _indent_width(ln)

        if cur_indent is None:
            # start new block
            cur_indent = ind
            buf = [ln]
        else:
            # if strong indent change (>= INDENT_STEP) -> end previous block, start new block
            if abs(ind - cur_indent) >= INDENT_STEP:
                flush()
                cur_indent = ind
                buf = [ln]
            else:
                buf.append(ln)

    flush()
    return blocks

# ===================== Split large blocks by sentences =====================
def _split_block_into_chunks_by_sentence(blk_text: str, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[str]:
    """
    If block is too large, split by sentences (don't break bullet points).
    """
    # preserve line breaks when there are bullets: treat each bullet-line as one unit
    parts: List[str] = []
    for line in blk_text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if BULLET_PATTERN.match(line):
            parts.append(line)
        else:
            # additionally split by sentences for regular lines
            segs = SPLIT_PATTERN.split(line)
            for s in segs:
                s = s.strip()
                if s:
                    parts.append(s)

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for p in parts:
        t = _tok_count(p)
        if t > max_tokens:
            # one sentence/bullet too long: push separately
            if cur:
                if cur_tok >= min_tokens:
                    chunks.append("\n".join(cur))
                    cur, cur_tok = [], 0
                else:
                    chunks.append("\n".join(cur + [p]))
                    cur, cur_tok = [], 0
            else:
                chunks.append(p)
            continue

        if cur and cur_tok + t > max_tokens:
            chunks.append("\n".join(cur))
            cur, cur_tok = [p], t
        else:
            cur.append(p)
            cur_tok += t

    if cur:
        if chunks and _tok_count(chunks[-1]) + cur_tok <= max_tokens:
            chunks[-1] = (chunks[-1] + "\n" + "\n".join(cur)).strip("\n")
        else:
            chunks.append("\n".join(cur))

    # if the last chunk is still < min -> merge backward if possible
    if len(chunks) >= 2 and _tok_count(chunks[-1]) < min_tokens:
        if _tok_count(chunks[-2]) + _tok_count(chunks[-1]) <= max_tokens:
            chunks[-2] = (chunks[-2] + "\n" + chunks[-1]).strip("\n")
            chunks.pop()

    return chunks

# ===================== Pack blocks into ~500 token chunks =====================
def _pack_blocks_into_chunks(blocks: List[Dict], headings: List[Dict], min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[Dict]:
    """
    Group blocks (in order) into ~500 token chunks, within [300, 500] when possible.
    - Oversized block -> split by sentences (don't break bullets).
    - Don't use headings; only rely on indent & blank lines to determine natural blocks.
    """
    out: List[Dict] = []
    cur_parts: List[str] = []
    cur_inds: List[int] = []
    cur_block_indices: List[int] = []
    cur_tok = 0

    def close():
        nonlocal cur_parts, cur_inds, cur_block_indices, cur_tok
        if cur_parts:
            payload = "\n".join(cur_parts).strip("\n")
            # Get breadcrumbs for the first block in this chunk
            first_block_idx = cur_block_indices[0] if cur_block_indices else 0
            breadcrumbs = _build_breadcrumbs(headings, first_block_idx)
            
            out.append({
                "payload": payload,
                "indent_levels": sorted(set(cur_inds)),
                "breadcrumbs": breadcrumbs,
                "block_indices": cur_block_indices.copy()
            })
            cur_parts = []
            cur_inds = []
            cur_block_indices = []
            cur_tok = 0

    for block_idx, blk in enumerate(blocks):
        txt = blk["text"]
        ind = blk["indent"]
        t = _tok_count(txt)

        # if block is too large -> split into smaller pieces within block
        if t > max_tokens:
            subchunks = _split_block_into_chunks_by_sentence(txt, min_tokens, max_tokens)
            for sc in subchunks:
                st = _tok_count(sc)
                if cur_tok > 0 and cur_tok + st <= max_tokens:
                    cur_parts.append(sc)
                    cur_inds.append(ind)
                    if block_idx not in cur_block_indices:
                        cur_block_indices.append(block_idx)
                    cur_tok += st
                    if cur_tok >= min_tokens:
                        close()
                else:
                    if cur_tok > 0:
                        close()
                    cur_parts.append(sc)
                    cur_inds.append(ind)
                    cur_block_indices.append(block_idx)
                    cur_tok = st
                    if cur_tok >= min_tokens:
                        close()
            continue

        # regular sized block: try to fit into current chunk
        if cur_tok + t <= max_tokens:
            cur_parts.append(txt)
            cur_inds.append(ind)
            cur_block_indices.append(block_idx)
            cur_tok += t
            # reached "nice" range [min,max] -> don't rush to close, unless next block would exceed
            # (no lookahead here to keep simple; chunk will close when adding next block exceeds max)
        else:
            # close current chunk if adding would exceed max
            close()
            # put block into new chunk
            cur_parts.append(txt)
            cur_inds.append(ind)
            cur_block_indices.append(block_idx)
            cur_tok = t
            if cur_tok >= min_tokens:
                close()

    close()
    return out

# ===================== Main API =====================
def txt_to_chunks_smart_indent(
    txt_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS
) -> List[Dict]:
    """
    - Read TXT -> text (preserve line breaks)  
    - Divide into blocks by "blank lines" + "indent change >= INDENT_STEP"  
    - Pack blocks into ~500 token chunks (within [300, 500] if possible)  
    - Automatically detect headings and create breadcrumbs  
    - Prepend heading context to the chunk content  
    """
    file_name = os.path.basename(txt_path)
    doc_text = _txt_to_text_with_lines(txt_path)
    blocks = _lines_to_blocks(doc_text)
    
    # Extract headings for breadcrumb generation
    headings = _extract_headings_from_blocks(blocks)

    packed = _pack_blocks_into_chunks(blocks, headings, min_tokens=min_tokens, max_tokens=max_tokens)

    results = []
    doc_id = os.path.splitext(file_name)[0]  # Remove extension for doc_id
    
    for idx, ch in enumerate(packed):
        # Build breadcrumb string
        breadcrumb_text = ""
        if ch["breadcrumbs"]:
            breadcrumb_text = " > ".join(ch["breadcrumbs"])
        
        # Prepend breadcrumbs to content if available
        content_with_context = ch["payload"]
        if breadcrumb_text:
            content_with_context = f"[Context: {breadcrumb_text}]\n\n{ch['payload']}"
        
        results.append({
            "chunk_id": idx,
            "file_name": file_name,
            "indent_levels": ch["indent_levels"],
            "content": content_with_context,
            "token_est": _tok_count(content_with_context),
            # New metadata fields
            "doc_id": doc_id,
            "source": os.path.abspath(txt_path),
            "section_title": ch["breadcrumbs"][-1] if ch["breadcrumbs"] else None,
            "heading_path": breadcrumb_text if breadcrumb_text else None,
            "chunk_index": idx,
            "breadcrumbs": ch["breadcrumbs"]
        })
    return results
