import re
import fitz  # PyMuPDF
import os
import sys
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.pattern import *
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from pattern import *

# ===================== Configuration =====================
MIN_TOKENS = 300
MAX_TOKENS = 500
USE_TIKTOKEN = False  # set True if tiktoken is installed for accurate token counting
TAB_WIDTH = 4         # 1 tab = 4 spaces when calculating indent
INDENT_STEP = 4       # indent difference threshold (>=) to consider as new "block"

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

# ===================== Table of Contents Detection =====================
def _is_toc_page(page_text: str) -> bool:
    """
    Enhanced TOC page detection based on common patterns
    """
    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
    if len(lines) < 5:  # Too short to be meaningful TOC
        return False
    
    # Check for explicit TOC indicators
    toc_indicators = 0
    for line in lines[:15]:  # Check first 15 lines for TOC title
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in [
            'table of contents', 'mục lục',
        ]):
            toc_indicators += 1
    
    # Count different types of TOC patterns
    page_number_lines = 0
    dot_lines = 0
    numbered_section_lines = 0
    
    for line in lines:
        # Pattern 1: Lines ending with page numbers (like "Failure Is Everywhere 205")
        if re.search(r'\s+\d{1,4}\s*$', line) and len(line) > 15:
            page_number_lines += 1
        
        # Pattern 2: Lines with many dots (like "11. Microservices at Scale. . . . . . . . . . 205")
        if line.count('.') >= 8:
            dot_lines += 1
            


    total_lines = len(lines)
    if total_lines == 0:
        return False
        
    page_number_ratio = page_number_lines / total_lines
    dot_ratio = dot_lines / total_lines
    numbered_ratio = numbered_section_lines / total_lines
    
    # Strong indicators
    if toc_indicators > 0:
        return True
    
    # Pattern-based detection with lower thresholds for better detection
    if page_number_ratio > 0.2 and dot_ratio > 0.1:
        return True
    
    if page_number_ratio > 0.4:  # Many lines end with page numbers
        return True
        
    if numbered_ratio > 0.1 and page_number_ratio > 0.15:  # Numbered sections with page numbers
        return True
        
    return False

# ===================== PDF -> text (preserve line breaks) =====================
def _pdf_to_text_with_lines(pdf_path: str) -> Tuple[str, List[int]]:
    """
    Returns:
    - doc_text: full document text, pages joined with '\n\n' (TOC pages already filtered out)
    """
    doc = fitz.open(pdf_path)
    pieces = []
    toc_index = []
    toc_pages_found = 0
    
    for pno in range(len(doc)):
        page = doc[pno]
        # get blocks to maintain proper reading order + preserve \n between lines in block
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1],1), round(b[0],1)))
        page_text = "\n".join(
            [blk[4] for blk in blocks if blk[4] and blk[4].strip() != ""]
        )
        
        # Check if this page is TOC and skip it
        if _is_toc_page(page_text):
            print(f"Skipping TOC page {pno + 1}")
            toc_index.append(pno + 1)
            toc_pages_found += 1
            continue
        if len(page_text) < 3:
            print(f"Skipping unmeaningful page {pno + 1}")
            toc_index.append(pno + 1)
            toc_pages_found += 1
            continue
        pieces.append(page_text)
        
    if toc_pages_found > 0:
        print(f"Filtered out {toc_pages_found} TOC pages from {len(doc)} total pages")

    return "\n\n<<<PAGE_BREAK>>>".join(pieces), toc_index

    # ===================== Heading detection =====================

def _is_heading(line: str, font_size: Optional[float] = None, is_bold: bool = False) -> Tuple[bool, int]:
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
    
    # Font-based detection (if available)
    if font_size and font_size > 12:
        return True, 2 if is_bold else 3
    
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
def _pack_blocks_into_chunks(blocks: List[Dict], headings: List[Dict], toc_index: List[int], min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[Dict]:
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
    cur_page = 1  # Track current page number

    def close():
        nonlocal cur_parts, cur_inds, cur_block_indices, cur_tok, cur_page
        if cur_parts:
            payload = "\n".join(cur_parts).strip("\n")
            # Get breadcrumbs for the first block in this chunk
            first_block_idx = cur_block_indices[0] if cur_block_indices else 0
            breadcrumbs = _build_breadcrumbs(headings, first_block_idx)
            page_increment = payload.count("<<<PAGE_BREAK>>>")
            while cur_page + page_increment in toc_index:
                cur_page += 1
            out.append({
                "payload": payload.replace("<<<PAGE_BREAK>>>", ""),
                "indent_levels": sorted(set(cur_inds)),
                "breadcrumbs": breadcrumbs,
                "block_indices": cur_block_indices.copy(),
                "page_from": cur_page,
                "page_to": cur_page + page_increment
            })
            cur_page += page_increment
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
def pdf_to_chunks_smart_indent(
    pdf_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS
) -> List[Dict]:
    """
    - Read PDF -> text (preserve line breaks)  
    - Divide into blocks by "blank lines" + "indent change >= INDENT_STEP"  
    - Pack blocks into ~500 token chunks (within [300, 500] if possible)  
    - Automatically detect headings and create breadcrumbs  
    - Prepend heading context to the chunk content  

    """
    file_name = os.path.basename(pdf_path)
    doc_text, toc_index = _pdf_to_text_with_lines(pdf_path)
    blocks = _lines_to_blocks(doc_text)
    
    # Extract headings for breadcrumb generation
    headings = _extract_headings_from_blocks(blocks)

    packed = _pack_blocks_into_chunks(blocks, headings, toc_index, min_tokens=min_tokens, max_tokens=max_tokens)

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
            "source": os.path.abspath(pdf_path),
            "section_title": ch["breadcrumbs"][-1] if ch["breadcrumbs"] else None,
            "heading_path": breadcrumb_text if breadcrumb_text else None,
            "chunk_index": idx,
            "page_from": ch["page_from"],
            "page_to": ch["page_to"],
            "breadcrumbs": ch["breadcrumbs"]
        })
    return results

# ===================== Demo =====================
if __name__ == "__main__":
    pdf_path = "./123.pdf"
    chunks = pdf_to_chunks_smart_indent(pdf_path, min_tokens=300, max_tokens=500)

    for ch in chunks:
        print("="*120)
        print(f"File: {ch['file_name']} | Pages: {f"{ch['page_from']}-{ch['page_to']}"} | Chunk: {ch['chunk_id']} | ~tokens: {ch['token_est']}")
        print(f"Indent levels in chunk: {ch['indent_levels']}")
        print("="*10)
        print(ch['content'])
