import re
import os
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
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

# ===================== Roman numeral utilities =====================
def _roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer"""
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
    }
    result = 0
    prev_value = 0
    
    for char in reversed(roman.upper()):
        if char not in roman_values:
            return 0
        value = roman_values[char]
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result

def _is_valid_roman(roman: str) -> bool:
    """Check if string is a valid Roman numeral"""
    if not roman:
        return False
    pattern = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(pattern, roman.upper()))

# ===================== MD -> text (clean images and preserve structure) =====================
def _md_to_text_with_structure(md_path: str) -> str:
    """
    Read MD file, remove images, and return content with structure preserved
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        content = None
        for encoding in encodings:
            try:
                with open(md_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file {md_path} with any common encoding")
    
    # Remove images: ![alt text](url) or ![alt text][ref] or ![](url)
    content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)  # ![alt](url)
    content = re.sub(r'!\[([^\]]*)\]\[[^\]]+\]', '', content)  # ![alt][ref]
    content = re.sub(r'!\[\]\([^\)]+\)', '', content)  # ![](url)
    
    # Clean up multiple empty lines left by image removal
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    return content

# ===================== Enhanced heading detection =====================
def _is_heading(line: str) -> Tuple[bool, int]:
    """
    Enhanced heading detection for Markdown with Roman numerals and Arabic numbers
    Returns: (is_heading, level)
    """
    line_clean = line.strip()
    if not line_clean or len(line_clean) < 2:
        return False, 0
    
    # 1. Markdown headers (# ## ### etc.)
    md_header_match = re.match(r'^(#{1,6})\s+(.+)$', line_clean)
    if md_header_match:
        return True, len(md_header_match.group(1))
    
    # 2. Roman numerals (I, II, III, IV, V, etc.)
    roman_match = re.match(r'^\s*([IVXLCDM]+)[\.\s]+(.+)$', line_clean)
    if roman_match:
        roman = roman_match.group(1)
        if _is_valid_roman(roman):
            roman_value = _roman_to_int(roman)
            # Level 1 for main sections (I, II, III...)
            return True, 1
    
    # 3. Arabic numbers (1, 2, 3, 1.1, 1.2, etc.)
    numbered_match = re.match(r'^\s*(\d+(?:\.\d+)*)[\.)\s]+(.+)$', line_clean)
    if numbered_match:
        number_part = numbered_match.group(1)
        dots = number_part.count('.')
        # Level based on depth: 1. = level 2, 1.1. = level 3, etc.
        return True, min(dots + 2, 6)
    
    # 4. Underlined headings (setext style)
    # This would need the next line, so we'll handle it separately
    
    # 5. All caps short lines (likely section headers)
    if line_clean.isupper() and len(line_clean) < 80 and not line_clean.endswith('.'):
        return True, 2
    
    # 6. Traditional patterns from utils.pattern
    for i, pattern in enumerate(HEADING_PATTERNS):
        if pattern.match(line_clean):
            if i == 0:  # numbered pattern already handled above
                continue
            level = 1 if i == 1 else (2 if len(line_clean) < 50 else 3)
            return True, level
    
    return False, 0

def _detect_setext_headings(lines: List[str]) -> Dict[int, Tuple[bool, int]]:
    """
    Detect setext-style headings (underlined with = or -)
    Returns dict mapping line_index -> (is_heading, level)
    """
    setext_headings = {}
    
    for i in range(len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        
        # Skip empty lines
        if not current_line or not next_line:
            continue
            
        # Check if next line is underline
        if re.match(r'^=+\s*$', next_line):  # === underline = level 1
            setext_headings[i] = (True, 1)
        elif re.match(r'^-+\s*$', next_line):  # --- underline = level 2
            setext_headings[i] = (True, 2)
    
    return setext_headings

# ===================== Cached breadcrumb building =====================
@lru_cache(maxsize=1000)
def _build_breadcrumb_path(headings_tuple: Tuple, current_block_index: int) -> Tuple[str, ...]:
    """
    Build breadcrumb path with caching for better performance
    headings_tuple: tuple of (text, level, block_index) for cacheable input
    """
    breadcrumbs = []
    current_levels = {}  # level -> heading text
    
    # Convert back to list of dicts and find relevant headings
    headings = [{"text": h[0], "level": h[1], "block_index": h[2]} for h in headings_tuple]
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
    
    return tuple(breadcrumbs)

def _extract_headings_from_blocks(blocks: List[Dict], lines: List[str]) -> List[Dict]:
    """
    Extract headings from blocks with enhanced detection
    """
    headings = []
    setext_headings = _detect_setext_headings(lines)
    line_to_block = {}  # Map line index to block index
    
    # Build line to block mapping
    current_line = 0
    for block_idx, block in enumerate(blocks):
        block_lines = block["text"].split('\n')
        for _ in block_lines:
            line_to_block[current_line] = block_idx
            current_line += 1
    
    # Check each block for headings
    for block_idx, block in enumerate(blocks):
        block_lines = block["text"].split('\n')
        
        for line_idx, line in enumerate(block_lines):
            # Check for regular heading patterns
            is_head, level = _is_heading(line.strip())
            if is_head:
                headings.append({
                    "text": line.strip(),
                    "level": level,
                    "block_index": block_idx,
                    "indent": block["indent"]
                })
                break  # Only take first heading from each block
        
        # Also check for setext headings
        for line_idx, (is_head, level) in setext_headings.items():
            if line_idx in line_to_block and line_to_block[line_idx] == block_idx:
                headings.append({
                    "text": lines[line_idx].strip(),
                    "level": level,
                    "block_index": block_idx,
                    "indent": block["indent"]
                })
                break
    
    return headings

def _build_breadcrumbs(headings: List[Dict], current_block_index: int) -> List[str]:
    """
    Build breadcrumbs using cached function
    """
    # Convert to tuple for caching
    headings_tuple = tuple((h["text"], h["level"], h["block_index"]) for h in headings)
    breadcrumb_tuple = _build_breadcrumb_path(headings_tuple, current_block_index)
    return list(breadcrumb_tuple)

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
    Enhanced for Markdown with code blocks and lists.
    """
    # preserve line breaks when there are bullets or code blocks
    parts: List[str] = []
    in_code_block = False
    
    for line in blk_text.splitlines():
        line = line.rstrip()
        if not line:
            continue
            
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            parts.append(line)
            continue
            
        # Don't split code blocks or bullet points
        if in_code_block or BULLET_PATTERN.match(line):
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
def md_to_chunks_smart_indent(
    md_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS
) -> List[Dict]:
    """
    - Read MD -> text (remove images, preserve structure)  
    - Divide into blocks by "blank lines" + "indent change >= INDENT_STEP"  
    - Pack blocks into ~500 token chunks (within [300, 500] if possible)  
    - Enhanced heading detection for Roman numerals (I, II, III) and Arabic numbers (1, 2, 3)
    - Use caching for better breadcrumb building performance
    - Prepend heading context to the chunk content  
    """
    file_name = os.path.basename(md_path)
    doc_text = _md_to_text_with_structure(md_path)
    lines = doc_text.splitlines()
    blocks = _lines_to_blocks(doc_text)
    
    # Extract headings with enhanced detection
    headings = _extract_headings_from_blocks(blocks, lines)

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
            "source": os.path.abspath(md_path),
            "section_title": ch["breadcrumbs"][-1] if ch["breadcrumbs"] else None,
            "heading_path": breadcrumb_text if breadcrumb_text else None,
            "chunk_index": idx,
            "breadcrumbs": ch["breadcrumbs"]
        })
    return results
