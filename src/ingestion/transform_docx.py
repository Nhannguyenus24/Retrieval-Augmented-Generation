import re
import os
from typing import List, Dict, Tuple, Optional
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
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

# ===================== DOCX -> text (preserve structure) =====================
def _docx_to_text_with_structure(docx_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text from DOCX with paragraph structure preserved
    Returns:
    - doc_text: full document text with paragraph breaks
    - paragraph_info: list of dicts with paragraph metadata
    """
    doc = Document(docx_path)
    pieces = []
    paragraph_info = []
    
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if not text:
            continue
            
        # Get style information
        style_name = paragraph.style.name if paragraph.style else "Normal"
        
        # Detect heading level from style
        heading_level = 0
        if "Heading" in style_name:
            try:
                heading_level = int(re.search(r'\d+', style_name).group())
            except:
                heading_level = 1
        
        # Calculate indent level (approximation from left indent)
        indent_level = 0
        if paragraph.paragraph_format.left_indent:
            # Convert to points and estimate indent level
            indent_points = paragraph.paragraph_format.left_indent.pt
            indent_level = int(indent_points / 18)  # Approximate: 18pt = 1 indent level
        
        paragraph_info.append({
            "index": i,
            "text": text,
            "style": style_name,
            "heading_level": heading_level,
            "indent": indent_level,
            "is_heading": heading_level > 0
        })
        
        pieces.append(text)
    
    return "\n\n".join(pieces), paragraph_info

# ===================== Heading detection for DOCX =====================
def _is_docx_heading(para_info: Dict) -> Tuple[bool, int]:
    """
    Detect if a paragraph is a heading based on DOCX metadata
    Returns: (is_heading, level)
    """
    # First check style-based headings
    if para_info["is_heading"] and para_info["heading_level"] > 0:
        return True, para_info["heading_level"]
    
    # Fallback to pattern-based detection
    text = para_info["text"].strip()
    if not text or len(text) < 3:
        return False, 0
    
    # Check numbered headings (1., 1.1., 1.1.1.)
    numbered_match = re.match(r"^\s*(\d+\.)+\s+(.+)$", text)
    if numbered_match:
        dots = numbered_match.group(1).count('.')
        return True, min(dots, 6)
    
    # Check other patterns from utils.pattern
    for i, pattern in enumerate(HEADING_PATTERNS):
        if pattern.match(text):
            if i == 0:  # numbered pattern already handled
                continue
            # ALL CAPS = level 1, others = level 2-3
            level = 1 if i == 1 else (2 if len(text) < 50 else 3)
            return True, level
    
    return False, 0

def _extract_headings_from_paragraphs(paragraph_info: List[Dict]) -> List[Dict]:
    """
    Extract headings from paragraph info and create heading hierarchy
    """
    headings = []
    
    for para in paragraph_info:
        is_head, level = _is_docx_heading(para)
        if is_head:
            headings.append({
                "text": para["text"],
                "level": level,
                "paragraph_index": para["index"],
                "indent": para["indent"]
            })
    
    return headings

def _build_breadcrumbs_docx(headings: List[Dict], current_para_index: int) -> List[str]:
    """
    Build breadcrumb path for a given paragraph based on preceding headings
    """
    breadcrumbs = []
    current_levels = {}  # level -> heading text
    
    # Find headings that precede current paragraph
    relevant_headings = [h for h in headings if h["paragraph_index"] < current_para_index]
    
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

# ===================== Convert paragraphs to blocks =====================
def _paragraphs_to_blocks(paragraph_info: List[Dict]) -> List[Dict]:
    """
    Convert paragraph info into blocks based on indent and content similarity
    """
    if not paragraph_info:
        return []
    
    blocks = []
    current_block = []
    current_indent = paragraph_info[0]["indent"] if paragraph_info else 0
    
    def flush_block():
        nonlocal current_block, current_indent
        if current_block:
            combined_text = "\n".join([p["text"] for p in current_block])
            blocks.append({
                "indent": current_indent,
                "text": combined_text,
                "paragraph_indices": [p["index"] for p in current_block]
            })
            current_block = []
    
    for para in paragraph_info:
        text = para["text"]
        indent = para["indent"]
        
        # Skip empty paragraphs
        if not text.strip():
            continue
        
        # If indent changes significantly, flush current block
        if abs(indent - current_indent) >= INDENT_STEP:
            flush_block()
            current_indent = indent
            current_block = [para]
        else:
            current_block.append(para)
    
    # Flush remaining block
    flush_block()
    return blocks

# ===================== Split large blocks by sentences =====================
def _split_docx_block_into_chunks(blk_text: str, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[str]:
    """
    Split a DOCX block into chunks if it's too large
    """
    # Split by paragraphs first, then by sentences if needed
    paragraphs = blk_text.split('\n')
    parts: List[str] = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if it's a bullet point
        if BULLET_PATTERN.match(para):
            parts.append(para)
        else:
            # Split by sentences for regular paragraphs
            segs = SPLIT_PATTERN.split(para)
            for s in segs:
                s = s.strip()
                if s:
                    parts.append(s)
    
    # Now group parts into chunks
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    
    for p in parts:
        t = _tok_count(p)
        if t > max_tokens:
            # Single part too long: push separately
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
    
    # Merge small final chunk if possible
    if len(chunks) >= 2 and _tok_count(chunks[-1]) < min_tokens:
        if _tok_count(chunks[-2]) + _tok_count(chunks[-1]) <= max_tokens:
            chunks[-2] = (chunks[-2] + "\n" + chunks[-1]).strip("\n")
            chunks.pop()
    
    return chunks

# ===================== Pack blocks into chunks =====================
def _pack_docx_blocks_into_chunks(blocks: List[Dict], headings: List[Dict], min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[Dict]:
    """
    Group DOCX blocks into ~500 token chunks
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
            first_para_idx = blocks[first_block_idx]["paragraph_indices"][0] if first_block_idx < len(blocks) and blocks[first_block_idx]["paragraph_indices"] else 0
            breadcrumbs = _build_breadcrumbs_docx(headings, first_para_idx)
            
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
        
        # If block is too large, split it
        if t > max_tokens:
            subchunks = _split_docx_block_into_chunks(txt, min_tokens, max_tokens)
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
        
        # Regular sized block
        if cur_tok + t <= max_tokens:
            cur_parts.append(txt)
            cur_inds.append(ind)
            cur_block_indices.append(block_idx)
            cur_tok += t
        else:
            # Close current chunk and start new one
            close()
            cur_parts.append(txt)
            cur_inds.append(ind)
            cur_block_indices.append(block_idx)
            cur_tok = t
            if cur_tok >= min_tokens:
                close()
    
    close()
    return out

# ===================== Main API =====================
def docx_to_chunks_smart_indent(
    docx_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS
) -> List[Dict]:
    """
    - Read DOCX -> text (preserve paragraph structure)
    - Divide into blocks by indent and content similarity
    - Pack blocks into ~500 token chunks (within [300, 500] if possible)
    - Automatically detect headings and create breadcrumbs
    - Prepend heading context to the chunk content
    """
    file_name = os.path.basename(docx_path)
    doc_text, paragraph_info = _docx_to_text_with_structure(docx_path)
    blocks = _paragraphs_to_blocks(paragraph_info)
    
    # Extract headings for breadcrumb generation
    headings = _extract_headings_from_paragraphs(paragraph_info)
    
    packed = _pack_docx_blocks_into_chunks(blocks, headings, min_tokens=min_tokens, max_tokens=max_tokens)
    
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
            "source": os.path.abspath(docx_path),
            "section_title": ch["breadcrumbs"][-1] if ch["breadcrumbs"] else None,
            "heading_path": breadcrumb_text if breadcrumb_text else None,
            "chunk_index": idx,
            "breadcrumbs": ch["breadcrumbs"]
        })
    
    return results
