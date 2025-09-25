import re
import fitz  # PyMuPDF
import os
from typing import List, Dict, Tuple, Optional
from utils.pattern import *
from dotenv import load_dotenv
import tiktoken
from PIL import Image
import io
import json
from utils.upload_image import upload_image_to_cloudinary

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
    return max(1, len(s) / 4)

def _tok_count(s: str) -> int:
    global _tokenc
    if USE_TIKTOKEN:
        try:
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
            'table of contents', 'mục lục'
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

# ===================== Image Detection & Extraction =====================
def _extract_page_images(page, page_num: int, doc_id: str, output_dir: str = "images") -> List[Dict]:
    """
    Extract all images from a specific page and return metadata with Cloudinary URLs
    """
    images_dir = os.path.join(output_dir, f"image_{doc_id}")
    os.makedirs(images_dir, exist_ok=True)
    
    extracted_images = []
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Create filename
            image_filename = f"page_{page_num}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save image locally
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Get image dimensions
            try:
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    width, height = pil_img.size
                    mode = pil_img.mode
            except:
                width = height = mode = None
            
            # Upload to Cloudinary
            cloudinary_url = ""
            try:
                cloudinary_url = upload_image_to_cloudinary(image_path)
                print(f"Uploaded to Cloudinary: {cloudinary_url}")
            except Exception as e:
                print(f"Failed to upload {image_filename} to Cloudinary: {e}")
            
            # Get image position on page
            img_rect = None
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 1 and block.get("number") == xref:
                    img_rect = {
                        "x0": block["bbox"][0],
                        "y0": block["bbox"][1], 
                        "x1": block["bbox"][2],
                        "y1": block["bbox"][3]
                    }
                    break
            
            image_metadata = {
                "image_id": f"{doc_id}_page_{page_num}_img_{img_index + 1}",
                "filename": image_filename,
                "file_path": os.path.abspath(image_path),
                "cloudinary_url": cloudinary_url,  # NEW: Cloudinary URL
                "page_number": page_num,
                "image_index": img_index + 1,
                "format": image_ext.upper(),
                "width": width,
                "height": height,
                "mode": mode,
                "size_bytes": len(image_bytes),
                "position": img_rect
            }
            
            extracted_images.append(image_metadata)
            print(f"Extracted: {image_filename} ({width}x{height}) -> {cloudinary_url}")
            
        except Exception as e:
            print(f"Error extracting image {img_index + 1} from page {page_num}: {e}")
            continue
    
    return extracted_images

def _get_image_context(page, image_rect: Dict, context_margin: int = 50) -> str:
    """
    Extract text context around an image position
    """
    if not image_rect:
        return ""
    
    # Get all text blocks
    blocks = page.get_text("dict")["blocks"]
    context_text = []
    
    img_y0 = image_rect["y0"]
    img_y1 = image_rect["y1"]
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            block_rect = block["bbox"]
            block_y0, block_y1 = block_rect[1], block_rect[3]
            
            # Check if text is near image (above or below)
            if (abs(block_y1 - img_y0) < context_margin or  # Text above image
                abs(block_y0 - img_y1) < context_margin):   # Text below image
                
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                    block_text += "\n"
                context_text.append(block_text.strip())
    
    return "\n\n".join(context_text)

# ===================== PDF -> text with image detection =====================
def _pdf_to_text_with_images(pdf_path: str) -> Tuple[str, List[int], Dict[int, List[Dict]]]:
    """
    Returns:
    - doc_text: full document text
    - toc_index: list of TOC page numbers
    - page_images: dict mapping page_num -> list of image metadata
    """
    doc = fitz.open(pdf_path)
    pieces = []
    toc_index = []
    page_images = {}
    toc_pages_found = 0
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {len(doc)}")
    
    for pno in range(len(doc)):
        page = doc[pno]
        page_num = pno + 1
        
        # Get page text
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1],1), round(b[0],1)))
        page_text = "\n".join(
            [blk[4] for blk in blocks if blk[4] and blk[4].strip() != ""]
        )
        
        # Check if this page is TOC and skip it
        if _is_toc_page(page_text):
            print(f"Skipping TOC page {page_num}")
            toc_index.append(page_num)
            toc_pages_found += 1
            continue
            
        if len(page_text) < 3:
            print(f"Skipping unmeaningful page {page_num}")
            toc_index.append(page_num)
            toc_pages_found += 1
            continue
        
        # Extract images from this page
        images_on_page = _extract_page_images(page, page_num, doc_id)
        if images_on_page:
            page_images[page_num] = images_on_page
            print(f"Found {len(images_on_page)} images on page {page_num}")
            
            # Add image context to page text
            for img_meta in images_on_page:
                if img_meta.get("position"):
                    img_context = _get_image_context(page, img_meta["position"])
                    if img_context:
                        # Add image reference to text
                        image_ref = f"\n[IMAGE: {img_meta['filename']} - {img_meta['width']}x{img_meta['height']}]\n"
                        if img_meta.get('cloudinary_url'):
                            image_ref += f"[IMAGE_URL: {img_meta['cloudinary_url']}]\n"
                        if img_context:
                            image_ref += f"[IMAGE_CONTEXT: {img_context[:200]}...]\n"
                        page_text += image_ref
        
        pieces.append(page_text)
    
    doc.close()
    
    if toc_pages_found > 0:
        print(f"Filtered out {toc_pages_found} TOC pages from {len(doc)} total pages")
    
    total_images = sum(len(imgs) for imgs in page_images.values())
    if total_images > 0:
        print(f"Extracted {total_images} images from {len(page_images)} pages")
        
        # Save images metadata
        images_dir = os.path.join("images", f"image_{doc_id}")
        metadata_file = os.path.join(images_dir, "images_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(page_images, f, indent=2, ensure_ascii=False)
        print(f"Images metadata saved to {metadata_file}")
    
    return "\n\n<<<PAGE_BREAK>>>".join(pieces), toc_index, page_images

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

# ===================== Enhanced chunk creation with image info =====================
def _pack_blocks_with_images(blocks: List[Dict], headings: List[Dict], toc_index: List[int], 
                            page_images: Dict[int, List[Dict]], min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS) -> List[Dict]:
    """
    Enhanced version that includes image information in chunks
    """
    out: List[Dict] = []
    cur_parts: List[str] = []
    cur_inds: List[int] = []
    cur_block_indices: List[int] = []
    cur_tok = 0
    cur_page = 1

    def close():
        nonlocal cur_parts, cur_inds, cur_block_indices, cur_tok, cur_page
        if cur_parts:
            payload = "\n".join(cur_parts).strip("\n")
            first_block_idx = cur_block_indices[0] if cur_block_indices else 0
            breadcrumbs = _build_breadcrumbs(headings, first_block_idx)
            page_increment = payload.count("<<<PAGE_BREAK>>>")
            
            # Collect images for this chunk
            chunk_images = []
            chunk_image_urls = []
            for page_num in range(cur_page, cur_page + page_increment + 1):
                if page_num in page_images:
                    for img in page_images[page_num]:
                        chunk_images.append(img)
                        if img.get('cloudinary_url'):
                            chunk_image_urls.append(img['cloudinary_url'])
            
            while cur_page + page_increment in toc_index:
                cur_page += 1
                
            out.append({
                "payload": payload.replace("<<<PAGE_BREAK>>>", ""),
                "indent_levels": sorted(set(cur_inds)),
                "breadcrumbs": breadcrumbs,
                "block_indices": cur_block_indices.copy(),
                "page_from": cur_page,
                "page_to": cur_page + page_increment,
                "images": chunk_images,  # NEW: Include images in chunk
                "image_urls": chunk_image_urls,  # NEW: Cloudinary URLs array
                "has_images": len(chunk_images) > 0  # NEW: Flag for quick check
            })
            
            cur_page += page_increment
            cur_parts = []
            cur_inds = []
            cur_block_indices = []
            cur_tok = 0

    # Rest of the logic remains the same as _pack_blocks_into_chunks
    for block_idx, blk in enumerate(blocks):
        txt = blk["text"]
        ind = blk["indent"]
        t = _tok_count(txt)

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

        if cur_tok + t <= max_tokens:
            cur_parts.append(txt)
            cur_inds.append(ind)
            cur_block_indices.append(block_idx)
            cur_tok += t
        else:
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
def pdf_to_chunks_smart_indent(
    pdf_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    extract_images: bool = True  # NEW parameter
) -> List[Dict]:
    """
    Enhanced version with image detection and extraction
    - Read PDF -> text (preserve line breaks)  
    - Divide into blocks by "blank lines" + "indent change >= INDENT_STEP"  
    - Pack blocks into ~500 token chunks (within [300, 500] if possible)  
    - Automatically detect headings and create breadcrumbs  
    - Prepend heading context to the chunk content  
    - NEW: Detect and extract images, upload to Cloudinary

    """
    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]
    
    if extract_images:
        # Use enhanced text extraction with image detection
        doc_text, toc_index, page_images = _pdf_to_text_with_images(pdf_path)
        blocks = _lines_to_blocks(doc_text)
        headings = _extract_headings_from_blocks(blocks)
        
        # Use enhanced packing with image info
        packed = _pack_blocks_with_images(blocks, headings, toc_index, page_images, min_tokens, max_tokens)
    else:
        # Original logic without images
        doc_text, toc_index = _pdf_to_text_with_lines(pdf_path)
        blocks = _lines_to_blocks(doc_text)
        headings = _extract_headings_from_blocks(blocks)
        packed = _pack_blocks_into_chunks(blocks, headings, toc_index, min_tokens, max_tokens)

    results = []
    
    for idx, ch in enumerate(packed):
        breadcrumb_text = ""
        if ch["breadcrumbs"]:
            breadcrumb_text = " > ".join(ch["breadcrumbs"])
        
        content_with_context = ch["payload"]
        if breadcrumb_text:
            content_with_context = f"[Context: {breadcrumb_text}]\n\n{ch['payload']}"
        
        # Enhanced metadata with image information
        chunk_data = {
            "chunk_id": idx,
            "file_name": file_name,
            "indent_levels": ch["indent_levels"],
            "content": content_with_context,
            "token_est": _tok_count(content_with_context),
            "doc_id": doc_id,
            "source": os.path.abspath(pdf_path),
            "section_title": ch["breadcrumbs"][-1] if ch["breadcrumbs"] else None,
            "heading_path": breadcrumb_text if breadcrumb_text else None,
            "chunk_index": idx,
            "page_from": ch["page_from"],
            "page_to": ch["page_to"],
            "breadcrumbs": ch["breadcrumbs"]
        }
        
        # Add image-related metadata if available
        if extract_images and ch.get("has_images"):
            chunk_data.update({
                "has_images": ch["has_images"],
                "images": ch["images"],
                "image_urls": ch["image_urls"],  # Array of Cloudinary URLs
                "image_count": len(ch["images"]),
                "image_filenames": [img["filename"] for img in ch["images"]]
            })
        else:
            chunk_data.update({
                "has_images": False,
                "images": [],
                "image_urls": [],
                "image_count": 0,
                "image_filenames": []
            })
        
        results.append(chunk_data)
    
    # Print summary
    if extract_images:
        total_chunks = len(results)
        chunks_with_images = sum(1 for r in results if r["has_images"])
        total_images = sum(r["image_count"] for r in results)
        
        print(f"\nCHUNKING SUMMARY:")
        print(f"Total chunks: {total_chunks}")
        print(f"Chunks with images: {chunks_with_images}")
        print(f"Total images detected: {total_images}")
    
    return results

# ===================== Utility function to get image info =====================
def get_chunk_images_info(chunks: List[Dict]) -> Dict:
    """
    Get summary of images across all chunks
    """
    image_summary = {
        "total_chunks": len(chunks),
        "chunks_with_images": 0,
        "total_images": 0,
        "images_by_page": {},
        "image_formats": {},
        "large_images": [],  # Images > 1MB
        "cloudinary_urls": []  # All Cloudinary URLs
    }
    
    for chunk in chunks:
        if chunk.get("has_images"):
            image_summary["chunks_with_images"] += 1
            
            # Add Cloudinary URLs
            image_summary["cloudinary_urls"].extend(chunk.get("image_urls", []))
            
            for img in chunk.get("images", []):
                image_summary["total_images"] += 1
                
                # Count by page
                page = img["page_number"]
                if page not in image_summary["images_by_page"]:
                    image_summary["images_by_page"][page] = 0
                image_summary["images_by_page"][page] += 1
                
                # Count by format
                fmt = img.get("format", "UNKNOWN")
                if fmt not in image_summary["image_formats"]:
                    image_summary["image_formats"][fmt] = 0
                image_summary["image_formats"][fmt] += 1
                
                # Track large images
                if img.get("size_bytes", 0) > 1024 * 1024:  # > 1MB
                    image_summary["large_images"].append({
                        "filename": img["filename"],
                        "cloudinary_url": img.get("cloudinary_url", ""),
                        "size_mb": round(img["size_bytes"] / (1024 * 1024), 2),
                        "dimensions": f"{img.get('width', 'unknown')}x{img.get('height', 'unknown')}"
                    })
    
    return image_summary
