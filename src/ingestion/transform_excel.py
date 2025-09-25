import re
import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from utils.pattern import *
from dotenv import load_dotenv
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

# ===================== Image Detection & Extraction from Excel =====================
def _extract_excel_images(excel_path: str, output_dir: str = "images") -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Extract all images from Excel file and return metadata with Cloudinary URLs
    Returns:
    - all_images: list of all image metadata
    - sheet_images: dict mapping sheet_name -> list of image metadata
    """
    all_images = []
    sheet_images = {}
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load workbook with openpyxl to access images
        workbook = openpyxl.load_workbook(excel_path)
        file_name = os.path.basename(excel_path)
        doc_id = os.path.splitext(file_name)[0]
        
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            sheet_images[sheet_name] = []
            
            # Extract images from worksheet
            if hasattr(worksheet, '_images') and worksheet._images:
                for idx, img in enumerate(worksheet._images):
                    try:
                        # Get image data
                        image_data = img._data()
                        if not image_data:
                            continue
                            
                        # Create PIL Image
                        pil_image = Image.open(io.BytesIO(image_data))
                        width, height = pil_image.size
                        
                        # Generate filename
                        img_filename = f"{doc_id}_{sheet_name}_img_{idx+1}.{pil_image.format.lower() if pil_image.format else 'png'}"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        # Save image locally
                        pil_image.save(img_path)
                        
                        # Upload to Cloudinary
                        cloudinary_url = upload_image_to_cloudinary(img_path, img_filename)
                        
                        # Get image position info
                        anchor = getattr(img, 'anchor', None)
                        cell_ref = anchor._from.row if anchor and hasattr(anchor, '_from') else 0
                        
                        # Create image metadata
                        img_metadata = {
                            "filename": img_filename,
                            "path": img_path,
                            "width": width,
                            "height": height,
                            "format": pil_image.format or "PNG",
                            "size": len(image_data),
                            "sheet": sheet_name,
                            "cell_row": cell_ref,
                            "cloudinary_url": cloudinary_url
                        }
                        
                        all_images.append(img_metadata)
                        sheet_images[sheet_name].append(img_metadata)
                        
                        print(f"Extracted image: {img_filename} from sheet '{sheet_name}' (Cloudinary: {bool(cloudinary_url)})")
                        
                    except Exception as e:
                        print(f"Error extracting image {idx+1} from sheet '{sheet_name}': {str(e)}")
                        continue
        
        workbook.close()
        print(f"Total images extracted: {len(all_images)}")
        
    except Exception as e:
        print(f"Error extracting images from Excel: {str(e)}")
    
    return all_images, sheet_images

def _get_image_context_excel(sheet_data: pd.DataFrame, row_idx: int, context_rows: int = 2) -> str:
    """
    Get textual context around an image location in Excel sheet
    """
    try:
        start_row = max(0, row_idx - context_rows)
        end_row = min(len(sheet_data), row_idx + context_rows + 1)
        
        context_lines = []
        for i in range(start_row, end_row):
            if i < len(sheet_data):
                row_data = sheet_data.iloc[i]
                # Convert row to string, filtering out NaN values
                row_text = " | ".join([str(val) for val in row_data.values if pd.notna(val) and str(val).strip()])
                if row_text:
                    context_lines.append(f"Row {i+1}: {row_text}")
        
        return " ".join(context_lines)
    except:
        return ""

# ===================== Excel -> text with structure =====================
def _excel_to_text_with_images(excel_path: str) -> Tuple[str, List[Dict], List[Dict], Dict[str, List[Dict]]]:
    """
    Convert Excel to text with images and return structured data
    Returns:
    - full_text: concatenated text from all sheets
    - sheet_info: list of sheet metadata
    - all_images: list of all image metadata  
    - sheet_images: dict mapping sheet_name -> list of image metadata
    """
    print(f"Processing Excel file: {excel_path}")
    
    # Extract images first
    all_images, sheet_images = _extract_excel_images(excel_path)
    
    # Process sheets with pandas
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        
        full_text = ""
        sheet_info = []
        
        for sheet_name in sheet_names:
            print(f"Processing sheet: {sheet_name}")
            
            # Read sheet data
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            
            # Clean the dataframe - remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                print(f"Sheet '{sheet_name}' is empty, skipping...")
                continue
            
            # Convert to text representation
            sheet_text_lines = []
            
            # Add sheet header
            sheet_text_lines.append(f"\n=== SHEET: {sheet_name} ===\n")
            
            # Process each row
            for row_idx, (_, row) in enumerate(df.iterrows()):
                # Filter out NaN values and create row text
                row_values = [str(cell) for cell in row.values if pd.notna(cell) and str(cell).strip()]
                
                if row_values:
                    row_text = " | ".join(row_values)
                    
                    # Check if this row has associated images
                    row_images = []
                    if sheet_name in sheet_images:
                        for img_meta in sheet_images[sheet_name]:
                            # Associate image if it's near this row (within 3 rows)
                            if abs(img_meta.get('cell_row', 0) - row_idx) <= 3:
                                row_images.append(img_meta)
                    
                    # Add image references to row text
                    if row_images:
                        image_context = _get_image_context_excel(df, row_idx)
                        for img_meta in row_images:
                            image_ref = f"\n[IMAGE: {img_meta['filename']} - {img_meta.get('width', 'unknown')}x{img_meta.get('height', 'unknown')}]"
                            if img_meta.get('cloudinary_url'):
                                image_ref += f"\n[IMAGE_URL: {img_meta['cloudinary_url']}]"
                            if image_context:
                                image_ref += f"\n[IMAGE_CONTEXT: {image_context[:200]}...]"
                            row_text += image_ref
                    
                    sheet_text_lines.append(f"Row {row_idx + 1}: {row_text}")
            
            # Join sheet text
            sheet_text = "\n".join(sheet_text_lines)
            
            # Add to full text
            full_text += sheet_text + "\n\n"
            
            # Store sheet metadata
            sheet_info.append({
                "sheet_name": sheet_name,
                "row_count": len(df),
                "col_count": len(df.columns),
                "has_images": len(sheet_images.get(sheet_name, [])) > 0,
                "image_count": len(sheet_images.get(sheet_name, [])),
                "images": sheet_images.get(sheet_name, []),
                "text": sheet_text
            })
            
            print(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns, {len(sheet_images.get(sheet_name, []))} images")
        
        excel_file.close()
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return "", [], all_images, sheet_images
    
    print(f"Excel processing complete. Total images: {len(all_images)}")
    return full_text, sheet_info, all_images, sheet_images

# ===================== Sheet structure analysis =====================
def _detect_excel_structure(sheet_info: List[Dict]) -> List[Dict]:
    """
    Analyze Excel structure to identify headers, data sections, and summaries
    """
    structured_blocks = []
    
    for sheet_meta in sheet_info:
        sheet_name = sheet_meta["sheet_name"]
        sheet_text = sheet_meta["text"]
        
        if not sheet_text.strip():
            continue
        
        # Split into lines for analysis
        lines = [line.strip() for line in sheet_text.split('\n') if line.strip()]
        
        current_block = {
            "type": "data_section",
            "sheet": sheet_name,
            "content": "",
            "heading_level": 0,
            "indent_level": 0,
            "has_images": sheet_meta["has_images"],
            "images": sheet_meta.get("images", []),
            "image_urls": [img.get("cloudinary_url", "") for img in sheet_meta.get("images", []) if img.get("cloudinary_url")],
            "image_count": sheet_meta["image_count"],
            "token_count": 0
        }
        
        for line in lines:
            # Detect different types of Excel content
            if line.startswith("=== SHEET:"):
                # Sheet header
                if current_block["content"]:
                    current_block["token_count"] = _tok_count(current_block["content"])
                    structured_blocks.append(current_block)
                
                current_block = {
                    "type": "sheet_header",
                    "sheet": sheet_name,
                    "content": line,
                    "heading_level": 1,
                    "indent_level": 0,
                    "has_images": False,
                    "images": [],
                    "image_urls": [],
                    "image_count": 0,
                    "token_count": _tok_count(line)
                }
                structured_blocks.append(current_block)
                
                # Reset for data content
                current_block = {
                    "type": "data_section",
                    "sheet": sheet_name,
                    "content": "",
                    "heading_level": 0,
                    "indent_level": 0,
                    "has_images": sheet_meta["has_images"],
                    "images": sheet_meta.get("images", []),
                    "image_urls": [img.get("cloudinary_url", "") for img in sheet_meta.get("images", []) if img.get("cloudinary_url")],
                    "image_count": sheet_meta["image_count"],
                    "token_count": 0
                }
                
            elif line.startswith("Row "):
                # Data row
                current_block["content"] += line + "\n"
            
            else:
                # Other content
                current_block["content"] += line + "\n"
        
        # Add final block
        if current_block["content"]:
            current_block["token_count"] = _tok_count(current_block["content"])
            structured_blocks.append(current_block)
    
    return structured_blocks

# ===================== Excel blocks to chunks =====================
def _pack_excel_blocks_into_chunks(blocks: List[Dict], min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> List[Dict]:
    """
    Pack Excel blocks into chunks with proper token management
    """
    if not blocks:
        return []
    
    chunks = []
    current_chunk = {
        "content": "",
        "token_count": 0,
        "sheets": set(),
        "has_images": False,
        "images": [],
        "image_urls": [],
        "image_count": 0,
        "metadata": {
            "block_types": [],
            "sheet_names": []
        }
    }
    
    for block in blocks:
        block_tokens = block.get("token_count", _tok_count(block["content"]))
        
        # If adding this block would exceed max_tokens and current chunk has content
        if (current_chunk["token_count"] + block_tokens > max_tokens and 
            current_chunk["content"] and 
            current_chunk["token_count"] >= min_tokens):
            
            # Finalize current chunk
            _finalize_excel_chunk(current_chunk, chunks)
            
            # Start new chunk
            current_chunk = {
                "content": "",
                "token_count": 0,
                "sheets": set(),
                "has_images": False,
                "images": [],
                "image_urls": [],
                "image_count": 0,
                "metadata": {
                    "block_types": [],
                    "sheet_names": []
                }
            }
        
        # Add block to current chunk
        if current_chunk["content"]:
            current_chunk["content"] += "\n\n"
            
        current_chunk["content"] += block["content"]
        current_chunk["token_count"] += block_tokens
        current_chunk["sheets"].add(block["sheet"])
        current_chunk["metadata"]["block_types"].append(block["type"])
        
        # Handle images
        if block.get("has_images", False):
            current_chunk["has_images"] = True
            current_chunk["images"].extend(block.get("images", []))
            current_chunk["image_urls"].extend(block.get("image_urls", []))
            current_chunk["image_count"] += block.get("image_count", 0)
    
    # Add final chunk
    if current_chunk["content"]:
        _finalize_excel_chunk(current_chunk, chunks)
    
    return chunks

def _finalize_excel_chunk(chunk: Dict, chunks: List[Dict]) -> None:
    """
    Finalize an Excel chunk and add it to the chunks list
    """
    # Convert sets to lists for JSON serialization
    chunk["metadata"]["sheet_names"] = list(chunk["sheets"])
    del chunk["sheets"]
    
    # Remove duplicate image URLs
    chunk["image_urls"] = list(set(chunk["image_urls"]))
    
    # Add chunk metadata
    chunk["metadata"]["source_type"] = "excel"
    chunk["metadata"]["has_data"] = any("data_section" in bt for bt in chunk["metadata"]["block_types"])
    chunk["metadata"]["has_headers"] = any("sheet_header" in bt for bt in chunk["metadata"]["block_types"])
    
    chunks.append(chunk)

# ===================== Main Excel processing function =====================
def excel_to_chunks_smart_indent(
    excel_path: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    extract_images: bool = True
) -> List[Dict]:
    """
    Enhanced Excel processor with image detection and extraction
    - Read Excel -> structured text (preserve sheet and cell structure)
    - Analyze sheets and detect data patterns
    - Extract and upload images to Cloudinary
    - Pack into chunks with proper token management
    - Include sheet metadata and image references
    """
    file_name = os.path.basename(excel_path)
    doc_id = os.path.splitext(file_name)[0]
    
    print(f"\n{'='*50}")
    print(f"Processing Excel: {file_name}")
    print(f"Extract images: {extract_images}")
    print(f"Token range: [{min_tokens}, {max_tokens}]")
    print(f"{'='*50}")
    
    try:
        # Convert Excel to structured text with images
        if extract_images:
            full_text, sheet_info, all_images, sheet_images = _excel_to_text_with_images(excel_path)
        else:
            # Simple processing without images
            excel_file = pd.ExcelFile(excel_path)
            sheet_names = excel_file.sheet_names
            
            full_text = ""
            sheet_info = []
            all_images = []
            sheet_images = {}
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                if not df.empty:
                    sheet_text = f"\n=== SHEET: {sheet_name} ===\n"
                    for row_idx, (_, row) in enumerate(df.iterrows()):
                        row_values = [str(cell) for cell in row.values if pd.notna(cell) and str(cell).strip()]
                        if row_values:
                            sheet_text += f"Row {row_idx + 1}: {' | '.join(row_values)}\n"
                    
                    full_text += sheet_text + "\n\n"
                    sheet_info.append({
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                        "col_count": len(df.columns),
                        "has_images": False,
                        "image_count": 0,
                        "images": [],
                        "text": sheet_text
                    })
            
            excel_file.close()
        
        if not full_text.strip():
            print("No content found in Excel file")
            return []
        
        # Detect Excel structure
        blocks = _detect_excel_structure(sheet_info)
        print(f"Created {len(blocks)} structural blocks")
        
        # Pack blocks into chunks
        chunks = _pack_excel_blocks_into_chunks(blocks, min_tokens, max_tokens)
        print(f"Packed into {len(chunks)} chunks")
        
        # Add final metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.update({
                "chunk_id": f"{doc_id}_chunk_{i+1}",
                "file_name": file_name,  # Add this missing field
                "source_file": file_name,
                "source_type": "excel",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_est": chunk.get("token_count", 0),  # Add token_est field
                "indent_levels": [],  # Add for consistency with other transforms
                "processing_metadata": {
                    "total_sheets": len(sheet_info),
                    "total_images": len(all_images),
                    "extraction_method": "smart_indent_with_images" if extract_images else "smart_indent",
                    "sheet_names": [s["sheet_name"] for s in sheet_info]
                }
            })
        
        # Summary
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        image_chunks = sum(1 for chunk in chunks if chunk.get("has_images", False))
        
        print(f"\n📊 Excel Processing Summary:")
        print(f"  📄 File: {file_name}")
        print(f"  📑 Sheets processed: {len(sheet_info)}")
        print(f"  🖼️  Total images: {len(all_images)}")
        print(f"  📦 Total chunks: {len(chunks)}")
        print(f"  🖼️  Chunks with images: {image_chunks}")
        print(f"  🔤 Total tokens: {total_tokens:,}")
        print(f"  📊 Avg tokens/chunk: {total_tokens/len(chunks):.1f}")
        print(f"  ✅ Processing complete!")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing Excel file: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# ===================== Utility functions =====================
def get_excel_info(excel_path: str) -> Dict:
    """
    Get basic information about an Excel file
    """
    try:
        excel_file = pd.ExcelFile(excel_path)
        sheet_info = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            sheet_info.append({
                "name": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "is_empty": df.empty
            })
        
        excel_file.close()
        
        return {
            "file_path": excel_path,
            "file_name": os.path.basename(excel_path),
            "total_sheets": len(sheet_info),
            "sheets": sheet_info
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "file_path": excel_path
        }

if __name__ == "__main__":
    # Test function
    test_file = "test.xlsx"
    if os.path.exists(test_file):
        chunks = excel_to_chunks_smart_indent(test_file)
        print(f"Generated {len(chunks)} chunks")
        if chunks:
            print("First chunk preview:")
            print(chunks[0]["content"][:200] + "...")