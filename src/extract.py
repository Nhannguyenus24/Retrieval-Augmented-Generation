import os
from PIL import Image
from typing import List, Dict, Optional, Set
import io
import fitz
def extract_images_with_context(pdf_path: str, output_dir: str = "images") -> List[Dict]:
    """
    Extract images with surrounding text context for better understanding
    """
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]
    
    # Create output directory
    images_dir = os.path.join(output_dir, f"anh_{doc_id}")
    os.makedirs(images_dir, exist_ok=True)
    
    extracted_images = []
    image_count = 0
    
    print(f"Extracting images with context from {file_name}...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get page text for context
        page_text = page.get_text()
        
        # Get text blocks for better context extraction
        text_blocks = page.get_text("dict")["blocks"]
        
        # Get list of images on this page
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                # Extract image data
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image position
                img_rect = None
                for block in text_blocks:
                    if block.get("type") == 1:  # Image block
                        if block.get("number") == xref:
                            img_rect = fitz.Rect(block["bbox"])
                            break
                
                # Extract surrounding text context
                context_text = ""
                if img_rect:
                    # Get text before and after image
                    context_blocks = []
                    for block in text_blocks:
                        if block.get("type") == 0:  # Text block
                            block_rect = fitz.Rect(block["bbox"])
                            # Check if text block is near the image
                            if (abs(block_rect.y1 - img_rect.y0) < 50 or  # Text above image
                                abs(block_rect.y0 - img_rect.y1) < 50):   # Text below image
                                block_text = ""
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        block_text += span.get("text", "")
                                    block_text += "\n"
                                context_blocks.append(block_text.strip())
                    
                    context_text = "\n\n".join(context_blocks)
                
                # Create filename
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image dimensions
                try:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        width, height = pil_img.size
                        mode = pil_img.mode
                except Exception:
                    width = height = mode = None
                
                # Store metadata with context
                image_metadata = {
                    "image_id": f"{doc_id}_page_{page_num + 1}_img_{img_index + 1}",
                    "filename": image_filename,
                    "file_path": os.path.abspath(image_path),
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "format": image_ext.upper(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": len(image_bytes),
                    "source_pdf": file_name,
                    "doc_id": doc_id,
                    "extraction_dir": images_dir,
                    "surrounding_text": context_text.strip() if context_text else "",
                    "page_text_preview": page_text[:500] + "..." if len(page_text) > 500 else page_text
                }
                
                extracted_images.append(image_metadata)
                image_count += 1
                
                print(f"Extracted with context: {image_filename} ({width}x{height})")
                
            except Exception as e:
                print(f"Error extracting image {img_index + 1} from page {page_num + 1}: {e}")
                continue
    
    doc.close()
    
    print(f"Successfully extracted {image_count} images with context to {images_dir}")
    
    # Save metadata to JSON file
    import json
    metadata_file = os.path.join(images_dir, "images_metadata_with_context.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(extracted_images, f, indent=2, ensure_ascii=False)
    
    print(f"Image metadata with context saved to {metadata_file}")
    
    return extracted_images
def extract_images_from_pdf(pdf_path: str, output_dir: str = "images") -> List[Dict]:
    """
    Extract all images from PDF and save to images/anh_index directory
    Returns list of extracted image metadata
    """
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]
    
    # Create output directory
    images_dir = os.path.join(output_dir, f"image_{doc_id}")
    os.makedirs(images_dir, exist_ok=True)
    
    extracted_images = []
    image_count = 0
    
    print(f"Extracting images from {file_name}...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get list of images on this page
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            # Get image data
            xref = img[0]
            try:
                # Extract image data
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create filename
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image dimensions and other metadata
                try:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        width, height = pil_img.size
                        mode = pil_img.mode
                except Exception as e:
                    width = height = mode = None
                    print(f"Warning: Could not get image dimensions for {image_filename}: {e}")
                
                # Store metadata
                image_metadata = {
                    "image_id": f"{doc_id}_page_{page_num + 1}_img_{img_index + 1}",
                    "filename": image_filename,
                    "file_path": os.path.abspath(image_path),
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "format": image_ext.upper(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": len(image_bytes),
                    "source_pdf": file_name,
                    "doc_id": doc_id,
                    "extraction_dir": images_dir
                }
                
                extracted_images.append(image_metadata)
                image_count += 1
                
                print(f"Extracted: {image_filename} ({width}x{height} pixels, {len(image_bytes)} bytes)")
                
            except Exception as e:
                print(f"Error extracting image {img_index + 1} from page {page_num + 1}: {e}")
                continue
    
    doc.close()
    
    print(f"Successfully extracted {image_count} images to {images_dir}")
    
    # Save metadata to JSON file
    import json
    metadata_file = os.path.join(images_dir, "images_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(extracted_images, f, indent=2, ensure_ascii=False)
    
    print(f"Image metadata saved to {metadata_file}")
    
    return extracted_images

def extract_images_batch(folder_path: str, output_dir: str = "images") -> Dict[str, List[Dict]]:
    """
    Extract images from all PDF files in a folder
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return {}
    
    all_extracted = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        
        try:
            # Extract images with context
            extracted = extract_images_with_context(pdf_path, output_dir)
            all_extracted[pdf_file] = extracted
            print(f"✅ Completed {pdf_file}: {len(extracted)} images extracted")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            all_extracted[pdf_file] = []
    
    # Create summary report
    total_images = sum(len(images) for images in all_extracted.values())
    print(f"\n📊 SUMMARY:")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total images extracted: {total_images}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "extraction_summary.json")
    summary = {
        "total_pdfs": len(pdf_files),
        "total_images": total_images,
        "extraction_details": all_extracted,
        "timestamp": str(datetime.now())
    }
    
    import json
    from datetime import datetime
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_file}")
    
    return all_extracted

# ===================== Usage Examples =====================
if __name__ == "__main__":
    # Example usage
    pdf_path = "./docs/money-rules.pdf"
    for filename in os.listdir("./docs"):
        file_path = os.path.join("./docs", filename)
    # Extract images only
        if filename.endswith(".pdf"):
            images = extract_images_from_pdf(file_path)
    
    # Batch extract from folder
    # batch_results = extract_images_batch("docs/")