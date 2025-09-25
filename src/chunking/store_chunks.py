import os
import sys
import json
from typing import Dict, List
from llm.provider import gemini
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from ingestion.transform_pdf import pdf_to_chunks_smart_indent
from ingestion.transform_txt import txt_to_chunks_smart_indent
from ingestion.transform_md import md_to_chunks_smart_indent
from ingestion.transform_docx import docx_to_chunks_smart_indent
from ingestion.transform_excel import excel_to_chunks_smart_indent
import logging

logger = logging.getLogger("store_chunks")

CHROMA_HOST = str(os.getenv("CHROMA_HOST", "chromadb"))
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME", "SOF_DOCUMENTATION"))
EMBEDDING_MODEL = str(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

class ChromaChunkStore:
    def __init__(self, host: str = CHROMA_HOST, port: int = CHROMA_PORT, collection_name: str = COLLECTION_NAME):
        """Initialize connection to ChromaDB"""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        embedding_func = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        # Connect to ChromaDB server
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port
            )
            print(f"Connected to ChromaDB at {host}:{port}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            print("Make sure ChromaDB server is running (docker-compose up)")
            sys.exit(1)
            
        # Create or get collection
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            # If not exists, create new one
            try:
                self.collection = self.client.create_collection(name=collection_name)
                print(f"Created new collection: {collection_name}")
            except Exception as e:
                print(f"Error creating collection: {e}")
                sys.exit(1)

    def get_summary(self, chunks: List[Dict]) -> str:
        """Generate a summary from the chunks and metadata"""
        if not chunks:
            return "No content available for summary."
        
        # Prepare content for summary generation
        content_text = "\n\n".join([
            f"File: {chunk.get('file_name', 'Unknown')}\n"
            f"Section: {chunk.get('section_title', 'N/A')}\n"
            f"Content: {chunk['content']}"
            for chunk in chunks[:100]
        ])
        
        return gemini.one_shot(content_text, "Generate a comprehensive summary", template_name="summary.jinja")

    def store_chunks(self, folder_path: str, min_tokens: int = 300, max_tokens: int = 500) -> bool:
        # Process all files in the folder into chunks and store
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return False
        
        try:
            documents = []
            metadatas = []
            ids = []

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith(".pdf"):
                    chunks = pdf_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
                elif filename.endswith(".txt"):
                    chunks = txt_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
                elif filename.endswith(".md"):
                    chunks = md_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
                elif filename.endswith(".docx"):
                    chunks = docx_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
                elif filename.endswith((".xlsx", ".xls")):
                    chunks = excel_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
                else:
                    print(f"Unsupported file type: {filename}")
                    continue
                
                if not chunks:
                    print(f"No valid chunks found for file: {filename}")
                    continue

                print(f"Created {len(chunks)} chunks from {filename}")
                
                # Generate summary for this document
                try:
                    document_summary = self.get_summary(chunks)
                    print(f"Generated summary for {filename}: {document_summary}")
                except Exception as e:
                    print(f"Error generating summary for {filename}: {e}")
                    document_summary = "Summary generation failed."

                for chunk in chunks:
                    chunk_id = f"{chunk['file_name']}_{chunk['chunk_id']}"
                    # Chunk content
                    documents.append(chunk['content'])

                    page_from = str(chunk.get('page_from', 'Unknown'))
                    page_to   = str(chunk.get('page_to', 'Unknown'))
                    page_range = page_from if page_from == page_to else f"{page_from}-{page_to}"

                    
                    # Enhanced metadata with breadcrumbs and new fields
                    metadata = {
                        "file_name": chunk['file_name'],
                        "chunk_id": chunk['chunk_id'],
                        "page_range": page_range,
                        "indent_levels": json.dumps(chunk['indent_levels']),
                        "token_count": chunk['token_est'],
                        "source_path": os.path.abspath(file_path),
                        "doc_id": chunk.get('doc_id'),
                        "source": chunk.get('source'),
                        "section_title": chunk.get('section_title'),
                        "heading_path": chunk.get('heading_path'),
                        "chunk_index": chunk.get('chunk_index'),
                        "page_from": chunk.get('page_from'),
                        "page_to": chunk.get('page_to'),
                        "document_summary": document_summary,
                        "image_urls": json.dumps(chunk.get('image_urls', []), ensure_ascii=False)
                    }
                    
                    # Remove None values to keep metadata clean
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    metadatas.append(metadata)
                    
                    ids.append(chunk_id)

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Added {len(documents)} documents to collection")
            return True
        except Exception as e:
            print(f"Error processing files in folder: {e}")
            return False

    def store_chunks_from_single_file(self, file_path: str, min_tokens: int = 300, max_tokens: int = 500) -> Dict:
        """
        Process a single file into chunks and store in ChromaDB
        Returns: dict with success status, message, and metadata
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"File not found: {file_path}",
                "chunks_stored": 0,
                "filename": os.path.basename(file_path),
                "error": "File not found"
            }
        
        filename = os.path.basename(file_path)
        filename_lower = filename.lower()
        
        try:
            # Determine file type and process accordingly
            chunks = []
            file_type = "unknown"
            
            if filename_lower.endswith(".pdf"):
                file_type = "pdf"
                chunks = pdf_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
            elif filename_lower.endswith(".txt"):
                file_type = "txt"
                chunks = txt_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
            elif filename_lower.endswith(".md"):
                file_type = "markdown"
                chunks = md_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
            elif filename_lower.endswith(".docx"):
                file_type = "docx"
                chunks = docx_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
            elif filename_lower.endswith((".xlsx", ".xls")):
                file_type = "excel"
                chunks = excel_to_chunks_smart_indent(file_path, min_tokens=min_tokens, max_tokens=max_tokens)
            else:
                return {
                    "success": False,
                    "message": f"Unsupported file type: {filename}",
                    "chunks_stored": 0,
                    "filename": filename,
                    "file_type": file_type,
                    "error": "Unsupported file type"
                }
                
            if not chunks:
                return {
                    "success": False,
                    "message": f"No valid chunks found for file: {filename}",
                    "chunks_stored": 0,
                    "filename": filename,
                    "file_type": file_type,
                    "error": "No chunks generated"
                }

            print(f"Created {len(chunks)} chunks from {filename}")
            
            # Generate summary for this document
            try:
                document_summary = self.get_summary(chunks)
                print(f"Generated summary for {filename}: {document_summary[:100]}...")
            except Exception as e:
                print(f"Error generating summary for {filename}: {e}")
                document_summary = "Summary generation failed."

            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []

            for chunk in chunks:
                chunk_id = f"{chunk.get('file_name', filename)}_{chunk.get('chunk_id', f'chunk_{len(documents)+1}')}"
                # Chunk content
                documents.append(chunk['content'])

                page_from = str(chunk.get('page_from', 'Unknown'))
                page_to   = str(chunk.get('page_to', 'Unknown'))
                page_range = page_from if page_from == page_to else f"{page_from}-{page_to}"

                # Enhanced metadata with breadcrumbs and new fields
                metadata = {
                    "file_name": chunk.get('file_name', filename),
                    "chunk_id": chunk.get('chunk_id', f'chunk_{len(documents)}'),
                    "page_range": page_range,
                    "indent_levels": json.dumps(chunk.get('indent_levels', [])),
                    "token_count": int(chunk.get('token_est', chunk.get('token_count', 0))),  # Ensure integer
                    "source_path": os.path.abspath(file_path),
                    "doc_id": chunk.get('doc_id', filename_lower.split('.')[0]),
                    "source": chunk.get('source', filename),
                    "section_title": chunk.get('section_title'),
                    "heading_path": chunk.get('heading_path'),
                    "chunk_index": chunk.get('chunk_index'),
                    "page_from": chunk.get('page_from'),
                    "page_to": chunk.get('page_to'),
                    "document_summary": document_summary,
                    "image_urls": json.dumps(chunk.get('image_urls', []), ensure_ascii=False),
                    "file_type": file_type
                }
                
                # Remove None values to keep metadata clean
                metadata = {k: v for k, v in metadata.items() if v is not None}
                metadatas.append(metadata)
                ids.append(chunk_id)

            # Store in ChromaDB
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Added {len(documents)} documents to collection from {filename}")
            
            return {
                "success": True,
                "message": f"Successfully processed and stored {filename}",
                "chunks_stored": len(chunks),
                "filename": filename,
                "file_type": file_type,
                "file_path": file_path,
                "document_summary": document_summary,
                "total_tokens": int(sum(chunk.get('token_est', chunk.get('token_count', 0)) for chunk in chunks)),  # Ensure integer
                "has_images": any(chunk.get('image_urls', []) for chunk in chunks),
                "image_count": sum(len(chunk.get('image_urls', [])) for chunk in chunks)
            }
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return {
                "success": False,
                "message": f"Error processing file: {filename}",
                "chunks_stored": 0,
                "filename": filename,
                "file_type": file_type,
                "error": str(e)
            }

    def clear_collection(self) -> bool:
        """Delete all documents in collection"""
        try:
            # Get all IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} documents")
            else:
                print("Collection is already empty")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def get_file_chunks(self, filename: str) -> Dict:
        """
        Get all chunks and metadata for a specific file
        Returns: dict with success status, message, and chunks data
        """
        try:
            # Query all chunks for this file
            results = self.collection.get(
                where={"file_name": filename}
            )
            
            if not results['ids']:
                return {
                    "success": False,
                    "message": f"No chunks found for file: {filename}",
                    "chunk_count": 0,
                    "filename": filename,
                    "chunks": [],
                    "error": "File not found in collection"
                }
            
            # Format chunk data for response
            chunks = []
            for i, chunk_id in enumerate(results['ids']):
                chunk_data = {
                    "id": chunk_id,
                    "content": results['documents'][i] if results['documents'] and i < len(results['documents']) else "",
                    "metadata": results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                }
                chunks.append(chunk_data)
            
            print(f"Retrieved {len(chunks)} chunks for file: {filename}")
            
            return {
                "success": True,
                "message": f"Successfully retrieved chunks for {filename}",
                "chunk_count": len(chunks),
                "filename": filename,
                "chunks": chunks
            }
            
        except Exception as e:
            print(f"Error getting file chunks for {filename}: {e}")
            return {
                "success": False,
                "message": f"Error retrieving chunks for {filename}",
                "chunk_count": 0,
                "filename": filename,
                "chunks": [],
                "error": str(e)
            }

    def delete_file_chunks(self, filename: str) -> Dict:
        """
        Delete all chunks and metadata for a specific file
        Returns: dict with success status, message, and deleted count
        """
        try:
            # Query all chunks for this file
            results = self.collection.get(
                where={"file_name": filename}
            )
            
            if not results['ids']:
                return {
                    "success": False,
                    "message": f"No chunks found for file: {filename}",
                    "deleted_count": 0,
                    "filename": filename,
                    "error": "File not found in collection"
                }
            
            # Delete all chunks for this file
            deleted_count = len(results['ids'])
            self.collection.delete(ids=results['ids'])
            
            print(f"Deleted {deleted_count} chunks for file: {filename}")
            
            return {
                "success": True,
                "message": f"Successfully deleted all chunks for {filename}",
                "deleted_count": deleted_count,
                "filename": filename,
                "chunk_ids": results['ids']
            }
            
        except Exception as e:
            print(f"Error deleting file chunks for {filename}: {e}")
            return {
                "success": False,
                "message": f"Error deleting chunks for {filename}",
                "deleted_count": 0,
                "filename": filename,
                "error": str(e)
            }

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            results = self.collection.get(limit=5)  # Get first 5 samples
            
            stats = {
                "total_documents": count,
                "collection_name": self.collection_name,
                "sample_documents": []
            }
            
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    stats["sample_documents"].append({
                        "id": results['ids'][i],
                        "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": metadata
                    })
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def get_document_summaries(self) -> Dict:
        """Get summaries of all documents in the collection"""
        try:
            results = self.collection.get()
            summaries = {}
            for metadata in results['metadatas']:
                file_name = metadata.get('file_name')
                document_summary = metadata.get('document_summary')
                if file_name and document_summary not in summaries:
                    summaries[file_name] = document_summary
            
            return summaries
        except Exception as e:
            print(f"Error getting document summaries: {e}")
            return {}
    
    def generate_suggested_questions(self, summaries: Dict = None) -> List[str]:
        """Generate 4 suggested questions based on document summaries"""
        if summaries is None:
            summaries = self.get_document_summaries()
        
        if not summaries:
            return [
                "What documents are available in this collection?",
                "Can you provide an overview of the content?",
                "What topics are covered in these documents?",
                "How can I search for specific information?"
            ]
        
        # Combine all summaries for context
        combined_summaries = "\n\n".join([
            f"Document: {file_name}\nSummary: {summary}"
            for file_name, summary in summaries.items()
        ])
        
        try:
            # Use Gemini to generate questions based on summaries
            suggestions_response = gemini.one_shot(
                combined_summaries, 
                "Generate 4 insightful questions that users might want to ask about these documents",
                template_name="suggestion.jinja"
            )
            
            # Parse the response to extract questions (assuming they're numbered or bullet points)
            questions = []
            lines = suggestions_response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and ('?' in line or line.startswith(('1.', '2.', '3.', '4.', '-', '•'))):
                    # Clean up the line (remove numbering, bullets)
                    clean_question = line.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '')
                    clean_question = clean_question.replace('-', '').replace('•', '').strip()
                    if clean_question and clean_question not in questions:
                        questions.append(clean_question)
                        
                        if len(questions) >= 4:
                            break
            
            # Ensure we have exactly 4 questions
            if len(questions) < 4:
                default_questions = [
                    "What are the main topics covered in these documents?",
                    "Can you explain the key concepts mentioned?",
                    "What are the important findings or conclusions?",
                    "How do these documents relate to each other?"
                ]
                questions.extend(default_questions[len(questions):4])
            
            return questions[:4]
            
        except Exception as e:
            print(f"Error generating suggested questions: {e}")
            return [
                "What are the main topics in these documents?",
                "Can you summarize the key findings?",
                "What specific information can I search for?",
                "How are these documents structured?"
            ]

    def validate_query(self, query: str) -> str:
        
        query = query.strip()
        
        # Get document summaries for context
        summaries_context = self.get_document_summaries()
        
        try:
            
            validation_result = gemini.one_shot(
                summaries_context,
                query,
                template_name="validate_query.jinja"
            )
            return validation_result
                
        except Exception as e:
            print(f"Error validating query: {e}")
            # Fallback validation using simple keyword matching
            return "error"