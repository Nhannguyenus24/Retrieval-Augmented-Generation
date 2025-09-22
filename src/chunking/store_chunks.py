import os
import sys
import json
from typing import Dict, List
from llm.prompt import gemini
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


from ingestion.transform_pdf import pdf_to_chunks_smart_indent
from ingestion.transform_txt import txt_to_chunks_smart_indent
from ingestion.transform_md import md_to_chunks_smart_indent
from ingestion.transform_docx import docx_to_chunks_smart_indent

# ===================== Configuration =====================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_chunk"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model - lightweight model
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # Sentence transformer model

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

    def get_summary(chunks: List[Dict]) -> str:
        """Generate a summary from the chunks and metadata"""
        return gemini.one_shot(str(chunks), "", template_name="summary.jinja")

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
                else:
                    print(f"Unsupported file type: {filename}")
                    continue
                
                if not chunks:
                    print(f"No valid chunks found for file: {filename}")
                    continue

                print(f"Created {len(chunks)} chunks from {filename}")

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
                        # New metadata fields
                        "doc_id": chunk.get('doc_id'),
                        "source": chunk.get('source'),
                        "section_title": chunk.get('section_title'),
                        "heading_path": chunk.get('heading_path'),
                        "chunk_index": chunk.get('chunk_index'),
                        "page_from": chunk.get('page_from'),
                        "page_to": chunk.get('page_to'),
                        "breadcrumbs": json.dumps(chunk.get('breadcrumbs', [])) if chunk.get('breadcrumbs') else None
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
