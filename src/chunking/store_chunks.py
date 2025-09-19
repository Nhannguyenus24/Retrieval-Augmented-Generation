#!/usr/bin/env python3
"""
Script to store PDF chunks into ChromaDB vector database
Uses transformpdf.py to create chunks and store them in ChromaDB
"""

import os
import sys
import json
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    from ingestion.transform_pdf import pdf_to_chunks_smart_indent
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
    from transform_pdf import pdf_to_chunks_smart_indent

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

    def store_pdf_chunks(self, pdf_path: str, min_tokens: int = 300, max_tokens: int = 500) -> bool:
        """
        Process PDF into chunks and store in ChromaDB
        
        Args:
            pdf_path: Path to PDF file
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            bool: True if successful, False if error
        """
        if not os.path.exists(pdf_path):
            print(f"File does not exist: {pdf_path}")
            return False
            
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Create chunks from PDF
            chunks = pdf_to_chunks_smart_indent(pdf_path, min_tokens=min_tokens, max_tokens=max_tokens)
            
            if not chunks:
                print("No chunks created from PDF")
                return False
                
            print(f"Created {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Unique ID for each chunk
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
                    "source_path": os.path.abspath(pdf_path),
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
            
            # Save to ChromaDB (will automatically create embeddings)
            print("Saving chunks to ChromaDB...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully saved {len(chunks)} chunks to collection '{self.collection_name}'")
            
            # Statistics
            total_docs = self.collection.count()
            print(f"Total documents in collection: {total_docs}")
            
            return True
            
        except Exception as e:
            print(f"Error saving chunks: {e}")
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

def main():
    """Main function to run from command line"""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {sys.argv[0]} <pdf_path> [min_tokens] [max_tokens]")
        print(f"  python {sys.argv[0]} --clear  # Clear all chunks")
        print(f"  python {sys.argv[0]} --stats  # View statistics")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} 123.pdf")
        print(f"  python {sys.argv[0]} 123.pdf 250 600")
        sys.exit(1)
    
    # Initialize store
    store = ChromaChunkStore()
    
    if sys.argv[1] == "--clear":
        store.clear_collection()
        return
    
    if sys.argv[1] == "--stats":
        stats = store.get_collection_stats()
        print("Collection Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    
    # Process PDF
    pdf_path = sys.argv[1]
    min_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    print(f"Configuration: min_tokens={min_tokens}, max_tokens={max_tokens}")
    
    success = store.store_pdf_chunks(pdf_path, min_tokens=min_tokens, max_tokens=max_tokens)
    
    if success:
        print("\nCompleted! Chunks have been saved to ChromaDB")
        print("You can now use search_chunks.py to search")
    else:
        print("\nError occurred during processing")
        sys.exit(1)

if __name__ == "__main__":
    main()
