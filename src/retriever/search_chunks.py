#!/usr/bin/env python3
"""
Script to search chunks in a ChromaDB vector database.
Takes input from console and returns the top-k most similar chunks.
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import chromadb
from chromadb.config import Settings

# ===================== Configuration =====================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_chunk"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

class ChromaChunkSearcher:
    def __init__(self, host: str = CHROMA_HOST, port: int = CHROMA_PORT, collection_name: str = COLLECTION_NAME):
        """Initialize connection to ChromaDB"""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        # Connect to ChromaDB server
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port
            )
            print(f"Connected to ChromaDB at {host}:{port}")
        except Exception as e:
            print(f"ChromaDB connection error: {e}")
            print("Make sure the ChromaDB server is running (docker-compose up)")
            sys.exit(1)
            
        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            doc_count = self.collection.count()
            print(f"Collection '{collection_name}' has {doc_count} documents")
            
            if doc_count == 0:
                print("Collection is empty. Run store_chunks.py first")
                
        except Exception as e:
            print(f"Failed to get collection '{collection_name}': {e}")
            print("Run store_chunks.py to create data first")
            sys.exit(1)

    def search_chunks(self, query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: The search question/keywords
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List[Dict]: List of found chunks with metadata and similarity score
        """
        if not query.strip():
            print("Query must not be empty")
            return []
            
        try:
            print(f"Searching: '{query}' (top-{top_k})")
            
            # Perform vector similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, MAX_TOP_K),
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                print("No results found")
                return []
            
            # Process results
            search_results = []
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] 
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                # Convert distance to similarity score (1.0 = most similar, 0.0 = least similar)
                similarity = max(0.0, 1.0 - distance)
                
                # Filter by similarity threshold
                if similarity < min_similarity:
                    continue
                
                # Parse indent_levels from JSON string
                try:
                    indent_levels = json.loads(metadata.get('indent_levels', '[]'))
                except:
                    indent_levels = []
                
                result = {
                    'rank': i + 1,
                    'id': doc_id,
                    'similarity': round(similarity, 4),
                    'distance': round(distance, 4),
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', -1),
                    'page_range': metadata.get('page_range', 'Unknown'),
                    'token_count': metadata.get('token_count', 0),
                    'indent_levels': indent_levels,
                    'content': doc,
                    'source_path': metadata.get('source_path', 'Unknown')

                }
                
                search_results.append(result)
            
            print(f"Found {len(search_results)} matching results")
            return search_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def delete_collection(self) -> bool:
        """
        Delete the entire collection and all its documents
        
        Returns:
            bool: True if successful, False if error
        """
        try:
            # Get document count before deletion
            doc_count = self.collection.count()
            
            if doc_count == 0:
                print("Collection is already empty")
                return True
            
            # Confirm deletion
            print(f"WARNING: This will permanently delete {doc_count} documents from collection '{self.collection_name}'")
            confirm = input("Type 'DELETE' to confirm: ").strip()
            
            if confirm != 'DELETE':
                print("Deletion cancelled")
                return False
            
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            print(f"Successfully deleted collection '{self.collection_name}' with {doc_count} documents")
            
            return True
            
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def print_search_results(self, results: List[Dict], show_content: bool = True, max_content_length: int = 500):
        """Pretty-print search results to console"""
        if not results:
            print("No results to display")
            return
            
        print("\n" + "="*120)
        print(f"SEARCH RESULTS ({len(results)} chunks)")
        print("="*120)

        for result in results:
            print(f"\n# {result['rank']} - {result['file_name']} (Chunk {result['chunk_id']})")
            print(f"   Page: {result['page_range']} | Similarity: {result['similarity']} | Tokens: {result['token_count']}")
            print(f"   ID: {result['id']}")
            
            if result['indent_levels']:
                print(f"   Indent levels: {result['indent_levels']}")
            
                print(f"   Heading path: {result['heading_path']}")
                print(f"   Section title: {result['heading_path']}")
                print(f"   Bread crumb: {result['heading_path']}") 
            if show_content:
                content = result['content']
                print(f"   Content:")
                # Indent content for readability
                for line in content.split('\n'):
                    print(f"      {line}")
            
            print("-" * 80)

    def interactive_search(self):
        """Interactive search mode from console"""
        print("\nWELCOME TO THE CHUNK SEARCH SYSTEM!")
        print("Enter a question or keywords to search within PDF chunks")
        print("Special commands:")
        print("   - 'quit' or 'exit': Exit")
        print("   - 'stats': Show collection statistics")
        print("   - 'config': Change search settings")
        print("   - 'delete': Delete entire collection")
        print("-" * 80)
        
        # Default settings
        current_top_k = DEFAULT_TOP_K
        current_min_similarity = 0.0
        show_content = True
        max_content_length = 500
        
        while True:
            try:
                # Enter query
                print(f"\nCurrent settings: top-k={current_top_k}, min_similarity={current_min_similarity:.2f}")
                query = input("Enter your question (or 'quit' to exit): ").strip()
                
                if not query:
                    continue
                    
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                elif query.lower() == 'stats':
                    count = self.collection.count()
                    print(f"Collection '{self.collection_name}' has {count} documents")
                    continue
                
                elif query.lower() == 'delete':
                    success = self.delete_collection()
                    if success:
                        print("Collection deleted successfully. Exiting...")
                        break
                    continue
                    
                elif query.lower() == 'config':
                    print("\nUPDATE SETTINGS:")
                    try:
                        new_k = input(f"Current top-k: {current_top_k}. Enter new value (Enter to keep): ").strip()
                        if new_k:
                            current_top_k = max(1, min(int(new_k), MAX_TOP_K))
                            
                        new_sim = input(f"Current min similarity: {current_min_similarity:.2f}. Enter new value 0.0-1.0 (Enter to keep): ").strip()
                        if new_sim:
                            current_min_similarity = max(0.0, min(float(new_sim), 1.0))
                            
                        show_choice = input(f"Show content: {'Yes' if show_content else 'No'}. Toggle? (y/n, Enter to keep): ").strip().lower()
                        if show_choice == 'y':
                            show_content = not show_content
                            
                        if show_content:
                            new_len = input(f"Max content length: {max_content_length}. Enter new value (Enter to keep): ").strip()
                            if new_len:
                                max_content_length = max(100, int(new_len))
                        
                        print("Settings updated.")
                    except ValueError:
                        print("Invalid value!")
                    continue
                
                # Execute search
                results = self.search_chunks(query, top_k=current_top_k, min_similarity=current_min_similarity)
                self.print_search_results(results, show_content=show_content, max_content_length=max_content_length)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function for command line execution"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage:")
        print(f"  python {sys.argv[0]}                             # Interactive mode")
        print(f"  python {sys.argv[0]} \"<query>\" [top_k]            # One-shot search")
        print(f"  python {sys.argv[0]} --delete                     # Delete entire collection")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]}                             # Run interactive mode")
        print(f"  python {sys.argv[0]} \"machine learning\" 3          # Search 3 chunks about ML")
        print(f"  python {sys.argv[0]} --delete                     # Delete all chunks")
        sys.exit(0)
    
    # Initialize searcher
    searcher = ChromaChunkSearcher()
    
    if len(sys.argv) == 1:
        # Interactive mode
        searcher.interactive_search()
    elif len(sys.argv) > 1 and sys.argv[1] == '--delete':
        # Delete collection
        searcher.delete_collection()
    else:
        # One-shot search
        query = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
        
        results = searcher.search_chunks(query, top_k=top_k)
        searcher.print_search_results(results)

if __name__ == "__main__":
    main()
