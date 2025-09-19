#!/usr/bin/env python3
"""
RAG CLI - Retrieval Augmented Generation Command Line Interface
A unified CLI tool for managing PDF document chunking, storage, and retrieval.

This tool combines functionality from:
- transform_pdf.py: PDF chunking with smart indentation
- store_chunks.py: Storing chunks in ChromaDB
- search_chunks.py: Searching and retrieving chunks
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chunking'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retriever'))

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

try:
    from ingestion.transform_pdf import pdf_to_chunks_smart_indent
    from chunking.store_chunks import ChromaChunkStore
    from retriever.search_chunks import ChromaChunkSearcher
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# ===================== Configuration =====================
DEFAULT_MIN_TOKENS = 300
DEFAULT_MAX_TOKENS = 500
DEFAULT_TOP_K = 5
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_chunk"

class RAGCLIApp:
    """Main CLI application class"""
    
    def __init__(self):
        self.store = None
        self.searcher = None
    
    def _init_store(self):
        """Initialize ChromaDB store connection"""
        if not self.store:
            try:
                self.store = ChromaChunkStore(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    collection_name=COLLECTION_NAME
                )
            except Exception as e:
                print(f"Error: Failed to connect to ChromaDB: {e}")
                print("Make sure ChromaDB server is running: docker-compose up -d")
                sys.exit(1)
        return self.store
    
    def _init_searcher(self):
        """Initialize ChromaDB searcher connection"""
        if not self.searcher:
            try:
                self.searcher = ChromaChunkSearcher(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    collection_name=COLLECTION_NAME
                )
            except Exception as e:
                print(f"Error: Failed to connect to ChromaDB: {e}")
                print("Make sure ChromaDB server is running: docker-compose up -d")
                sys.exit(1)
        return self.searcher

    def chunk_pdf(self, pdf_path: str, min_tokens: int = DEFAULT_MIN_TOKENS, 
                  max_tokens: int = DEFAULT_MAX_TOKENS, output_file: Optional[str] = None) -> bool:
        """
        Chunk a PDF file into smaller pieces
        
        Args:
            pdf_path: Path to PDF file
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            output_file: Optional JSON file to save chunks
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(pdf_path):
            print(f"Error: File does not exist: {pdf_path}")
            return False
        
        if not pdf_path.lower().endswith('.pdf'):
            print(f"Error: File must be a PDF: {pdf_path}")
            return False
        
        try:
            print(f"Processing PDF: {pdf_path}")
            print(f"Token range: {min_tokens}-{max_tokens}")
            
            # Create chunks
            chunks = pdf_to_chunks_smart_indent(
                pdf_path, 
                min_tokens=min_tokens, 
                max_tokens=max_tokens
            )
            
            if not chunks:
                print("Warning: No chunks created from PDF")
                return False
            
            print(f"Successfully created {len(chunks)} chunks")
            
            # Display summary
            total_tokens = sum(chunk['token_est'] for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            print(f"Summary:")
            print(f"  - Total chunks: {len(chunks)}")
            print(f"  - Total tokens: {total_tokens}")
            print(f"  - Average tokens per chunk: {avg_tokens:.1f}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                print(f"Chunks saved to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False

    def store_pdf(self, pdf_path: str, min_tokens: int = DEFAULT_MIN_TOKENS, 
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> bool:
        """
        Process and store PDF chunks in ChromaDB
        
        Args:
            pdf_path: Path to PDF file
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            bool: True if successful
        """
        store = self._init_store()
        
        if not os.path.exists(pdf_path):
            print(f"Error: File does not exist: {pdf_path}")
            return False
        
        print(f"Storing PDF chunks: {pdf_path}")
        success = store.store_pdf_chunks(pdf_path, min_tokens, max_tokens)
        
        if success:
            print("✓ PDF chunks successfully stored in ChromaDB")
        else:
            print("✗ Failed to store PDF chunks")
        
        return success

    def search_chunks(self, query: str, top_k: int = DEFAULT_TOP_K, 
                     min_similarity: float = 0.0, show_content: bool = True) -> bool:
        """
        Search for chunks similar to query
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            show_content: Whether to show chunk content
            
        Returns:
            bool: True if successful
        """
        searcher = self._init_searcher()
        
        if not query.strip():
            print("Error: Search query cannot be empty")
            return False
        
        print(f"Searching for: '{query}'")
        print(f"Parameters: top-k={top_k}, min_similarity={min_similarity}")
        
        try:
            results = searcher.search_chunks(query, top_k, min_similarity)
            
            if not results:
                print("No matching chunks found")
                return True
            
            # Display results
            searcher.print_search_results(results, show_content=show_content)
            return True
            
        except Exception as e:
            print(f"Error during search: {e}")
            return False

    def interactive_search(self):
        """Start interactive search mode"""
        searcher = self._init_searcher()
        print("Starting interactive search mode...")
        searcher.interactive_search()

    def show_stats(self) -> bool:
        """Show collection statistics"""
        store = self._init_store()
        
        try:
            stats = store.get_collection_stats()
            
            print("Collection Statistics:")
            print("=" * 50)
            print(f"Collection Name: {stats.get('collection_name', 'Unknown')}")
            print(f"Total Documents: {stats.get('total_documents', 0)}")
            
            if stats.get('sample_documents'):
                print(f"\nSample Documents:")
                for i, doc in enumerate(stats['sample_documents'], 1):
                    print(f"\n{i}. ID: {doc['id']}")
                    print(f"   Preview: {doc['content_preview']}")
                    if doc.get('metadata'):
                        print(f"   File: {doc['metadata'].get('file_name', 'Unknown')}")
                        print(f"   Pages: {doc['metadata'].get('page_range', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        store = self._init_store()
        
        try:
            # Get count first
            count = store.collection.count()
            if count == 0:
                print("Collection is already empty")
                return True
            
            print(f"WARNING: This will delete {count} documents from the collection")
            confirm = input("Type 'yes' to confirm deletion: ").strip().lower()
            
            if confirm != 'yes':
                print("Deletion cancelled")
                return False
            
            success = store.clear_collection()
            if success:
                print("✓ Collection cleared successfully")
            else:
                print("✗ Failed to clear collection")
            
            return success
            
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def delete_collection(self) -> bool:
        """Delete entire collection"""
        searcher = self._init_searcher()
        
        try:
            success = searcher.delete_collection()
            if success:
                print("✓ Collection deleted successfully")
            else:
                print("✗ Failed to delete collection")
            
            return success
            
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        prog='rag-cli',
        description='RAG CLI - Manage PDF document chunking, storage, and retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chunk a PDF file (preview only)
  python rag_cli.py chunk document.pdf
  
  # Chunk with custom token limits
  python rag_cli.py chunk document.pdf --min-tokens 250 --max-tokens 600
  
  # Store PDF chunks in ChromaDB
  python rag_cli.py store document.pdf
  
  # Search for chunks
  python rag_cli.py search "machine learning algorithms"
  
  # Search with custom parameters
  python rag_cli.py search "neural networks" --top-k 10 --min-similarity 0.5
  
  # Interactive search mode
  python rag_cli.py interactive
  
  # Show collection statistics
  python rag_cli.py stats
  
  # Clear all chunks from collection
  python rag_cli.py clear
  
  # Delete entire collection
  python rag_cli.py delete
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk PDF file into smaller pieces')
    chunk_parser.add_argument('pdf_path', help='Path to PDF file')
    chunk_parser.add_argument('--min-tokens', type=int, default=DEFAULT_MIN_TOKENS,
                             help=f'Minimum tokens per chunk (default: {DEFAULT_MIN_TOKENS})')
    chunk_parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                             help=f'Maximum tokens per chunk (default: {DEFAULT_MAX_TOKENS})')
    chunk_parser.add_argument('--output', '-o', help='Save chunks to JSON file')
    
    # Store command
    store_parser = subparsers.add_parser('store', help='Process and store PDF chunks in ChromaDB')
    store_parser.add_argument('pdf_path', help='Path to PDF file')
    store_parser.add_argument('--min-tokens', type=int, default=DEFAULT_MIN_TOKENS,
                             help=f'Minimum tokens per chunk (default: {DEFAULT_MIN_TOKENS})')
    store_parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                             help=f'Maximum tokens per chunk (default: {DEFAULT_MAX_TOKENS})')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for chunks similar to query')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                              help=f'Number of results to return (default: {DEFAULT_TOP_K})')
    search_parser.add_argument('--min-similarity', type=float, default=0.0,
                              help='Minimum similarity threshold (0.0-1.0, default: 0.0)')
    search_parser.add_argument('--no-content', action='store_true',
                              help='Hide chunk content in results')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive search mode')
    
    # Stats command
    subparsers.add_parser('stats', help='Show collection statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all documents from collection')
    
    # Delete command
    subparsers.add_parser('delete', help='Delete entire collection')
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    app = RAGCLIApp()
    success = True
    
    try:
        if args.command == 'chunk':
            success = app.chunk_pdf(
                args.pdf_path, 
                args.min_tokens, 
                args.max_tokens,
                args.output
            )
        
        elif args.command == 'store':
            success = app.store_pdf(
                args.pdf_path,
                args.min_tokens,
                args.max_tokens
            )
        
        elif args.command == 'search':
            success = app.search_chunks(
                args.query,
                args.top_k,
                args.min_similarity,
                show_content=not args.no_content
            )
        
        elif args.command == 'interactive':
            app.interactive_search()
        
        elif args.command == 'stats':
            success = app.show_stats()
        
        elif args.command == 'clear':
            success = app.clear_collection()
        
        elif args.command == 'delete':
            success = app.delete_collection()
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            success = False
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        success = False
    except Exception as e:
        print(f"Unexpected error: {e}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
