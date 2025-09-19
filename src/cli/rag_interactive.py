#!/usr/bin/env python3
"""
RAG Interactive CLI - Beautiful Interactive Interface
A comprehensive interactive CLI tool for RAG operations with enhanced visual presentation.

Features:
- Beautiful ASCII interface with symbols and borders
- Continuous loop operation
- Full RAG functionality (chunk, store, search)
- Enhanced visual feedback
- Comprehensive information display
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
    from llm.provider.gemini import one_shot
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

# ===================== Visual Constants =====================
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

class Symbols:
    """ASCII symbols for visual enhancement"""
    ARROW_RIGHT = ">"
    ARROW_LEFT = "<"
    BULLET = "*"
    DIAMOND = "*"
    STAR = "*"
    CHECK = "+"
    CROSS = "X"
    WARNING = "!"
    INFO = "i"
    SEARCH = "?"
    FOLDER = "DIR"
    FILE = "FILE"
    DATABASE = "DB"
    GEAR = "CFG"
    CHART = "STAT"
    TRASH = "DEL"
    
    # Box drawing
    BOX_H = "-"
    BOX_V = "|"
    BOX_TL = "+"
    BOX_TR = "+"
    BOX_BL = "+"
    BOX_BR = "+"
    BOX_CROSS = "+"
    BOX_T = "+"
    BOX_B = "+"
    BOX_L = "+"
    BOX_R = "+"

class RAGInteractiveCLI:
    """Enhanced interactive CLI for RAG operations"""
    
    def __init__(self):
        self.store = None
        self.searcher = None
        self.settings = {
            'min_tokens': DEFAULT_MIN_TOKENS,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'top_k': DEFAULT_TOP_K,
            'min_similarity': 0.0,
            'show_content': True,
            'max_content_length': 500,
            'use_gemini': True,
            'show_raw_chunks': False
        }
    
    def _init_store(self):
        """Initialize ChromaDB store connection"""
        if not self.store:
            try:
                self.store = ChromaChunkStore(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    collection_name=COLLECTION_NAME
                )
                self._print_success("Connected to ChromaDB for storage operations")
            except Exception as e:
                self._print_error(f"Failed to connect to ChromaDB: {e}")
                self._print_warning("Make sure ChromaDB server is running: docker-compose up -d")
                return None
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
                self._print_success("Connected to ChromaDB for search operations")
            except Exception as e:
                self._print_error(f"Failed to connect to ChromaDB: {e}")
                self._print_warning("Make sure ChromaDB server is running: docker-compose up -d")
                return None
        return self.searcher

    # ===================== Visual Helper Methods =====================
    
    def _print_header(self, title: str, width: int = 80):
        """Print a beautiful header with borders"""
        print(f"\n{Colors.BRIGHT_CYAN}{Symbols.BOX_TL}{Symbols.BOX_H * (width-2)}{Symbols.BOX_TR}")
        padding = (width - len(title) - 2) // 2
        print(f"{Symbols.BOX_V}{' ' * padding}{Colors.BOLD}{title}{Colors.RESET}{Colors.BRIGHT_CYAN}{' ' * (width - len(title) - padding - 2)}{Symbols.BOX_V}")
        print(f"{Symbols.BOX_BL}{Symbols.BOX_H * (width-2)}{Symbols.BOX_BR}{Colors.RESET}\n")
    
    def _print_box(self, content: str, width: int = 80, color: str = Colors.CYAN):
        """Print content in a box"""
        lines = content.split('\n')
        print(f"{color}{Symbols.BOX_TL}{Symbols.BOX_H * (width-2)}{Symbols.BOX_TR}")
        for line in lines:
            padding = width - len(line) - 4
            print(f"{Symbols.BOX_V} {line}{' ' * padding} {Symbols.BOX_V}")
        print(f"{Symbols.BOX_BL}{Symbols.BOX_H * (width-2)}{Symbols.BOX_BR}{Colors.RESET}")
    
    def _print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.BRIGHT_GREEN}{Symbols.CHECK} {message}{Colors.RESET}")
    
    def _print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.BRIGHT_RED}{Symbols.CROSS} {message}{Colors.RESET}")
    
    def _print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.BRIGHT_YELLOW}{Symbols.WARNING} {message}{Colors.RESET}")
    
    def _print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BRIGHT_BLUE}{Symbols.INFO} {message}{Colors.RESET}")
    
    def _print_separator(self, char: str = "─", width: int = 80, color: str = Colors.DIM):
        """Print a separator line"""
        print(f"{color}{char * width}{Colors.RESET}")
    
    def _print_gemini_response(self, response: str, query: str):
        """Print Gemini AI response with beautiful formatting"""
        print(f"\n{Colors.BRIGHT_MAGENTA}🤖 AI RESPONSE{Colors.RESET}")
        self._print_separator("═", width=100, color=Colors.BRIGHT_MAGENTA)
        
        print(f"{Colors.BRIGHT_CYAN}Query:{Colors.RESET} {Colors.BOLD}{query}{Colors.RESET}")
        print(f"\n{Colors.BRIGHT_YELLOW}Answer:{Colors.RESET}")
        
        # Format the response with proper indentation
        for line in response.split('\n'):
            if line.strip():
                print(f"  {line}")
            else:
                print()
        
        self._print_separator("═", width=100, color=Colors.BRIGHT_MAGENTA)
    
    def _get_gemini_response(self, results: list, query: str) -> str:
        """Get AI response from Gemini using search results"""
        try:
            if not results:
                return "No search results available to generate response."
            
            # Format results for Gemini
            formatted_results = []
            for result in results:
                formatted_result = {
                    'file_name': result['file_name'],
                    'page_range': result['page_range'],
                    'similarity': result['similarity'],
                    'content': result['content']
                }
                formatted_results.append(formatted_result)
            
            response = one_shot(str(formatted_results), query)
            return response
        except Exception as e:
            return f"Error generating AI response: {e}"
    
    def _print_menu_item(self, number: str, icon: str, title: str, description: str):
        """Print a formatted menu item"""
        print(f"  {Colors.BRIGHT_CYAN}[{number}]{Colors.RESET} {icon} {Colors.BOLD}{title}{Colors.RESET}")
        print(f"      {Colors.DIM}{description}{Colors.RESET}")

    # ===================== Main Interface Methods =====================
    
    def show_welcome(self):
        """Display welcome screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        welcome_art = f"""
{Colors.BRIGHT_MAGENTA}
    ██████╗  █████╗  ██████╗     ██╗███╗   ██╗████████╗███████╗██████╗  █████╗  ██████╗████████╗██╗██╗   ██╗███████╗
    ██╔══██╗██╔══██╗██╔════╝     ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██╔════╝
    ██████╔╝███████║██║  ███╗    ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝███████║██║        ██║   ██║██║   ██║█████╗  
    ██╔══██╗██╔══██║██║   ██║    ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██║╚██╗ ██╔╝██╔══╝  
    ██║  ██║██║  ██║╚██████╔╝    ██║██║ ╚████║   ██║   ███████╗██║  ██║██║  ██║╚██████╗   ██║   ██║ ╚████╔╝ ███████╗
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝
{Colors.RESET}
        """
        
        print(welcome_art)
        
        welcome_text = f"""
            Welcome to RAG Interactive CLI

            Features:
            {Symbols.BULLET} Process PDF documents into intelligent chunks
            {Symbols.BULLET} Store chunks in ChromaDB vector database  
            {Symbols.BULLET} Perform semantic search with similarity scoring
            {Symbols.BULLET} Interactive search with real-time results
            {Symbols.BULLET} Collection management and statistics
        """
        
        self._print_box(welcome_text.strip(), width=90, color=Colors.BRIGHT_BLUE)
        
        # Show connection status
        print(f"\n{Colors.BRIGHT_YELLOW}System Status:{Colors.RESET}")
        print(f"{Symbols.DATABASE} ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
        print(f"{Symbols.FOLDER} Collection: {COLLECTION_NAME}")
        
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")

    def show_main_menu(self):
        """Display the main menu"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self._print_header("RAG INTERACTIVE CLI - MAIN MENU")
        
        print(f"{Colors.BRIGHT_YELLOW}Document Operations:{Colors.RESET}")
        self._print_menu_item("1", Symbols.FILE, "Chunk PDF", "Process PDF into text chunks with smart indentation")
        self._print_menu_item("2", Symbols.DATABASE, "Store PDF", "Process and store PDF chunks in ChromaDB")
        
        print(f"\n{Colors.BRIGHT_YELLOW}Search Operations:{Colors.RESET}")
        self._print_menu_item("3", Symbols.SEARCH, "Search Chunks", "Search for chunks with similarity scoring")
        self._print_menu_item("4", "LOOP", "Interactive Search", "Continuous search mode with live results")
        
        print(f"\n{Colors.BRIGHT_YELLOW}Management:{Colors.RESET}")
        self._print_menu_item("5", Symbols.CHART, "Show Statistics", "Display collection stats and sample data")
        self._print_menu_item("6", Symbols.GEAR, "Settings", "Configure search and processing parameters")
        self._print_menu_item("7", "CLEAR", "Clear Collection", "Remove all documents from collection")
        self._print_menu_item("8", Symbols.TRASH, "Delete Collection", "Permanently delete the entire collection")
        
        print(f"\n{Colors.BRIGHT_YELLOW}System:{Colors.RESET}")
        self._print_menu_item("0", "EXIT", "Exit", "Exit the application")
        
        self._print_separator()
        
        # Show current settings
        gemini_status = "ON" if self.settings['use_gemini'] else "OFF"
        print(f"{Colors.DIM}Current Settings: min_tokens={self.settings['min_tokens']}, max_tokens={self.settings['max_tokens']}, top_k={self.settings['top_k']}, Gemini AI: {gemini_status}{Colors.RESET}")

    def chunk_pdf_interactive(self):
        """Interactive PDF chunking"""
        self._print_header("PDF CHUNKING")
        
        # Get PDF path
        pdf_path = input(f"{Symbols.FILE} Enter PDF file path: ").strip()
        if not pdf_path:
            self._print_warning("No file path provided")
            return
        
        if not os.path.exists(pdf_path):
            self._print_error(f"File does not exist: {pdf_path}")
            return
        
        if not pdf_path.lower().endswith('.pdf'):
            self._print_error("File must be a PDF")
            return
        
        # Get parameters
        print(f"\n{Colors.YELLOW}Chunking Parameters:{Colors.RESET}")
        min_tokens = input(f"Min tokens per chunk [{self.settings['min_tokens']}]: ").strip()
        min_tokens = int(min_tokens) if min_tokens else self.settings['min_tokens']
        
        max_tokens = input(f"Max tokens per chunk [{self.settings['max_tokens']}]: ").strip()
        max_tokens = int(max_tokens) if max_tokens else self.settings['max_tokens']
        
        save_to_file = input(f"Save chunks to JSON file? (y/N): ").strip().lower() == 'y'
        output_file = None
        if save_to_file:
            output_file = input("Output JSON file path: ").strip()
        
        print(f"\n{Colors.BRIGHT_BLUE}Processing PDF...{Colors.RESET}")
        
        try:
            # Create chunks
            chunks = pdf_to_chunks_smart_indent(pdf_path, min_tokens=min_tokens, max_tokens=max_tokens)
            
            if not chunks:
                self._print_warning("No chunks created from PDF")
                return
            
            # Display results
            total_tokens = sum(chunk['token_est'] for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            results = f"""
                Processing Results:
                {Symbols.CHECK} Successfully processed: {os.path.basename(pdf_path)}
                {Symbols.BULLET} Total chunks created: {len(chunks)}
                {Symbols.BULLET} Total tokens: {total_tokens:,}
                {Symbols.BULLET} Average tokens per chunk: {avg_tokens:.1f}
                {Symbols.BULLET} Token range: {min_tokens} - {max_tokens}
                            """
            
            self._print_box(results.strip(), color=Colors.BRIGHT_GREEN)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                self._print_success(f"Chunks saved to: {output_file}")
            
            # Show sample chunks
            show_samples = input(f"\nShow sample chunks? (y/N): ").strip().lower() == 'y'
            if show_samples:
                print(f"\n{Colors.BRIGHT_YELLOW}Sample Chunks:{Colors.RESET}")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"\n{Colors.CYAN}Chunk {i+1}:{Colors.RESET}")
                    print(f"  Page: {chunk.get('page_range', 'Unknown')}")
                    print(f"  Tokens: {chunk.get('token_est', 0)}")
                    content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                    print(f"  Content: {content_preview}")
                    self._print_separator(width=60)
        
        except Exception as e:
            self._print_error(f"Error processing PDF: {e}")

    def store_pdf_interactive(self):
        """Interactive PDF storage"""
        self._print_header("STORE PDF CHUNKS")
        
        store = self._init_store()
        if not store:
            return
        
        # Get PDF path
        pdf_path = input(f"{Symbols.FILE} Enter PDF file path: ").strip()
        if not pdf_path:
            self._print_warning("No file path provided")
            return
        
        if not os.path.exists(pdf_path):
            self._print_error(f"File does not exist: {pdf_path}")
            return
        
        # Get parameters
        print(f"\n{Colors.YELLOW}Processing Parameters:{Colors.RESET}")
        min_tokens = input(f"Min tokens per chunk [{self.settings['min_tokens']}]: ").strip()
        min_tokens = int(min_tokens) if min_tokens else self.settings['min_tokens']
        
        max_tokens = input(f"Max tokens per chunk [{self.settings['max_tokens']}]: ").strip()
        max_tokens = int(max_tokens) if max_tokens else self.settings['max_tokens']
        
        print(f"\n{Colors.BRIGHT_BLUE}Processing and storing PDF chunks...{Colors.RESET}")
        
        try:
            success = store.store_pdf_chunks(pdf_path, min_tokens, max_tokens)
            
            if success:
                self._print_success("PDF chunks successfully stored in ChromaDB")
                
                # Show updated collection stats
                count = store.collection.count()
                self._print_info(f"Collection now contains {count} documents")
            else:
                self._print_error("Failed to store PDF chunks")
        
        except Exception as e:
            self._print_error(f"Error storing PDF: {e}")

    def search_chunks_interactive(self):
        """Interactive chunk search"""
        self._print_header("SEARCH CHUNKS")
        
        searcher = self._init_searcher()
        if not searcher:
            return
        
        # Get search parameters
        query = input(f"{Symbols.SEARCH} Enter your search query: ").strip()
        if not query:
            self._print_warning("No search query provided")
            return
        
        print(f"\n{Colors.YELLOW}Search Parameters:{Colors.RESET}")
        top_k = input(f"Number of results [{self.settings['top_k']}]: ").strip()
        top_k = int(top_k) if top_k else self.settings['top_k']
        
        min_similarity = input(f"Minimum similarity (0.0-1.0) [{self.settings['min_similarity']}]: ").strip()
        min_similarity = float(min_similarity) if min_similarity else self.settings['min_similarity']
        
        show_raw_chunks = input(f"Show raw chunks? (y/N): ").strip().lower() == 'y'
        show_content = show_raw_chunks  # For compatibility
        
        print(f"\n{Colors.BRIGHT_BLUE}Searching for: '{query}'...{Colors.RESET}")
        
        try:
            results = searcher.search_chunks(query, top_k, min_similarity)
            
            if not results:
                self._print_warning("No matching chunks found")
                return
            
            # Get AI response first if enabled
            if self.settings['use_gemini']:
                print(f"\n{Colors.BRIGHT_BLUE}Generating AI response...{Colors.RESET}")
                ai_response = self._get_gemini_response(results, query)
                self._print_gemini_response(ai_response, query)
            
            # Display enhanced results if requested
            if self.settings['show_raw_chunks']:
                self._print_success(f"Found {len(results)} matching chunks")
                self._print_separator(width=100)
                
                for result in results:
                    print(f"\n{Colors.BRIGHT_CYAN}#{result['rank']} - {result['file_name']} (Chunk {result['chunk_id']}){Colors.RESET}")
                    print(f"   {Symbols.DIAMOND} Page: {result['page_range']} | Similarity: {Colors.BRIGHT_GREEN}{result['similarity']:.4f}{Colors.RESET} | Tokens: {result['token_count']}")
                    print(f"   {Symbols.DIAMOND} ID: {Colors.DIM}{result['id']}{Colors.RESET}")
                    
                    if result.get('indent_levels'):
                        print(f"   {Symbols.DIAMOND} Structure levels: {result['indent_levels']}")
                    
                    if show_content:
                        print(f"\n   {Colors.YELLOW}Content:{Colors.RESET}")
                        # Format content with proper indentation
                        for line in result['content'].split('\n'):
                            if line.strip():
                                print(f"   {Colors.DIM}│{Colors.RESET} {line}")
                    
                    self._print_separator("─", width=100, color=Colors.DIM)
            elif not self.settings['use_gemini']:
                # Show basic results if Gemini is disabled
                self._print_success(f"Found {len(results)} matching chunks")
                for result in results:
                    print(f"\n{Colors.BRIGHT_CYAN}#{result['rank']} - {result['file_name']} (Chunk {result['chunk_id']}){Colors.RESET}")
                    print(f"   {Symbols.DIAMOND} Page: {result['page_range']} | Similarity: {Colors.BRIGHT_GREEN}{result['similarity']:.4f}{Colors.RESET}")
        
        except Exception as e:
            self._print_error(f"Search error: {e}")

    def interactive_search_mode(self):
        """Continuous interactive search mode"""
        self._print_header("INTERACTIVE SEARCH MODE")
        
        searcher = self._init_searcher()
        if not searcher:
            return
        
        print(f"{Colors.BRIGHT_YELLOW}Interactive Search Commands:{Colors.RESET}")
        print(f"  {Symbols.BULLET} Enter any question to search")
        print(f"  {Symbols.BULLET} 'config' - Change search settings")
        print(f"  {Symbols.BULLET} 'stats' - Show collection statistics")
        print(f"  {Symbols.BULLET} 'gemini' - Toggle AI response on/off")
        print(f"  {Symbols.BULLET} 'chunks' - Toggle raw chunks display")
        print(f"  {Symbols.BULLET} 'back' - Return to main menu")
        print(f"  {Symbols.BULLET} 'quit' - Exit application")
        
        self._print_separator()
        
        while True:
            try:
                gemini_status = "ON" if self.settings['use_gemini'] else "OFF"
                chunks_status = "ON" if self.settings['show_raw_chunks'] else "OFF"
                print(f"\n{Colors.DIM}Settings: top-k={self.settings['top_k']}, similarity={self.settings['min_similarity']:.2f}, AI={gemini_status}, Chunks={chunks_status}{Colors.RESET}")
                query = input(f"{Colors.BRIGHT_CYAN}Search{Colors.RESET} {Symbols.ARROW_RIGHT} ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['back', 'return', 'menu']:
                    break
                elif query.lower() in ['quit', 'exit', 'q']:
                    sys.exit(0)
                elif query.lower() == 'stats':
                    self.show_collection_stats()
                    continue
                elif query.lower() == 'config':
                    self.configure_search_settings()
                    continue
                elif query.lower() == 'gemini':
                    self.settings['use_gemini'] = not self.settings['use_gemini']
                    status = "enabled" if self.settings['use_gemini'] else "disabled"
                    self._print_success(f"AI response {status}")
                    continue
                elif query.lower() == 'chunks':
                    self.settings['show_raw_chunks'] = not self.settings['show_raw_chunks']
                    status = "enabled" if self.settings['show_raw_chunks'] else "disabled"
                    self._print_success(f"Raw chunks display {status}")
                    continue
                
                # Perform search
                print(f"{Colors.BRIGHT_BLUE}Searching...{Colors.RESET}")
                results = searcher.search_chunks(
                    query, 
                    self.settings['top_k'], 
                    self.settings['min_similarity']
                )
                
                if not results:
                    self._print_warning("No matching chunks found")
                    continue
                
                # Get AI response if enabled
                if self.settings['use_gemini']:
                    ai_response = self._get_gemini_response(results, query)
                    self._print_gemini_response(ai_response, query)
                
                # Display compact results if raw chunks are enabled
                if self.settings['show_raw_chunks']:
                    print(f"\n{Colors.BRIGHT_GREEN}Found {len(results)} results:{Colors.RESET}")
                    
                    for result in results:
                        print(f"\n{Colors.CYAN}#{result['rank']}{Colors.RESET} {Colors.BOLD}{result['file_name']}{Colors.RESET} (p.{result['page_range']}) - {Colors.BRIGHT_GREEN}{result['similarity']:.3f}{Colors.RESET}")
                        
                        if self.settings['show_content']:
                            # Show truncated content
                            content = result['content']
                            if len(content) > self.settings['max_content_length']:
                                content = content[:self.settings['max_content_length']] + "..."
                            
                            for line in content.split('\n')[:3]:  # Show first 3 lines
                                if line.strip():
                                    print(f"    {Colors.DIM}{line}{Colors.RESET}")
                            if len(result['content'].split('\n')) > 3:
                                print(f"    {Colors.DIM}... (truncated){Colors.RESET}")
                elif not self.settings['use_gemini']:
                    # Show basic summary if Gemini is disabled
                    print(f"\n{Colors.BRIGHT_GREEN}Found {len(results)} matching chunks from:{Colors.RESET}")
                    files = list(set([r['file_name'] for r in results]))
                    for file in files:
                        print(f"  {Colors.CYAN}• {file}{Colors.RESET}")
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Returning to main menu...{Colors.RESET}")
                break
            except Exception as e:
                self._print_error(f"Search error: {e}")

    def show_collection_stats(self):
        """Display collection statistics"""
        self._print_header("COLLECTION STATISTICS")
        
        store = self._init_store()
        if not store:
            return
        
        try:
            stats = store.get_collection_stats()
            
            stats_text = f"""
                Collection Information:
                {Symbols.DATABASE} Name: {stats.get('collection_name', 'Unknown')}
                {Symbols.CHART} Total Documents: {stats.get('total_documents', 0):,}
                {Symbols.GEAR} Host: {CHROMA_HOST}:{CHROMA_PORT}
                            """
            
            self._print_box(stats_text.strip(), color=Colors.BRIGHT_BLUE)
            
            # Show sample documents
            if stats.get('sample_documents'):
                print(f"\n{Colors.BRIGHT_YELLOW}Sample Documents:{Colors.RESET}")
                for i, doc in enumerate(stats['sample_documents'], 1):
                    print(f"\n{Colors.CYAN}Sample {i}:{Colors.RESET}")
                    print(f"  {Symbols.BULLET} ID: {Colors.DIM}{doc['id']}{Colors.RESET}")
                    print(f"  {Symbols.BULLET} Preview: {doc['content_preview']}")
                    if doc.get('metadata'):
                        print(f"  {Symbols.BULLET} File: {Colors.BOLD}{doc['metadata'].get('file_name', 'Unknown')}{Colors.RESET}")
                        print(f"  {Symbols.BULLET} Pages: {doc['metadata'].get('page_range', 'Unknown')}")
                    self._print_separator(width=60, color=Colors.DIM)
        
        except Exception as e:
            self._print_error(f"Error getting statistics: {e}")

    def configure_search_settings(self):
        """Configure search and processing settings"""
        self._print_header("SETTINGS CONFIGURATION")
        
        print(f"{Colors.BRIGHT_YELLOW}Current Settings:{Colors.RESET}")
        print(f"  {Symbols.GEAR} Min tokens per chunk: {self.settings['min_tokens']}")
        print(f"  {Symbols.GEAR} Max tokens per chunk: {self.settings['max_tokens']}")
        print(f"  {Symbols.GEAR} Search results (top-k): {self.settings['top_k']}")
        print(f"  {Symbols.GEAR} Minimum similarity: {self.settings['min_similarity']:.2f}")
        print(f"  {Symbols.GEAR} Use Gemini AI: {'Yes' if self.settings['use_gemini'] else 'No'}")
        print(f"  {Symbols.GEAR} Show raw chunks: {'Yes' if self.settings['show_raw_chunks'] else 'No'}")
        print(f"  {Symbols.GEAR} Show content: {'Yes' if self.settings['show_content'] else 'No'}")
        print(f"  {Symbols.GEAR} Max content length: {self.settings['max_content_length']}")
        
        print(f"\n{Colors.YELLOW}Update Settings (press Enter to keep current value):{Colors.RESET}")
        
        try:
            # Processing settings
            new_min = input(f"Min tokens [{self.settings['min_tokens']}]: ").strip()
            if new_min:
                self.settings['min_tokens'] = max(50, int(new_min))
            
            new_max = input(f"Max tokens [{self.settings['max_tokens']}]: ").strip()
            if new_max:
                self.settings['max_tokens'] = max(self.settings['min_tokens'], int(new_max))
            
            # Search settings
            new_k = input(f"Search results (top-k) [{self.settings['top_k']}]: ").strip()
            if new_k:
                self.settings['top_k'] = max(1, min(int(new_k), 50))
            
            new_sim = input(f"Minimum similarity (0.0-1.0) [{self.settings['min_similarity']:.2f}]: ").strip()
            if new_sim:
                self.settings['min_similarity'] = max(0.0, min(float(new_sim), 1.0))
            
            # AI settings
            use_gemini = input(f"Use Gemini AI ({'Y' if self.settings['use_gemini'] else 'N'}): ").strip().lower()
            if use_gemini in ['y', 'yes', 'true']:
                self.settings['use_gemini'] = True
            elif use_gemini in ['n', 'no', 'false']:
                self.settings['use_gemini'] = False
            
            show_raw_chunks = input(f"Show raw chunks ({'Y' if self.settings['show_raw_chunks'] else 'N'}): ").strip().lower()
            if show_raw_chunks in ['y', 'yes', 'true']:
                self.settings['show_raw_chunks'] = True
            elif show_raw_chunks in ['n', 'no', 'false']:
                self.settings['show_raw_chunks'] = False
            
            # Display settings
            show_content = input(f"Show content ({'Y' if self.settings['show_content'] else 'N'}): ").strip().lower()
            if show_content in ['y', 'yes', 'true']:
                self.settings['show_content'] = True
            elif show_content in ['n', 'no', 'false']:
                self.settings['show_content'] = False
            
            if self.settings['show_content']:
                new_len = input(f"Max content length [{self.settings['max_content_length']}]: ").strip()
                if new_len:
                    self.settings['max_content_length'] = max(100, int(new_len))
            
            self._print_success("Settings updated successfully")
        
        except ValueError:
            self._print_error("Invalid value entered")
        except Exception as e:
            self._print_error(f"Error updating settings: {e}")

    def clear_collection_interactive(self):
        """Interactive collection clearing"""
        self._print_header("CLEAR COLLECTION")
        
        store = self._init_store()
        if not store:
            return
        
        try:
            count = store.collection.count()
            if count == 0:
                self._print_info("Collection is already empty")
                return
            
            warning_text = f"""
                WARNING: DESTRUCTIVE OPERATION
                This will permanently delete {count:,} documents from the collection.
                This action cannot be undone.

                Collection: {COLLECTION_NAME}
                Documents to delete: {count:,}
            """
            
            self._print_box(warning_text.strip(), color=Colors.BRIGHT_RED)
            
            confirm = input(f"\n{Colors.BRIGHT_RED}Type 'DELETE' to confirm: {Colors.RESET}").strip()
            
            if confirm != 'DELETE':
                self._print_info("Operation cancelled")
                return
            
            print(f"\n{Colors.BRIGHT_BLUE}Clearing collection...{Colors.RESET}")
            success = store.clear_collection()
            
            if success:
                self._print_success("Collection cleared successfully")
            else:
                self._print_error("Failed to clear collection")
        
        except Exception as e:
            self._print_error(f"Error clearing collection: {e}")

    def delete_collection_interactive(self):
        """Interactive collection deletion"""
        self._print_header("DELETE COLLECTION")
        
        searcher = self._init_searcher()
        if not searcher:
            return
        
        try:
            count = searcher.collection.count()
            
            warning_text = f"""
                CRITICAL WARNING: PERMANENT DELETION
                This will permanently delete the entire collection and all its data.
                This action is IRREVERSIBLE and will require recreating the collection.

                Collection: {COLLECTION_NAME}
                Documents: {count:,}
            """
            
            self._print_box(warning_text.strip(), color=Colors.BRIGHT_RED)
            
            print(f"\n{Colors.BRIGHT_RED}This is a destructive operation that cannot be undone!{Colors.RESET}")
            confirm1 = input(f"Type the collection name '{COLLECTION_NAME}' to continue: ").strip()
            
            if confirm1 != COLLECTION_NAME:
                self._print_info("Operation cancelled")
                return
            
            confirm2 = input(f"Type 'DELETE FOREVER' to confirm: ").strip()
            
            if confirm2 != 'DELETE FOREVER':
                self._print_info("Operation cancelled")
                return
            
            print(f"\n{Colors.BRIGHT_BLUE}Deleting collection...{Colors.RESET}")
            success = searcher.delete_collection()
            
            if success:
                self._print_success("Collection deleted successfully")
                self._print_info("You will need to recreate the collection to store new data")
            else:
                self._print_error("Failed to delete collection")
        
        except Exception as e:
            self._print_error(f"Error deleting collection: {e}")

    def run(self):
        """Main application loop"""
        try:
            self.show_welcome()
            
            while True:
                try:
                    self.show_main_menu()
                    
                    choice = input(f"\n{Colors.BRIGHT_CYAN}Select option [0-8]: {Colors.RESET}").strip()
                    
                    if choice == '0':
                        print(f"\n{Colors.BRIGHT_GREEN}Thank you for using RAG Interactive CLI!{Colors.RESET}")
                        break
                    
                    elif choice == '1':
                        self.chunk_pdf_interactive()
                    
                    elif choice == '2':
                        self.store_pdf_interactive()
                    
                    elif choice == '3':
                        self.search_chunks_interactive()
                    
                    elif choice == '4':
                        self.interactive_search_mode()
                    
                    elif choice == '5':
                        self.show_collection_stats()
                    
                    elif choice == '6':
                        self.configure_search_settings()
                    
                    elif choice == '7':
                        self.clear_collection_interactive()
                    
                    elif choice == '8':
                        self.delete_collection_interactive()
                    
                    else:
                        self._print_warning("Invalid option. Please choose 0-8.")
                    
                    if choice != '0':
                        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
                
                except KeyboardInterrupt:
                    print(f"\n\n{Colors.YELLOW}Operation interrupted. Returning to main menu...{Colors.RESET}")
                    time.sleep(1)
                    continue
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_GREEN}Thank you for using RAG Interactive CLI!{Colors.RESET}")
            print(f"{Colors.DIM}Application terminated by user! {Symbols.STAR}{Colors.RESET}")
        
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            print(f"{Colors.DIM}Application will now exit.{Colors.RESET}")

def main():
    try:
        app = RAGInteractiveCLI()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
