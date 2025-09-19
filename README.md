# RAG Interactive CLI

**Retrieval Augmented Generation** - Document Processing & Search System

## Technologies Used
- **Python 3.12+** - Core programming language
- **ChromaDB** - Vector database for embeddings storage
- **Sentence Transformers** - Text embedding generation
- **PyMuPDF** - PDF document processing
- **Docker** - ChromaDB containerization

## Installation

1. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start ChromaDB server:**
```bash
docker-compose up -d
```

4. **Add environment variable:**
- Create .env file in root folder
- Add GEMINI_API_KEY
```bash
GEMINI_API_KEY: your_gemini_api_key
```

## Usage

**Run RAG Interactive CLI:**
```bash
python src/cli/rag_interactive.py
```

## Current Features
- PDF document chunking with smart indentation (detect TOC page, recover heading for metadata, build breadcrumbs, indent level, avoid break sentence)
- Vector storage in ChromaDB (all-MiniLM-L6-v2)
- Semantic search with similarity scoring 
- Interactive search mode with real-time results
- Collection management (stats, clear, delete)
- Configurable settings (token limits, search parameters)

## Upcoming Features
- Improve chunking metadata (better recover heading pattern, page index for each chunks)
- Switch model function (all-MiniLM-L6-v2 for fast response, all-mpnet-base-v2 for better accuracy, paraphrase-multilingual-MiniLM-L12-v2 for multilanguage)
- Support Docx
- Support Markdown file
- Support TXT file
- Rerank strategy for better chunk selection
- One-shot query to llm
- Pipeline AI agent (multicall to llm for better answer)
- CI pipeline
- Logs for debug


