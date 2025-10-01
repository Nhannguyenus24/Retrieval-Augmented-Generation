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

4. **Configure environment variables:**
Create a `.env` file in the root folder with the following configuration:

```bash
# AI API Keys (Required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Cloudinary Configuration (Required for image processing)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# ChromaDB Configuration (Optional - defaults provided)
CHROMA_HOST=chromadb
CHROMA_PORT=8000
COLLECTION_NAME=SOF_DOCUMENTATION
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Configuration (Optional - defaults provided)
MIN_TOKENS=300
MAX_TOKENS=500
USE_TIKTOKEN=false
TAB_WIDTH=4
INDENT_STEP=4
```

5. **Run locally:**
```bash
uvicorn --app-dir src main:app --reload --host 127.0.0.1 --port 1234
```

## Usage

### **API Server:**
Access the API documentation at: `http://127.0.0.1:1234/docs`

**Available Endpoints:**
- `GET /health` - Health check
- `GET /chunks` - Process all files in docs folder
- `POST /query` - Query documents with RAG
- `GET /suggestions` - Get suggested questions
- `DELETE /chunks` - Clear all chunks
- `POST /process-file` - **NEW**: Process and store single file from docs folder with summary generation
- `GET /list-files` - **NEW**: List all available files in docs folder
- `DELETE /delete-file` - **NEW**: Delete all chunks for a specific file
- `POST /get-file-chunks` - **NEW**: Get all chunks for a specific file

**Single File Processing Example:**
```bash
# List available files
curl http://127.0.0.1:1234/list-files

# Process specific file
curl -X POST http://127.0.0.1:1234/process-file \
  -H "Content-Type: application/json" \
  -d '{"filename": "your-document.pdf"}'

# Get all chunks for specific file
curl -X POST http://127.0.0.1:1234/get-file-chunks \
  -H "Content-Type: application/json" \
  -d '{"filename": "your-document.pdf"}'

# Delete all chunks for specific file
curl -X DELETE http://127.0.0.1:1234/delete-file \
  -H "Content-Type: application/json" \
  -d '{"filename": "your-document.pdf"}'
```

### **Interactive CLI:**
```bash
python src/cli/rag_interactive.py
```
## Current Features
- PDF document chunking with smart indentation (detect TOC page, recover heading for metadata, build breadcrumbs, indent level, avoid break sentence)
- Vector storage in ChromaDB (all-MiniLM-L6-v2)
- Semantic search with similarity scoring 
- Index to document in retrieve answer
- Support Docx, Markdown file, TXT file, Excel files (.xlsx, .xls)
- One-shot query to llm 
- Backend server with OPENAPI v3
- Suggestion 4 question
- Rerank strategy for better chunk selection

## Upcoming Features
- Improve chunking metadata (better recover heading pattern, page index for each chunks, detect unnecessary pages)
- Switch model function (all-MiniLM-L6-v2 for fast response, all-mpnet-base-v2 for better accuracy)
- Pipeline AI agent (multicall to llm for better answer)
- CI pipeline
- Logs for debug
- Cronjob for question suggestion
- Upload/Download/Delete document
- Answer in selected document
- Strict search in English or Vietnamese
- AI thinking based on context (gemini only)
