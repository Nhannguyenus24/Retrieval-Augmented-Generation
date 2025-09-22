from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import os
from chunking.store_chunks import ChromaChunkStore
from retriever.top_k import *
from llm.provider import gemini, openai

router = APIRouter()
log = logging.getLogger("uvicorn")

class NameRequest(BaseModel):
    name: str

class ChunkRequest(BaseModel):
    min_tokens: Optional[int] = 300
    max_tokens: Optional[int] = 500

class ChunkResponse(BaseModel):
    success: bool
    message: str
    total_chunks: Optional[int] = None
    total_documents: Optional[int] = None

class QueryRequest(BaseModel):
    request_message: str
    model: str  # "gemini" or "openai"
    top_k: Optional[int] = 5
    min_similarity: Optional[float] = 0.0

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    chunks_found: Optional[int] = None

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/chunks", response_model=ChunkResponse)
def process_and_store_chunks(request: ChunkRequest):
    """
    Process and store all documents from the docs folder into ChromaDB
    """
    try:
        log.info(f"Received /chunks request with min_tokens={request.min_tokens}, max_tokens={request.max_tokens}")
        # Initialize the chunk store
        store = ChromaChunkStore()
        
        # Get the docs folder path (relative to project root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        docs_folder = os.path.join(project_root, 'docs')
        
        if not os.path.exists(docs_folder):
            log.error(f"Docs folder not found: {docs_folder}")
            raise HTTPException(status_code=404, detail=f"Docs folder not found: {docs_folder}")
        
        # Process all files in docs folder
        success = store.store_chunks(
            docs_folder, 
            min_tokens=request.min_tokens, 
            max_tokens=request.max_tokens
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process and store chunks")
        
        # Get collection statistics
        stats = store.get_collection_stats()
        total_chunks = stats.get('total_documents', 0)
        log.info(f"Collection stats: {stats}")
        
        return ChunkResponse(
            success=True,
            message=f"Successfully processed and stored chunks from docs folder",
            total_chunks=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error in /chunks endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing chunks: {str(e)}")

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query documents using RAG with specified model (gemini or openai)
    """
    try:
        log.info(f"Received /query request: model={request.model}, top_k={request.top_k}, min_similarity={request.min_similarity}")
        # Validate model parameter
        if request.model.lower() not in ["gemini", "openai"]:
            log.warning(f"Invalid model: {request.model}")
            raise HTTPException(status_code=400, detail="Model must be either 'gemini' or 'openai'")
        
        # Search for relevant chunks
        chunks = get_top_k(
            query=request.request_message,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        log.info(f"Found {len(chunks) if chunks else 0} chunks for query")
        if not chunks:
            return QueryResponse(
                success=True,
                response="No relevant documents found for your query.",
                chunks_found=0
            )
        
        # Prepare content for LLM
        contents = ""
        for i, chunk in enumerate(chunks, 1):
            contents += f"Document {i}:\n{chunk['content']}\n\n"
        
        # Generate response using specified model
        if request.model.lower() == "gemini":
            log.debug("Using Gemini model for response generation")
            response_text = gemini.one_shot(
                contents=contents,
                user_query=request.request_message,
                template_name="one_shot.jinja"
            )
        else:  # openai
            log.debug("Using OpenAI model for response generation")
            response_text = openai.one_shot(
                contents=contents,
                user_query=request.request_message,
                template_name="one_shot.jinja"
            )
        log.info("LLM response generated successfully")

        return QueryResponse(
            success=True,
            response=response_text,
            chunks_found=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error in /query endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
