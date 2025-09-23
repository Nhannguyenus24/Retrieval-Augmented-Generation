from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
from chunking.store_chunks import ChromaChunkStore
from retriever.top_k import *
from llm.provider import gemini, openai

router = APIRouter()
log = logging.getLogger("router")

class NameRequest(BaseModel):
    name: str
class ChunkResponse(BaseModel):
    success: bool
    message: str
    total_chunks: Optional[int] = None
    total_documents: Optional[int] = None

class QueryRequest(BaseModel):
    request_message: str
    model: str  # "gemini" or "openai"

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    chunks_found: Optional[int] = None

class SuggestionsResponse(BaseModel):
    success: bool
    summaries: Optional[Dict[str, str]] = None
    suggested_questions: Optional[List[str]] = None
    error: Optional[str] = None

class DeleteResponse(BaseModel):
    success: bool
    message: str
    deleted_count: Optional[int] = None

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/chunks", response_model=ChunkResponse)
def process_and_store_chunks():
    """
    Process and store all documents from the docs folder into ChromaDB
    """
    try:
        log.info(f"Received /chunks request")
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
            docs_folder
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
        store = ChromaChunkStore()
        log.info(f"Received /query request: model={request.model}")
        # Validate model parameter
        if request.model.lower() not in ["gemini", "openai"]:
            log.warning(f"Invalid model: {request.model}")
            raise HTTPException(status_code=400, detail="Model must be either 'gemini' or 'openai'")
        
        # Search for relevant chunks
        chunks = get_expanded_results(
            query=request.request_message,
        )
        log.info(f"Found {len(chunks) if chunks else 0} chunks for query")
        if not chunks:
            return QueryResponse(
                success=True,
                response="No relevant documents found for your query.",
                chunks_found=0
            )
        
        transform_request = store.validate_query(request.request_message)
        log.info(transform_request)
        if transform_request.startswith("VALID:"):
            if request.model.lower() == "gemini":
                log.debug("Using Gemini model for response generation")
                response_text = gemini.one_shot(
                    contents=str(chunks),
                    user_query=transform_request.replace("VALID:", "").strip(),
                    template_name="one_shot.jinja"
                )
            else:  # openai
                log.debug("Using OpenAI model for response generation")
                response_text = openai.one_shot(
                    contents=str(chunks),
                    user_query=transform_request.replace("VALID:", "").strip(),
                    template_name="one_shot.jinja"
                )
            return QueryResponse(
                success=True,
                response=response_text,
                
                chunks_found=len(chunks)
            )
        elif transform_request.startswith("IRRELEVANT:"):
            return QueryResponse(
                success=False,
                error=transform_request.replace("IRRELEVANT:", "").strip(),
                chunks_found=0
            )
        elif transform_request.startswith("SUGGESTION:"):
            return QueryResponse(
                success=False,
                error="Bạn có thể tham khảo một số gợi ý sau: " + transform_request.replace("SUGGESTION:", "").strip(),
                chunks_found=len(chunks)
            )
        # Generate response using specified model
        # if request.model.lower() == "gemini":
        #     log.debug("Using Gemini model for response generation")
        #     response_text = gemini.one_shot(
        #         contents=contents,
        #         user_query=request.request_message,
        #         template_name="one_shot.jinja"
        #     )
        # else:  # openai
        #     log.debug("Using OpenAI model for response generation")
        #     response_text = openai.one_shot(
        #         contents=contents,
        #         user_query=request.request_message,
        #         template_name="one_shot.jinja"
        #     )
        log.info("LLM response generated successfully")

        # return QueryResponse(
        #     success=True,
        #     response=response_text,
        #     chunks_found=len(chunks)
        # )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error in /query endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/suggestions", response_model=SuggestionsResponse)
def get_suggestions():
    """
    Get document summaries and 4 suggested questions based on the content
    """
    try:
        log.info("Received /suggestions request")
        
        # Initialize the chunk store
        store = ChromaChunkStore()
        
        # Get document summaries
        summaries = store.get_document_summaries()
        log.info(f"Retrieved summaries for {len(summaries)} documents")
        
        # Generate suggested questions
        suggested_questions = store.generate_suggested_questions(summaries)
        log.info(f"Generated {len(suggested_questions)} suggested questions")
        
        return SuggestionsResponse(
            success=True,
            summaries=summaries,
            suggested_questions=suggested_questions
        )
        
    except Exception as e:
        log.exception("Unhandled error in /suggestions endpoint")
        raise HTTPException(status_code=500, detail=f"Error getting suggestions: {str(e)}")

@router.delete("/chunks", response_model=DeleteResponse)
def delete_all_chunks():
    """
    Delete all chunks from the ChromaDB collection
    """
    try:
        log.info("Received /chunks DELETE request")
        
        # Initialize the chunk store
        store = ChromaChunkStore()
        
        # Get current count before deletion
        stats = store.get_collection_stats()
        current_count = stats.get('total_documents', 0)
        
        if current_count == 0:
            log.info("Collection is already empty")
            return DeleteResponse(
                success=True,
                message="Collection is already empty",
                deleted_count=0
            )
        
        # Clear the collection
        success = store.clear_collection()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete chunks from collection")
        
        log.info(f"Successfully deleted {current_count} chunks from collection")
        
        return DeleteResponse(
            success=True,
            message=f"Successfully deleted all chunks from collection",
            deleted_count=current_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error in /chunks DELETE endpoint")
        raise HTTPException(status_code=500, detail=f"Error deleting chunks: {str(e)}")
