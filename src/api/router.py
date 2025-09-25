from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
from chunking.store_chunks import ChromaChunkStore
from retriever.top_k import *
from llm.provider import gemini, openai
import re
import ast

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
    img: Optional[List[str]] = []

class SuggestionsResponse(BaseModel):
    success: bool
    summaries: Optional[Dict[str, str]] = None
    suggested_questions: Optional[List[str]] = None
    error: Optional[str] = None

class DeleteResponse(BaseModel):
    success: bool
    message: str
    deleted_count: Optional[int] = None

class FileProcessRequest(BaseModel):
    filename: str

class FileDeleteRequest(BaseModel):
    filename: str

class FileDeleteResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    deleted_count: Optional[int] = None
    chunk_ids: Optional[List[str]] = None
    error: Optional[str] = None

class FileGetChunksRequest(BaseModel):
    filename: str

class ChunkData(BaseModel):
    id: str
    content: str
    metadata: Dict

class FileGetChunksResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    chunk_count: Optional[int] = None
    chunks: Optional[List[ChunkData]] = None
    error: Optional[str] = None

class FileProcessResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    file_path: Optional[str] = None
    chunks_stored: Optional[int] = None
    file_type: Optional[str] = None
    processing_time: Optional[float] = None
    document_summary: Optional[str] = None
    total_tokens: Optional[int] = None
    has_images: Optional[bool] = None
    image_count: Optional[int] = None
    error: Optional[str] = None

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
            response_text, img_urls = split_text_and_urls(response_text)
            return QueryResponse(
                success=True,
                response=response_text,
                img=img_urls,
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

@router.post("/process-file", response_model=FileProcessResponse)
def process_single_file(request: FileProcessRequest):
    """
    Process a single file from the docs folder and store chunks in ChromaDB with summary generation
    """
    import time
    
    start_time = time.time()
    
    try:
        log.info(f"Received /process-file request for: {request.filename}")
        
        # Initialize the chunk store
        store = ChromaChunkStore()
        
        # Get the docs folder path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        docs_folder = os.path.join(project_root, 'docs')
        file_path = os.path.join(docs_folder, request.filename)
        
        # Check if docs folder exists
        if not os.path.exists(docs_folder):
            log.error(f"Docs folder not found: {docs_folder}")
            raise HTTPException(status_code=404, detail=f"Docs folder not found: {docs_folder}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            log.error(f"File not found: {request.filename}")
            raise HTTPException(status_code=404, detail=f"File not found in docs folder: {request.filename}")
        
        # Use the store_chunks_from_single_file method
        result = store.store_chunks_from_single_file(file_path)
        
        processing_time = time.time() - start_time
        
        if result["success"]:
            log.info(f"Successfully processed {request.filename}: {result['chunks_stored']} chunks stored in {processing_time:.2f}s")
            
            return FileProcessResponse(
                success=True,
                message=result["message"],
                filename=result["filename"],
                file_path=result["file_path"],
                chunks_stored=result["chunks_stored"],
                file_type=result["file_type"],
                processing_time=processing_time,
                document_summary=result.get("document_summary", ""),
                total_tokens=int(result.get("total_tokens", 0)),  # Convert to int
                has_images=result.get("has_images", False),
                image_count=result.get("image_count", 0)
            )
        else:
            log.error(f"Failed to process {request.filename}: {result.get('error', 'Unknown error')}")
            
            return FileProcessResponse(
                success=False,
                message=result["message"],
                filename=result["filename"],
                file_path=file_path,
                chunks_stored=result["chunks_stored"],
                file_type=result.get("file_type", "unknown"),
                processing_time=processing_time,
                error=result.get("error", "Processing failed")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        log.exception(f"Unhandled error in /process-file endpoint for {request.filename}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error processing {request.filename}: {str(e)}"
        )

@router.get("/list-files", response_model=Dict[str, List[str]])
def list_available_files():
    """
    List all available files in the docs folder that can be processed
    """
    try:
        log.info("Received /list-files request")
        
        # Get the docs folder path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        docs_folder = os.path.join(project_root, 'docs')
        
        if not os.path.exists(docs_folder):
            log.error(f"Docs folder not found: {docs_folder}")
            raise HTTPException(status_code=404, detail=f"Docs folder not found: {docs_folder}")
        
        # Get all files in docs folder
        all_files = [f for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))]
        
        # Categorize files by type
        supported_files = {
            "pdf": [],
            "txt": [],
            "markdown": [],
            "docx": [],
            "excel": [],
            "unsupported": []
        }
        
        for filename in all_files:
            filename_lower = filename.lower()
            if filename_lower.endswith(".pdf"):
                supported_files["pdf"].append(filename)
            elif filename_lower.endswith(".txt"):
                supported_files["txt"].append(filename)
            elif filename_lower.endswith(".md"):
                supported_files["markdown"].append(filename)
            elif filename_lower.endswith(".docx"):
                supported_files["docx"].append(filename)
            elif filename_lower.endswith((".xlsx", ".xls")):
                supported_files["excel"].append(filename)
            else:
                supported_files["unsupported"].append(filename)
        
        # Add summary
        total_supported = sum(len(files) for key, files in supported_files.items() if key != "unsupported")
        supported_files["summary"] = [
            f"Total files: {len(all_files)}",
            f"Supported files: {total_supported}",
            f"Unsupported files: {len(supported_files['unsupported'])}"
        ]
        
        log.info(f"Listed {len(all_files)} files from docs folder")
        return supported_files
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error in /list-files endpoint")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.delete("/delete-file", response_model=FileDeleteResponse)
def delete_file_chunks(request: FileDeleteRequest):
    """
    Delete all chunks and metadata for a specific file from ChromaDB
    """
    try:
        log.info(f"Received /delete-file request for: {request.filename}")
        
        # Initialize the chunk store
        store = ChromaChunkStore()
        
        # Delete all chunks for this file
        result = store.delete_file_chunks(request.filename)
        
        if result["success"]:
            log.info(f"Successfully deleted {result['deleted_count']} chunks for {request.filename}")
            
            return FileDeleteResponse(
                success=True,
                message=result["message"],
                filename=result["filename"],
                deleted_count=result["deleted_count"],
                chunk_ids=result.get("chunk_ids", [])
            )
        else:
            log.warning(f"Failed to delete chunks for {request.filename}: {result.get('error', 'Unknown error')}")
            
            return FileDeleteResponse(
                success=False,
                message=result["message"],
                filename=result["filename"],
                deleted_count=result["deleted_count"],
                error=result.get("error", "Deletion failed")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Unhandled error in /delete-file endpoint for {request.filename}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error deleting chunks for {request.filename}: {str(e)}"
        )


@router.post("/get-file-chunks", response_model=FileGetChunksResponse)
async def get_file_chunks(request: FileGetChunksRequest):
    """
    Get all chunks for a specific file by filename
    """
    try:
        log.info(f"Getting chunks for file: {request.filename}")
        
        # Initialize ChromaDB store
        store = ChromaChunkStore()
        
        # Get chunks for the specific file
        result = store.get_file_chunks(request.filename)
        
        if result["success"]:
            log.info(f"Successfully retrieved {result['chunk_count']} chunks for {request.filename}")
            
            # Convert chunks to Pydantic models
            chunk_data = []
            for chunk in result["chunks"]:
                chunk_data.append(ChunkData(
                    id=chunk["id"],
                    content=chunk["content"],
                    metadata=chunk["metadata"]
                ))
            
            return FileGetChunksResponse(
                success=True,
                message=result["message"],
                filename=result["filename"],
                chunk_count=result["chunk_count"],
                chunks=chunk_data
            )
        else:
            log.warning(f"No chunks found for {request.filename}: {result.get('message', 'Unknown error')}")
            
            return FileGetChunksResponse(
                success=False,
                message=result["message"],
                filename=result["filename"],
                chunk_count=result["chunk_count"],
                chunks=[],
                error=result.get("error", "No chunks found")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Unhandled error in /get-file-chunks endpoint for {request.filename}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error retrieving chunks for {request.filename}: {str(e)}"
        )


def split_text_and_urls(s: str):
    # Regex: tìm list bắt đầu bằng [ và bên trong có 'http'
    match = re.search(r"\[(?:'|\")https?://.*\]$", s.strip())
    if not match:
        return s.strip(), []

    list_str = match.group(0)   # phần mảng url
    text_part = s[:match.start()].strip().rstrip(",")  # phần text trước đó

    try:
        urls = ast.literal_eval(list_str)
    except Exception:
        urls = []
    
    return text_part, urls