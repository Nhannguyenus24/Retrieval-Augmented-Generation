import os
import sys
import json
from typing import List, Dict, Set
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import chromadb
from chromadb.config import Settings

# ===================== Configuration =====================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_chunk"
DEFAULT_TOP_K = 5
MAX_TOP_K = 100
DEFAULT_NEIGHBOR_WINDOW = 2  # Number of neighbors on each side

class TopKRetriever:
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
        except Exception as e:
            raise ConnectionError(f"ChromaDB connection error: {e}")
            
        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Failed to get collection '{collection_name}': {e}")

    def get_top_k_chunks(self, query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = 0.0) -> List[Dict]:
        """
        Get top-k most similar chunks for a query.
        
        Args:
            query: The search question/keywords
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List[Dict]: List of top-k chunks with metadata and similarity score
        """
        if not query.strip():
            return []
            
        try:
            # Perform vector similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, MAX_TOP_K),
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Process results
            chunks = []
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
                
                # Parse metadata
                try:
                    indent_levels = json.loads(metadata.get('indent_levels', '[]'))
                except:
                    indent_levels = []

                page_from = metadata.get('page_from', 'Unknown')
                page_to = metadata.get('page_to', 'Unknown')
                page_range = str(page_from) if page_from == page_to else f"{page_from}-{page_to}"

                chunk = {
                    'id': doc_id,
                    'similarity': round(similarity, 4),
                    'distance': round(distance, 4),
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'doc_id': metadata.get('doc_id', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', -1),
                    'chunk_index': metadata.get('chunk_index', -1),
                    'page_range': page_range,
                    'page_from': page_from,
                    'page_to': page_to,
                    'token_count': metadata.get('token_count', 0),
                    'indent_levels': indent_levels,
                    'content': doc,
                    'source_path': metadata.get('source_path', 'Unknown'),
                    'section_title': metadata.get('section_title'),
                    'heading_path': metadata.get('heading_path'),
                    'breadcrumbs': json.loads(metadata.get('breadcrumbs', '[]')) if metadata.get('breadcrumbs') else []
                }
                
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Search error: {e}")

    def get_neighbors_by_chunk_index(self, doc_id: str, chunk_index: int, window: int = DEFAULT_NEIGHBOR_WINDOW) -> List[Dict]:
        """
        Get neighboring chunks by chunk_index within the same document.
        
        Args:
            doc_id: Document ID to search within
            chunk_index: Center chunk index
            window: Number of neighbors on each side (total = 2*window + 1)
            
        Returns:
            List[Dict]: List of neighboring chunks sorted by chunk_index
        """
        try:
            # Calculate range of chunk indices to retrieve
            start_idx = max(0, chunk_index - window)
            end_idx = chunk_index + window
            
            # Build where clause for filtering
            where_clause = {
            "doc_id": doc_id,
            "chunk_index": {"$gte": start_idx, "$lte": end_idx}
            }
            
            # Query neighbors
            results = self.collection.get(
                where=where_clause,
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return []
            
            # Process and sort results
            neighbors = []
            documents = results['documents']
            metadatas = results['metadatas']
            ids = results['ids']
            
            for doc, metadata, doc_id in zip(documents, metadatas, ids):
                try:
                    indent_levels = json.loads(metadata.get('indent_levels', '[]'))
                except:
                    indent_levels = []

                page_from = metadata.get('page_from', 'Unknown')
                page_to = metadata.get('page_to', 'Unknown')
                page_range = str(page_from) if page_from == page_to else f"{page_from}-{page_to}"

                neighbor = {
                    'id': doc_id,
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'doc_id': metadata.get('doc_id', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', -1),
                    'chunk_index': metadata.get('chunk_index', -1),
                    'page_range': page_range,
                    'page_from': page_from,
                    'page_to': page_to,
                    'token_count': metadata.get('token_count', 0),
                    'indent_levels': indent_levels,
                    'content': doc,
                    'source_path': metadata.get('source_path', 'Unknown'),
                    'section_title': metadata.get('section_title'),
                    'heading_path': metadata.get('heading_path'),
                    'breadcrumbs': json.loads(metadata.get('breadcrumbs', '[]')) if metadata.get('breadcrumbs') else []
                }
                
                neighbors.append(neighbor)
            
            # Sort by chunk_index
            neighbors.sort(key=lambda x: x['chunk_index'])
            
            return neighbors
            
        except Exception as e:
            raise RuntimeError(f"Neighbor search error: {e}")

    def get_expanded_chunks(self, query: str, top_k: int = DEFAULT_TOP_K, 
                          neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW, 
                          min_similarity: float = 0.0) -> List[Dict]:
        """
        Get top-k chunks and expand each with neighboring chunks.
        
        Args:
            query: The search question/keywords
            top_k: Number of top similar chunks to find
            neighbor_window: Number of neighbors on each side of each top-k chunk
            min_similarity: Minimum similarity threshold for top-k chunks
            
        Returns:
            List[Dict]: Combined list of top-k chunks and their neighbors, deduplicated
        """
        # Get top-k chunks
        top_chunks = self.get_top_k_chunks(query, top_k, min_similarity)
        
        if not top_chunks:
            return []
        
        # Collect all chunks (top-k + neighbors)
        all_chunks = []
        seen_ids: Set[str] = set()
        
        for chunk in top_chunks:
            doc_id = chunk['doc_id']
            chunk_index = chunk['chunk_index']
            
            # Get neighbors for this chunk
            neighbors = self.get_neighbors_by_chunk_index(doc_id, chunk_index, neighbor_window)
            
            # Add all chunks (including the original chunk and neighbors)
            for neighbor in neighbors:
                if neighbor['id'] not in seen_ids:
                    # Mark original top-k chunks
                    neighbor['is_top_k'] = neighbor['id'] == chunk['id']
                    if neighbor['is_top_k']:
                        neighbor['similarity'] = chunk['similarity']
                        neighbor['distance'] = chunk['distance']
                    else:
                        neighbor['is_top_k'] = False
                        neighbor['similarity'] = None
                        neighbor['distance'] = None
                    
                    all_chunks.append(neighbor)
                    seen_ids.add(neighbor['id'])
        
        # Sort by document and chunk index for better readability
        all_chunks.sort(key=lambda x: (x['doc_id'], x['chunk_index']))
        
        return all_chunks

    def get_contextual_chunks(self, query: str, top_k: int = DEFAULT_TOP_K,
                            context_window: int = 1, min_similarity: float = 0.0) -> List[Dict]:
        """
        Get top-k chunks with minimal context (usually 1 chunk before and after).
        More focused than get_expanded_chunks.
        
        Args:
            query: The search question/keywords
            top_k: Number of top similar chunks to find
            context_window: Number of context chunks on each side (default: 1)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Dict]: List of chunks with context, sorted by relevance then position
        """
        return self.get_expanded_chunks(query, top_k, context_window, min_similarity)

# ===================== Convenience Functions =====================

def get_top_k(query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = 0.0,
              host: str = CHROMA_HOST, port: int = CHROMA_PORT, 
              collection_name: str = COLLECTION_NAME) -> List[Dict]:
    """
    Convenience function to get top-k chunks.
    
    Returns:
        List[Dict]: Array of top-k chunks
    """
    retriever = TopKRetriever(host, port, collection_name)
    return retriever.get_top_k_chunks(query, top_k, min_similarity)

def get_neighbors(doc_id: str, chunk_index: int, window: int = DEFAULT_NEIGHBOR_WINDOW,
                 host: str = CHROMA_HOST, port: int = CHROMA_PORT,
                 collection_name: str = COLLECTION_NAME) -> List[Dict]:
    """
    Convenience function to get neighboring chunks.
    
    Returns:
        List[Dict]: Array of neighboring chunks sorted by position
    """
    retriever = TopKRetriever(host, port, collection_name)
    return retriever.get_neighbors_by_chunk_index(doc_id, chunk_index, window)

def get_expanded_results(query: str, top_k: int = DEFAULT_TOP_K, 
                        neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
                        min_similarity: float = 0.0,
                        host: str = CHROMA_HOST, port: int = CHROMA_PORT,
                        collection_name: str = COLLECTION_NAME) -> List[Dict]:
    """
    Convenience function to get top-k chunks expanded with neighbors.
    
    Returns:
        List[Dict]: Array of chunks including top-k and their neighbors
    """
    retriever = TopKRetriever(host, port, collection_name)
    return retriever.get_expanded_chunks(query, top_k, neighbor_window, min_similarity)