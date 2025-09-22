#!/usr/bin/env python3
"""
Re-ranking module using BM25 + RRF fusion followed by Cross-Encoder re-ranking.
Combines lexical (BM25) and semantic (vector) search results for better retrieval quality.
"""
import os
import sys
import hashlib
from typing import List, Dict, Optional, Set
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required libraries
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"Missing required packages. Please install: pip install rank-bm25 sentence-transformers")
    print(f"Error: {e}")
    sys.exit(1)

# ===================== Configuration =====================
DEFAULT_TOP_M_FUSION = 30      # Top-M after RRF fusion
DEFAULT_TOP_K_FINAL = 12       # Final top-K after Cross-Encoder
DEFAULT_RRF_K = 60             # RRF parameter
DEFAULT_MAX_CHUNKS_PER_FILE = 3  # Max chunks per file in final results

# Global Cross-Encoder instance (reuse for efficiency)
_CE = None

def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Get or create Cross-Encoder instance (singleton pattern)"""
    global _CE
    if _CE is None:
        print(f"Loading Cross-Encoder model: {model_name}")
        _CE = CrossEncoder(model_name)
    return _CE

# ===================== Text Processing =====================
def _tokenize(text: str) -> List[str]:
    """
    Basic tokenizer for BM25. 
    For Vietnamese, consider using more sophisticated tokenizers.
    """
    # Basic preprocessing
    text = text.lower()
    # Remove common punctuation
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split and filter empty tokens
    tokens = [token for token in text.split() if token.strip()]
    return tokens

def _remove_stopwords_vi(tokens: List[str]) -> List[str]:
    """
    Remove common Vietnamese stopwords (optional enhancement)
    """
    # Common Vietnamese stopwords
    vi_stopwords = {
        'là', 'của', 'và', 'có', 'được', 'với', 'trong', 'cho', 'để', 'một',
        'các', 'này', 'đó', 'như', 'từ', 'về', 'theo', 'sau', 'trước', 'khi',
        'nếu', 'nhưng', 'hoặc', 'cũng', 'đã', 'sẽ', 'bằng', 'tại', 'trên', 'dưới'
    }
    return [token for token in tokens if token not in vi_stopwords]

def _advanced_tokenize_vi(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Advanced Vietnamese tokenization with stopword removal
    """
    tokens = _tokenize(text)
    if remove_stopwords:
        tokens = _remove_stopwords_vi(tokens)
    return tokens

# ===================== BM25 Scoring =====================
def _bm25_scores(query: str, results: List[Dict], use_advanced_tokenizer: bool = True) -> List[float]:
    """
    Calculate BM25 scores for query against corpus
    
    Args:
        query: Search query
        results: List of search results with 'content' field
        use_advanced_tokenizer: Use Vietnamese-aware tokenizer
        
    Returns:
        List[float]: BM25 scores (higher is better)
    """
    if not results:
        return []
    
    tokenizer = _advanced_tokenize_vi if use_advanced_tokenizer else _tokenize
    
    # Build corpus and tokenize
    corpus = [r["content"] for r in results]
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    
    # Filter out empty documents
    valid_docs = [(i, tokens) for i, tokens in enumerate(tokenized_corpus) if tokens]
    if not valid_docs:
        return [0.0] * len(results)
    
    # Create BM25 index
    valid_tokens = [tokens for _, tokens in valid_docs]
    bm25 = BM25Okapi(valid_tokens)
    
    # Score query
    query_tokens = tokenizer(query)
    if not query_tokens:
        return [0.0] * len(results)
    
    valid_scores = bm25.get_scores(query_tokens)
    
    # Map back to original order
    scores = [0.0] * len(results)
    for (original_idx, _), score in zip(valid_docs, valid_scores):
        scores[original_idx] = float(score)
    
    return scores

# ===================== Ranking Utilities =====================
def _to_rank_map(order_ids: List[str]) -> Dict[str, int]:
    """Convert ordered list to rank mapping (1 is best)"""
    return {doc_id: rank for rank, doc_id in enumerate(order_ids, start=1)}

def _rrf_fuse(rank_maps: List[Dict[str, int]], k: int = DEFAULT_RRF_K) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF)
    
    Args:
        rank_maps: List of rank mappings from different sources
        k: RRF parameter (default: 60)
        
    Returns:
        Dict[str, float]: Fused scores (higher is better)
    """
    fused = {}
    for rank_map in rank_maps:
        for doc_id, rank in rank_map.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused

# ===================== Cross-Encoder Re-ranking =====================
def _crossencoder_rerank(query: str, results: List[Dict], 
                        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Dict]:
    """
    Re-rank results using Cross-Encoder
    
    Args:
        query: Search query
        results: List of search results
        model_name: Cross-Encoder model name
        
    Returns:
        List[Dict]: Re-ranked results with ce_score and updated rank
    """
    if not results:
        return []
    
    # Get Cross-Encoder model
    ce_model = _get_cross_encoder(model_name)
    
    # Prepare query-document pairs
    pairs = [(query, r["content"]) for r in results]
    
    # Get Cross-Encoder scores
    try:
        ce_scores = ce_model.predict(pairs)
    except Exception as e:
        print(f"Cross-Encoder prediction error: {e}")
        # Fallback: keep original order
        ce_scores = [1.0 - i * 0.01 for i in range(len(results))]
    
    # Add scores to results
    for result, score in zip(results, ce_scores):
        result["ce_score"] = float(score)
    
    # Sort by Cross-Encoder score (descending)
    results.sort(key=lambda x: x["ce_score"], reverse=True)
    
    # Update ranks
    for i, result in enumerate(results, 1):
        result["rank"] = i
    
    return results

# ===================== Main Re-ranking Function =====================
def rerank_fusion_then_crossencoder(
    query: str,
    search_results: List[Dict],
    top_m_after_fusion: int = DEFAULT_TOP_M_FUSION,
    top_k_final: int = DEFAULT_TOP_K_FINAL,
    rrf_k: int = DEFAULT_RRF_K,
    use_advanced_tokenizer: bool = True,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> List[Dict]:
    """
    Main re-ranking function using BM25 + RRF fusion followed by Cross-Encoder re-ranking.
    
    Process:
    1. Calculate BM25 scores on search_results corpus
    2. Create rank maps for vector (original order) and BM25 rankings
    3. Apply RRF fusion to combine rankings
    4. Take top-M after fusion
    5. Apply Cross-Encoder re-ranking on top-M
    6. Return final top-K results
    
    Args:
        query: Search query string
        search_results: List of search result dicts with required fields:
            - id: str (unique identifier)
            - content: str (text content)
            - similarity: float (optional, from vector search)
            - Other metadata fields are preserved
        top_m_after_fusion: Number of top results after RRF fusion
        top_k_final: Final number of results after Cross-Encoder
        rrf_k: RRF parameter (default: 60)
        use_advanced_tokenizer: Use Vietnamese-aware tokenizer
        cross_encoder_model: Cross-Encoder model name
        
    Returns:
        List[Dict]: Top-K re-ranked results with additional debug fields:
            - rank: int (final rank, 1 is best)
            - ce_score: float (Cross-Encoder score)
            - rrf_score: float (RRF fusion score)
            - bm25_rank: int (BM25 rank)
            - vector_rank: int (original vector rank)
    """
    if not search_results:
        return []
    
    print(f"Re-ranking {len(search_results)} results for query: '{query[:50]}...'")
    
    # Make copies to avoid modifying original data
    results = [result.copy() for result in search_results]
    
    # 1) Vector rank from current order (1 is best)
    vector_rank_ids = [r["id"] for r in results]
    vector_rank_map = _to_rank_map(vector_rank_ids)
    
    # 2) BM25 ranking
    print("  Computing BM25 scores...")
    bm25_scores = _bm25_scores(query, results, use_advanced_tokenizer)
    
    # Sort by BM25 score (descending) to get BM25 ranking
    bm25_order = sorted(range(len(results)), key=lambda i: -bm25_scores[i])
    bm25_rank_map = _to_rank_map([results[i]["id"] for i in bm25_order])
    
    # 3) RRF Fusion
    print("  Applying RRF fusion...")
    fused_scores = _rrf_fuse([vector_rank_map, bm25_rank_map], k=rrf_k)
    
    # Sort by RRF score (descending) and take top-M
    fused_sorted = sorted(results, key=lambda r: fused_scores.get(r["id"], 0.0), reverse=True)
    fused_top_m = fused_sorted[:top_m_after_fusion]
    
    print(f"  Selected top-{len(fused_top_m)} after fusion")
    
    # 4) Cross-Encoder re-ranking on top-M
    print("  Applying Cross-Encoder re-ranking...")
    ce_reranked = _crossencoder_rerank(query, fused_top_m, cross_encoder_model)
    
    # Take final top-K
    final_results = ce_reranked[:top_k_final]
    
    # 5) Add debug information
    bm25_rank_lookup = {results[i]["id"]: (rank + 1) for rank, i in enumerate(bm25_order)}
    
    for result in final_results:
        result_id = result["id"]
        result["rrf_score"] = float(fused_scores.get(result_id, 0.0))
        result["bm25_rank"] = bm25_rank_lookup.get(result_id, len(results) + 1)
        result["vector_rank"] = vector_rank_map.get(result_id, len(results) + 1)
        result["bm25_score"] = bm25_scores[vector_rank_ids.index(result_id)] if result_id in vector_rank_ids else 0.0
    
    print(f"  Final top-{len(final_results)} results selected")
    return final_results

# ===================== Post-processing Utilities =====================
def cap_results_per_file(results: List[Dict], max_per_file: int = DEFAULT_MAX_CHUNKS_PER_FILE) -> List[Dict]:
    """
    Cap number of chunks per file to ensure diversity
    
    Args:
        results: List of search results
        max_per_file: Maximum chunks per file
        
    Returns:
        List[Dict]: Filtered results
    """
    if max_per_file <= 0:
        return results
    
    file_counts = defaultdict(int)
    filtered_results = []
    
    for result in results:
        file_name = result.get("file_name", "unknown")
        if file_counts[file_name] < max_per_file:
            filtered_results.append(result)
            file_counts[file_name] += 1
    
    print(f"  Capped results: {len(results)} -> {len(filtered_results)} (max {max_per_file} per file)")
    return filtered_results

def deduplicate_by_content(results: List[Dict], hash_length: int = 16) -> List[Dict]:
    """
    Remove duplicate results based on content hash
    
    Args:
        results: List of search results
        hash_length: Length of content hash for comparison
        
    Returns:
        List[Dict]: Deduplicated results
    """
    seen_hashes: Set[str] = set()
    deduped_results = []
    
    for result in results:
        content = result.get("content", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:hash_length]
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped_results.append(result)
    
    if len(deduped_results) < len(results):
        print(f"  Removed duplicates: {len(results)} -> {len(deduped_results)}")
    
    return deduped_results

def postprocess_results(results: List[Dict], 
                       max_per_file: int = DEFAULT_MAX_CHUNKS_PER_FILE,
                       remove_duplicates: bool = True) -> List[Dict]:
    """
    Apply post-processing filters to results
    
    Args:
        results: List of search results
        max_per_file: Maximum chunks per file (0 = no limit)
        remove_duplicates: Remove duplicate content
        
    Returns:
        List[Dict]: Post-processed results
    """
    processed = results
    
    if remove_duplicates:
        processed = deduplicate_by_content(processed)
    
    if max_per_file > 0:
        processed = cap_results_per_file(processed, max_per_file)
    
    # Update final ranks
    for i, result in enumerate(processed, 1):
        result["final_rank"] = i
    
    return processed

# ===================== Complete Pipeline Function =====================
def rerank_and_postprocess(
    query: str,
    search_results: List[Dict],
    top_m_after_fusion: int = DEFAULT_TOP_M_FUSION,
    top_k_final: int = DEFAULT_TOP_K_FINAL,
    rrf_k: int = DEFAULT_RRF_K,
    max_per_file: int = DEFAULT_MAX_CHUNKS_PER_FILE,
    remove_duplicates: bool = True,
    use_advanced_tokenizer: bool = True
) -> List[Dict]:
    """
    Complete re-ranking and post-processing pipeline
    
    Args:
        query: Search query
        search_results: Input search results
        top_m_after_fusion: Top-M after RRF fusion
        top_k_final: Top-K after Cross-Encoder
        rrf_k: RRF parameter
        max_per_file: Max chunks per file (0 = no limit)
        remove_duplicates: Remove duplicate content
        use_advanced_tokenizer: Use Vietnamese tokenizer
        
    Returns:
        List[Dict]: Final processed results
    """
    print(f"=== RERANK PIPELINE ===")
    print(f"Input: {len(search_results)} results")
    print(f"Pipeline: Vector+BM25 -> RRF -> Top-{top_m_after_fusion} -> CrossEncoder -> Top-{top_k_final}")
    
    # Step 1: Re-ranking
    reranked = rerank_fusion_then_crossencoder(
        query=query,
        search_results=search_results,
        top_m_after_fusion=top_m_after_fusion,
        top_k_final=top_k_final,
        rrf_k=rrf_k,
        use_advanced_tokenizer=use_advanced_tokenizer
    )
    
    # Step 2: Post-processing
    print("  Applying post-processing...")
    final_results = postprocess_results(
        reranked,
        max_per_file=max_per_file,
        remove_duplicates=remove_duplicates
    )
    
    print(f"=== PIPELINE COMPLETE ===")
    print(f"Final output: {len(final_results)} results")
    
    return final_results

# ===================== Debug and Analysis =====================
def analyze_reranking_results(results: List[Dict]) -> None:
    """Print analysis of re-ranking results"""
    if not results:
        print("No results to analyze")
        return
    
    print("\n=== RE-RANKING ANALYSIS ===")
    print(f"Total results: {len(results)}")
    
    # Rank changes analysis
    if all('vector_rank' in r for r in results):
        rank_changes = []
        for r in results:
            change = r['vector_rank'] - r['rank']
            rank_changes.append(change)
        
        print(f"Average rank change: {sum(rank_changes) / len(rank_changes):.2f}")
        print(f"Max rank improvement: {max(rank_changes)} positions")
        print(f"Max rank decline: {min(rank_changes)} positions")
    
    # Score distribution
    if all('ce_score' in r for r in results):
        ce_scores = [r['ce_score'] for r in results]
        print(f"CE score range: {min(ce_scores):.3f} - {max(ce_scores):.3f}")
    
    # File distribution
    files = [r.get('file_name', 'unknown') for r in results]
    file_counts = defaultdict(int)
    for f in files:
        file_counts[f] += 1
    
    print(f"Files represented: {len(file_counts)}")
    for file, count in sorted(file_counts.items()):
        print(f"  {file}: {count} chunks")

# ===================== Example Usage =====================
def main():
    """Example usage of re-ranking functions"""
    # Mock search results for demonstration
    mock_results = [
        {
            "id": "doc1_chunk1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "similarity": 0.85,
            "file_name": "ml_basics.pdf",
            "chunk_id": 1
        },
        {
            "id": "doc1_chunk2", 
            "content": "Deep learning uses neural networks with multiple layers to learn representations.",
            "similarity": 0.82,
            "file_name": "ml_basics.pdf",
            "chunk_id": 2
        },
        {
            "id": "doc2_chunk1",
            "content": "Natural language processing enables computers to understand human language.",
            "similarity": 0.78,
            "file_name": "nlp_guide.pdf",
            "chunk_id": 1
        }
    ]
    
    query = "what is machine learning and deep learning"
    
    print("=== EXAMPLE RE-RANKING ===")
    reranked = rerank_fusion_then_crossencoder(
        query=query,
        search_results=mock_results,
        top_m_after_fusion=3,
        top_k_final=2
    )
    
    print("\n=== RESULTS ===")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. {result['file_name']} (ID: {result['id']})")
        print(f"   CE Score: {result['ce_score']:.3f} | RRF: {result['rrf_score']:.3f}")
        print(f"   Content: {result['content'][:80]}...")
        print()

if __name__ == "__main__":
    main()
