"""
src/retrieval/hybrid_fusion.py
Reciprocal Rank Fusion (RRF) module for hybrid retrieval.
Merges ranked lists from Dense Semantic Search (ChromaDB) and Sparse Keyword Search (BM25)
using the standard Cormack et al. (2009) formula:
    RRF(d) = sum(1 / (k + rank_m(d))) for m in {Dense, Sparse}
where k = 60 by default.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.retrieval.bm25_search import BM25Index
from src.retrieval.vector_search import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Parameters:
        dense_results: Ranked list of chunks from ChromaVectorStore.
        sparse_results: Ranked list of chunks from BM25Index.
        rrf_k: Smoothing constant preventing high ranks from disproportionately dominating (default 60).
        top_n: Number of top fused documents to return (default 20).

    Returns:
        Deduplicated, re-ranked list of chunk dicts sorted by rrf_score descending.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}

    # Process Dense results (1-based ranking)
    for rank, item in enumerate(dense_results, start=1):
        doc_id = item["chunk_id"]
        doc_map[doc_id] = item
        dense_ranks[doc_id] = rank
        dense_scores[doc_id] = item.get("score", 0.0)
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Process Sparse (BM25) results (1-based ranking)
    for rank, item in enumerate(sparse_results, start=1):
        doc_id = item["chunk_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = item
        sparse_ranks[doc_id] = rank
        sparse_scores[doc_id] = item.get("score", 0.0)
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sort documents by RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda d_id: scores[d_id], reverse=True)

    # Slice top_n candidates
    top_doc_ids = sorted_doc_ids[:top_n]

    fused_results: List[Dict[str, Any]] = []
    for doc_id in top_doc_ids:
        base_item = doc_map[doc_id]
        fused_item = {
            "chunk_id": doc_id,
            "text": base_item.get("text", ""),
            "metadata": base_item.get("metadata", {}),
            "pmid": base_item.get("pmid", ""),
            "title": base_item.get("title", ""),
            "url": base_item.get("url", ""),
            "rrf_score": round(scores[doc_id], 6),
            "dense_rank": dense_ranks.get(doc_id),
            "sparse_rank": sparse_ranks.get(doc_id),
            "dense_score": dense_scores.get(doc_id),
            "sparse_score": sparse_scores.get(doc_id),
        }
        fused_results.append(fused_item)

    logger.debug(
        f"RRF Fusion: merged {len(dense_results)} dense + {len(sparse_results)} sparse items "
        f"into {len(scores)} unique chunks -> top {len(fused_results)} returned."
    )
    return fused_results


class HybridRetriever:
    """
    High-level orchestrator querying both ChromaVectorStore and BM25Index in parallel
    and applying Reciprocal Rank Fusion to return the most relevant candidates.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: Optional[int] = None,
    ):
        cfg = get_config()
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k if rrf_k is not None else cfg.retrieval.rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Dense semantic vector search via ChromaDB (Top K_dense).
        2. Sparse keyword BM25 search (Top K_sparse).
        3. Reciprocal Rank Fusion (RRF) -> Top K_final.
        """
        cfg = get_config()
        k_dense = dense_top_k or cfg.retrieval.dense_top_k
        k_sparse = sparse_top_k or cfg.retrieval.sparse_top_k
        k_final = final_top_k or cfg.retrieval.dense_top_k

        logger.info(f"Hybrid retrieval query: '{query}' (dense_k={k_dense}, sparse_k={k_sparse})")

        # 1. Fetch dense candidates
        dense_results = self.vector_store.search(query=query, top_k=k_dense)

        # 2. Fetch sparse BM25 candidates
        sparse_results = self.bm25_index.search(query=query, top_k=k_sparse)

        # 3. Fuse rankings via RRF
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=self.rrf_k,
            top_n=k_final,
        )

        logger.info(f"Hybrid retrieval returned {len(fused)} candidates for query: '{query}'.")
        return fused


if __name__ == "__main__":
    retriever = HybridRetriever()
    print("=" * 60)
    print("HYBRID RETRIEVER STATUS:")
    print(f"ChromaDB Chunks: {retriever.vector_store.count()}")
    print(f"BM25 Chunks    : {retriever.bm25_index.count()}")
    print(f"RRF k parameter: {retriever.rrf_k}")
    print("=" * 60)

    if retriever.vector_store.count() > 0 and retriever.bm25_index.count() > 0:
        sample_query = "What is the first-line medication for type 2 diabetes?"
        print(f"\nDEMO HYBRID RETRIEVAL (RRF) for: '{sample_query}' (Top 5):")
        hits = retriever.retrieve(sample_query, final_top_k=5)
        for rank, hit in enumerate(hits, 1):
            d_rank = f"#{hit['dense_rank']}" if hit['dense_rank'] else "N/A"
            s_rank = f"#{hit['sparse_rank']}" if hit['sparse_rank'] else "N/A"
            print(f"[{rank}] PMID: {hit['pmid']} | RRF Score: {hit['rrf_score']} | (Dense: {d_rank}, BM25: {s_rank})")
            print(f"    Title: {hit['title']}")
            print(f"    Text: {hit['text'][:110]}...\n")
