"""
src/retrieval/reranker.py
Cross-Encoder Reranker for Ask_Med.
Performs deep semantic cross-attention between a query and retrieved passage candidates
to accurately score and re-rank the candidates, selecting the top N most relevant passages.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks document candidates using a pre-trained Cross-Encoder model.
    Cross-encoders process (query, passage) pairs simultaneously through transformer layers,
    allowing full cross-attention across all tokens in both query and passage,
    yielding significantly higher ranking accuracy than bi-encoder cosine similarity alone.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initializes the CrossEncoder model.

        Parameters:
            model_name: HuggingFace model path (defaults to config: cross-encoder/ms-marco-MiniLM-L-6-v2).
            top_n: Number of reranked candidates to return (defaults to config: 5).
            device: 'cuda', 'cpu', or None (auto-detected).
        """
        cfg = get_config()
        self.model_name = model_name or cfg.models.reranker.model_name
        self.top_n = top_n or cfg.models.reranker.top_n

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Loading CrossEncoder reranker model: '{self.model_name}' on device '{self.device}'..."
        )
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info(f"CrossEncoder model '{self.model_name}' successfully loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a clinical query.

        Parameters:
            query: The user or clinical question.
            candidates: List of candidate dictionaries, each containing at least 'text' and 'chunk_id'.
            top_n: Optional override for the number of top passages to return.

        Returns:
            List of top_n candidates sorted by rerank_score descending.
        """
        if not candidates:
            logger.warning("Reranker received an empty candidate list. Returning empty list.")
            return []

        limit = top_n if top_n is not None else self.top_n

        # Prepare (query, passage) pairs
        pairs: List[Tuple[str, str]] = []
        valid_candidates: List[Dict[str, Any]] = []

        for cand in candidates:
            passage_text = cand.get("text", "")
            # If text is empty, check metadata or title as fallback
            if not passage_text and isinstance(cand.get("metadata"), dict):
                passage_text = cand["metadata"].get("text", "")
            pairs.append((query, passage_text))
            valid_candidates.append(cand)

        logger.debug(f"Scoring {len(pairs)} pairs with CrossEncoder...")
        scores = self.model.predict(pairs)

        # Attach scores and rank
        scored_candidates: List[Dict[str, Any]] = []
        for cand, score in zip(valid_candidates, scores):
            # Shallow copy to avoid side-effects on input data
            item = dict(cand)
            item["rerank_score"] = round(float(score), 4)
            scored_candidates.append(item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        reranked_results = scored_candidates[:limit]
        logger.info(
            f"Reranked {len(candidates)} candidates -> selected top {len(reranked_results)} "
            f"(Score range: {reranked_results[0]['rerank_score']} to {reranked_results[-1]['rerank_score']})."
        )
        return reranked_results


if __name__ == "__main__":
    import argparse
    from src.retrieval.hybrid_fusion import HybridRetriever

    parser = argparse.ArgumentParser(description="Test Cross-Encoder Reranker on Ask_Med retrieval.")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the first-line medication for type 2 diabetes?",
        help="Medical query to retrieve and rerank.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of reranked candidates to return (default 5).",
    )
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"Testing Hybrid Retrieval + Cross-Encoder Reranking")
    print(f"Query: {args.query}")
    print(f"=======================================================\n")

    print("[1/2] Executing Hybrid Retrieval (Dense Top 20 + Sparse Top 20 -> RRF Top 20)...")
    retriever = HybridRetriever()
    rrf_candidates = retriever.retrieve(args.query, final_top_k=20)
    print(f"-> Retrieved {len(rrf_candidates)} candidates via Reciprocal Rank Fusion.")

    print(f"\n[2/2] Running Cross-Encoder Reranker (Top {args.top_n})...")
    reranker = CrossEncoderReranker(top_n=args.top_n)
    reranked = reranker.rerank(args.query, rrf_candidates, top_n=args.top_n)

    print(f"\n================ Top {len(reranked)} Reranked Passages ================")
    for rank, item in enumerate(reranked, start=1):
        print(f"\n#{rank} | PMID: {item.get('pmid')} | Cross-Encoder Score: {item['rerank_score']}")
        print(f"    RRF Rank Prior: RRF Score = {item.get('rrf_score')} (Dense Rank: {item.get('dense_rank')}, Sparse Rank: {item.get('sparse_rank')})")
        print(f"    Title: {item.get('title')}")
        snippet = item.get("text", "").replace("\n", " ")[:200]
        print(f"    Snippet: {snippet}...")
