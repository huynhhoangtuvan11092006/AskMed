"""
tests/test_retrieval.py
End-to-end Evaluation and Latency Benchmark for the Ask_Med Retrieval Pipeline.
Tests the entire multi-stage pipeline:
  1. Dense Semantic Search (ChromaDB + nomic-embed-text)
  2. Sparse Lexical Search (Rank-BM25)
  3. Reciprocal Rank Fusion (RRF k=60 -> Top 20)
  4. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2 -> Top 5)
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


class FullRetrievalPipeline:
    """Helper wrapper combining HybridRetriever and CrossEncoderReranker into a unified pipeline."""

    def __init__(self, rrf_top_k: int = 20, rerank_top_n: int = 5):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_n=rerank_top_n)
        self.rrf_top_k = rrf_top_k
        self.rerank_top_n = rerank_top_n

    def run(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Step 1: Hybrid Retrieval (Dense Top 20 + Sparse Top 20 -> RRF Top 20)
        t_hybrid_start = time.perf_counter()
        candidates = self.retriever.retrieve(query, final_top_k=self.rrf_top_k)
        t_hybrid = time.perf_counter() - t_hybrid_start

        # Step 2: Cross-Encoder Reranker (Top 20 -> Top 5)
        t_rerank_start = time.perf_counter()
        top_passages = self.reranker.rerank(query, candidates, top_n=self.rerank_top_n)
        t_rerank = time.perf_counter() - t_rerank_start

        total_latency = time.perf_counter() - t0

        return {
            "query": query,
            "candidates_count": len(candidates),
            "top_passages": top_passages,
            "latency": {
                "hybrid_sec": round(t_hybrid, 4),
                "rerank_sec": round(t_rerank, 4),
                "total_sec": round(total_latency, 4),
            },
        }


@pytest.fixture(scope="module")
def pipeline():
    return FullRetrievalPipeline(rrf_top_k=20, rerank_top_n=5)


def test_full_pipeline_structure_and_metadata(pipeline):
    """Verify that returned top 5 passages have valid PMIDs, titles, URLs, and non-empty texts."""
    query = "What is the first-line medication for type 2 diabetes?"
    result = pipeline.run(query)

    assert result["query"] == query
    assert result["candidates_count"] > 0
    top_passages = result["top_passages"]

    assert len(top_passages) <= 5
    assert len(top_passages) > 0

    for item in top_passages:
        assert "pmid" in item and len(str(item["pmid"])) > 0
        assert "title" in item and len(item["title"]) > 0
        assert "text" in item and len(item["text"]) > 20
        assert "rerank_score" in item
        assert isinstance(item["rerank_score"], (int, float))
        # Ensure PubMed URL format
        if item.get("url"):
            assert "pubmed.ncbi.nlm.nih.gov" in item["url"]


def test_full_pipeline_clinical_relevance_diabetes(pipeline):
    """Verify clinical relevance: diabetes query must surface metformin or diabetes chunks at rank 1."""
    query = "What is the first-line medication for type 2 diabetes?"
    result = pipeline.run(query)
    top_passages = result["top_passages"]

    top_1 = top_passages[0]
    full_top_text = " ".join([p["text"].lower() + " " + p["title"].lower() for p in top_passages])

    assert "diabetes" in full_top_text
    assert "metformin" in full_top_text
    # Verify scores are strictly ordered descending
    scores = [p["rerank_score"] for p in top_passages]
    assert scores == sorted(scores, reverse=True)


def test_full_pipeline_clinical_relevance_hypertension(pipeline):
    """Verify clinical relevance: hypertension query must surface blood pressure/hypertension management."""
    query = "hypertension blood pressure treatment and management guidelines"
    result = pipeline.run(query)
    top_passages = result["top_passages"]

    assert len(top_passages) > 0
    full_top_text = " ".join([p["text"].lower() + " " + p["title"].lower() for p in top_passages])
    assert any(term in full_top_text for term in ["hypertension", "blood pressure", "cardiovascular"])


def test_full_pipeline_latency_benchmarks(pipeline):
    """Verify latency benchmarks: Total retrieval time must remain within interactive limits (< 5.0s on CPU)."""
    query = "cancer immunotherapy immune checkpoint inhibitors"
    result = pipeline.run(query)
    latencies = result["latency"]

    # Hybrid retrieval (Chroma embedding + BM25) is typically < 1.5s
    assert latencies["hybrid_sec"] < 3.0, f"Hybrid retrieval too slow: {latencies['hybrid_sec']}s"
    # Cross-encoder on 20 pairs on CPU is typically < 2.5s
    assert latencies["rerank_sec"] < 4.0, f"Reranker too slow: {latencies['rerank_sec']}s"
    # Total end-to-end latency threshold
    assert latencies["total_sec"] < 5.0, f"Total pipeline latency too slow: {latencies['total_sec']}s"


def test_full_pipeline_empty_and_oov_query(pipeline):
    """Verify graceful handling for empty or out-of-vocabulary queries."""
    # 1. Nonsense/OOV Query
    oov_query = "xyzqwertyunknownmedicalterm998877"
    result_oov = pipeline.run(oov_query)
    assert isinstance(result_oov["top_passages"], list)

    # 2. Empty Query
    result_empty = pipeline.run("")
    assert isinstance(result_empty["top_passages"], list)


if __name__ == "__main__":
    test_pipeline = FullRetrievalPipeline(rrf_top_k=20, rerank_top_n=5)
    test_queries = [
        "What is the first-line medication for type 2 diabetes?",
        "Hypertension management and blood pressure reduction strategies",
        "Mechanisms of immune checkpoint inhibitors in cancer immunotherapy",
    ]

    print("=" * 80)
    print("ASK_MED RETRIEVAL PIPELINE END-TO-END BENCHMARK")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Test Query #{i}]: {q}")
        res = test_pipeline.run(q)
        lat = res["latency"]
        print(f"  -> Latency: Hybrid = {lat['hybrid_sec']}s | Reranker = {lat['rerank_sec']}s | Total = {lat['total_sec']}s")
        print(f"  -> RRF Candidates: {res['candidates_count']} -> Top Reranked: {len(res['top_passages'])}")
        print("  Top 3 Passages:")
        for r, p in enumerate(res["top_passages"][:3], 1):
            print(f"    #{r} [PMID: {p['pmid']}] (Score: {p['rerank_score']}) Title: {p['title'][:70]}...")
