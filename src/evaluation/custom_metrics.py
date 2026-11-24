"""
src/evaluation/custom_metrics.py
Information Retrieval (IR) Evaluation Metrics for Ask_Med.
Implements standard retrieval benchmarks:
  - Hit Rate@K (Hit Rate@1, Hit Rate@3, Hit Rate@5): Fraction of queries with at least 1 relevant paper retrieved in Top K.
  - MRR (Mean Reciprocal Rank): Measures reciprocal rank of the first relevant document.
  - Average Rerank Score: Deep cross-attention relevance score distribution from Cross-Encoder.
  - Precision@K & Recall@K: Standard document-level precision and recall.
"""

import logging
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_hit_rate(retrieved_pmids: List[str], expected_pmids: List[str], k: int = 5) -> float:
    """
    Computes Hit Rate@K for a single query.
    Returns 1.0 if at least one expected PMID appears in the top K retrieved PMIDs, else 0.0.
    """
    if not expected_pmids or not retrieved_pmids or k <= 0:
        return 0.0

    top_k_set: Set[str] = set(str(p).strip() for p in retrieved_pmids[:k] if p)
    expected_set: Set[str] = set(str(p).strip() for p in expected_pmids if p)

    return 1.0 if bool(top_k_set & expected_set) else 0.0


def compute_reciprocal_rank(retrieved_pmids: List[str], expected_pmids: List[str], max_k: int = 5) -> float:
    """
    Computes Reciprocal Rank (RR) for a single query.
    Finds the 1-based rank 'r' of the first relevant document in top max_k.
    Returns 1/r if found, else 0.0.
    """
    if not expected_pmids or not retrieved_pmids:
        return 0.0

    expected_set: Set[str] = set(str(p).strip() for p in expected_pmids if p)

    for rank, pmid in enumerate(retrieved_pmids[:max_k], start=1):
        clean_pmid = str(pmid).strip()
        if clean_pmid in expected_set:
            return round(1.0 / rank, 4)

    return 0.0


def compute_average_rerank_score(retrieved_items: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Computes the average Cross-Encoder reranking score among top K passages.
    """
    if not retrieved_items or k <= 0:
        return 0.0

    top_k_items = retrieved_items[:k]
    scores = []
    for item in top_k_items:
        score = item.get("rerank_score")
        if score is not None and isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 4)


def compute_precision_at_k(retrieved_pmids: List[str], expected_pmids: List[str], k: int = 5) -> float:
    """
    Computes Precision@K: Fraction of top K retrieved PMIDs that are relevant.
    """
    if not retrieved_pmids or not expected_pmids or k <= 0:
        return 0.0

    top_k_pmids = [str(p).strip() for p in retrieved_pmids[:k] if p]
    expected_set = set(str(p).strip() for p in expected_pmids if p)

    relevant_count = sum(1 for p in top_k_pmids if p in expected_set)
    return round(relevant_count / min(k, len(top_k_pmids)), 4)


def compute_recall_at_k(retrieved_pmids: List[str], expected_pmids: List[str], k: int = 5) -> float:
    """
    Computes Recall@K: Fraction of all expected PMIDs captured in top K retrieved PMIDs.
    """
    if not retrieved_pmids or not expected_pmids or k <= 0:
        return 0.0

    top_k_set = set(str(p).strip() for p in retrieved_pmids[:k] if p)
    expected_set = set(str(p).strip() for p in expected_pmids if p)

    if not expected_set:
        return 0.0

    captured_count = len(top_k_set & expected_set)
    return round(captured_count / len(expected_set), 4)


class RetrievalMetricsEvaluator:
    """
    High-level evaluator for computing both single-sample and batch Information Retrieval metrics.
    """

    @classmethod
    def evaluate_sample(
        cls,
        retrieved_items: List[Dict[str, Any]],
        expected_pmids: List[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluates a single query sample against its ground truth expected PMIDs.

        Parameters:
            retrieved_items: Ranked list of retrieved passage dicts containing 'pmid' and 'rerank_score'.
            expected_pmids: List of ground-truth PMIDs.
            k: Evaluation cutoff (default 5).

        Returns:
            Dict containing hit_rate@1, hit_rate@3, hit_rate@5, mrr, precision@5, recall@5, avg_rerank_score.
        """
        pmids = [str(item.get("pmid", "")).strip() for item in retrieved_items if item.get("pmid")]

        hit_1 = compute_hit_rate(pmids, expected_pmids, k=1)
        hit_3 = compute_hit_rate(pmids, expected_pmids, k=3)
        hit_5 = compute_hit_rate(pmids, expected_pmids, k=k)

        mrr = compute_reciprocal_rank(pmids, expected_pmids, max_k=k)
        precision_5 = compute_precision_at_k(pmids, expected_pmids, k=k)
        recall_5 = compute_recall_at_k(pmids, expected_pmids, k=k)
        avg_score = compute_average_rerank_score(retrieved_items, k=k)

        top1_score = retrieved_items[0].get("rerank_score", 0.0) if retrieved_items else 0.0

        return {
            "hit_rate@1": hit_1,
            "hit_rate@3": hit_3,
            f"hit_rate@{k}": hit_5,
            "mrr": mrr,
            f"precision@{k}": precision_5,
            f"recall@{k}": recall_5,
            "avg_rerank_score": avg_score,
            "top1_rerank_score": round(float(top1_score), 4) if isinstance(top1_score, (int, float)) else 0.0,
        }

    @classmethod
    def evaluate_batch(cls, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregates individual query metrics into dataset-wide macro averages.

        Parameters:
            batch_metrics: List of metric dicts returned by evaluate_sample.

        Returns:
            Dict of averaged metrics across all samples.
        """
        if not batch_metrics:
            return {}

        total = len(batch_metrics)
        keys = batch_metrics[0].keys()
        macro_averages: Dict[str, float] = {}

        for key in keys:
            total_sum = sum(sample.get(key, 0.0) for sample in batch_metrics)
            macro_averages[key] = round(total_sum / total, 4)

        return macro_averages


if __name__ == "__main__":
    # Demo sample test
    sample_retrieved = [
        {"pmid": "42976769", "title": "Metformin-supportive nutrition", "rerank_score": 7.6549},
        {"pmid": "38711446", "title": "Dulaglutide in type 2 diabetes", "rerank_score": 7.5677},
        {"pmid": "37111272", "title": "Fenugreek and Cinnamon in T2DM", "rerank_score": 7.1230},
        {"pmid": "42970163", "title": "Dapagliflozin Metformin Combination", "rerank_score": 6.2325},
        {"pmid": "42925073", "title": "GLUT1 and MATE1 polymorphisms", "rerank_score": 5.4943},
    ]
    sample_expected = ["42976769", "42970163"]

    print("=" * 60)
    print("Testing RetrievalMetricsEvaluator (Step 5.3 Demo)")
    print("=" * 60)

    sample_metrics = RetrievalMetricsEvaluator.evaluate_sample(sample_retrieved, sample_expected, k=5)
    for m, val in sample_metrics.items():
        if "score" in m:
            print(f"  • {m:<18}: {val}")
        else:
            print(f"  • {m:<18}: {val * 100:.2f}% ({val})")

