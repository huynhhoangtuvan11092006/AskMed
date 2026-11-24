"""
tests/test_custom_metrics.py
Unit tests for Information Retrieval evaluation metrics in Ask_Med (src.evaluation.custom_metrics).
"""

import pytest
from src.evaluation.custom_metrics import (
    RetrievalMetricsEvaluator,
    compute_average_rerank_score,
    compute_hit_rate,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)


def test_compute_hit_rate():
    retrieved = ["101", "102", "103", "104", "105"]
    expected_hit = ["103", "999"]
    expected_miss = ["888", "999"]

    # Hit@5
    assert compute_hit_rate(retrieved, expected_hit, k=5) == 1.0
    # Miss@5
    assert compute_hit_rate(retrieved, expected_miss, k=5) == 0.0
    # Hit@1 vs Hit@3
    assert compute_hit_rate(retrieved, ["101"], k=1) == 1.0
    assert compute_hit_rate(retrieved, ["103"], k=1) == 0.0
    assert compute_hit_rate(retrieved, ["103"], k=3) == 1.0

    # Edge cases
    assert compute_hit_rate([], ["101"]) == 0.0
    assert compute_hit_rate(["101"], []) == 0.0
    assert compute_hit_rate(["101"], ["101"], k=0) == 0.0


def test_compute_reciprocal_rank():
    retrieved = ["101", "102", "103", "104", "105"]

    # Rank 1 -> RR = 1.0
    assert compute_reciprocal_rank(retrieved, ["101", "999"], max_k=5) == 1.0

    # Rank 2 -> RR = 0.5
    assert compute_reciprocal_rank(retrieved, ["102"], max_k=5) == 0.5

    # Rank 4 -> RR = 0.25
    assert compute_reciprocal_rank(retrieved, ["104"], max_k=5) == 0.25

    # Not found in top 5 -> RR = 0.0
    assert compute_reciprocal_rank(retrieved, ["999"], max_k=5) == 0.0

    # Found at rank 4, but cutoff max_k=3 -> RR = 0.0
    assert compute_reciprocal_rank(retrieved, ["104"], max_k=3) == 0.0


def test_compute_average_rerank_score():
    items = [
        {"pmid": "1", "rerank_score": 8.0},
        {"pmid": "2", "rerank_score": 6.0},
        {"pmid": "3", "rerank_score": 4.0},
    ]
    avg = compute_average_rerank_score(items, k=3)
    assert avg == 6.0

    # Top 2 only
    avg_2 = compute_average_rerank_score(items, k=2)
    assert avg_2 == 7.0

    # Missing or empty
    assert compute_average_rerank_score([], k=5) == 0.0


def test_precision_and_recall_at_k():
    retrieved = ["101", "102", "103", "104", "105"]
    expected = ["102", "104", "999"]  # 2 hits out of 3 expected

    # Precision@5 = 2 / 5 = 0.4
    prec = compute_precision_at_k(retrieved, expected, k=5)
    assert prec == 0.4

    # Recall@5 = 2 / 3 = 0.6667
    rec = compute_recall_at_k(retrieved, expected, k=5)
    assert rec == 0.6667


def test_evaluator_sample_and_batch():
    sample1 = [
        {"pmid": "101", "rerank_score": 7.5},
        {"pmid": "102", "rerank_score": 6.0},
        {"pmid": "103", "rerank_score": 5.0},
        {"pmid": "104", "rerank_score": 4.0},
        {"pmid": "105", "rerank_score": 3.0},
    ]
    expected1 = ["101"]

    sample2 = [
        {"pmid": "201", "rerank_score": 6.5},
        {"pmid": "202", "rerank_score": 5.5},
        {"pmid": "203", "rerank_score": 4.5},
        {"pmid": "204", "rerank_score": 3.5},
        {"pmid": "205", "rerank_score": 2.5},
    ]
    expected2 = ["202"]

    res1 = RetrievalMetricsEvaluator.evaluate_sample(sample1, expected1, k=5)
    assert res1["hit_rate@1"] == 1.0
    assert res1["hit_rate@5"] == 1.0
    assert res1["mrr"] == 1.0
    assert res1["top1_rerank_score"] == 7.5

    res2 = RetrievalMetricsEvaluator.evaluate_sample(sample2, expected2, k=5)
    assert res2["hit_rate@1"] == 0.0
    assert res2["hit_rate@5"] == 1.0
    assert res2["mrr"] == 0.5  # Rank 2

    # Batch macro average
    batch_avg = RetrievalMetricsEvaluator.evaluate_batch([res1, res2])
    assert batch_avg["hit_rate@1"] == 0.5  # (1.0 + 0.0) / 2
    assert batch_avg["hit_rate@5"] == 1.0  # (1.0 + 1.0) / 2
    assert batch_avg["mrr"] == 0.75        # (1.0 + 0.5) / 2
