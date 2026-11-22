"""
tests/test_hybrid_fusion.py
Unit tests for Reciprocal Rank Fusion (RRF) and HybridRetriever.
"""

from unittest.mock import MagicMock
import pytest

from src.retrieval.hybrid_fusion import HybridRetriever, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_math():
    dense_results = [
        {"chunk_id": "chunk_A", "pmid": "111", "title": "Doc A", "text": "Text A", "score": 0.95},
        {"chunk_id": "chunk_B", "pmid": "222", "title": "Doc B", "text": "Text B", "score": 0.85},
    ]
    sparse_results = [
        {"chunk_id": "chunk_A", "pmid": "111", "title": "Doc A", "text": "Text A", "score": 12.5},
        {"chunk_id": "chunk_C", "pmid": "333", "title": "Doc C", "text": "Text C", "score": 10.0},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60, top_n=10)

    assert len(fused) == 3

    # chunk_A is rank 1 in both: RRF = 1/61 + 1/61 = 2/61 ≈ 0.032787
    top_hit = fused[0]
    assert top_hit["chunk_id"] == "chunk_A"
    expected_score_a = round((1.0 / 61) + (1.0 / 61), 6)
    assert top_hit["rrf_score"] == pytest.approx(expected_score_a, rel=1e-5)
    assert top_hit["dense_rank"] == 1
    assert top_hit["sparse_rank"] == 1

    # chunk_B is rank 2 in dense, not in sparse: RRF = 1/62 ≈ 0.016129
    # chunk_C is rank 2 in sparse, not in dense: RRF = 1/62 ≈ 0.016129
    doc_ids = [item["chunk_id"] for item in fused]
    assert "chunk_B" in doc_ids
    assert "chunk_C" in doc_ids


def test_rrf_deduplication():
    # Both lists contain the exact same documents
    dense_results = [
        {"chunk_id": "doc_1", "pmid": "100", "title": "Title 1", "text": "Text 1", "score": 0.9},
        {"chunk_id": "doc_2", "pmid": "200", "title": "Title 2", "text": "Text 2", "score": 0.8},
    ]
    sparse_results = [
        {"chunk_id": "doc_2", "pmid": "200", "title": "Title 2", "text": "Text 2", "score": 15.0},
        {"chunk_id": "doc_1", "pmid": "100", "title": "Title 1", "text": "Text 1", "score": 10.0},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60, top_n=20)

    # Should be deduplicated to exactly 2 chunks
    assert len(fused) == 2
    fused_ids = {item["chunk_id"] for item in fused}
    assert fused_ids == {"doc_1", "doc_2"}


def test_rrf_top_n_truncation():
    dense_results = [
        {"chunk_id": f"chunk_dense_{i}", "pmid": f"{i}", "title": f"T{i}", "text": f"D{i}", "score": 0.5}
        for i in range(15)
    ]
    sparse_results = [
        {"chunk_id": f"chunk_sparse_{i}", "pmid": f"{i+100}", "title": f"T{i+100}", "text": f"S{i}", "score": 5.0}
        for i in range(15)
    ]

    # Total unique chunks = 30, request top 20
    fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60, top_n=20)
    assert len(fused) == 20


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], [], rrf_k=60, top_n=10) == []

    dense_only = [{"chunk_id": "d1", "pmid": "1", "title": "T1", "text": "D1", "score": 0.9}]
    fused_dense = reciprocal_rank_fusion(dense_only, [], rrf_k=60, top_n=10)
    assert len(fused_dense) == 1
    assert fused_dense[0]["dense_rank"] == 1
    assert fused_dense[0]["sparse_rank"] is None

    sparse_only = [{"chunk_id": "s1", "pmid": "2", "title": "T2", "text": "S1", "score": 8.0}]
    fused_sparse = reciprocal_rank_fusion([], sparse_only, rrf_k=60, top_n=10)
    assert len(fused_sparse) == 1
    assert fused_sparse[0]["dense_rank"] is None
    assert fused_sparse[0]["sparse_rank"] == 1


def test_hybrid_retriever_orchestration():
    mock_vector_store = MagicMock()
    mock_bm25_index = MagicMock()

    mock_vector_store.search.return_value = [
        {"chunk_id": "c1", "pmid": "10", "title": "T1", "text": "Text 1", "score": 0.9},
    ]
    mock_bm25_index.search.return_value = [
        {"chunk_id": "c1", "pmid": "10", "title": "T1", "text": "Text 1", "score": 12.0},
        {"chunk_id": "c2", "pmid": "20", "title": "T2", "text": "Text 2", "score": 8.0},
    ]

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_index=mock_bm25_index,
        rrf_k=60,
    )

    results = retriever.retrieve("metformin diabetes", dense_top_k=20, sparse_top_k=20, final_top_k=10)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    mock_vector_store.search.assert_called_once_with(query="metformin diabetes", top_k=20)
    mock_bm25_index.search.assert_called_once_with(query="metformin diabetes", top_k=20)
