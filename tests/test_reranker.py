"""
tests/test_reranker.py
Unit tests for the Cross-Encoder Reranker module (src.retrieval.reranker).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def sample_candidates():
    return [
        {
            "chunk_id": "chunk_001",
            "pmid": "11111",
            "title": "Astrophysics and Stellar Evolution",
            "text": "Stellar evolution describes the process by which a star changes over time through nuclear fusion.",
            "rrf_score": 0.015,
        },
        {
            "chunk_id": "chunk_002",
            "pmid": "22222",
            "title": "First-line Pharmacotherapy for Type 2 Diabetes",
            "text": "Metformin remains the universally recommended initial pharmacological agent for the treatment of type 2 diabetes mellitus.",
            "rrf_score": 0.012,
        },
        {
            "chunk_id": "chunk_003",
            "pmid": "33333",
            "title": "Dietary Management in Glycemic Control",
            "text": "Lifestyle modifications and dietary interventions complement metformin in managing blood glucose levels in diabetic patients.",
            "rrf_score": 0.010,
        },
    ]


def test_reranker_initialization_defaults():
    with patch("src.retrieval.reranker.CrossEncoder") as mock_ce:
        reranker = CrossEncoderReranker(model_name="dummy-model", top_n=3)
        assert reranker.model_name == "dummy-model"
        assert reranker.top_n == 3
        mock_ce.assert_called_once()


def test_reranker_empty_candidates():
    with patch("src.retrieval.reranker.CrossEncoder"):
        reranker = CrossEncoderReranker(model_name="dummy-model")
        results = reranker.rerank(query="diabetes", candidates=[], top_n=5)
        assert results == []


def test_reranker_scoring_and_ordering(sample_candidates):
    with patch("src.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        # Mock predict to return scores: chunk_001 -> -4.5, chunk_002 -> 8.2, chunk_003 -> 3.1
        mock_model.predict.return_value = [-4.5, 8.2, 3.1]
        mock_ce.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="dummy-model", top_n=2)
        results = reranker.rerank(
            query="What is the first-line medication for type 2 diabetes?",
            candidates=sample_candidates,
            top_n=2,
        )

        assert len(results) == 2
        # chunk_002 must be rank 1 (score 8.2)
        assert results[0]["chunk_id"] == "chunk_002"
        assert results[0]["rerank_score"] == 8.2
        # chunk_003 must be rank 2 (score 3.1)
        assert results[1]["chunk_id"] == "chunk_003"
        assert results[1]["rerank_score"] == 3.1

        # Verify pairs passed to model.predict
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[1][0] == "What is the first-line medication for type 2 diabetes?"
        assert "Metformin" in pairs[1][1]


def test_reranker_metadata_text_fallback():
    candidates_with_meta = [
        {
            "chunk_id": "meta_001",
            "text": "",
            "metadata": {"text": "Metformin therapy guidelines."},
        }
    ]
    with patch("src.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        mock_ce.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="dummy-model")
        results = reranker.rerank("diabetes", candidates_with_meta)
        assert len(results) == 1
        assert results[0]["rerank_score"] == 5.0
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][1] == "Metformin therapy guidelines."


@pytest.mark.live
def test_reranker_live_model():
    """Live test downloading/using the real cross-encoder model to score medical vs non-medical text."""
    reranker = CrossEncoderReranker(top_n=2)
    query = "What is the primary medication for type 2 diabetes?"
    candidates = [
        {
            "chunk_id": "doc_astronomy",
            "text": "Jupiter is the largest planet in our solar system, with a giant red spot storm.",
            "title": "Jupiter planetary study",
        },
        {
            "chunk_id": "doc_diabetes",
            "text": "Metformin is considered the first-line drug of choice for patients with type 2 diabetes.",
            "title": "ADA Clinical Practice Recommendations",
        },
    ]

    reranked = reranker.rerank(query, candidates, top_n=2)
    assert len(reranked) == 2
    # The diabetes chunk MUST score higher than the astronomy chunk
    assert reranked[0]["chunk_id"] == "doc_diabetes"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
