"""
tests/test_rag_pipeline.py
Unit and live integration tests for the AskMedPipeline end-to-end orchestration.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.generation.rag_pipeline import AskMedPipeline, ask_med


@pytest.fixture
def mock_pipeline():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_llm = MagicMock()

    # Mock retrieval return
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "c1", "pmid": "42976769", "title": "Metformin Study", "text": "Metformin first-line."}
    ]

    # Mock rerank return
    mock_reranker.rerank.return_value = [
        {"chunk_id": "c1", "pmid": "42976769", "title": "Metformin Study", "text": "Metformin first-line.", "rerank_score": 7.5}
    ]

    # Mock LLM return
    mock_llm.generate.return_value = "Metformin is recommended as first-line [PMID: 42976769]."

    pipeline = AskMedPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        llm_client=mock_llm,
    )
    return pipeline, mock_retriever, mock_reranker, mock_llm


def test_pipeline_orchestration(mock_pipeline):
    pipeline, mock_retriever, mock_reranker, mock_llm = mock_pipeline
    query = "What is the first-line medication for type 2 diabetes?"

    res = pipeline.ask(query)

    assert res["query"] == query
    assert "Metformin is recommended" in res["raw_answer"]
    assert "[[PMID: 42976769]](https://pubmed.ncbi.nlm.nih.gov/42976769/)" in res["formatted_answer"]

    # Verify citations
    assert len(res["citations"]) == 1
    c = res["citations"][0]
    assert c["pmid"] == "42976769"
    assert c["title"] == "Metformin Study"

    # Verify grounding audit
    assert res["grounding"]["is_grounded"] is True
    assert res["grounding"]["grounding_score"] == 1.0

    # Verify latency
    assert "retrieval_sec" in res["latency"]
    assert "rerank_sec" in res["latency"]
    assert "llm_sec" in res["latency"]
    assert "total_sec" in res["latency"]

    mock_retriever.retrieve.assert_called_once()
    mock_reranker.rerank.assert_called_once()
    mock_llm.generate.assert_called_once()


def test_pipeline_streaming(mock_pipeline):
    pipeline, _, _, mock_llm = mock_pipeline
    mock_llm.generate.return_value = iter(["Metformin ", "is ", "first-line."])

    token_iter, metadata = pipeline.ask_stream("diabetes medication")
    tokens = list(token_iter)

    assert "".join(tokens) == "Metformin is first-line."
    assert metadata["query"] == "diabetes medication"
    assert len(metadata["top_passages"]) == 1


@pytest.mark.live
def test_pipeline_live_e2e():
    """Live full end-to-end integration test querying the actual ChromaDB + BM25 + CrossEncoder + Llama 3.1."""
    query = "What is the first-line medication for type 2 diabetes?"
    res = ask_med(query)

    assert res["query"] == query
    assert isinstance(res["raw_answer"], str)
    assert len(res["raw_answer"]) > 10

    # Must contain markdown citations
    assert "https://pubmed.ncbi.nlm.nih.gov/" in res["formatted_answer"]

    # Must contain citations list
    assert isinstance(res["citations"], list)
    assert len(res["citations"]) > 0

    # Top passages count
    assert len(res["retrieved_passages"]) <= 5

    # Check timing
    assert res["latency"]["total_sec"] > 0
