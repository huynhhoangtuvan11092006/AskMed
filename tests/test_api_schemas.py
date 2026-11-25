"""
tests/test_api_schemas.py
-------------------------
Unit tests verifying Pydantic v2 schemas for the AskMed API.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    CitationItem,
    ContextItem,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)


def test_citation_item_valid():
    item = CitationItem(
        pmid="37111272",
        title="Metformin in T2D",
        url="https://pubmed.ncbi.nlm.nih.gov/37111272/",
        journal="Lancet",
        pub_year="2023"
    )
    assert item.pmid == "37111272"
    assert item.url == "https://pubmed.ncbi.nlm.nih.gov/37111272/"
    assert item.title == "Metformin in T2D"


def test_context_item_valid():
    ctx = ContextItem(
        pmid="37111272",
        chunk_id="chunk_01",
        text="Clinical findings show that metformin is effective...",
        score=2.34
    )
    assert ctx.pmid == "37111272"
    assert ctx.score == 2.34


def test_query_request_valid():
    req = QueryRequest(
        query="What is the first-line medication for type 2 diabetes?",
        top_k=5,
        include_raw_contexts=True
    )
    assert req.query.startswith("What is")
    assert req.top_k == 5
    assert req.include_raw_contexts is True
    assert req.stream is False


def test_query_request_validation_error():
    # Query too short (< 3 chars)
    with pytest.raises(ValidationError):
        QueryRequest(query="hi")

    # top_k out of range (< 1 or > 20)
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query text", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query text", top_k=25)


def test_query_response_valid():
    res = QueryResponse(
        query="What is first-line T2D drug?",
        answer="Metformin is foundational [PMID: 37111272].",
        citations=[
            CitationItem(
                pmid="37111272",
                title="Metformin study",
                url="https://pubmed.ncbi.nlm.nih.gov/37111272/"
            )
        ],
        is_grounded=True,
        latency_seconds=1.45
    )
    assert res.is_grounded is True
    assert len(res.citations) == 1
    assert res.latency_seconds == 1.45


def test_ingest_request_valid():
    req = IngestRequest(
        keywords=["diabetes mellitus", "metformin"],
        max_articles_per_keyword=15
    )
    assert len(req.keywords) == 2
    assert req.max_articles_per_keyword == 15

    # Empty keywords list should fail
    with pytest.raises(ValidationError):
        IngestRequest(keywords=[])


def test_stats_and_health_response():
    stats = StatsResponse(
        total_articles=620,
        total_chunks=703,
        vector_db_status="connected",
        bm25_status="loaded",
        embedding_model="nomic-embed-text",
        reranker_model="ms-marco-MiniLM-L-6-v2",
        llm_model="llama3.1:8b"
    )
    assert stats.total_articles == 620

    health = HealthResponse(
        status="ok",
        ollama_connected=True,
        vector_db_ready=True,
        bm25_ready=True
    )
    assert health.status == "ok"
    assert health.version == "1.0.0"
