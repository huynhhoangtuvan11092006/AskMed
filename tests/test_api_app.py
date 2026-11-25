"""
tests/test_api_app.py
---------------------
Unit and integration tests for FastAPI application endpoints in AskMed.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "AskMed API"
    assert "/docs" in data["docs_url"]


def test_health_endpoints():
    for endpoint in ["/health", "/api/v1/health"]:
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "1.0.0"


def test_system_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_articles" in data
    assert "total_chunks" in data
    assert data["total_articles"] >= 0
    assert "llm_model" in data
    assert "embedding_model" in data


def test_paper_details_known_and_unknown():
    # Test unknown paper returns 404
    res_404 = client.get("/api/v1/paper/99999999999999")
    assert res_404.status_code == 404

    # Test known paper from bm25.pkl (e.g., 37111272)
    res_known = client.get("/api/v1/paper/37111272")
    if res_known.status_code == 200:
        data = res_known.json()
        assert data["pmid"] == "37111272"
        assert "url" in data


def test_ingest_endpoint_background():
    payload = {
        "keywords": ["test disease", "test drug"],
        "max_articles_per_keyword": 2
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "in_progress"
    assert "test disease" in data["message"]


def test_query_endpoint_mocked():
    mock_pipeline = MagicMock()
    mock_pipeline.ask.return_value = {
        "query": "What is metformin?",
        "answer": "Metformin is foundational first-line treatment [PMID: 37111272].",
        "citations": [
            {
                "pmid": "37111272",
                "title": "Metformin Guideline",
                "url": "https://pubmed.ncbi.nlm.nih.gov/37111272/",
                "journal": "Lancet",
                "pub_year": "2023"
            }
        ],
        "retrieved_passages": [
            {
                "pmid": "37111272",
                "chunk_id": "chunk_01",
                "text": "Metformin is first-line...",
                "rerank_score": 2.5
            }
        ],
        "grounding": {"is_grounded": True}
    }

    with patch("src.api.app.get_pipeline", return_value=mock_pipeline):
        with patch("src.api.app._pipeline_instance", mock_pipeline):
            payload = {
                "query": "What is metformin?",
                "top_k": 5,
                "include_raw_contexts": True,
                "stream": False
            }
            response = client.post("/api/v1/query", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is metformin?"
            assert "37111272" in data["answer"]
            assert data["is_grounded"] is True
            assert len(data["citations"]) == 1
            assert len(data["raw_contexts"]) == 1
            assert data["latency_seconds"] >= 0.0


def test_process_time_header():
    """Step 6.3: Verify X-Process-Time header is injected by middleware."""
    response = client.get("/")
    assert response.status_code == 200
    assert "x-process-time" in response.headers
    assert response.headers["x-process-time"].endswith("s")


def test_validation_error_custom_format():
    """Step 6.3: Verify RequestValidationError returns standardized 422 JSON."""
    payload = {"query": "ab"}  # Too short (< 3 chars)
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Request Validation Failed"
    assert "details" in data
    assert data["status_code"] == 422


def test_ollama_outage_returns_503():
    """Step 6.3: Verify connection failure to Ollama returns 503 Service Unavailable."""
    mock_pipeline = MagicMock()
    mock_pipeline.ask.side_effect = ConnectionRefusedError("Failed to connect to Ollama at http://127.0.0.1:11434")

    with patch("src.api.app.get_pipeline", return_value=mock_pipeline):
        with patch("src.api.app._pipeline_instance", mock_pipeline):
            payload = {"query": "What is metformin?", "top_k": 5}
            response = client.post("/api/v1/query", json=payload)
            assert response.status_code == 503
            data = response.json()
            assert "Ollama" in data["detail"] or "503" in str(data)


