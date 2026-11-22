"""
tests/test_smoke.py
Smoke test suite for the Ask_Med foundation setup (Phase 1 - Step 1.5).
Verifies:
1. Configuration loading from configs/config.yaml
2. Ollama connection and model availability (llama3.1:8b, nomic-embed-text)
3. Embedding generation with nomic-embed-text
4. LLM inference with llama3.1:8b
5. ChromaDB vector storage & similarity query
6. Rank-BM25 sparse keyword indexing & query
"""

import os
import sys
from pathlib import Path
import pytest
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


class TestFoundationSmoke:
    """Smoke tests for foundation components."""

    @pytest.fixture
    def config(self):
        """Loads and caches application configuration."""
        cfg = get_config()
        assert cfg is not None
        assert cfg.project.name == "Ask_Med"
        return cfg

    def test_01_config_loading(self, config):
        """Verify configuration fields and automatic directory creation."""
        print(f"\n[1/6] Config Loaded: {config.project.name} v{config.project.version}")
        assert config.paths.data_dir.exists()
        assert config.paths.raw_dir.exists()
        assert config.paths.chroma_db_dir.exists()
        assert config.models.ollama.llm_model == "llama3.1:8b"
        assert config.models.ollama.embed_model == "nomic-embed-text"
        assert config.ingestion.chunk_size == 512
        assert config.retrieval.rrf_k == 60

    def test_02_ollama_service_online(self, config):
        """Verify local Ollama service is reachable on configured port."""
        base_url = config.models.ollama.base_url
        print(f"\n[2/6] Checking Ollama health at: {base_url}/api/tags")
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        assert resp.status_code == 200, f"Ollama returned HTTP {resp.status_code}"
        data = resp.json()
        assert "models" in data, "No 'models' key in Ollama response"

        # Check required models are present
        installed_models = [m.get("name", "") for m in data.get("models", [])]
        print(f"      Installed models in Ollama: {installed_models}")

        has_llm = any(config.models.ollama.llm_model in m for m in installed_models)
        has_embed = any(config.models.ollama.embed_model in m for m in installed_models)

        assert has_llm, f"Required LLM '{config.models.ollama.llm_model}' not found in Ollama!"
        assert has_embed, f"Required Embed model '{config.models.ollama.embed_model}' not found in Ollama!"

    def test_03_embedding_generation(self, config):
        """Verify nomic-embed-text generates non-empty dense vectors."""
        import ollama

        client = ollama.Client(host=config.models.ollama.base_url)
        sample_medical_text = "Metformin is a first-line medication for the treatment of type 2 diabetes."

        print(f"\n[3/6] Generating embedding with model: {config.models.ollama.embed_model}")
        res = client.embeddings(
            model=config.models.ollama.embed_model,
            prompt=f"search_document: {sample_medical_text}",
        )
        embedding = res.get("embedding", [])
        assert isinstance(embedding, list)
        assert len(embedding) > 0, "Embedding vector is empty"
        print(f"      Vector dimensions: {len(embedding)} floats (Expected: 768)")
        assert len(embedding) == 768, f"Expected 768 dimensions for nomic-embed-text, got {len(embedding)}"

    def test_04_llm_inference(self, config):
        """Verify llama3.1:8b produces valid medical reasoning output."""
        import ollama

        client = ollama.Client(host=config.models.ollama.base_url)
        test_prompt = "What is hypertension? Answer in one short sentence of at most 20 words."

        print(f"\n[4/6] Querying LLM: {config.models.ollama.llm_model}")
        res = client.generate(
            model=config.models.ollama.llm_model,
            prompt=test_prompt,
            options={
                "temperature": config.models.ollama.temperature,
                "max_tokens": 50,
            },
        )
        response_text = res.get("response", "").strip()
        assert len(response_text) > 0, "LLM returned empty response"
        print(f"      LLM Response: '{response_text}'")

    def test_05_chromadb_dense_storage(self, config):
        """Verify ChromaDB can create an ephemeral collection, add a document, and query it."""
        import chromadb

        print("\n[5/6] Testing ChromaDB in-memory vector collection")
        client = chromadb.EphemeralClient()
        collection = client.create_collection(name="smoke_test_collection")

        # Insert dummy item with mock vector (dim 768)
        dummy_vector = [0.01] * 768
        collection.add(
            ids=["pmid_12345"],
            embeddings=[dummy_vector],
            documents=["Clinical study on cardiovascular disease."],
            metadatas=[{"pmid": "12345", "title": "Cardiovascular Study"}],
        )

        query_res = collection.query(
            query_embeddings=[dummy_vector],
            n_results=1,
        )
        assert len(query_res["ids"][0]) == 1
        assert query_res["ids"][0][0] == "pmid_12345"
        print(f"      ChromaDB query match: {query_res['ids'][0][0]} (OK)")

    def test_06_bm25_sparse_search(self):
        """Verify Rank-BM25 indexing and keyword ranking."""
        from rank_bm25 import BM25Okapi

        print("\n[6/6] Testing Rank-BM25 sparse keyword ranking")
        corpus = [
            "aspirin reduces risk of heart attack in coronary artery disease",
            "metformin is prescribed for type 2 diabetes mellitus glycemic control",
            "penicillin is an antibiotic used for bacterial infections",
        ]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        query = "coronary artery aspirin".lower().split()
        scores = bm25.get_scores(query)

        # The first document must have the highest score
        top_idx = int(scores.argmax())
        assert top_idx == 0, f"Expected document 0 to rank highest, got {top_idx}"
        print(f"      BM25 top matched doc index: {top_idx} (Score: {scores[top_idx]:.4f}) (OK)")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

