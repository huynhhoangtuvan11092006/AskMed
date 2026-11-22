"""
tests/test_vector_search.py
Unit tests for ChromaDB vector indexing and dense semantic search (Phase 2 - Step 2.4).
"""

from pathlib import Path
import pytest

from src.ingestion.chunker import MedicalChunk
from src.retrieval.vector_search import ChromaVectorStore, OllamaEmbeddingClient


class TestVectorSearch:
    """Test suite for ChromaVectorStore and OllamaEmbeddingClient."""

    @pytest.fixture
    def embedding_client(self):
        """Initializes the Ollama embedding client."""
        return OllamaEmbeddingClient()

    @pytest.fixture
    def temp_vector_store(self, tmp_path, embedding_client):
        """Initializes an isolated persistent ChromaDB instance inside tmp_path."""
        return ChromaVectorStore(
            persist_dir=tmp_path / "test_chroma",
            collection_name="test_medical_knowledge",
            embedding_client=embedding_client,
        )

    def test_01_embedding_generation(self, embedding_client):
        """Test document and query embedding generation."""
        doc_vecs = embedding_client.embed_documents(["Hypertension is elevated arterial blood pressure."])
        assert len(doc_vecs) == 1
        assert len(doc_vecs[0]) == 768

        query_vec = embedding_client.embed_query("high blood pressure treatment")
        assert len(query_vec) == 768

    def test_02_add_and_search_chunks(self, temp_vector_store):
        """Test adding chunks and retrieving relevant ones via dense search."""
        chunks = [
            MedicalChunk(
                chunk_id="111_chunk_0",
                pmid="111",
                title="Efficacy of ACE Inhibitors in Hypertension",
                text="Title: Efficacy of ACE Inhibitors in Hypertension\n\nAbstract: ACE inhibitors like lisinopril significantly lower blood pressure.",
                chunk_index=0,
                total_chunks=1,
                token_count=25,
                journal="Hypertension Journal",
                pub_date="2024",
                doi="10.1000/111",
            ),
            MedicalChunk(
                chunk_id="222_chunk_0",
                pmid="222",
                title="Management of Type 2 Diabetes with Metformin",
                text="Title: Management of Type 2 Diabetes with Metformin\n\nAbstract: Metformin is the primary oral hypoglycemic agent for type 2 diabetes.",
                chunk_index=0,
                total_chunks=1,
                token_count=26,
                journal="Diabetes Care",
                pub_date="2024",
                doi="10.1000/222",
            ),
        ]

        # Add chunks
        added_count = temp_vector_store.add_chunks(chunks)
        assert added_count == 2
        assert temp_vector_store.count() == 2

        # Search for hypertension
        results = temp_vector_store.search(query="blood pressure medications lisinopril", top_k=2)
        assert len(results) == 2

        # The hypertension chunk (PMID 111) must be ranked top with highest similarity score
        top_result = results[0]
        assert top_result["pmid"] == "111"
        assert top_result["score"] > 0.6
        assert top_result["title"] == "Efficacy of ACE Inhibitors in Hypertension"
        assert top_result["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"

    def test_03_get_pmids_and_delete(self, temp_vector_store):
        """Test retrieving unique PMIDs and deleting by PMID."""
        chunks = [
            MedicalChunk(
                chunk_id="111_chunk_0",
                pmid="111",
                title="Hypertension Study",
                text="Title: Hypertension Study\n\nAbstract: Text about blood pressure.",
                chunk_index=0,
                total_chunks=1,
                token_count=15,
                journal="Journal A",
                pub_date="2024",
            ),
            MedicalChunk(
                chunk_id="222_chunk_0",
                pmid="222",
                title="Diabetes Study",
                text="Title: Diabetes Study\n\nAbstract: Text about blood glucose.",
                chunk_index=0,
                total_chunks=1,
                token_count=15,
                journal="Journal B",
                pub_date="2024",
            ),
        ]
        temp_vector_store.add_chunks(chunks)

        pmids = temp_vector_store.get_all_pmids()
        assert "111" in pmids
        assert "222" in pmids

        # Delete PMID 111
        deleted = temp_vector_store.delete_by_pmid("111")
        assert deleted == 1
        assert temp_vector_store.count() == 1
        assert "111" not in temp_vector_store.get_all_pmids()
        assert "222" in temp_vector_store.get_all_pmids()


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
