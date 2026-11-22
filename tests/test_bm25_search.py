"""
tests/test_bm25_search.py
Unit tests for BM25Index and tokenize_medical_text.
"""

from pathlib import Path
import pytest
from src.ingestion.chunker import MedicalChunk
from src.retrieval.bm25_search import BM25Index, tokenize_medical_text


def test_tokenize_medical_text():
    text = "Patient diagnosed with COVID-19 and elevated HbA1c, treated with Metformin 500mg."
    tokens = tokenize_medical_text(text)

    # Hyphenated medical compound should be preserved
    assert "covid-19" in tokens
    # Alphanumeric biomarker should be preserved
    assert "hba1c" in tokens
    assert "metformin" in tokens
    assert "500mg" in tokens

    # Stopwords should be filtered
    assert "with" not in tokens
    assert "and" not in tokens

    # Empty text
    assert tokenize_medical_text("") == []


@pytest.fixture
def sample_chunks():
    return [
        MedicalChunk(
            chunk_id="chunk_1",
            pmid="12345678",
            title="Metformin Therapy in Type 2 Diabetes",
            text="Metformin reduces hepatic glucose production and is the first-line medication for type 2 diabetes mellitus.",
            chunk_index=0,
            total_chunks=1,
            token_count=18,
            journal="Diabetes Care",
            pub_date="2023-01-15",
            doi="10.2337/dc22-0001",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        ),
        MedicalChunk(
            chunk_id="chunk_2",
            pmid="87654321",
            title="Hypertension Management Guidelines",
            text="ACE inhibitors and angiotensin receptor blockers are recommended initial antihypertensive therapies.",
            chunk_index=0,
            total_chunks=1,
            token_count=13,
            journal="Hypertension",
            pub_date="2023-05-20",
            doi="10.1161/hyp.0002",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
        ),
        MedicalChunk(
            chunk_id="chunk_3",
            pmid="11223344",
            title="GLP-1 Receptor Agonists for Obesity",
            text="Semaglutide is a GLP-1 receptor agonist effective for significant weight reduction in non-diabetic obese patients.",
            chunk_index=0,
            total_chunks=1,
            token_count=17,
            journal="New England Journal of Medicine",
            pub_date="2024-02-10",
            doi="10.1056/nejmoa0003",
            url="https://pubmed.ncbi.nlm.nih.gov/11223344/",
        ),
    ]


def test_bm25_build_and_search(tmp_path: Path, sample_chunks):
    test_index_file = tmp_path / "bm25_test.pkl"
    index = BM25Index(index_file=test_index_file)

    assert index.count() == 0

    count = index.build_index(sample_chunks, auto_save=True)
    assert count == 3
    assert index.count() == 3
    assert test_index_file.exists()

    # Search query
    query = "first-line medication for diabetes glucose"
    results = index.search(query, top_k=2)

    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["chunk_id"] == "chunk_1"
    assert top_hit["pmid"] == "12345678"
    assert top_hit["score"] > 0
    assert "metformin" in top_hit["text"].lower()
    assert "metadata" in top_hit


def test_bm25_save_and_load(tmp_path: Path, sample_chunks):
    test_index_file = tmp_path / "bm25_persist.pkl"
    index1 = BM25Index(index_file=test_index_file)
    index1.build_index(sample_chunks, auto_save=True)

    # Initialize a second index instance pointing to the same file
    index2 = BM25Index(index_file=test_index_file)
    assert index2.count() == 3

    results = index2.search("Semaglutide obesity weight loss", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_3"
    assert results[0]["pmid"] == "11223344"


def test_bm25_empty_or_no_match(tmp_path: Path, sample_chunks):
    test_index_file = tmp_path / "bm25_empty.pkl"
    index = BM25Index(index_file=test_index_file)

    # Search on empty index
    assert index.search("anything") == []

    # Search with completely unrelated terms
    index.build_index(sample_chunks, auto_save=False)
    results = index.search("astronomy mars rover galaxy spacecraft")
    assert results == []
