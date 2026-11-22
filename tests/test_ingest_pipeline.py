"""
tests/test_ingest_pipeline.py
Unit tests for the end-to-end IngestionPipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.ingestion.chunker import MedicalChunker
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.ingestion.pubmed_fetcher import PubMedArticle, PubMedFetcher
from src.retrieval.bm25_search import BM25Index
from src.retrieval.vector_search import ChromaVectorStore


@pytest.fixture
def mock_articles():
    return [
        PubMedArticle(
            pmid="11111111",
            title="Metformin and Glycemic Control in Adults",
            abstract=(
                "BACKGROUND: Metformin is the initial drug of choice for type 2 diabetes. "
                "METHODS: In a randomized controlled clinical trial involving 500 patients with type 2 diabetes, "
                "metformin monotherapy was compared with placebo over a period of 52 weeks. "
                "RESULTS: Glycated hemoglobin levels were significantly reduced by 1.2% in the metformin cohort. "
                "CONCLUSIONS: Metformin provides sustained glycemic improvement with favorable safety profile."
            ),
            journal="Diabetes Care",
            pub_date="2023-01-10",
            doi="10.2337/dc23-0001",
            url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
        ),
        PubMedArticle(
            pmid="22222222",
            title="Lisinopril in Essential Hypertension",
            abstract=(
                "BACKGROUND: Angiotensin-converting enzyme inhibitors are widely utilized for blood pressure regulation. "
                "METHODS: We assessed systolic and diastolic blood pressure changes in 400 hypertensive adults treated with lisinopril 20mg daily. "
                "RESULTS: Mean blood pressure decreased from 150/95 mmHg to 125/80 mmHg after 12 weeks of treatment. "
                "CONCLUSIONS: Lisinopril is highly efficacious for controlling moderate essential hypertension."
            ),
            journal="Hypertension",
            pub_date="2023-03-15",
            doi="10.1161/hyp.0002",
            url="https://pubmed.ncbi.nlm.nih.gov/22222222/",
        ),
    ]


@pytest.fixture
def test_pipeline(tmp_path: Path):
    chroma_dir = tmp_path / "chroma_test"
    bm25_file = tmp_path / "bm25_test.pkl"

    # Use a dummy embedding client that returns 768-dim zero vectors to avoid hitting Ollama in fast unit test
    mock_embed_client = MagicMock()
    mock_embed_client.embed_documents.side_effect = lambda texts: [[0.01] * 768 for _ in texts]
    mock_embed_client.embed_query.side_effect = lambda query: [0.01] * 768

    vector_store = ChromaVectorStore(
        persist_dir=chroma_dir,
        collection_name="test_ingestion_col",
        embedding_client=mock_embed_client,
    )
    bm25_index = BM25Index(index_file=bm25_file)
    chunker = MedicalChunker(chunk_size=512, chunk_overlap=128)
    fetcher = MagicMock(spec=PubMedFetcher)

    return IngestionPipeline(
        fetcher=fetcher,
        chunker=chunker,
        vector_store=vector_store,
        bm25_index=bm25_index,
        min_abstract_words=30,
    )


def test_process_and_index_articles(test_pipeline, mock_articles):
    result = test_pipeline.process_and_index_articles(mock_articles)

    assert result["raw_count"] == 2
    assert result["cleaned_count"] == 2
    assert result["chunks_count"] >= 2
    assert result["vector_store_total"] >= 2
    assert result["bm25_total"] >= 2

    # Verify search in BM25
    bm25_results = test_pipeline.bm25_index.search("Metformin diabetes", top_k=1)
    assert len(bm25_results) == 1
    assert bm25_results[0]["pmid"] == "11111111"


def test_ingest_pmids_with_mock_fetcher(test_pipeline, mock_articles):
    test_pipeline.fetcher.fetch_articles_by_pmids.return_value = mock_articles

    result = test_pipeline.ingest_pmids(["11111111", "22222222"])

    assert result["pmids_found"] == 2
    assert result["cleaned_count"] == 2
    test_pipeline.fetcher.fetch_articles_by_pmids.assert_called_once_with(["11111111", "22222222"])


def test_ingest_file(test_pipeline, mock_articles, tmp_path: Path):
    test_pipeline.fetcher.fetch_articles_by_pmids.return_value = mock_articles

    pmid_file = tmp_path / "pmids.txt"
    pmid_file.write_text("# Test PMIDs\n11111111\n22222222\n\n", encoding="utf-8")

    result = test_pipeline.ingest_file(pmid_file)

    assert result["pmids_found"] == 2
    assert result["cleaned_count"] == 2


def test_pipeline_stats(test_pipeline, mock_articles):
    test_pipeline.process_and_index_articles(mock_articles)
    stats = test_pipeline.get_stats()

    assert stats["chroma_chunks"] >= 2
    assert stats["chroma_pmids"] == 2
    assert stats["bm25_chunks"] >= 2

