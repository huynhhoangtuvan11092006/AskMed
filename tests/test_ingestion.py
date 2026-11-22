"""
tests/test_ingestion.py
Unit tests for data ingestion components (Phase 2 - Step 2.1).
Tests PubMedFetcher with live NCBI Entrez queries and JSON persistence.
"""

from pathlib import Path
import pytest

from src.ingestion.pubmed_fetcher import PubMedArticle, PubMedFetcher


class TestPubMedFetcher:
    """Test suite for PubMedFetcher module."""

    def test_01_pubmed_article_dataclass(self):
        """Test PubMedArticle initialization and dictionary serialization."""
        art = PubMedArticle(
            pmid="12345678",
            title="Efficacy of Metformin in Type 2 Diabetes",
            abstract="Metformin is widely considered the first-line medication for type 2 diabetes mellitus.",
            journal="New England Journal of Medicine",
            pub_date="2023-05-15",
            doi="10.1056/NEJMoa123456",
            mesh_terms=["Diabetes Mellitus, Type 2", "Metformin"],
        )
        assert art.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        assert art.pmid == "12345678"

        # Test to_dict and from_dict
        d = art.to_dict()
        assert d["pmid"] == "12345678"
        assert d["doi"] == "10.1056/NEJMoa123456"

        restored = PubMedArticle.from_dict(d)
        assert restored.title == art.title
        assert restored.url == art.url
        assert restored.mesh_terms == art.mesh_terms

    def test_02_pubmed_search_and_fetch_live(self):
        """Test live NCBI Entrez search and fetching of real medical abstracts."""
        fetcher = PubMedFetcher(min_abstract_words=20)

        # Search for a well-known, high-volume medical query
        query = "hypertension ACE inhibitors clinical trial"
        articles = fetcher.search_and_fetch(query=query, max_results=3, save_raw=True)

        assert len(articles) > 0, "Expected at least 1 article from live search"

        for art in articles:
            assert len(art.pmid) > 0, "PMID should not be empty"
            assert len(art.title) > 0, "Title should not be empty"
            assert len(art.abstract) > 50, "Abstract should contain meaningful text"
            assert art.url.startswith("https://pubmed.ncbi.nlm.nih.gov/"), "Invalid PubMed URL"
            print(f"\n[Verified Article] PMID: {art.pmid} | Title: {art.title[:60]}... | Date: {art.pub_date}")

    def test_03_save_and_load_json(self, tmp_path):
        """Test persisting fetched articles to raw JSON and reloading them."""
        fetcher = PubMedFetcher()
        sample_articles = [
            PubMedArticle(
                pmid="99990001",
                title="Sample Medical Paper 1",
                abstract="This is a detailed abstract of sample medical paper 1 with sufficient length.",
                journal="The Lancet",
                pub_date="2024-01-01",
            ),
            PubMedArticle(
                pmid="99990002",
                title="Sample Medical Paper 2",
                abstract="This is another detailed abstract of sample medical paper 2 with clinical trial data.",
                journal="BMJ",
                pub_date="2024-02-01",
            ),
        ]

        # Save to custom JSON file in temporary directory
        output_file = tmp_path / "test_raw_articles.json"
        fetcher.raw_dir = tmp_path
        saved_path = fetcher.save_to_json(sample_articles, custom_filename="test_raw_articles.json")

        assert saved_path.exists()

        # Load back and verify integrity
        loaded = PubMedFetcher.load_from_json(saved_path)
        assert len(loaded) == 2
        assert loaded[0].pmid == "99990001"
        assert loaded[1].title == "Sample Medical Paper 2"
        assert loaded[0].journal == "The Lancet"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

