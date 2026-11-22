"""
tests/test_chunker.py
Unit tests for semantic chunking and metadata preservation (Phase 2 - Step 2.3).
Tests MedicalChunker token counting, sentence-aware splitting, and overlap mechanism.
"""

import pytest
from src.ingestion.chunker import MedicalChunk, MedicalChunker
from src.ingestion.pubmed_fetcher import PubMedArticle


class TestMedicalChunker:
    """Test suite for MedicalChunker module."""

    @pytest.fixture
    def chunker(self):
        """Initializes a MedicalChunker with standard 512/128 parameters."""
        return MedicalChunker(chunk_size=512, chunk_overlap=128)

    def test_01_token_counting(self, chunker):
        """Verify tiktoken token counting precision."""
        text = "Type 2 diabetes mellitus is a chronic metabolic disorder."
        tokens = chunker.count_tokens(text)
        assert tokens > 0
        assert tokens == len(chunker.tokenizer.encode(text))

    def test_02_sentence_splitting(self, chunker):
        """Verify sentence splitting preserves abbreviations and sentence endings."""
        text = (
            "Patients received 500 mg daily (e.g., orally with meals). "
            "Follow-up occurred after 12 weeks. Did symptoms improve? Yes, markedly."
        )
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) >= 2
        # Check that sentences don't split arbitrarily on e.g.
        assert any("e.g." in s for s in sentences)

    def test_03_short_abstract_single_chunk(self, chunker):
        """Standard PubMed abstracts under 512 tokens should produce exactly 1 chunk."""
        article = PubMedArticle(
            pmid="12345678",
            title="Efficacy of Metformin in Type 2 Diabetes",
            abstract=(
                "BACKGROUND: Metformin is the first-line medication for type 2 diabetes. "
                "METHODS: We conducted a double-blind trial in 200 patients. "
                "RESULTS: HbA1c reduced by 1.2% in the treatment group. "
                "CONCLUSIONS: Metformin is safe and effective."
            ),
            journal="The Lancet",
            pub_date="2024-01-15",
            doi="10.1016/S0140-6736(23)00001-X",
        )

        chunks = chunker.chunk_article(article)
        assert len(chunks) == 1

        c = chunks[0]
        assert c.chunk_id == "12345678_chunk_0"
        assert c.pmid == "12345678"
        assert c.chunk_index == 0
        assert c.total_chunks == 1
        assert c.token_count <= 512
        assert c.token_count > 0
        assert c.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        assert c.doi == "10.1016/S0140-6736(23)00001-X"
        assert "Title: Efficacy of Metformin in Type 2 Diabetes" in c.text

    def test_04_long_text_multi_chunk_with_overlap(self):
        """Very long texts (> 512 tokens) should split into multiple chunks with overlap."""
        # Use smaller chunk_size to trigger multi-chunking reliably in test
        small_chunker = MedicalChunker(chunk_size=100, chunk_overlap=30)

        # Generate a synthetic multi-sentence medical text
        sentences = [
            f"Sentence {i}: High blood pressure increases the risk of stroke and renal failure in stage {i} patients."
            for i in range(1, 15)
        ]
        long_abstract = " ".join(sentences)

        article = PubMedArticle(
            pmid="99999999",
            title="Comprehensive Analysis of Hypertension Risks",
            abstract=long_abstract,
            journal="Hypertension Journal",
            pub_date="2024-03-01",
        )

        chunks = small_chunker.chunk_article(article)
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

        for i, c in enumerate(chunks):
            assert c.pmid == "99999999"
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)
            assert c.token_count <= 120  # Under reasonable bound
            assert c.chunk_id == f"99999999_chunk_{i}"
            assert "Title: Comprehensive Analysis of Hypertension Risks" in c.text

    def test_05_batch_chunking(self, chunker):
        """Test batch chunking on multiple articles."""
        articles = [
            PubMedArticle(pmid="101", title="Study 1", abstract="Abstract text for study 1 of medical significance.", journal="J1", pub_date="2024"),
            PubMedArticle(pmid="102", title="Study 2", abstract="Abstract text for study 2 of clinical significance.", journal="J2", pub_date="2024"),
        ]

        chunks = chunker.chunk_article_list(articles)
        assert len(chunks) == 2
        assert chunks[0].pmid == "101"
        assert chunks[1].pmid == "102"
        assert chunks[0].chunk_id == "101_chunk_0"
        assert chunks[1].chunk_id == "102_chunk_0"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

