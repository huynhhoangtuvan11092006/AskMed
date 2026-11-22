"""
tests/test_cleaner.py
Unit tests for data cleaning and filtering (Phase 2 - Step 2.2).
Tests clean_text, clean_abstract, clean_title, and article validation rules.
"""

import pytest
from src.ingestion.cleaner import (
    clean_text,
    clean_title,
    clean_abstract,
    is_valid_article,
    clean_article,
    clean_article_list,
)
from src.ingestion.pubmed_fetcher import PubMedArticle


class TestTextCleaner:
    """Test suite for string-level text cleaning."""

    def test_01_html_unescape_and_tag_stripping(self):
        """Test removing HTML/XML tags and unescaping entities."""
        raw = "<p>Patient received <b>500 mg</b> of amoxicillin &amp; clavulanate (P &lt; 0.05).<br/></p>"
        cleaned = clean_text(raw)
        assert cleaned == "Patient received 500 mg of amoxicillin & clavulanate (P < 0.05)."
        assert "<" not in cleaned and ">" not in cleaned or ("<" in cleaned and "P < 0.05" in cleaned)
        assert "<b>" not in cleaned
        assert "<p>" not in cleaned

    def test_02_unicode_normalization(self):
        """Test Unicode NFKC normalization and dash standardization."""
        raw = "Dose: 50–100 µg/mL; Glucose level: −1.5 mmol/L; ‘quoted’ phrase"
        cleaned = clean_text(raw)
        assert "-" in cleaned  # Dash converted to ASCII hyphen
        assert "'" in cleaned  # Curly quote standardized
        assert "µg/mL" in cleaned or "μg/mL" in cleaned

    def test_03_whitespace_normalization(self):
        """Test collapsing excess whitespace and excessive blank lines."""
        raw = "Line 1    with    multiple   spaces.\n\n\n\n\nLine 2 after big gap."
        cleaned = clean_text(raw)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned
        assert cleaned == "Line 1 with multiple spaces.\n\nLine 2 after big gap."

    def test_04_clean_title(self):
        """Test title sanitation, period stripping, and bracket removal."""
        raw_title = "[Clinical trials of SGLT2 inhibitors in heart failure]... "
        cleaned = clean_title(raw_title)
        assert cleaned == "Clinical trials of SGLT2 inhibitors in heart failure"
        assert not cleaned.endswith(".")

    def test_05_clean_abstract_sections(self):
        """Test structuring abstract headers onto clean paragraphs."""
        raw_abstract = (
            "BACKGROUND: Heart disease is prevalent. METHODS: We conducted a trial. "
            "RESULTS: Mortality dropped significantly. CONCLUSIONS: Treatment is effective."
        )
        cleaned = clean_abstract(raw_abstract)
        assert "BACKGROUND: Heart disease is prevalent." in cleaned
        assert "\n\nMETHODS:" in cleaned
        assert "\n\nRESULTS:" in cleaned
        assert "\n\nCONCLUSIONS:" in cleaned


class TestArticleValidationAndCleaning:
    """Test suite for article-level validation and quality filtering."""

    def test_06_filter_short_abstract(self):
        """Articles with abstract fewer than 50 words must be rejected."""
        short_art = PubMedArticle(
            pmid="11111",
            title="Short Study",
            abstract="This study evaluated a drug. It worked well.",  # Only 8 words
            journal="Test Journal",
            pub_date="2024",
        )
        assert not is_valid_article(short_art, min_words=50)
        assert clean_article(short_art, min_words=50) is None

    def test_07_filter_placeholder_or_retracted(self):
        """Articles with placeholder text or retraction notices must be rejected."""
        placeholder_art = PubMedArticle(
            pmid="22222",
            title="Placeholder Study",
            abstract="No abstract available.",
            journal="Test Journal",
            pub_date="2024",
        )
        assert not is_valid_article(placeholder_art)

        retracted_art = PubMedArticle(
            pmid="33333",
            title="Retracted Study",
            abstract="This article has been retracted by the editor due to scientific misconduct and data errors.",
            journal="Test Journal",
            pub_date="2024",
        )
        assert not is_valid_article(retracted_art)

    def test_08_clean_article_list_batch(self):
        """Test batch cleaning of articles list."""
        valid_abstract = (
            "BACKGROUND: Type 2 diabetes mellitus requires rigorous glycemic control. "
            "METHODS: We randomized 500 adult patients to receive either metformin monotherapy "
            "or lifestyle modification alone for a duration of 24 weeks. Primary endpoint was change in HbA1c. "
            "RESULTS: Patients receiving metformin achieved an average reduction in HbA1c of 1.4% compared "
            "to 0.4% in the control group (P < 0.001). Adverse events were predominantly mild gastrointestinal symptoms. "
            "CONCLUSIONS: Metformin demonstrates superior efficacy and acceptable safety in newly diagnosed patients."
        )
        articles = [
            PubMedArticle(
                pmid="10001",
                title="<b>Metformin Trial</b>.",
                abstract=f"<jats:p>{valid_abstract}</jats:p>",
                journal="<i>Diabetes Care</i>",
                pub_date="2024-01-01",
            ),
            PubMedArticle(
                pmid="10002",
                title="Empty Study",
                abstract="No abstract available",
                journal="Journal",
                pub_date="2024",
            ),
        ]

        cleaned_list = clean_article_list(articles, min_words=50)
        assert len(cleaned_list) == 1
        assert cleaned_list[0].pmid == "10001"
        assert cleaned_list[0].title == "Metformin Trial"  # Stripped <b> and period
        assert cleaned_list[0].journal == "Diabetes Care"  # Stripped <i>
        assert "<jats:p>" not in cleaned_list[0].abstract


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

