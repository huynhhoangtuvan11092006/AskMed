"""
tests/test_citation_parser.py
Unit tests for the CitationParser module (src.generation.citation_parser).
"""

import pytest
from src.generation.citation_parser import Citation, CitationParser


@pytest.fixture
def sample_passages():
    return [
        {
            "pmid": "42976769",
            "title": "Metformin-supportive nutrition: When gut microbes eat drugs",
            "text": "Metformin is a first-line treatment for type 2 diabetes in adults and children.",
            "metadata": {"journal": "Lancet Diabetes", "pub_date": "2024"},
        },
        {
            "pmid": "42970163",
            "title": "Efficacy and Safety of Fixed-Dose Combination Tablet in Type 2 Diabetes",
            "text": "Dapagliflozin and metformin combination significantly reduced HbA1c levels.",
            "metadata": {"journal": "Diabetes Care", "pub_date": "2023"},
        },
    ]


def test_extract_pmids_empty_and_no_match():
    assert CitationParser.extract_pmids("") == []
    assert CitationParser.extract_pmids("No citations in this text.") == []


def test_extract_pmids_deduplication_and_order():
    text = (
        "Claim A [PMID: 11111]. Claim B [PMID: 22222]. "
        "Claim C [PMID: 11111]. Claim D [PMID: 33333]."
    )
    pmids = CitationParser.extract_pmids(text)
    # Order must be 11111, 22222, 33333 with no duplicates
    assert pmids == ["11111", "22222", "33333"]


def test_extract_pmids_formatting_variants():
    text = "Variant 1 [PMID:12345], Variant 2 [pmid: 67890], Variant 3 [PMID:   99999]."
    pmids = CitationParser.extract_pmids(text)
    assert pmids == ["12345", "67890", "99999"]


def test_build_pubmed_url():
    url = CitationParser.build_pubmed_url("42976769")
    assert url == "https://pubmed.ncbi.nlm.nih.gov/42976769/"


def test_parse_citations_enrichment(sample_passages):
    text = (
        "Metformin is primary therapy [PMID: 42976769]. "
        "Combined with dapagliflozin [PMID: 42970163]."
    )
    citations = CitationParser.parse_citations(text, source_passages=sample_passages)

    assert len(citations) == 2
    c1 = citations[0]
    assert c1.pmid == "42976769"
    assert "Metformin-supportive nutrition" in c1.title
    assert c1.journal == "Lancet Diabetes"
    assert c1.pub_date == "2024"
    assert c1.url == "https://pubmed.ncbi.nlm.nih.gov/42976769/"

    c2 = citations[1]
    assert c2.pmid == "42970163"
    assert c2.journal == "Diabetes Care"


def test_make_markdown_links():
    text = "Metformin is first-line [PMID: 42976769]."
    linked = CitationParser.make_markdown_links(text)
    expected = "Metformin is first-line [[PMID: 42976769]](https://pubmed.ncbi.nlm.nih.gov/42976769/)."
    assert linked == expected


def test_validate_grounding_clean(sample_passages):
    text = "Metformin therapy [PMID: 42976769] and dapagliflozin [PMID: 42970163]."
    res = CitationParser.validate_grounding(text, sample_passages)

    assert res["is_grounded"] is True
    assert res["grounding_score"] == 1.0
    assert len(res["hallucinated_pmids"]) == 0
    assert set(res["valid_pmids"]) == {"42976769", "42970163"}


def test_validate_grounding_hallucination_detected(sample_passages):
    # LLM hallucinated a PMID 99999999 that does NOT exist in sample_passages
    text = (
        "Metformin therapy [PMID: 42976769], but also fake treatment [PMID: 99999999]."
    )
    res = CitationParser.validate_grounding(text, sample_passages)

    assert res["is_grounded"] is False
    assert res["grounding_score"] == 0.5  # 1 valid out of 2 cited
    assert res["hallucinated_pmids"] == ["99999999"]
    assert res["valid_pmids"] == ["42976769"]
