"""
src/ingestion/cleaner.py
Module responsible for sanitizing, normalizing, and filtering raw medical texts
and PubMedArticle records prior to semantic chunking and indexing.
"""

import html
import logging
import re
import unicodedata
from typing import List, Optional

from src.ingestion.pubmed_fetcher import PubMedArticle

logger = logging.getLogger(__name__)

# Regular expression to match HTML and XML tags (e.g., <i>, <b>, <sup>, <jats:p>)
RE_HTML_XML_TAGS = re.compile(r"<[^>]+>", re.IGNORECASE)

# Regular expression to match excessive whitespace (tabs, consecutive spaces)
RE_MULTI_SPACES = re.compile(r"[^\S\r\n]+")

# Regular expression to match 3 or more consecutive newlines
RE_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

# Common placeholders indicating empty or non-viable abstracts
PLACEHOLDER_ABSTRACTS = {
    "no abstract",
    "no abstract available",
    "abstract not available",
    "abstract unavailable",
    "none",
    "n/a",
}


def clean_text(text: Optional[str]) -> str:
    """
    Cleans and normalizes an arbitrary text string:
    1. Strips HTML/XML tags first (before unescaping entities so < and > in data are preserved).
    2. Unescapes HTML entities (e.g., &amp; -> &, &lt; -> <, &#160; -> space).
    3. Performs Unicode NFKC normalization (standardizes ligatures, symbols like µ, ±, ®, ©).
    4. Normalizes various unicode dashes (en-dash, em-dash, minus) to ASCII hyphen.
    5. Strips non-printable control characters.
    6. Collapses redundant spaces and newlines.
    """
    if not text:
        return ""

    # 1. Strip HTML/XML tags first
    text = RE_HTML_XML_TAGS.sub(" ", text)

    # 2. Unescape HTML entities
    text = html.unescape(text)

    # 3. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 4. Standardize unicode dashes and quotes
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)

    # 5. Remove non-printable control characters except newline and tab
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (unicodedata.category(ch)[0] != "C"))

    # 6. Normalize whitespace
    text = RE_MULTI_SPACES.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = RE_EXCESSIVE_NEWLINES.sub("\n\n", text)

    return text.strip()


def clean_title(title: Optional[str]) -> str:
    """Cleans an article title, stripping HTML tags, extra spaces, and trailing periods."""
    cleaned = clean_text(title)
    # Remove trailing period often present in MEDLINE titles
    cleaned = cleaned.rstrip(".").strip()
    # Remove wrapping brackets if any (e.g., [Study on diabetes])
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def clean_abstract(abstract: Optional[str]) -> str:
    """
    Cleans an abstract and standardizes clinical structured section headers
    (e.g., 'BACKGROUND:', 'METHODS:', 'RESULTS:', 'CONCLUSIONS:').
    """
    cleaned = clean_text(abstract)
    if not cleaned:
        return ""

    # Ensure structured section labels start on a new paragraph if preceded by a period
    section_headers = [
        "BACKGROUND",
        "OBJECTIVE",
        "OBJECTIVES",
        "METHODS",
        "PATIENTS AND METHODS",
        "MATERIALS AND METHODS",
        "RESULTS",
        "CONCLUSION",
        "CONCLUSIONS",
        "DISCUSSION",
        "SIGNIFICANCE",
    ]
    pattern = r"(?<=\.)\s+(" + "|".join(section_headers) + r"): "
    cleaned = re.sub(pattern, r"\n\n\1: ", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def is_valid_article(
    article: PubMedArticle,
    min_words: int = 50,
) -> bool:
    """
    Validates whether an article meets minimum quality criteria for medical RAG:
    1. Must have a valid PMID.
    2. Title must be non-empty.
    3. Abstract must contain at least `min_words` words.
    4. Must not be a placeholder abstract or a retracted study notice.
    """
    if not article.pmid or not article.pmid.strip():
        return False

    if not article.title or not article.title.strip():
        return False

    abstract = article.abstract.strip()
    if not abstract:
        return False

    # Check for placeholder abstract text
    cleaned_lower = abstract.lower().strip().rstrip(".")
    if cleaned_lower in PLACEHOLDER_ABSTRACTS:
        return False

    # Check for retraction notices
    if "this article has been retracted" in cleaned_lower or "retraction notice" in cleaned_lower:
        return False

    # Word count check
    words = abstract.split()
    if len(words) < min_words:
        return False

    return True


def clean_article(
    article: PubMedArticle,
    min_words: int = 50,
) -> Optional[PubMedArticle]:
    """
    Sanitizes a PubMedArticle instance and validates quality.
    Returns cleaned PubMedArticle if valid, otherwise returns None.
    """
    cleaned_title = clean_title(article.title)
    cleaned_abstract = clean_abstract(article.abstract)
    cleaned_journal = clean_text(article.journal)

    cleaned_article = PubMedArticle(
        pmid=article.pmid.strip(),
        title=cleaned_title,
        abstract=cleaned_abstract,
        journal=cleaned_journal,
        pub_date=article.pub_date.strip(),
        doi=article.doi.strip() if article.doi else None,
        url=article.url.strip(),
        mesh_terms=[clean_text(m) for m in (article.mesh_terms or []) if clean_text(m)],
    )

    if is_valid_article(cleaned_article, min_words=min_words):
        return cleaned_article

    return None


def clean_article_list(
    articles: List[PubMedArticle],
    min_words: int = 50,
) -> List[PubMedArticle]:
    """
    Cleans a collection of PubMedArticle objects in batch,
    filtering out invalid or truncated abstracts.
    """
    valid_articles: List[PubMedArticle] = []
    skipped_count = 0

    for art in articles:
        cleaned = clean_article(art, min_words=min_words)
        if cleaned:
            valid_articles.append(cleaned)
        else:
            skipped_count += 1

    logger.info(
        f"Cleaned {len(articles)} articles: {len(valid_articles)} kept, {skipped_count} skipped "
        f"(criteria: >= {min_words} words, valid metadata)."
    )
    return valid_articles


if __name__ == "__main__":
    # Self-test demonstration
    sample_dirty_abstract = (
        "<b>BACKGROUND:</b> Type 2 diabetes mellitus is associated with high cardiovascular risk.<br/>"
        "<i>METHODS:</i> We evaluated &alpha;-glucosidase inhibitors &amp; Metformin in 1,200 patients.&#160;&#160;"
        "RESULTS: HbA1c decreased by &minus;1.2% (P &lt; 0.001).   "
        "CONCLUSIONS: Metformin demonstrates superior glycemic control."
    )
    cleaned = clean_abstract(sample_dirty_abstract)
    print("=" * 60)
    print("ORIGINAL DIRTY ABSTRACT:")
    print(sample_dirty_abstract)
    print("\n" + "=" * 60)
    print("CLEANED ABSTRACT:")
    print(cleaned)
    print("=" * 60)
