"""
src/generation/citation_parser.py
Citation Parser and Validator for Ask_Med.
Extracts in-text medical citations matching [PMID: <pmid>], deduplicates them,
enriches them with scientific metadata from retrieved passages, converts tags
to clickable PubMed hyperlinks, and validates against citation hallucination.
"""

from dataclasses import asdict, dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches [PMID: 12345678], [pmid:12345], [PMID:  999]
PMID_REGEX = re.compile(r"\[PMID:\s*(\d+)\]", re.IGNORECASE)


@dataclass
class Citation:
    """Represents an evidence-grounded scientific citation extracted from LLM text."""
    pmid: str
    url: str
    title: str = "PubMed Article"
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts Citation to a serializable dictionary."""
        return asdict(self)


class CitationParser:
    """
    Parses and verifies scientific PubMed citations in medical answers.
    Ensures that citations match the retrieved literature and formats them
    for interactive user display.
    """

    PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov"

    @classmethod
    def extract_pmids(cls, text: str) -> List[str]:
        """
        Scans text for all [PMID: <digits>] patterns and returns unique PMIDs
        preserving order of appearance.

        Parameters:
            text: Raw generated response from the LLM.

        Returns:
            List of unique PMID strings (e.g., ['42976769', '42970163']).
        """
        if not text:
            return []

        matches = PMID_REGEX.findall(text)
        seen: Set[str] = set()
        unique_pmids: List[str] = []

        for pmid in matches:
            clean_pmid = pmid.strip()
            if clean_pmid not in seen:
                seen.add(clean_pmid)
                unique_pmids.append(clean_pmid)

        return unique_pmids

    @classmethod
    def build_pubmed_url(cls, pmid: str) -> str:
        """Constructs an official NCBI PubMed URL for a given PMID."""
        return f"{cls.PUBMED_BASE_URL}/{pmid.strip()}/"

    @classmethod
    def parse_citations(
        cls,
        text: str,
        source_passages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Citation]:
        """
        Extracts cited PMIDs and enriches them with title, journal, and snippet
        from the retrieved candidate passages.

        Parameters:
            text: Generated answer containing [PMID: ...] citations.
            source_passages: List of passage dictionaries retrieved from knowledge base.

        Returns:
            List of enriched Citation objects.
        """
        pmids = cls.extract_pmids(text)
        if not pmids:
            return []

        # Create quick lookup map from retrieved passages
        passage_map: Dict[str, Dict[str, Any]] = {}
        if source_passages:
            for p in source_passages:
                pid = str(p.get("pmid", "")).strip()
                if pid and pid not in passage_map:
                    passage_map[pid] = p

        citations: List[Citation] = []
        for pmid in pmids:
            url = cls.build_pubmed_url(pmid)
            meta = passage_map.get(pmid, {})

            title = meta.get("title") or "PubMed Biomedical Paper"
            metadata_dict = meta.get("metadata", {})
            journal = metadata_dict.get("journal") if isinstance(metadata_dict, dict) else None
            pub_date = metadata_dict.get("pub_date") if isinstance(metadata_dict, dict) else None

            snippet = meta.get("text", "")
            if snippet and len(snippet) > 200:
                snippet = snippet[:197] + "..."

            citations.append(
                Citation(
                    pmid=pmid,
                    url=url,
                    title=title,
                    journal=journal,
                    pub_date=pub_date,
                    snippet=snippet,
                )
            )

        return citations

    @classmethod
    def make_markdown_links(cls, text: str) -> str:
        """
        Transforms raw [PMID: 12345] tags into clickable Markdown hyperlinks:
        [[PMID: 12345]](https://pubmed.ncbi.nlm.nih.gov/12345/)

        Parameters:
            text: Raw LLM text.

        Returns:
            Formatted text with clickable markdown citations.
        """
        if not text:
            return ""

        def replace_with_link(match: re.Match) -> str:
            pmid = match.group(1).strip()
            url = cls.build_pubmed_url(pmid)
            return f"[[PMID: {pmid}]]({url})"

        return PMID_REGEX.sub(replace_with_link, text)

    @classmethod
    def validate_grounding(
        cls,
        text: str,
        source_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verifies whether all PMIDs cited by the LLM were actually present in the
        source passages provided to it (Anti-Hallucination verification).

        Parameters:
            text: LLM response.
            source_passages: Passages provided in the context.

        Returns:
            Dictionary containing:
                - cited_pmids: all PMIDs extracted from text
                - valid_pmids: PMIDs that exist in source_passages
                - hallucinated_pmids: PMIDs cited by LLM but not in source passages
                - is_grounded: True if 100% of cited PMIDs are valid
                - grounding_score: Ratio of valid PMIDs / total cited PMIDs (0.0 to 1.0)
        """
        cited_pmids = cls.extract_pmids(text)
        available_pmids = {
            str(p.get("pmid", "")).strip() for p in source_passages if p.get("pmid")
        }

        valid_pmids = [p for p in cited_pmids if p in available_pmids]
        hallucinated_pmids = [p for p in cited_pmids if p not in available_pmids]

        total = len(cited_pmids)
        score = 1.0 if total == 0 else round(len(valid_pmids) / total, 4)
        is_grounded = len(hallucinated_pmids) == 0

        if not is_grounded:
            logger.warning(
                f"Detected {len(hallucinated_pmids)} hallucinated citation(s): {hallucinated_pmids}"
            )

        return {
            "cited_pmids": cited_pmids,
            "valid_pmids": valid_pmids,
            "hallucinated_pmids": hallucinated_pmids,
            "is_grounded": is_grounded,
            "grounding_score": score,
        }


if __name__ == "__main__":
    sample_answer = (
        "Metformin is the first-line medication for type 2 diabetes [PMID: 42976769]. "
        "Additionally, fixed-dose combinations with dapagliflozin improve glycemic control [PMID: 42970163]. "
        "Uncited claims may also reference [PMID: 42976769] redundantly."
    )
    passages = [
        {
            "pmid": "42976769",
            "title": "Metformin-supportive nutrition in Type 2 Diabetes",
            "text": "Metformin is a first-line treatment for diabetes mellitus.",
            "metadata": {"journal": "Lancet Diabetes", "pub_date": "2024"},
        },
        {
            "pmid": "42970163",
            "title": "Dapagliflozin and Metformin Combinations",
            "text": "Efficacy and safety of fixed-dose combination in clinical trials.",
            "metadata": {"journal": "Diabetes Care", "pub_date": "2023"},
        },
    ]

    print("=" * 60)
    print("Testing CitationParser (Step 4.3 Demo)")
    print("=" * 60)

    pmids = CitationParser.extract_pmids(sample_answer)
    print(f"\n1. Extracted PMIDs: {pmids}")

    citations = CitationParser.parse_citations(sample_answer, passages)
    print(f"\n2. Enriched Citations ({len(citations)}):")
    for c in citations:
        print(f"   - [PMID: {c.pmid}] {c.title} ({c.journal}, {c.pub_date})")
        print(f"     URL: {c.url}")

    linked_text = CitationParser.make_markdown_links(sample_answer)
    print(f"\n3. Markdown with Clickable Links:\n{linked_text}")

    grounding = CitationParser.validate_grounding(sample_answer, passages)
    print(f"\n4. Grounding Validation:\n   Is Grounded: {grounding['is_grounded']} (Score: {grounding['grounding_score']})")
