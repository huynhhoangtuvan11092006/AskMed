"""
src/ingestion/pubmed_fetcher.py
Module responsible for querying PubMed via NCBI Bio.Entrez E-utilities API
with an automatic transparent fallback to Europe PMC (PubMed Global Mirror).
Fetches abstracts, titles, publication dates, journals, DOIs, and saves them
into structured JSON format in data/raw/.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from Bio import Entrez
import requests

from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """Represents a structured scientific paper abstract fetched from PubMed."""
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str
    doi: Optional[str] = None
    url: str = ""
    mesh_terms: Optional[List[str]] = None

    def __post_init__(self):
        if not self.url and self.pmid:
            self.url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.mesh_terms is None:
            self.mesh_terms = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert article data to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PubMedArticle":
        """Instantiate article object from dictionary."""
        return cls(
            pmid=str(data.get("pmid", "")),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            journal=data.get("journal", ""),
            pub_date=data.get("pub_date", ""),
            doi=data.get("doi"),
            url=data.get("url", ""),
            mesh_terms=data.get("mesh_terms", []),
        )


class PubMedFetcher:
    """
    Fetcher client wrapping Bio.Entrez with automated Europe PMC fallback.
    Ensures 100% resilience even when NCBI temporarily rate-limits or blocks client IP.
    """

    EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(
        self,
        email: Optional[str] = None,
        tool: Optional[str] = None,
        min_abstract_words: Optional[int] = None,
    ):
        cfg = get_config()
        self.email = email or cfg.ingestion.entrez_email
        self.tool = tool or cfg.ingestion.entrez_tool
        self.min_abstract_words = (
            min_abstract_words if min_abstract_words is not None else cfg.ingestion.min_abstract_words
        )
        self.raw_dir = cfg.paths.raw_dir

        # Configure Biopython Entrez credentials
        Entrez.email = self.email
        Entrez.tool = self.tool
        logger.info(f"Initialized PubMedFetcher (Email: {self.email}, Tool: {self.tool})")

    def search_pmids(
        self,
        query: str,
        max_results: int = 50,
        sort: str = "relevance",
    ) -> List[str]:
        """
        Execute an Entrez esearch query against PubMed database.
        Returns a list of PubMed IDs (PMIDs).
        """
        logger.info(f"Searching PubMed via NCBI Entrez for: '{query}' (limit: {max_results})")
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort=sort,
            )
            raw_content = handle.read()
            handle.close()

            # Detect NCBI Access Denied HTML response
            if b"NCBI - WWW Error Blocked Diagnostic" in raw_content or b"Access Denied" in raw_content:
                raise PermissionError("NCBI IP Access Blocked. Switching to Europe PMC mirror.")

            from io import BytesIO
            record = Entrez.read(BytesIO(raw_content))
            pmids = record.get("IdList", [])
            logger.info(f"NCBI Entrez found {len(pmids)} articles.")
            return pmids
        except Exception as e:
            logger.warning(f"NCBI Entrez esearch unavailable ({e}). Using Europe PMC fallback.")
            return self._search_pmids_europe_pmc(query=query, max_results=max_results)

    def fetch_articles_by_pmids(
        self,
        pmids: List[str],
        batch_size: int = 50,
    ) -> List[PubMedArticle]:
        """
        Download and parse XML details for a given list of PMIDs using Entrez efetch.
        Falls back to Europe PMC if NCBI is blocked.
        """
        if not pmids:
            return []

        articles: List[PubMedArticle] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            logger.info(f"Fetching details for batch {i // batch_size + 1} ({len(batch)} PMIDs)...")

            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    retmode="xml",
                )
                raw_xml = handle.read()
                handle.close()

                if b"NCBI - WWW Error Blocked Diagnostic" in raw_xml or b"Access Denied" in raw_xml:
                    raise PermissionError("NCBI efetch IP Blocked. Falling back to Europe PMC.")

                from io import BytesIO
                records = Entrez.read(BytesIO(raw_xml))
                pubmed_articles = records.get("PubmedArticle", [])
                for item in pubmed_articles:
                    parsed = self._parse_pubmed_article(item)
                    if parsed:
                        articles.append(parsed)

                time.sleep(0.35)
            except Exception as e:
                logger.warning(f"NCBI efetch batch failed ({e}). Fetching batch via Europe PMC...")
                fallback_articles = self._fetch_pmids_europe_pmc(batch)
                articles.extend(fallback_articles)

        logger.info(f"Successfully retrieved {len(articles)} articles with valid abstracts.")
        return articles

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 50,
        save_raw: bool = True,
    ) -> List[PubMedArticle]:
        """
        High-level pipeline: search PMIDs, fetch details, and optionally save to raw JSON.
        """
        # Try NCBI first, if blocked or empty, Europe PMC handles the entire query directly
        try:
            pmids = self.search_pmids(query=query, max_results=max_results)
            articles = self.fetch_articles_by_pmids(pmids=pmids)
        except Exception as e:
            logger.warning(f"NCBI pipeline encountered error: {e}. Executing direct Europe PMC query.")
            articles = self._direct_search_europe_pmc(query=query, max_results=max_results)

        # If NCBI returned empty abstracts (or failed to parse), fallback to Europe PMC
        if not articles:
            logger.info("Executing Europe PMC direct retrieval to ensure complete abstracts...")
            articles = self._direct_search_europe_pmc(query=query, max_results=max_results)

        if save_raw and articles:
            output_file = self.save_to_json(articles, query=query)
            logger.info(f"Saved {len(articles)} articles to: {output_file}")

        return articles

    def save_to_json(
        self,
        articles: List[PubMedArticle],
        query: Optional[str] = None,
        custom_filename: Optional[str] = None,
    ) -> Path:
        """Saves a list of PubMedArticle objects into a structured JSON file in data/raw/."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if custom_filename:
            filename = custom_filename if custom_filename.endswith(".json") else f"{custom_filename}.json"
        else:
            query_slug = re.sub(r"[^\w\-_]", "_", query.lower()) if query else "articles"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pubmed_{query_slug[:30]}_{timestamp}.json"

        output_path = self.raw_dir / filename
        data = [a.to_dict() for a in articles]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path

    @staticmethod
    def load_from_json(json_path: Path) -> List[PubMedArticle]:
        """Loads a list of PubMedArticle objects from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PubMedArticle.from_dict(item) for item in data]

    # --------------------------------------------------------------------------
    # Europe PMC Fallback Implementation (Official PubMed Global Mirror)
    # --------------------------------------------------------------------------

    def _search_pmids_europe_pmc(self, query: str, max_results: int = 50) -> List[str]:
        """Searches PMIDs via Europe PMC REST API."""
        params = {
            "query": f"({query}) AND (SRC:MED)",
            "format": "json",
            "pageSize": min(max_results, 100),
            "resultType": "lite",
        }
        try:
            resp = requests.get(self.EUROPE_PMC_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("resultList", {}).get("result", [])
            pmids = [str(r.get("pmid")) for r in results if r.get("pmid")]
            logger.info(f"Europe PMC returned {len(pmids)} PMIDs.")
            return pmids
        except Exception as e:
            logger.error(f"Europe PMC search failed: {e}")
            return []

    def _fetch_pmids_europe_pmc(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetches full article metadata for a list of PMIDs via Europe PMC."""
        if not pmids:
            return []

        query = " OR ".join([f"ext_id:{p}" for p in pmids])
        return self._direct_search_europe_pmc(query=query, max_results=len(pmids))

    def _direct_search_europe_pmc(self, query: str, max_results: int = 50) -> List[PubMedArticle]:
        """Direct search against Europe PMC retrieving full abstract and metadata."""
        params = {
            "query": f"({query}) AND (HAS_ABSTRACT:y) AND (SRC:MED)",
            "format": "json",
            "resultType": "core",
            "pageSize": min(max_results, 100),
        }
        articles: List[PubMedArticle] = []
        try:
            resp = requests.get(self.EUROPE_PMC_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("resultList", {}).get("result", [])

            for item in items:
                pmid = str(item.get("pmid", "")).strip()
                if not pmid:
                    continue

                abstract = str(item.get("abstractText", "")).strip()
                # Clean simple HTML tags inside abstractText
                abstract = re.sub(r"<[^>]+>", "", abstract)
                if not abstract or len(abstract.split()) < self.min_abstract_words:
                    continue

                title = str(item.get("title", "")).strip().rstrip(".")
                journal = str(item.get("journalInfo", {}).get("journal", {}).get("title", "Unknown Journal")).strip()
                pub_date = str(item.get("firstPublicationDate", item.get("pubYear", "Unknown")))
                doi = item.get("doi")

                mesh_terms = []
                for mesh in item.get("meshHeadingList", {}).get("meshHeading", []):
                    descriptor = mesh.get("descriptorName")
                    if descriptor:
                        mesh_terms.append(descriptor)

                articles.append(
                    PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        journal=journal,
                        pub_date=pub_date,
                        doi=doi,
                        mesh_terms=mesh_terms,
                    )
                )
        except Exception as e:
            logger.error(f"Error in direct Europe PMC search: {e}")

        return articles

    # --------------------------------------------------------------------------
    # NCBI XML Parsing Helpers
    # --------------------------------------------------------------------------

    def _parse_pubmed_article(self, article_dict: Dict[str, Any]) -> Optional[PubMedArticle]:
        """Safely parses a single PubmedArticle XML node into a PubMedArticle dataclass."""
        try:
            medline = article_dict.get("MedlineCitation", {})
            pmid = str(medline.get("PMID", "")).strip()

            article = medline.get("Article", {})
            title = str(article.get("ArticleTitle", "")).strip()

            abstract = self._extract_abstract_text(article.get("Abstract", {}))
            if not abstract or len(abstract.split()) < self.min_abstract_words:
                return None

            journal_info = article.get("Journal", {})
            journal = str(journal_info.get("Title", journal_info.get("ISOAbbreviation", "Unknown Journal"))).strip()
            pub_date = self._extract_pub_date(article, medline)
            doi = self._extract_doi(article_dict)
            mesh_terms = self._extract_mesh_headings(medline)

            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                mesh_terms=mesh_terms,
            )
        except Exception as e:
            logger.debug(f"Failed to parse article node: {e}")
            return None

    @staticmethod
    def _extract_abstract_text(abstract_node: Dict[str, Any]) -> str:
        """Flattens abstract sections into a unified clean string."""
        if not abstract_node:
            return ""

        abstract_text_field = abstract_node.get("AbstractText", "")
        if isinstance(abstract_text_field, str):
            return abstract_text_field.strip()

        if isinstance(abstract_text_field, list):
            sections = []
            for sec in abstract_text_field:
                sec_str = str(sec).strip()
                label = getattr(sec, "attributes", {}).get("Label", "")
                if label:
                    sections.append(f"{label}: {sec_str}")
                else:
                    sections.append(sec_str)
            return "\n\n".join(sections)

        return str(abstract_text_field).strip()

    @staticmethod
    def _extract_pub_date(article: Dict[str, Any], medline: Dict[str, Any]) -> str:
        """Extracts publication year/date from ArticleDate or Journal PubDate."""
        article_dates = article.get("ArticleDate", [])
        if article_dates and isinstance(article_dates, list):
            first_date = article_dates[0]
            year = first_date.get("Year", "")
            month = first_date.get("Month", "")
            day = first_date.get("Day", "")
            if year:
                return f"{year}-{month.zfill(2) if month else '01'}-{day.zfill(2) if day else '01'}"

        journal_issue = article.get("Journal", {}).get("JournalIssue", {})
        pub_date_node = journal_issue.get("PubDate", {})
        year = pub_date_node.get("Year", "")
        medline_date = pub_date_node.get("MedlineDate", "")

        if year:
            month = pub_date_node.get("Month", "")
            return f"{year}-{month}" if month else str(year)
        if medline_date:
            return str(medline_date)

        return "Unknown"

    @staticmethod
    def _extract_doi(article_dict: Dict[str, Any]) -> Optional[str]:
        """Extracts DOI from PubmedData ArticleIdList."""
        pubmed_data = article_dict.get("PubmedData", {})
        article_id_list = pubmed_data.get("ArticleIdList", [])
        for item in article_id_list:
            if getattr(item, "attributes", {}).get("IdType") == "doi":
                return str(item).strip()
        return None

    @staticmethod
    def _extract_mesh_headings(medline: Dict[str, Any]) -> List[str]:
        """Extracts Medical Subject Headings (MeSH terms)."""
        headings = []
        mesh_list = medline.get("MeshHeadingList", [])
        for heading in mesh_list:
            descriptor = heading.get("DescriptorName", "")
            if descriptor:
                headings.append(str(descriptor))
        return headings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts by query.")
    parser.add_argument("--query", type=str, default="hypertension first-line treatment", help="Medical search term")
    parser.add_argument("--limit", type=int, default=3, help="Number of articles to fetch")
    args = parser.parse_args()

    fetcher = PubMedFetcher()
    results = fetcher.search_and_fetch(query=args.query, max_results=args.limit)

    print("\n" + "=" * 70)
    print(f"SEARCH RESULTS FOR: '{args.query}' (Retrieved {len(results)} valid articles)")
    print("=" * 70)
    for idx, art in enumerate(results, 1):
        print(f"\n[{idx}] PMID: {art.pmid} | DOI: {art.doi}")
        print(f"    Title   : {art.title}")
        print(f"    Journal : {art.journal} ({art.pub_date})")
        print(f"    URL     : {art.url}")
        print(f"    Abstract: {art.abstract[:180]}...")
    print("\nDone!")
