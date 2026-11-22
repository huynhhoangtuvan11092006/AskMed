"""
src/ingestion/ingest_pipeline.py
End-to-end ingestion pipeline module for Ask_Med.
Orchestrates:
1. Fetching scientific articles from PubMed NCBI (with Europe PMC failover).
2. Cleaning, unescaping, and quality filtering of abstracts and titles.
3. Sentence-aware token chunking (512 tokens / 128 overlap) with metadata attachment.
4. Synchronous indexing into ChromaDB (Dense Vector Store) and BM25 (Sparse Keyword Index).
5. Command Line Interface (CLI) for ad-hoc queries, PMID batch ingestion, and seeding.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.ingestion.cleaner import clean_article_list
from src.ingestion.chunker import MedicalChunk, MedicalChunker
from src.ingestion.pubmed_fetcher import PubMedArticle, PubMedFetcher
from src.retrieval.bm25_search import BM25Index
from src.retrieval.vector_search import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Core clinical topics for seeding foundational medical knowledge base (5 x 300 = 1,500 articles)
DEFAULT_SEED_QUERIES = [
    "diabetes mellitus treatment",
    "hypertension management",
    "COVID-19 clinical outcomes",
    "cancer immunotherapy",
    "antibiotic resistance",
]


class IngestionPipeline:
    """
    Unified Ingestion Pipeline executing Fetch -> Clean -> Chunk -> Store (Dense + Sparse).
    """

    def __init__(
        self,
        fetcher: Optional[PubMedFetcher] = None,
        chunker: Optional[MedicalChunker] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        min_abstract_words: Optional[int] = None,
    ):
        cfg = get_config()
        self.fetcher = fetcher or PubMedFetcher()
        self.chunker = chunker or MedicalChunker()
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.min_abstract_words = (
            min_abstract_words if min_abstract_words is not None else cfg.ingestion.min_abstract_words
        )

    def process_and_index_articles(
        self,
        raw_articles: List[PubMedArticle],
    ) -> Dict[str, Any]:
        """
        Takes raw PubMedArticle objects, cleans them, splits into chunks,
        and saves synchronously into both ChromaDB and BM25 index.
        """
        start_time = time.time()
        logger.info(f"--- Ingestion Pipeline: Processing {len(raw_articles)} raw articles ---")

        if not raw_articles:
            logger.warning("No articles provided to process_and_index_articles().")
            return {
                "raw_count": 0,
                "cleaned_count": 0,
                "chunks_count": 0,
                "vector_store_total": self.vector_store.count(),
                "bm25_total": self.bm25_index.count(),
                "elapsed_seconds": 0.0,
            }

        # Step 1: Cleaning & Quality Filtering
        cleaned_articles = clean_article_list(raw_articles, min_words=self.min_abstract_words)
        logger.info(f"Step 1 (Clean): {len(raw_articles)} raw -> {len(cleaned_articles)} valid articles.")

        if not cleaned_articles:
            logger.warning("All articles were filtered out during cleaning step.")
            return {
                "raw_count": len(raw_articles),
                "cleaned_count": 0,
                "chunks_count": 0,
                "vector_store_total": self.vector_store.count(),
                "bm25_total": self.bm25_index.count(),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }

        # Step 2: Sentence-Aware Chunking (512 tokens / 128 overlap)
        chunks = self.chunker.chunk_article_list(cleaned_articles)
        logger.info(f"Step 2 (Chunk): {len(cleaned_articles)} articles -> {len(chunks)} chunks.")

        if not chunks:
            logger.warning("No chunks produced from cleaned articles.")
            return {
                "raw_count": len(raw_articles),
                "cleaned_count": len(cleaned_articles),
                "chunks_count": 0,
                "vector_store_total": self.vector_store.count(),
                "bm25_total": self.bm25_index.count(),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }

        # Step 3: Indexing into Dense ChromaDB
        logger.info(f"Step 3 (ChromaDB): Storing {len(chunks)} chunks into vector store...")
        added_vectors = self.vector_store.add_chunks(chunks)
        logger.info(f"ChromaDB updated: added {added_vectors} chunks (total: {self.vector_store.count()}).")

        # Step 4: Indexing into Sparse BM25
        logger.info(f"Step 4 (BM25): Adding {len(chunks)} chunks into BM25 index...")
        bm25_total = self.bm25_index.add_chunks(chunks)
        logger.info(f"BM25 updated: total documents = {bm25_total}.")

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "raw_count": len(raw_articles),
            "cleaned_count": len(cleaned_articles),
            "chunks_count": len(chunks),
            "vector_store_total": self.vector_store.count(),
            "bm25_total": self.bm25_index.count(),
            "elapsed_seconds": elapsed,
        }
        logger.info(f"Ingestion batch completed successfully in {elapsed}s: {summary}")
        return summary

    def ingest_query(self, query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Searches PubMed by query, fetches full abstracts, cleans, chunks, and indexes.
        """
        logger.info(f"Starting query-based ingestion for: '{query}' (limit: {max_results})")
        pmids = self.fetcher.search_pmids(query=query, max_results=max_results)
        if not pmids:
            logger.warning(f"No PMIDs found for query: '{query}'")
            return {"query": query, "pmids_found": 0, "chunks_count": 0}

        raw_articles = self.fetcher.fetch_articles_by_pmids(pmids)
        result = self.process_and_index_articles(raw_articles)
        result["query"] = query
        result["pmids_found"] = len(pmids)
        return result

    def ingest_pmids(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Ingests a specific list of PMIDs into the knowledge base.
        """
        clean_pmids = [p.strip() for p in pmids if p.strip()]
        logger.info(f"Starting PMID-based ingestion for {len(clean_pmids)} PMIDs.")
        if not clean_pmids:
            return {"pmids_found": 0, "chunks_count": 0}

        raw_articles = self.fetcher.fetch_articles_by_pmids(clean_pmids)
        result = self.process_and_index_articles(raw_articles)
        result["pmids_found"] = len(clean_pmids)
        return result

    def ingest_file(self, file_path: Path | str) -> Dict[str, Any]:
        """
        Reads PMIDs from a text file (one PMID per line) and ingests them.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PMID file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        pmids = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        logger.info(f"Loaded {len(pmids)} PMIDs from {path}.")
        return self.ingest_pmids(pmids)

    def seed_knowledge_base(
        self,
        queries: Optional[List[str]] = None,
        limit_per_query: int = 300,
    ) -> Dict[str, Any]:
        """
        Executes a seed run over multiple foundational medical topics
        to populate the initial knowledge base with 200-300 articles.
        """
        seed_queries = queries or DEFAULT_SEED_QUERIES
        logger.info(
            f"Seeding knowledge base with {len(seed_queries)} clinical topics "
            f"({limit_per_query} articles each, target ~{len(seed_queries) * limit_per_query} papers)..."
        )

        overall_start = time.time()
        total_raw = 0
        total_cleaned = 0
        total_chunks = 0
        topic_summaries: List[Dict[str, Any]] = []

        for idx, q in enumerate(seed_queries, 1):
            logger.info(f"[{idx}/{len(seed_queries)}] Seeding topic: '{q}'")
            try:
                res = self.ingest_query(query=q, max_results=limit_per_query)
                total_raw += res.get("raw_count", 0)
                total_cleaned += res.get("cleaned_count", 0)
                total_chunks += res.get("chunks_count", 0)
                topic_summaries.append(res)
            except Exception as e:
                logger.error(f"Error seeding topic '{q}': {e}", exc_info=True)

        total_elapsed = round(time.time() - overall_start, 2)
        summary = {
            "topics_processed": len(seed_queries),
            "total_raw": total_raw,
            "total_cleaned": total_cleaned,
            "total_chunks_added": total_chunks,
            "vector_store_total": self.vector_store.count(),
            "bm25_total": self.bm25_index.count(),
            "total_elapsed_seconds": total_elapsed,
            "topic_details": topic_summaries,
        }
        logger.info(f"Seeding finished in {total_elapsed}s. Total chunks indexed: {self.vector_store.count()}.")
        return summary

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of both ChromaDB and BM25 knowledge stores."""
        return {
            "chroma_chunks": self.vector_store.count(),
            "chroma_pmids": len(self.vector_store.get_all_pmids()),
            "bm25_chunks": self.bm25_index.count(),
        }


def build_cli_parser() -> argparse.ArgumentParser:
    """Builds the command line interface argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.ingest_pipeline",
        description="Ask_Med: Scientific Ingestion Pipeline (PubMed -> ChromaDB & BM25)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-q", "--query",
        type=str,
        help="Search PubMed by query string and ingest matching articles.",
    )
    group.add_argument(
        "-p", "--pmids",
        type=str,
        help="Comma-separated list of PMIDs to fetch and ingest (e.g. '34567890,34567891').",
    )
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a text file containing PMIDs (one per line).",
    )
    group.add_argument(
        "--seed",
        action="store_true",
        help="Seed the knowledge base with ~200-300 foundational medical papers.",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Display current statistics of ChromaDB and BM25 index.",
    )

    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=300,
        help="Maximum articles to fetch per query (default: 300).",
    )
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.stats:
        stats = pipeline.get_stats()
        print("=" * 60)
        print("ASK_MED KNOWLEDGE STORE STATISTICS:")
        print(f"  ChromaDB Chunks  : {stats['chroma_chunks']}")
        print(f"  ChromaDB PMIDs   : {stats['chroma_pmids']}")
        print(f"  BM25 Chunks      : {stats['bm25_chunks']}")
        print("=" * 60)
        sys.exit(0)

    if args.seed:
        print("Starting foundational medical knowledge base seeding...")
        result = pipeline.seed_knowledge_base(limit_per_query=args.limit)
        print("\n" + "=" * 60)
        print("SEEDING COMPLETED:")
        print(f"  Topics Processed  : {result['topics_processed']}")
        print(f"  Raw Articles      : {result['total_raw']}")
        print(f"  Cleaned Articles  : {result['total_cleaned']}")
        print(f"  Total Chunks      : {result['total_chunks_added']}")
        print(f"  ChromaDB Total    : {result['vector_store_total']}")
        print(f"  BM25 Total        : {result['bm25_total']}")
        print(f"  Time Elapsed      : {result['total_elapsed_seconds']}s")
        print("=" * 60)
        sys.exit(0)

    if args.query:
        print(f"Ingesting PubMed articles for query: '{args.query}' (limit: {args.limit})...")
        res = pipeline.ingest_query(query=args.query, max_results=args.limit)
        print("\n" + "=" * 60)
        print("INGESTION FINISHED:")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        sys.exit(0)

    if args.pmids:
        pmid_list = [p.strip() for p in args.pmids.split(",") if p.strip()]
        print(f"Ingesting {len(pmid_list)} specific PMIDs: {pmid_list}...")
        res = pipeline.ingest_pmids(pmids=pmid_list)
        print("\n" + "=" * 60)
        print("INGESTION FINISHED:")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        sys.exit(0)

    if args.file:
        print(f"Ingesting PMIDs from file: {args.file}...")
        res = pipeline.ingest_file(file_path=args.file)
        print("\n" + "=" * 60)
        print("INGESTION FINISHED:")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()

