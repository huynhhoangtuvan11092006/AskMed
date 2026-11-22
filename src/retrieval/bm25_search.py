"""
src/retrieval/bm25_search.py
Sparse keyword indexing and retrieval module using Rank-BM25 (Okapi BM25).
Handles clinical tokenization, index serialization to data/bm25_index/bm25.pkl,
and fast keyword-based passage retrieval.
"""

import logging
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from src.config import get_config
from src.ingestion.chunker import MedicalChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard English stopwords to filter out non-informative noise words
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves",
}


def tokenize_medical_text(text: str) -> List[str]:
    """
    Tokenizes medical text for BM25 keyword matching:
    1. Lowercases text.
    2. Preserves clinical alphanumeric terms and hyphenated biomarkers (e.g. 'covid-19', 'hba1c', 'il-6').
    3. Removes standard stopwords.
    4. Filters out standalone single punctuation or very short non-informative tokens.
    """
    if not text:
        return []

    # Regex capturing words, numbers, and hyphenated medical compounds
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text.lower())

    # Filter stopwords and 1-letter noise (unless it's an important biomarker like 'b' in vitamin b)
    tokens = [t for t in raw_tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
    return tokens


class BM25Index:
    """
    Manages Okapi BM25 sparse index, chunk metadata mapping,
    and persistence to data/bm25_index/bm25.pkl.
    """

    def __init__(self, index_file: Optional[Path | str] = None):
        cfg = get_config()
        self.index_file = Path(index_file) if index_file else cfg.paths.bm25_index_file
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[MedicalChunk] = []

        # Auto-load existing index file if present
        if self.index_file.exists():
            self.load()

    def count(self) -> int:
        """Returns the number of chunks currently indexed in BM25."""
        return len(self.chunks)

    def build_index(self, chunks: List[MedicalChunk], auto_save: bool = True) -> int:
        """
        Builds the BM25 index from a list of MedicalChunk objects.
        Tokenizes the corpus, fits the Okapi BM25 model, and saves to disk.
        """
        if not chunks:
            logger.warning("Empty chunk list provided to BM25Index.build_index().")
            return 0

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        self.chunks = chunks

        # Tokenize each chunk text
        tokenized_corpus = [tokenize_medical_text(c.text) for c in self.chunks]

        # Fit BM25Okapi model (k1=1.5, b=0.75 are standard defaults)
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index successfully constructed ({len(self.chunks)} documents).")

        if auto_save:
            self.save()

        return len(self.chunks)

    def add_chunks(self, new_chunks: List[MedicalChunk], auto_save: bool = True) -> int:
        """
        Merges new chunks into the existing index (deduplicating by chunk_id)
        and re-fits the BM25Okapi model.
        """
        if not new_chunks:
            return len(self.chunks)

        seen_ids = {c.chunk_id for c in self.chunks}
        to_add: List[MedicalChunk] = []
        for c in new_chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                to_add.append(c)

        if not to_add:
            logger.info("All new chunks already exist in BM25 index. Skipping rebuild.")
            return len(self.chunks)

        combined_chunks = self.chunks + to_add
        logger.info(f"Adding {len(to_add)} new unique chunks to BM25 index (total: {len(combined_chunks)})...")
        return self.build_index(combined_chunks, auto_save=auto_save)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs BM25 keyword retrieval.
        Returns a ranked list of matched chunks with their BM25 scores and metadata.
        """
        cfg = get_config()
        k = top_k or cfg.retrieval.sparse_top_k

        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty or uninitialized. Returning empty list.")
            return []

        tokenized_query = tokenize_medical_text(query)
        if not tokenized_query:
            return []

        # Calculate BM25 scores for all documents in corpus
        scores = self.bm25.get_scores(tokenized_query)
        query_token_set = set(tokenized_query)

        # Pair each chunk with its score and sort descending
        scored_pairs = []
        for idx, score in enumerate(scores):
            doc_tokens = set(tokenize_medical_text(self.chunks[idx].text))
            matched = query_token_set.intersection(doc_tokens)
            if not matched:
                continue

            # In normal corpora score > 0; for tiny N <= 2 edge cases, fallback to matched count
            effective_score = float(score) if score > 0 else float(len(matched))
            scored_pairs.append((self.chunks[idx], effective_score))

        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Slice top_k
        top_pairs = scored_pairs[:k]

        # Format results matching the output structure of vector_search
        results: List[Dict[str, Any]] = []
        for chunk, score in top_pairs:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.to_dict(),
                    "pmid": chunk.pmid,
                    "title": chunk.title,
                    "url": chunk.url,
                    "score": round(score, 4),
                }
            )

        return results

    def save(self, output_path: Optional[Path | str] = None) -> Path:
        """Serializes BM25 model and chunk metadata to a pickle file."""
        target_path = Path(output_path) if output_path else self.index_file
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "bm25": self.bm25,
            "chunks": [c.to_dict() for c in self.chunks],
        }

        with open(target_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved BM25 index to: {target_path} ({len(self.chunks)} chunks)")
        return target_path

    def load(self, input_path: Optional[Path | str] = None) -> bool:
        """Loads BM25 model and chunk metadata from a pickle file."""
        target_path = Path(input_path) if input_path else self.index_file
        if not target_path.exists():
            logger.warning(f"BM25 index file not found at: {target_path}")
            return False

        try:
            with open(target_path, "rb") as f:
                payload = pickle.load(f)

            self.bm25 = payload.get("bm25")
            raw_chunks = payload.get("chunks", [])
            self.chunks = [MedicalChunk.from_dict(d) for d in raw_chunks]
            logger.info(f"Loaded BM25 index from '{target_path}' ({len(self.chunks)} chunks).")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {target_path}: {e}")
            return False

    def clear(self) -> None:
        """Clears memory and deletes index file on disk."""
        self.bm25 = None
        self.chunks = []
        if self.index_file.exists():
            self.index_file.unlink()
            logger.info(f"Removed BM25 index file: {self.index_file}")


if __name__ == "__main__":
    index = BM25Index()
    print("=" * 60)
    print("BM25 SPARSE INDEX STATUS:")
    print(f"Index File Path : {index.index_file}")
    print(f"Total Chunks    : {index.count()}")
    print("=" * 60)

