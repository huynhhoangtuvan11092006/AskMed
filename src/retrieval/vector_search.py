"""
src/retrieval/vector_search.py
Dense Vector Store and Retrieval module using ChromaDB PersistentClient
and local Ollama embedding model (nomic-embed-text).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
import ollama

from src.config import get_config
from src.ingestion.chunker import MedicalChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Helper client for generating embeddings using local Ollama nomic-embed-text."""

    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        cfg = get_config()
        self.base_url = base_url or cfg.models.ollama.base_url
        self.model_name = model_name or cfg.models.ollama.embed_model
        self.client = ollama.Client(host=self.base_url)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of document chunks using batched inference for high performance.
        Prepends 'search_document: ' task prefix as recommended by Nomic AI for MTEB retrieval.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompts = [f"search_document: {t}" for t in batch]
            try:
                # Use high-speed batched embedding endpoint supported in Ollama v0.3+
                res = self.client.embed(model=self.model_name, input=prompts)
                embeddings.extend(res["embeddings"])
            except Exception:
                # Transparent fallback to single sequential embedding if batching fails
                for prompt in prompts:
                    res = self.client.embeddings(model=self.model_name, prompt=prompt)
                    embeddings.append(res["embedding"])

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single user search query.
        Prepends 'search_query: ' task prefix for optimal semantic matching.
        """
        prompt = f"search_query: {query}"
        res = self.client.embeddings(model=self.model_name, prompt=prompt)
        return res["embedding"]


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store for medical knowledge chunks.
    Manages indexing, metadata preservation, and dense similarity search.
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_client: Optional[OllamaEmbeddingClient] = None,
    ):
        cfg = get_config()
        self.persist_dir = Path(persist_dir) if persist_dir else cfg.paths.chroma_db_dir
        self.collection_name = collection_name or cfg.retrieval.collection_name

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize Embedding client
        self.embedding_client = embedding_client or OllamaEmbeddingClient()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Ask_Med PubMed Medical Chunks", "hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaVectorStore at '{self.persist_dir}' "
            f"(collection: '{self.collection_name}', items: {self.count()})"
        )

    def count(self) -> int:
        """Returns the total number of chunks currently stored in the collection."""
        return self.collection.count()

    def add_chunks(self, chunks: List[MedicalChunk], batch_size: int = 50) -> int:
        """
        Embeds and upserts MedicalChunk objects into ChromaDB in batches.
        Preserves all associated metadata (PMID, title, URL, journal, etc.).
        Returns the count of successfully added chunks.
        """
        if not chunks:
            return 0

        total_added = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "pmid": c.pmid,
                    "title": c.title,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "token_count": c.token_count,
                    "journal": c.journal or "Unknown",
                    "pub_date": c.pub_date or "Unknown",
                    "doi": c.doi or "",
                    "url": c.url,
                }
                for c in batch
            ]

            logger.info(f"Generating embeddings for batch {i // batch_size + 1} ({len(batch)} chunks)...")
            embeddings = self.embedding_client.embed_documents(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_added += len(batch)
            logger.info(f"Upserted {total_added}/{total_chunks} chunks into ChromaDB.")

        return total_added

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs dense semantic vector search on ChromaDB.
        Returns a list of dictionaries with chunk metadata, content, and similarity score.
        """
        cfg = get_config()
        k = top_k or cfg.retrieval.dense_top_k

        if self.count() == 0:
            logger.warning("ChromaDB collection is empty. Returning empty search results.")
            return []

        # Generate query vector with 'search_query: ' prefix
        query_vector = self.embedding_client.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance range is [0, 2], where 0 is identical
            # Convert distance to similarity score: similarity = 1 - (dist / 2)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "pmid": meta.get("pmid", ""),
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "score": round(similarity, 4),
                    "distance": round(dist, 4),
                }
            )

        return formatted_results

    def get_all_pmids(self) -> List[str]:
        """Returns a sorted list of unique PMIDs currently indexed in the vector store."""
        if self.count() == 0:
            return []
        all_data = self.collection.get(include=["metadatas"])
        pmids = {m["pmid"] for m in all_data.get("metadatas", []) if "pmid" in m}
        return sorted(list(pmids))

    def delete_by_pmid(self, pmid: str) -> int:
        """Deletes all chunks belonging to a specific PMID."""
        target_pmid = str(pmid).strip()
        matched = self.collection.get(where={"pmid": target_pmid})
        ids_to_delete = matched.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for PMID: {target_pmid}")
        return len(ids_to_delete)

    def clear(self) -> None:
        """Resets the collection (deletes all indexed records)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Ask_Med PubMed Medical Chunks", "hnsw:space": "cosine"},
        )
        logger.info(f"Cleared ChromaDB collection: '{self.collection_name}'")


if __name__ == "__main__":
    store = ChromaVectorStore()
    print("=" * 60)
    print("CHROMA VECTOR STORE STATUS:")
    print(f"Persist Directory: {store.persist_dir}")
    print(f"Collection Name  : {store.collection_name}")
    print(f"Total Chunks     : {store.count()}")
    print(f"Unique PMIDs     : {len(store.get_all_pmids())}")
    print("=" * 60)

    if store.count() > 0:
        sample_query = "What is the first-line medication for type 2 diabetes?"
        print(f"\nDEMO DENSE RETRIEVAL for query: '{sample_query}' (Top 3):")
        hits = store.search(sample_query, top_k=3)
        for rank, hit in enumerate(hits, 1):
            print(f"[{rank}] PMID: {hit['pmid']} | Score: {hit['score']} | Dist: {hit['distance']}")
            print(f"    Title: {hit['title']}")
            print(f"    Text: {hit['text'][:120]}...\n")

