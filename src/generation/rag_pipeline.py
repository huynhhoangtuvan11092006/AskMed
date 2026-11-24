"""
src/generation/rag_pipeline.py
End-to-End Evidence-Based Medical Question Answering Pipeline for Ask_Med.
Orchestrates the entire multi-stage RAG flow:
  1. Hybrid Retrieval: Dense Semantic (ChromaDB) + Sparse Lexical (Rank-BM25)
  2. Reciprocal Rank Fusion (RRF k=60 -> Top 20)
  3. Deep Cross-Attention Reranking (Cross-Encoder -> Top 5)
  4. Evidence-Grounded Medical Prompt Synthesis (prompts.py)
  5. Local LLM Deterministic Inference (llama3.1:8b via Ollama)
  6. Citation Extraction, Markdown Hyperlinking & Grounding Validation (citation_parser.py)
"""

from functools import lru_cache
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.generation.citation_parser import Citation, CitationParser
from src.generation.llm_client import OllamaLLMClient
from src.generation.prompts import build_medical_rag_prompt
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AskMedPipeline:
    """
    Unified End-to-End MedRAG Pipeline coordinator.
    Connects retrieval, reranking, generation, and citation grounding.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[OllamaLLMClient] = None,
    ):
        logger.info("Initializing AskMedPipeline...")
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or OllamaLLMClient()
        logger.info("AskMedPipeline successfully initialized and ready.")

    def ask(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_top_k: Optional[int] = 20,
        rerank_top_n: Optional[int] = 5,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes non-streaming end-to-end question answering.

        Parameters:
            query: Clinical question from user or medical practitioner.
            dense_top_k: Number of dense vectors to retrieve.
            sparse_top_k: Number of BM25 documents to retrieve.
            rrf_top_k: Number of candidates after RRF fusion (default 20).
            rerank_top_n: Number of top reranked passages to supply to LLM (default 5).
            temperature: Sampling temperature override (default 0.0).

        Returns:
            Dictionary containing raw answer, formatted markdown answer with links,
            enriched citations, retrieved passages, grounding audit, and latency metrics.
        """
        t_total_start = time.perf_counter()

        # Step 1: Hybrid Retrieval (Dense Top 20 + Sparse Top 20 -> RRF Top 20)
        t_retrieval_start = time.perf_counter()
        rrf_candidates = self.retriever.retrieve(
            query=query,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            final_top_k=rrf_top_k,
        )
        t_retrieval = time.perf_counter() - t_retrieval_start

        # Step 2: Cross-Encoder Reranking (Top 20 -> Top 5)
        t_rerank_start = time.perf_counter()
        top_passages = self.reranker.rerank(
            query=query,
            candidates=rrf_candidates,
            top_n=rerank_top_n,
        )
        t_rerank = time.perf_counter() - t_rerank_start

        # Step 3: Medical Prompt Formatting
        sys_prompt, user_prompt = build_medical_rag_prompt(
            query=query,
            passages=top_passages,
        )

        # Step 4: Deterministic LLM Inference
        t_llm_start = time.perf_counter()
        raw_answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            stream=False,
            temperature=temperature,
        )
        t_llm = time.perf_counter() - t_llm_start

        # Step 5: Citation Parsing, Linking & Grounding Audit
        citations = CitationParser.parse_citations(raw_answer, top_passages)
        formatted_answer = CitationParser.make_markdown_links(raw_answer)
        grounding_audit = CitationParser.validate_grounding(raw_answer, top_passages)

        t_total = time.perf_counter() - t_total_start

        return {
            "query": query,
            "raw_answer": raw_answer,
            "formatted_answer": formatted_answer,
            "citations": [c.to_dict() for c in citations],
            "retrieved_passages": top_passages,
            "grounding": grounding_audit,
            "latency": {
                "retrieval_sec": round(t_retrieval, 4),
                "rerank_sec": round(t_rerank, 4),
                "llm_sec": round(t_llm, 4),
                "total_sec": round(t_total, 4),
            },
        }

    def ask_stream(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_top_k: Optional[int] = 20,
        rerank_top_n: Optional[int] = 5,
        temperature: Optional[float] = None,
    ) -> Tuple[Iterator[str], Dict[str, Any]]:
        """
        Executes streaming question answering for interactive UI display.

        Returns:
            Tuple of:
              - token_iterator: generator yielding tokens as they are produced by LLM.
              - metadata: dictionary containing retrieved passages, prompts, and timing.
        """
        t_retrieval_start = time.perf_counter()
        rrf_candidates = self.retriever.retrieve(
            query=query,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            final_top_k=rrf_top_k,
        )
        t_retrieval = time.perf_counter() - t_retrieval_start

        t_rerank_start = time.perf_counter()
        top_passages = self.reranker.rerank(
            query=query,
            candidates=rrf_candidates,
            top_n=rerank_top_n,
        )
        t_rerank = time.perf_counter() - t_rerank_start

        sys_prompt, user_prompt = build_medical_rag_prompt(
            query=query,
            passages=top_passages,
        )

        token_iterator = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            stream=True,
            temperature=temperature,
        )

        metadata = {
            "query": query,
            "top_passages": top_passages,
            "system_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "latency": {
                "retrieval_sec": round(t_retrieval, 4),
                "rerank_sec": round(t_rerank, 4),
            },
        }
        return token_iterator, metadata


# Global singleton instance for high-efficiency reuse
_DEFAULT_PIPELINE: Optional[AskMedPipeline] = None


def get_pipeline() -> AskMedPipeline:
    """Returns a shared AskMedPipeline singleton instance."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = AskMedPipeline()
    return _DEFAULT_PIPELINE


def ask_med(query: str, **kwargs) -> Dict[str, Any]:
    """
    Standard high-level entry point for Ask_Med.
    Takes a clinical question and returns a fully grounded medical response.

    Parameters:
        query: Clinical question.
        **kwargs: Optional pipeline parameter overrides.

    Returns:
        Structured response dictionary.
    """
    pipeline = get_pipeline()
    return pipeline.ask(query, **kwargs)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ask_Med End-to-End Medical RAG Pipeline CLI.")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the first-line medication for type 2 diabetes?",
        help="Medical query to process through the pipeline.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ASK_MED: EVIDENCE-BASED MEDICAL QUESTION ANSWERING (MedRAG)")
    print(f"QUERY: {args.query}")
    print("=" * 70 + "\n")

    result = ask_med(args.query)

    print("------------------------- CLINICAL ANSWER -------------------------")
    print(result["formatted_answer"])
    print("\n------------------------- EVIDENCE CITATIONS ----------------------")
    if result["citations"]:
        for c in result["citations"]:
            print(f"• [PMID: {c['pmid']}] {c['title']} ({c.get('journal', 'PubMed')})")
            print(f"  URL: {c['url']}")
    else:
        print("No citations extracted.")

    print("\n------------------------- GROUNDING AUDIT -------------------------")
    g = result["grounding"]
    print(f"Grounding Score: {g['grounding_score'] * 100}% | Is Fully Grounded: {g['is_grounded']}")
    if g["hallucinated_pmids"]:
        print(f"WARNING - Hallucinated PMIDs detected: {g['hallucinated_pmids']}")

    print("\n------------------------- LATENCY BREAKDOWN -----------------------")
    lat = result["latency"]
    print(f"Retrieval: {lat['retrieval_sec']}s | Reranker: {lat['rerank_sec']}s | LLM: {lat['llm_sec']}s | Total: {lat['total_sec']}s")
    print("=" * 70 + "\n")
