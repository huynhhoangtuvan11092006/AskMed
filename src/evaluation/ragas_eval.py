"""
src/evaluation/ragas_eval.py
Implementation of the 4 Core RAGAs Evaluation Metrics for Ask_Med.
Measures RAG system performance across:
  1. Faithfulness (Groundedness / Anti-Hallucination)
  2. Answer Relevancy (Question-Answer Alignment)
  3. Context Precision (Quality of Passage Ranking)
  4. Context Recall (Completeness of Retrieved Evidence against Ground Truth)

Fully self-contained using local Ollama LLM (llama3.1:8b) and Embeddings (nomic-embed-text),
enabling completely offline, reproducible scientific benchmarking without external API keys.
"""

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from src.generation.llm_client import OllamaLLMClient
from src.retrieval.vector_search import OllamaEmbeddingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two numeric vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


class RagasMedicalEvaluator:
    """
    Evaluator executing the 4 core RAGAs metrics tailored for medical evidence-based QA:
      - Faithfulness: Fraction of answer claims grounded in retrieved PubMed context.
      - Answer Relevancy: Degree to which the answer directly addresses the clinical query.
      - Context Precision: Weighted precision of relevant contexts placed at top ranks.
      - Context Recall: Fraction of ground truth clinical facts captured in retrieved contexts.
    """

    def __init__(
        self,
        llm_client: Optional[OllamaLLMClient] = None,
        embed_client: Optional[OllamaEmbeddingClient] = None,
    ):
        self.llm = llm_client or OllamaLLMClient()
        self.embed = embed_client or OllamaEmbeddingClient()
        logger.info("Initialized RagasMedicalEvaluator with local Ollama judge and embeddings.")

    def split_into_statements(self, text: str) -> List[str]:
        """Splits an answer into atomic factual sentences or statements."""
        # Clean markdown citations for atomic evaluation
        cleaned = re.sub(r"\[\[?PMID:\s*\d+\]?\](?:\([^\)]+\))?", "", text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 10]
        return sentences if sentences else [text.strip()]

    def compute_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        RAGAs Faithfulness metric.
        Measures the ratio of factual statements in the answer that can be inferred
        directly from the retrieved context (0.0 = complete hallucination, 1.0 = fully grounded).
        """
        if not answer or not contexts:
            return 0.0

        statements = self.split_into_statements(answer)
        if not statements:
            return 1.0

        combined_context = "\n---\n".join(contexts)

        # Batch prompt to LLM judge for deterministic verification
        prompt = f"""You are an expert medical evaluation judge.
Given the following MEDICAL CONTEXT, evaluate whether each STATEMENT can be directly inferred from the context.

MEDICAL CONTEXT:
{combined_context}

STATEMENTS TO EVALUATE:
"""
        for i, stmt in enumerate(statements, start=1):
            prompt += f"{i}. {stmt}\n"

        prompt += """
For each statement, output a verdict: '1' if supported by the context, or '0' if unsupported.
Respond strictly in JSON format as a list of integers: [1, 0, 1, ...]"""

        try:
            resp = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a strict, objective evaluation judge. Output ONLY a valid JSON list of 0 and 1 integers.",
                stream=False,
                temperature=0.0,
            )
            match = re.search(r"\[[\s\d,]+\]", resp)
            if match:
                verdicts = json.loads(match.group(0))
                if verdicts and len(verdicts) > 0:
                    supported = sum(1 for v in verdicts if v == 1)
                    return round(min(1.0, supported / len(statements)), 4)
        except Exception as e:
            logger.warning(f"Faithfulness LLM parsing fallback: {e}")

        # Heuristic fallback: check keyword overlap
        supported = 0
        context_text = combined_context.lower()
        for stmt in statements:
            words = [w.lower() for w in re.findall(r"\w+", stmt) if len(w) > 4]
            if words and sum(1 for w in words if w in context_text) / len(words) >= 0.5:
                supported += 1
        return round(supported / len(statements), 4)

    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        RAGAs Answer Relevancy metric.
        Measures how directly the answer addresses the question using embedding semantic similarity
        penalized if the answer is evasive or off-topic.
        """
        if not question or not answer:
            return 0.0

        q_vec = self.embed.embed_query(question)
        a_vec = self.embed.embed_query(answer)
        sim = cosine_similarity(q_vec, a_vec)

        # Scale semantic similarity from embedding space to [0.0, 1.0] range
        relevancy_score = max(0.0, min(1.0, (sim - 0.4) / 0.6 if sim > 0.4 else sim * 0.8))
        return round(relevancy_score, 4)

    def compute_context_precision(
        self,
        question: str,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """
        RAGAs Context Precision metric.
        Calculates whether the most relevant contexts are ranked at the top ranks (Rank 1, Rank 2...).
        Formula: Mean Average Precision (MAP) across relevant ranks.
        """
        if not contexts or not ground_truth:
            return 0.0

        gt_keywords = set(w.lower() for w in re.findall(r"\w+", ground_truth) if len(w) > 3)
        if not gt_keywords:
            return 1.0

        # Assess relevance of each chunk at rank r (1-based)
        relevance_flags: List[int] = []
        for ctx in contexts:
            ctx_text = ctx.lower()
            overlap = sum(1 for kw in gt_keywords if kw in ctx_text)
            # A chunk is considered relevant if it captures >= 25% of ground truth keywords
            is_relevant = 1 if (overlap / len(gt_keywords)) >= 0.25 else 0
            relevance_flags.append(is_relevant)

        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            return 0.0

        # Compute Precision@k summation
        running_rel = 0
        precision_sum = 0.0
        for rank, flag in enumerate(relevance_flags, start=1):
            if flag == 1:
                running_rel += 1
                precision_at_rank = running_rel / rank
                precision_sum += precision_at_rank

        context_precision = precision_sum / total_relevant
        return round(context_precision, 4)

    def compute_context_recall(
        self,
        question: str,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """
        RAGAs Context Recall metric.
        Measures the extent to which the ground truth can be attributed to the retrieved context.
        Formula: (Ground truth facts present in context) / (Total ground truth facts).
        """
        if not ground_truth:
            return 1.0
        if not contexts:
            return 0.0

        gt_statements = self.split_into_statements(ground_truth)
        if not gt_statements:
            return 1.0

        combined_context = "\n---\n".join(contexts)

        # Prompt LLM judge to classify attribution of each ground truth statement
        prompt = f"""You are an objective medical evaluation judge.
Given the following RETRIEVED CONTEXT, analyze whether each GROUND TRUTH CLAIM can be attributed to or supported by the context.

RETRIEVED CONTEXT:
{combined_context}

GROUND TRUTH CLAIMS:
"""
        for i, stmt in enumerate(gt_statements, start=1):
            prompt += f"{i}. {stmt}\n"

        prompt += """
For each claim, output '1' if attributed to the context or '0' if not.
Respond strictly in JSON format as a list of integers: [1, 0, 1, ...]"""

        try:
            resp = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a strict, objective evaluation judge. Output ONLY a valid JSON list of 0 and 1 integers.",
                stream=False,
                temperature=0.0,
            )
            match = re.search(r"\[[\s\d,]+\]", resp)
            if match:
                verdicts = json.loads(match.group(0))
                if verdicts and len(verdicts) > 0:
                    attributed = sum(1 for v in verdicts if v == 1)
                    return round(min(1.0, attributed / len(gt_statements)), 4)
        except Exception as e:
            logger.warning(f"Context Recall LLM parsing fallback: {e}")

        # Heuristic fallback: check semantic keyword overlap
        ctx_lower = combined_context.lower()
        attributed_count = 0
        for stmt in gt_statements:
            stmt_words = [w.lower() for w in re.findall(r"\w+", stmt) if len(w) > 3]
            if not stmt_words:
                attributed_count += 1
                continue
            matched = sum(1 for w in stmt_words if w in ctx_lower)
            if (matched / len(stmt_words)) >= 0.25:
                attributed_count += 1

        context_recall = attributed_count / len(gt_statements)
        return round(min(1.0, context_recall), 4)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        Executes all 4 RAGAs metrics on a single sample and returns a comprehensive score dict.
        """
        faithfulness = self.compute_faithfulness(question, answer, contexts)
        answer_relevancy = self.compute_answer_relevancy(question, answer)
        context_precision = self.compute_context_precision(question, ground_truth, contexts)
        context_recall = self.compute_context_recall(question, ground_truth, contexts)

        # Average composite RAGAs score
        ragas_composite = round(
            (faithfulness + answer_relevancy + context_precision + context_recall) / 4.0, 4
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "ragas_score": ragas_composite,
        }


if __name__ == "__main__":
    evaluator = RagasMedicalEvaluator()

    sample_q = "What is the recommended first-line pharmacological treatment for type 2 diabetes mellitus?"
    sample_ctx = [
        "Metformin is a first-line treatment for type 2 diabetes in adults and children [PMID: 42976769].",
        "Efficacy and safety of dapagliflozin and metformin fixed-dose combination in type 2 diabetes [PMID: 42970163].",
    ]
    sample_ans = "Metformin is universally recommended as the first-line medication for type 2 diabetes mellitus [PMID: 42976769]."
    sample_gt = "Metformin is universally recommended as the foundational first-line pharmacological agent for patients with type 2 diabetes mellitus."

    print("=" * 60)
    print("Testing RagasMedicalEvaluator (Step 5.2 Demo)")
    print("=" * 60)
    scores = evaluator.evaluate(
        question=sample_q,
        answer=sample_ans,
        contexts=sample_ctx,
        ground_truth=sample_gt,
    )
    for metric, score in scores.items():
        print(f"  • {metric:<18}: {score * 100:.2f}% ({score})")
