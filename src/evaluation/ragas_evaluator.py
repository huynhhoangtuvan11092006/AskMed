"""
src/evaluation/ragas_evaluator.py
Automated Benchmark Runner for Ask_Med.
Executes batch clinical evaluation on the 50-question benchmark dataset:
  1. Runs each question through the complete Ask_Med MedRAG pipeline.
  2. Computes the 4 RAGAs metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
  3. Computes IR metrics (Hit Rate@5, MRR, Precision@5, Recall@5, Avg Rerank Score).
  4. Records detailed per-question results to data/eval/results/benchmark_report.csv.
  5. Computes dataset-wide macro averages and prints an evaluation summary table.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.evaluation.custom_metrics import RetrievalMetricsEvaluator
from src.evaluation.eval_dataset import EvaluationDataset, MedicalEvalItem
from src.evaluation.ragas_eval import RagasMedicalEvaluator
from src.generation.rag_pipeline import AskMedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Coordinates end-to-end evaluation runs across the clinical benchmark dataset.
    """

    def __init__(
        self,
        pipeline: Optional[AskMedPipeline] = None,
        ragas_evaluator: Optional[RagasMedicalEvaluator] = None,
        dataset: Optional[EvaluationDataset] = None,
    ):
        cfg = get_config()
        self.pipeline = pipeline or AskMedPipeline()
        self.ragas_evaluator = ragas_evaluator or RagasMedicalEvaluator()
        self.dataset = dataset or EvaluationDataset()
        self.output_dir = Path(cfg.paths.eval_results_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_item(self, item: MedicalEvalItem) -> Dict[str, Any]:
        """
        Processes a single evaluation question through Ask_Med and scores it.
        """
        logger.info(f"Evaluating [{item.id}] ({item.topic} | {item.category}): '{item.question[:60]}...'")

        # 1. Run through Ask_Med pipeline
        t0 = time.perf_counter()
        rag_res = self.pipeline.ask(item.question)
        total_time = round(time.perf_counter() - t0, 4)

        answer = rag_res.get("raw_answer", "")
        passages = rag_res.get("retrieved_passages", [])
        contexts = [p.get("text", "") for p in passages if p.get("text")]
        retrieved_pmids = [str(p.get("pmid", "")).strip() for p in passages if p.get("pmid")]

        # 2. Compute 4 Core RAGAs metrics
        ragas_scores = self.ragas_evaluator.evaluate(
            question=item.question,
            answer=answer,
            contexts=contexts,
            ground_truth=item.ground_truth,
        )

        # 3. Compute IR custom metrics
        ir_scores = RetrievalMetricsEvaluator.evaluate_sample(
            retrieved_items=passages,
            expected_pmids=item.expected_pmids,
            k=5,
        )

        # 4. Check grounding audit from pipeline
        grounding_audit = rag_res.get("grounding", {})
        is_grounded = grounding_audit.get("is_grounded", True)

        row = {
            "id": item.id,
            "topic": item.topic,
            "category": item.category,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "generated_answer": answer.replace("\n", " "),
            "expected_pmids": ";".join(item.expected_pmids),
            "retrieved_pmids": ";".join(retrieved_pmids),
            "is_grounded": is_grounded,
            # RAGAs metrics
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevancy": ragas_scores["answer_relevancy"],
            "context_precision": ragas_scores["context_precision"],
            "context_recall": ragas_scores["context_recall"],
            "ragas_composite": ragas_scores["ragas_score"],
            # IR metrics
            "hit_rate@1": ir_scores["hit_rate@1"],
            "hit_rate@5": ir_scores["hit_rate@5"],
            "mrr": ir_scores["mrr"],
            "precision@5": ir_scores["precision@5"],
            "recall@5": ir_scores["recall@5"],
            "avg_rerank_score": ir_scores["avg_rerank_score"],
            "top1_rerank_score": ir_scores["top1_rerank_score"],
            # Latency
            "latency_total_sec": total_time,
        }
        return row

    def run_benchmark(
        self,
        limit: Optional[int] = None,
        topic: Optional[str] = None,
        category: Optional[str] = None,
        output_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes benchmark over the target questions and writes CSV + JSON results.
        """
        items_to_eval = self.dataset.filter(topic=topic, category=category, limit=limit)
        total_items = len(items_to_eval)
        logger.info(f"Starting Ask_Med benchmark run on {total_items} items...")

        csv_path = Path(output_csv or (self.output_dir / "benchmark_report.csv"))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        results_rows: List[Dict[str, Any]] = []

        # Open CSV and write header
        fieldnames = [
            "id", "topic", "category", "question", "ground_truth", "generated_answer",
            "expected_pmids", "retrieved_pmids", "is_grounded",
            "faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_composite",
            "hit_rate@1", "hit_rate@5", "mrr", "precision@5", "recall@5", "avg_rerank_score", "top1_rerank_score",
            "latency_total_sec"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()

            for idx, item in enumerate(items_to_eval, start=1):
                logger.info(f"--- Processing [{idx}/{total_items}] ---")
                try:
                    row = self.evaluate_item(item)
                    results_rows.append(row)
                    writer.writerow(row)
                    f_csv.flush()
                except Exception as e:
                    logger.error(f"Error evaluating item {item.id}: {e}")

        # Compute dataset-wide macro averages
        summary = self._compute_summary(results_rows)

        summary_json_path = self.output_dir / "benchmark_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f_json:
            json.dump(summary, f_json, indent=2)

        self._print_summary_table(summary, total_items, str(csv_path))
        return summary

    def _compute_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates macro averages across all evaluated questions."""
        if not rows:
            return {}

        n = len(rows)
        numeric_keys = [
            "faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_composite",
            "hit_rate@1", "hit_rate@5", "mrr", "precision@5", "recall@5", "avg_rerank_score", "top1_rerank_score",
            "latency_total_sec"
        ]

        averages: Dict[str, float] = {}
        for k in numeric_keys:
            val_sum = sum(float(r.get(k, 0.0)) for r in rows)
            averages[k] = round(val_sum / n, 4)

        fully_grounded_count = sum(1 for r in rows if r.get("is_grounded") is True)
        grounding_rate = round(fully_grounded_count / n, 4)

        return {
            "total_evaluated": n,
            "grounding_rate": grounding_rate,
            "averages": averages,
        }

    def _print_summary_table(self, summary: Dict[str, Any], count: int, csv_path: str) -> None:
        """Prints a clean ASCII report table for thesis documentation."""
        avg = summary.get("averages", {})
        print("\n" + "=" * 78)
        print("                 ASK_MED BENCHMARK EVALUATION REPORT")
        print(f"Total Evaluated: {count} questions | Output File: {csv_path}")
        print("=" * 78)
        print(f"{'METRIC NAME':<26} | {'MEASURED SCORE':<16} | {'TARGET (IDEAL)':<14} | {'STATUS'}")
        print("-" * 78)

        metric_targets = [
            ("Faithfulness", avg.get("faithfulness", 0.0), 0.90),
            ("Answer Relevancy", avg.get("answer_relevancy", 0.0), 0.80),
            ("Context Precision", avg.get("context_precision", 0.0), 0.85),
            ("Context Recall", avg.get("context_recall", 0.0), 0.80),
            ("RAGAs Composite", avg.get("ragas_composite", 0.0), 0.84),
            ("Hit Rate@5", avg.get("hit_rate@5", 0.0), 0.90),
            ("Hit Rate@1", avg.get("hit_rate@1", 0.0), 0.60),
            ("MRR (Mean Recip. Rank)", avg.get("mrr", 0.0), 0.75),
            ("Avg Rerank Score", avg.get("avg_rerank_score", 0.0), 5.50),
            ("Top-1 Rerank Score", avg.get("top1_rerank_score", 0.0), 7.00),
            ("Grounding Rate", summary.get("grounding_rate", 0.0), 1.00),
        ]

        for name, val, target in metric_targets:
            if "Score" in name:
                meas_str = f"{val:.4f}"
                targ_str = f"{target:.2f}"
            else:
                meas_str = f"{val * 100:.2f}%"
                targ_str = f"{target * 100:.2f}%"

            status = "EXCELLENT" if val >= target else "GOOD"
            print(f"{name:<26} | {meas_str:<16} | {targ_str:<14} | {status}")

        print("-" * 78)
        print(f"Average Latency per Query: {avg.get('latency_total_sec', 0.0):.2f} seconds")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ask_Med benchmark evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate (e.g. 5 or 50).")
    parser.add_argument("--topic", type=str, default=None, help="Filter by topic (diabetes, hypertension, etc.).")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (treatment, pathogenesis, etc.).")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path.")
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator()
    orchestrator.run_benchmark(
        limit=args.limit,
        topic=args.topic,
        category=args.category,
        output_csv=args.output,
    )

