"""
tests/test_ragas_evaluator.py
Unit tests for src/evaluation/ragas_evaluator.py (BenchmarkOrchestrator).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.evaluation.eval_dataset import EvaluationDataset, MedicalEvalItem
from src.evaluation.ragas_evaluator import BenchmarkOrchestrator


@pytest.fixture
def sample_eval_item():
    return MedicalEvalItem(
        id="EVAL_001",
        topic="diabetes",
        category="treatment",
        question="What is the first-line oral pharmacological therapy for type 2 diabetes?",
        ground_truth="Metformin is recommended as the initial first-line pharmacological agent.",
        expected_pmids=["12345678", "23456789"],
    )


@pytest.fixture
def mock_pipeline():
    pipe = MagicMock()
    pipe.ask.return_value = {
        "raw_answer": "Metformin is the first-line drug [PMID: 12345678].",
        "retrieved_passages": [
            {
                "text": "Metformin is considered first-line for type 2 diabetes.",
                "pmid": "12345678",
                "rerank_score": 8.5,
            },
            {
                "text": "Lifestyle changes along with metformin reduce HbA1c.",
                "pmid": "99999999",
                "rerank_score": 5.2,
            },
        ],
        "grounding": {
            "is_grounded": True,
            "cited_pmids": ["12345678"],
            "retrieved_pmids": ["12345678", "99999999"],
            "hallucinated_pmids": [],
        },
    }
    return pipe


@pytest.fixture
def mock_ragas_evaluator():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.95,
        "context_precision": 1.0,
        "context_recall": 1.0,
        "ragas_score": 0.9875,
    }
    return evaluator


class TestBenchmarkOrchestrator:

    def test_init_custom_components(self, tmp_path, mock_pipeline, mock_ragas_evaluator):
        mock_ds = MagicMock(spec=EvaluationDataset)
        orch = BenchmarkOrchestrator(
            pipeline=mock_pipeline,
            ragas_evaluator=mock_ragas_evaluator,
            dataset=mock_ds,
        )
        assert orch.pipeline == mock_pipeline
        assert orch.ragas_evaluator == mock_ragas_evaluator
        assert orch.dataset == mock_ds
        assert orch.output_dir.exists()

    def test_evaluate_item(self, sample_eval_item, mock_pipeline, mock_ragas_evaluator):
        orch = BenchmarkOrchestrator(
            pipeline=mock_pipeline,
            ragas_evaluator=mock_ragas_evaluator,
            dataset=MagicMock(spec=EvaluationDataset),
        )

        row = orch.evaluate_item(sample_eval_item)

        assert row["id"] == "EVAL_001"
        assert row["topic"] == "diabetes"
        assert row["category"] == "treatment"
        assert "Metformin" in row["generated_answer"]
        assert row["expected_pmids"] == "12345678;23456789"
        assert row["retrieved_pmids"] == "12345678;99999999"
        assert row["is_grounded"] is True
        assert row["faithfulness"] == 1.0
        assert row["answer_relevancy"] == 0.95
        assert row["context_precision"] == 1.0
        assert row["context_recall"] == 1.0
        assert row["ragas_composite"] == 0.9875
        assert row["hit_rate@1"] == 1.0
        assert row["hit_rate@5"] == 1.0
        assert row["mrr"] == 1.0
        assert row["avg_rerank_score"] == pytest.approx(6.85, rel=1e-2)
        assert row["top1_rerank_score"] == 8.5
        assert row["latency_total_sec"] >= 0.0

    def test_compute_summary(self):
        orch = BenchmarkOrchestrator(
            pipeline=MagicMock(),
            ragas_evaluator=MagicMock(),
            dataset=MagicMock(),
        )

        rows = [
            {
                "faithfulness": 1.0,
                "answer_relevancy": 0.8,
                "context_precision": 1.0,
                "context_recall": 0.8,
                "ragas_composite": 0.9,
                "hit_rate@1": 1.0,
                "hit_rate@5": 1.0,
                "mrr": 1.0,
                "precision@5": 0.2,
                "recall@5": 0.5,
                "avg_rerank_score": 6.0,
                "top1_rerank_score": 8.0,
                "latency_total_sec": 1.5,
                "is_grounded": True,
            },
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.8,
                "context_precision": 0.6,
                "context_recall": 0.6,
                "ragas_composite": 0.7,
                "hit_rate@1": 0.0,
                "hit_rate@5": 1.0,
                "mrr": 0.5,
                "precision@5": 0.2,
                "recall@5": 0.5,
                "avg_rerank_score": 4.0,
                "top1_rerank_score": 6.0,
                "latency_total_sec": 2.5,
                "is_grounded": False,
            },
        ]

        summary = orch._compute_summary(rows)

        assert summary["total_evaluated"] == 2
        assert summary["grounding_rate"] == 0.5
        assert summary["averages"]["faithfulness"] == 0.9
        assert summary["averages"]["hit_rate@1"] == 0.5
        assert summary["averages"]["hit_rate@5"] == 1.0
        assert summary["averages"]["mrr"] == 0.75
        assert summary["averages"]["latency_total_sec"] == 2.0

    def test_compute_summary_empty(self):
        orch = BenchmarkOrchestrator(
            pipeline=MagicMock(),
            ragas_evaluator=MagicMock(),
            dataset=MagicMock(),
        )
        assert orch._compute_summary([]) == {}

    def test_run_benchmark(self, tmp_path, sample_eval_item, mock_pipeline, mock_ragas_evaluator):
        mock_ds = MagicMock(spec=EvaluationDataset)
        mock_ds.filter.return_value = [sample_eval_item]

        orch = BenchmarkOrchestrator(
            pipeline=mock_pipeline,
            ragas_evaluator=mock_ragas_evaluator,
            dataset=mock_ds,
        )
        orch.output_dir = tmp_path

        out_csv = tmp_path / "test_report.csv"
        summary = orch.run_benchmark(output_csv=str(out_csv))

        assert summary["total_evaluated"] == 1
        assert summary["grounding_rate"] == 1.0
        assert out_csv.exists()

        # Verify CSV contents
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2  # 1 header + 1 row
        assert "EVAL_001" in lines[1]
        assert "diabetes" in lines[1]

        # Verify summary JSON
        summary_json_path = tmp_path / "benchmark_summary.json"
        assert summary_json_path.exists()
        with open(summary_json_path, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["total_evaluated"] == 1
        assert saved_json["averages"]["faithfulness"] == 1.0

    def test_print_summary_table(self, capsys):
        orch = BenchmarkOrchestrator(
            pipeline=MagicMock(),
            ragas_evaluator=MagicMock(),
            dataset=MagicMock(),
        )
        summary = {
            "total_evaluated": 10,
            "grounding_rate": 1.0,
            "averages": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.88,
                "context_precision": 0.90,
                "context_recall": 0.85,
                "ragas_composite": 0.895,
                "hit_rate@5": 0.95,
                "hit_rate@1": 0.70,
                "mrr": 0.80,
                "avg_rerank_score": 6.2,
                "top1_rerank_score": 8.1,
                "latency_total_sec": 1.25,
            },
        }
        orch._print_summary_table(summary, count=10, csv_path="dummy.csv")
        captured = capsys.readouterr()
        assert "ASK_MED BENCHMARK EVALUATION REPORT" in captured.out
        assert "Faithfulness" in captured.out
        assert "EXCELLENT" in captured.out

