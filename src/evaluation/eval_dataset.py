"""
src/evaluation/eval_dataset.py
Medical Benchmark Dataset Loader and Validator for Ask_Med.
Handles loading, validation, filtering, and summary statistics for the 50 clinical
evaluation benchmark questions designed for RAGAs and Information Retrieval evaluations.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MedicalEvalItem:
    """Represents a single clinical benchmark item."""
    id: str
    category: str              # treatment, symptoms_diagnosis, drug_interactions, pathogenesis
    topic: str                 # diabetes, hypertension, covid19, cancer_immunotherapy, antibiotic_resistance
    question: str
    ground_truth: str
    expected_pmids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationDataset:
    """
    Manages loading, validation, and slicing of the clinical evaluation dataset.
    """

    def __init__(self, filepath: Optional[str] = None):
        cfg = get_config()
        self.filepath = Path(filepath or cfg.paths.eval_dataset_file)
        self.items: List[MedicalEvalItem] = []
        if self.filepath.exists():
            self.load()

    def load(self, filepath: Optional[str] = None) -> List[MedicalEvalItem]:
        """Loads and parses the JSON evaluation dataset."""
        target_path = Path(filepath or self.filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.items = []
        for idx, entry in enumerate(raw_data, start=1):
            item = MedicalEvalItem(
                id=entry.get("id", f"EVAL_{idx:03d}"),
                category=entry.get("category", "general"),
                topic=entry.get("topic", "general"),
                question=entry.get("question", "").strip(),
                ground_truth=entry.get("ground_truth", "").strip(),
                expected_pmids=[str(p).strip() for p in entry.get("expected_pmids", [])],
            )
            self.items.append(item)

        logger.info(f"Loaded {len(self.items)} evaluation questions from {target_path}")
        return self.items

    def save(self, filepath: Optional[str] = None) -> None:
        """Saves current items to JSON with UTF-8 indentation."""
        target_path = Path(filepath or self.filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = [item.to_dict() for item in self.items]
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.items)} items to {target_path}")

    def validate(self) -> Dict[str, Any]:
        """
        Validates structural integrity, non-emptiness, and minimum quality criteria.
        """
        errors: List[str] = []
        if len(self.items) < 50:
            errors.append(f"Dataset has {len(self.items)} questions, expected at least 50.")

        for item in self.items:
            if not item.question:
                errors.append(f"[{item.id}] Question is empty.")
            if not item.ground_truth:
                errors.append(f"[{item.id}] Ground truth is empty.")
            if not item.expected_pmids:
                errors.append(f"[{item.id}] Missing expected_pmids.")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "total_items": len(self.items),
            "error_count": len(errors),
            "errors": errors[:10],  # Truncate first 10 errors for brevity
        }

    def filter(
        self,
        category: Optional[str] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MedicalEvalItem]:
        """Filters items by category, topic, and limits results."""
        results = self.items
        if category:
            results = [i for i in results if i.category.lower() == category.lower()]
        if topic:
            results = [i for i in results if i.topic.lower() == topic.lower()]
        if limit is not None:
            results = results[:limit]
        return results

    def summary(self) -> Dict[str, Any]:
        """Calculates breakdown across categories and topics."""
        cat_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}

        for item in self.items:
            cat_counts[item.category] = cat_counts.get(item.category, 0) + 1
            topic_counts[item.topic] = topic_counts.get(item.topic, 0) + 1

        return {
            "total_questions": len(self.items),
            "categories": cat_counts,
            "topics": topic_counts,
        }


if __name__ == "__main__":
    dataset = EvaluationDataset()
    print("=" * 60)
    print("Ask_Med Clinical Benchmark Dataset Summary")
    print("=" * 60)
    stats = dataset.summary()
    print(f"Total Questions: {stats['total_questions']}")
    print("\nBreakdown by Category:")
    for cat, count in stats["categories"].items():
        print(f"  - {cat}: {count} questions")
    print("\nBreakdown by Topic:")
    for top, count in stats["topics"].items():
        print(f"  - {top}: {count} questions")

    validation = dataset.validate()
    print(f"\nValidation Result: {'VALID' if validation['is_valid'] else 'INVALID'}")
