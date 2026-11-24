"""
tests/test_eval_dataset.py
Unit tests for the Clinical Evaluation Dataset (src.evaluation.eval_dataset).
"""

from pathlib import Path
import pytest

from src.evaluation.eval_dataset import EvaluationDataset, MedicalEvalItem


@pytest.fixture
def dataset():
    ds = EvaluationDataset()
    return ds


def test_dataset_exists_and_loads(dataset):
    assert len(dataset.items) == 50
    assert dataset.filepath.exists()


def test_dataset_structure_and_types(dataset):
    for item in dataset.items:
        assert isinstance(item, MedicalEvalItem)
        assert item.id.startswith("MED_EVAL_")
        assert len(item.question) > 10
        assert len(item.ground_truth) > 20
        assert isinstance(item.expected_pmids, list)
        assert len(item.expected_pmids) > 0
        for pmid in item.expected_pmids:
            assert pmid.isdigit(), f"Invalid PMID: {pmid}"


def test_dataset_topic_and_category_coverage(dataset):
    stats = dataset.summary()
    assert stats["total_questions"] == 50

    expected_topics = {"diabetes", "hypertension", "covid19", "cancer_immunotherapy", "antibiotic_resistance"}
    assert set(stats["topics"].keys()) == expected_topics
    for topic, count in stats["topics"].items():
        assert count == 10, f"Topic {topic} expected 10 items, got {count}"

    expected_categories = {"treatment", "symptoms_diagnosis", "drug_interactions", "pathogenesis"}
    assert set(stats["categories"].keys()).issubset(expected_categories)
    assert len(stats["categories"]) >= 3


def test_dataset_validation(dataset):
    val = dataset.validate()
    assert val["is_valid"] is True
    assert val["error_count"] == 0
    assert val["total_items"] == 50


def test_dataset_filter_functionality(dataset):
    diabetes_items = dataset.filter(topic="diabetes")
    assert len(diabetes_items) == 10
    for item in diabetes_items:
        assert item.topic == "diabetes"

    treatment_items = dataset.filter(category="treatment")
    assert len(treatment_items) > 0
    for item in treatment_items:
        assert item.category == "treatment"

    limited_items = dataset.filter(limit=5)
    assert len(limited_items) == 5
