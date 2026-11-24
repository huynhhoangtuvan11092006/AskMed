"""
tests/test_ragas_eval.py
Unit tests for the 4 core RAGAs evaluation metrics in Ask_Med (src.evaluation.ragas_eval).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.evaluation.ragas_eval import RagasMedicalEvaluator, cosine_similarity


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert cosine_similarity(v1, v2) == 1.0
    assert cosine_similarity(v1, v3) == 0.0
    assert cosine_similarity([0.0], [0.0]) == 0.0


def test_split_into_statements():
    evaluator = RagasMedicalEvaluator()
    text = (
        "Metformin is the first-line drug [[PMID: 42976769]](https://pubmed.ncbi.nlm.nih.gov/42976769/). "
        "It improves glycemic control and reduces cardiovascular risk. "
        "However, gastrointestinal side effects can occur."
    )
    statements = evaluator.split_into_statements(text)
    assert len(statements) == 3
    assert "Metformin is the first-line drug" in statements[0]
    # Check that markdown links were stripped from statements
    assert "https://pubmed" not in statements[0]


def test_faithfulness_grounded():
    evaluator = RagasMedicalEvaluator()
    contexts = [
        "Metformin is the first-line treatment for type 2 diabetes mellitus in adults.",
        "It reduces hepatic glucose production and increases insulin sensitivity."
    ]
    answer = "Metformin is used as first-line therapy for type 2 diabetes. It lowers hepatic glucose production."

    score = evaluator.compute_faithfulness("diabetes treatment", answer, contexts)
    assert score >= 0.5
    assert score <= 1.0


def test_answer_relevancy():
    evaluator = RagasMedicalEvaluator()
    question = "What is the primary medication for type 2 diabetes?"

    relevant_ans = "Metformin is the recommended first-line pharmacological treatment for type 2 diabetes."
    irrelevant_ans = "Jupiter is the largest planet in our solar system and possesses a massive red storm."

    rel_score = evaluator.compute_answer_relevancy(question, relevant_ans)
    irrel_score = evaluator.compute_answer_relevancy(question, irrelevant_ans)

    assert rel_score > irrel_score
    assert rel_score >= 0.5
    assert irrel_score < 0.4


def test_context_precision_ranking():
    evaluator = RagasMedicalEvaluator()
    question = "What is first-line for diabetes?"
    ground_truth = "Metformin is the first-line medication for type 2 diabetes mellitus."

    # Relevant context at Rank 1 (High precision)
    contexts_rank1 = [
        "Metformin is first-line medication for type 2 diabetes.",
        "Astrophysics study of black holes and gravitational waves.",
    ]
    score_rank1 = evaluator.compute_context_precision(question, ground_truth, contexts_rank1)

    # Relevant context at Rank 2 (Lower precision)
    contexts_rank2 = [
        "Astrophysics study of black holes and gravitational waves.",
        "Metformin is first-line medication for type 2 diabetes.",
    ]
    score_rank2 = evaluator.compute_context_precision(question, ground_truth, contexts_rank2)

    assert score_rank1 >= score_rank2
    assert score_rank1 == 1.0  # Perfect precision at rank 1


def test_context_recall():
    evaluator = RagasMedicalEvaluator()
    question = "What is the role of metformin and dapagliflozin?"
    ground_truth = "Metformin reduces hepatic glucose production. Dapagliflozin promotes urinary glucose excretion."

    # Full context containing both facts
    full_context = [
        "Metformin reduces hepatic glucose production and improves peripheral insulin sensitivity.",
        "Dapagliflozin inhibits SGLT2 in renal tubules, promoting urinary glucose excretion.",
    ]
    recall_full = evaluator.compute_context_recall(question, ground_truth, full_context)
    assert recall_full >= 0.9

    # Partial context containing only 1 fact
    partial_context = [
        "Metformin reduces hepatic glucose production.",
        "Completely unrelated text about viral transmission.",
    ]
    recall_partial = evaluator.compute_context_recall(question, ground_truth, partial_context)
    assert recall_partial < recall_full


def test_evaluate_full_suite():
    evaluator = RagasMedicalEvaluator()
    question = "What is the first-line medication for type 2 diabetes?"
    answer = "Metformin is the first-line medication for type 2 diabetes [PMID: 42976769]."
    contexts = [
        "Metformin is a first-line treatment for type 2 diabetes in adults and children.",
        "SGLT2 inhibitors provide complementary cardiorenal protection.",
    ]
    ground_truth = "Metformin is universally recommended as the first-line therapy for type 2 diabetes mellitus."

    res = evaluator.evaluate(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_score"]:
        assert k in res
        assert 0.0 <= res[k] <= 1.0
