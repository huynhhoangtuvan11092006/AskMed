"""
tests/test_prompts.py
Unit tests and live validation for medical prompts formatting in Ask_Med.
"""

import pytest
from src.generation.llm_client import OllamaLLMClient
from src.generation.prompts import (
    MEDICAL_SYSTEM_PROMPT,
    build_medical_rag_prompt,
    create_rag_messages,
    format_context,
)


@pytest.fixture
def sample_passages():
    return [
        {
            "pmid": "42976769",
            "title": "Metformin-supportive nutrition: When gut microbes eat drugs",
            "text": "Metformin is a first-line treatment for type 2 diabetes in adults and children.",
        },
        {
            "pmid": "42970163",
            "title": "Efficacy of Fixed-Dose Combination Tablet in Type 2 Diabetes",
            "text": "Dapagliflozin and metformin combination significantly reduced HbA1c levels.",
        },
    ]


def test_format_context_empty():
    assert format_context([]) == "[No scientific literature available in context]"


def test_format_context_structure(sample_passages):
    formatted = format_context(sample_passages)
    assert "[DOCUMENT 1 | PMID: 42976769]" in formatted
    assert "[DOCUMENT 2 | PMID: 42970163]" in formatted
    assert "Metformin is a first-line treatment" in formatted
    assert "Dapagliflozin and metformin combination" in formatted


def test_format_context_metadata_fallback():
    passages = [
        {
            "pmid": "12345",
            "title": "Sample Study",
            "text": "",
            "metadata": {"text": "Fallback content inside metadata."},
        }
    ]
    formatted = format_context(passages)
    assert "Fallback content inside metadata." in formatted


def test_build_medical_rag_prompt(sample_passages):
    query = "What is the primary medication for diabetes?"
    sys_p, usr_p = build_medical_rag_prompt(query, sample_passages)

    assert sys_p == MEDICAL_SYSTEM_PROMPT
    assert "MANDATORY CITATIONS" in sys_p
    assert "[PMID: <pmid>]" in sys_p
    assert query in usr_p
    assert "42976769" in usr_p
    assert "42970163" in usr_p


def test_create_rag_messages(sample_passages):
    query = "What is the primary medication for diabetes?"
    messages = create_rag_messages(query, sample_passages)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]


def test_live_llm_adherence_to_citation_prompt(sample_passages):
    """
    Live test checking that llama3.1:8b correctly outputs [PMID: 42976769]
    when prompted with the sample passages.
    """
    client = OllamaLLMClient()
    if not client.is_healthy():
        pytest.skip("Local Ollama daemon not active.")

    query = "What is the first-line medication for type 2 diabetes?"
    sys_p, usr_p = build_medical_rag_prompt(query, sample_passages)

    answer = client.generate(prompt=usr_p, system_prompt=sys_p, stream=False)
    assert isinstance(answer, str)
    assert "metformin" in answer.lower()
    # The LLM MUST cite the PMID provided in the prompt
    assert "[PMID: 42976769]" in answer or "42976769" in answer
