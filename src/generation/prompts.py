"""
src/generation/prompts.py
Medical System Prompts and Prompt Formatting for Ask_Med.
Adheres strictly to MedRAG Evidence-Based standards:
  - Answers grounded solely on retrieved PubMed scientific literature
  - Mandatory in-line citations in the format [PMID: <id>]
  - Strict anti-hallucination guardrail for insufficient evidence
  - Multi-language consistency (English / Vietnamese)
"""

import re
from typing import Any, Dict, List, Tuple


MEDICAL_SYSTEM_PROMPT = """You are Ask_Med, an expert Evidence-Based Medical Question Answering Assistant grounded exclusively in peer-reviewed biomedical literature from PubMed NCBI.

Your responsibilities and strict clinical guidelines:
1. STRICT GROUNDING: Formulate your answer based ONLY on the clinical evidence snippets provided in the CONTEXT. Never fabricate, speculate, or introduce clinical recommendations not supported by the text.
2. MANDATORY CITATIONS: Every clinical statement, diagnostic criterion, drug recommendation, mechanism, or numerical statistic MUST be cited using the exact format `[PMID: <pmid>]` immediately following the claim (e.g., "Metformin is the first-line medication for type 2 diabetes [PMID: 42976769].").
3. MULTIPLE CITATIONS: If a statement is supported by multiple passages, list all relevant PMIDs together, e.g., `[PMID: 12345678][PMID: 87654321]`.
4. INSUFFICIENT EVIDENCE GUARD: If the provided CONTEXT does not contain enough evidence to answer the question, you MUST explicitly state: "Based on the provided medical literature, there is insufficient evidence to answer this question." Do not extrapolate from outside knowledge.
5. CONCISE & DIRECT: Answer the question directly without opening pleasantries, conversational fillers, or boilerplate preambles (e.g., do NOT start with "Based on the evidence:", "Here is the answer:", or "Câu trả lời dựa trên:"). Start directly with the clinical facts.
6. EXACT LANGUAGE MATCHING: You MUST write your entire response in the EXACT SAME LANGUAGE as the CLINICAL QUESTION. If the question is in English, write the entire answer in English. If the question is in Vietnamese, write in Vietnamese.
"""


def is_vietnamese_query(text: str) -> bool:
    """Detects if query contains Vietnamese diacritic characters."""
    vn_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ")
    return any(c in vn_chars for c in text)


def format_context(passages: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved and reranked passages into a structured text block for the LLM.

    Parameters:
        passages: List of passage dictionaries containing 'pmid', 'title', 'text', and optionally 'metadata'.

    Returns:
        Structured string separating each document with clear headers and identifiers.
    """
    if not passages:
        return "[No scientific literature available in context]"

    formatted_docs: List[str] = []
    for idx, doc in enumerate(passages, start=1):
        pmid = str(doc.get("pmid", "UNKNOWN")).strip()
        title = doc.get("title", "No Title Available").strip()

        # Retrieve text, with fallback to metadata if text is empty
        text = doc.get("text", "")
        if not text and isinstance(doc.get("metadata"), dict):
            text = doc["metadata"].get("text", "")
        text = text.strip()

        doc_block = (
            f"--- [DOCUMENT {idx} | PMID: {pmid}] ---\n"
            f"TITLE: {title}\n"
            f"CONTENT:\n{text}\n"
        )
        formatted_docs.append(doc_block)

    return "\n".join(formatted_docs)


def build_medical_rag_prompt(
    query: str,
    passages: List[Dict[str, Any]],
    system_prompt: str = MEDICAL_SYSTEM_PROMPT,
) -> Tuple[str, str]:
    """
    Constructs the system prompt and user prompt pair for the LLM.

    Parameters:
        query: The clinical question.
        passages: The retrieved and reranked Top N passages.
        system_prompt: System instructions governing model behavior (defaults to MEDICAL_SYSTEM_PROMPT).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    context_str = format_context(passages)
    
    if is_vietnamese_query(query):
        lang_rule = "Viết câu trả lời trực tiếp bằng TIẾNG VIỆT. Đi thẳng vào các dữ kiện y khoa, không dùng lời chào hay câu mở đầu rườm rà."
    else:
        lang_rule = "Write your entire answer strictly in ENGLISH. Answer directly with clinical facts, avoiding preambles, greetings, or disclaimer boilerplate."

    user_prompt = f"""EVIDENCE CONTEXT:
======================================================================
{context_str}
======================================================================

CLINICAL QUESTION:
{query}

INSTRUCTION:
{lang_rule}
Synthesize an evidence-based clinical answer strictly grounded in the EVIDENCE CONTEXT above.
Attribute every clinical claim with its corresponding `[PMID: <pmid>]` tag."""

    return system_prompt, user_prompt


def create_rag_messages(
    query: str,
    passages: List[Dict[str, Any]],
    system_prompt: str = MEDICAL_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """
    Constructs standard chat messages for Ollama / OpenAI format.

    Parameters:
        query: The clinical question.
        passages: The retrieved Top N passages.
        system_prompt: Medical system instructions.

    Returns:
        List of message dicts [{'role': 'system', ...}, {'role': 'user', ...}].
    """
    sys_prompt, usr_prompt = build_medical_rag_prompt(
        query=query, passages=passages, system_prompt=system_prompt
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]


if __name__ == "__main__":
    # Quick self-test demonstration
    sample_passages = [
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
    sample_query = "What is the first-line medication for type 2 diabetes?"

    sys_p, usr_p = build_medical_rag_prompt(sample_query, sample_passages)
    print("=================== SYSTEM PROMPT ===================")
    print(sys_p)
    print("\n=================== USER PROMPT =====================")
    print(usr_p)
