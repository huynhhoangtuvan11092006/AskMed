"""
src/ui/pages/1_Chat.py
----------------------
Interactive Clinical Chat Interface for AskMed (Evidence-Based Medical RAG).
Features:
  - Real-time conversational interface (st.chat_message, st.chat_input)
  - Streaming token responses via Server-Sent Events (SSE) or local pipeline
  - Clickable [PMID: ...] badges hyperlinking directly to PubMed NCBI
  - Expandable Cross-Encoder Evidence Passage Inspector (raw contexts & scores)
  - Grounding validation badge and latency metrics
  - Pre-configured clinical prompt shortcuts for instant testing
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.request

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AskMed - Khung Chat Y Khoa",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom Clinical CSS Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main typography */
    .main {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Page Header */
    .page-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* Clinical Citation Chip */
    .citation-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background-color: #f0fdfa;
        border: 1px solid #0d9488;
        border-radius: 16px;
        color: #0f766e !important;
        font-weight: 600;
        font-size: 0.82rem;
        text-decoration: none !important;
        margin-right: 8px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .citation-chip:hover {
        background-color: #0d9488;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.3);
    }

    /* Safety & Grounding Badges */
    .grounding-badge-ok {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 12px;
        color: #047857;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .grounding-badge-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 12px;
        color: #b45309;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Latency & Metadata Tag */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        background-color: #f1f5f9;
        border-radius: 10px;
        color: #64748b;
        font-size: 0.78rem;
    }

    /* Passage Card in Accordion */
    .passage-card {
        background-color: #f8fafc;
        border-left: 4px solid #028090;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .passage-meta {
        font-weight: 600;
        color: #0f766e;
        margin-bottom: 6px;
        font-size: 0.8rem;
    }

    /* Preset prompt button styling */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 3. Backend Communication Helpers
# -----------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def check_api_health() -> Dict[str, Any]:
    """Probes the FastAPI backend /health endpoint."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "offline", "ollama_connected": False, "vector_db_ready": False}


def query_backend_stream(
    query: str,
    top_k: int = 5,
    include_raw_contexts: bool = True
):
    """
    Calls FastAPI /api/v1/query with stream=True using Server-Sent Events (SSE).
    Yields text tokens as they arrive, and finishes with metadata.
    """
    url = f"{API_BASE_URL}/api/v1/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "include_raw_contexts": include_raw_contexts,
        "stream": True
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                    ev_type = event.get("type")
                    if ev_type == "token":
                        yield ("token", event.get("content", ""))
                    elif ev_type == "done":
                        yield ("done", event)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        yield ("error", str(exc))


def query_backend_sync(
    query: str,
    top_k: int = 5,
    include_raw_contexts: bool = True
) -> Dict[str, Any]:
    """
    Non-streaming fallback to FastAPI /api/v1/query.
    """
    url = f"{API_BASE_URL}/api/v1/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "include_raw_contexts": include_raw_contexts,
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def query_pipeline_direct(
    query: str,
    top_k: int = 5,
    include_raw_contexts: bool = True
) -> Dict[str, Any]:
    """
    Direct in-process fallback using AskMedPipeline when FastAPI server is unavailable.
    """
    from src.generation.rag_pipeline import get_pipeline
    pipeline = get_pipeline()
    t0 = time.perf_counter()
    res = pipeline.ask(query=query, rerank_top_n=top_k)
    duration = time.perf_counter() - t0

    citations = [
        {
            "pmid": str(c.get("pmid", "")),
            "title": c.get("title", ""),
            "journal": c.get("journal", ""),
            "pub_year": str(c.get("pub_year", "")) if c.get("pub_year") else None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{c.get('pmid', '')}/"
        }
        for c in res.get("citations", [])
    ]

    contexts = []
    if include_raw_contexts:
        for p in res.get("retrieved_passages", []):
            contexts.append({
                "pmid": str(p.get("pmid") or p.get("metadata", {}).get("pmid", "")),
                "chunk_id": str(p.get("chunk_id") or p.get("metadata", {}).get("chunk_id", "")),
                "text": p.get("text", "") or p.get("metadata", {}).get("text", ""),
                "score": p.get("rerank_score") or p.get("score")
            })

    return {
        "query": query,
        "answer": res.get("answer", ""),
        "citations": citations,
        "contexts": contexts,
        "processing_time_sec": round(duration, 3),
        "is_grounded": res.get("grounding", {}).get("is_grounded", len(citations) > 0),
        "total_contexts_retrieved": len(contexts)
    }


# -----------------------------------------------------------------------------
# 4. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Real-Time Probes
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 AskMed Control Panel")
    
    # Live System Health Probe
    health = check_api_health()
    api_online = health.get("status") in ["ok", "degraded"]
    ollama_online = health.get("ollama_connected", False)

    if api_online:
        st.success("🟢 FastAPI Backend: Sẵn sàng (8000)")
    else:
        st.warning("🟡 FastAPI Offline (Sử dụng Direct In-Process)")

    if ollama_online:
        st.success("🟢 Local Ollama (llama3.1:8b): Kết nối")
    else:
        st.error("🔴 Ollama LLM: Mất kết nối")

    st.markdown("---")
    st.markdown("#### ⚙️ Cấu hình RAG Retrieval")

    top_k_slider = st.slider(
        "Số đoạn trích dẫn (Top-K Chunks):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Số lượng đoạn văn bản được Cross-Encoder Reranker chấm điểm cao nhất gửi tới LLM."
    )

    use_streaming = st.checkbox(
        "Hiển thị phản hồi dạng Streaming",
        value=True,
        help="Hiển thị từng token câu trả lời mượt mà theo thời gian thực."
    )

    show_raw_contexts = st.checkbox(
        "Trích xuất Bằng chứng gốc (Passage Inspector)",
        value=True,
        help="Hiển thị Accordion chi tiết các đoạn văn bản khoa học từ PubMed dùng để sinh câu trả lời."
    )

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:20px; line-height:1.4;">
        <b>Lưu ý:</b> AskMed là hệ thống AI hỗ trợ học thuật & nghiên cứu y học chứng cứ.
        Không thay thế quyết định chẩn đoán hoặc kê đơn của bác sĩ chuyên khoa.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 6. Page Header & Preset Clinical Prompts
# -----------------------------------------------------------------------------
st.markdown('<div class="page-title">💬 Khung Chat Hỏi Đáp Y Khoa Lâm Sàng</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Hệ thống RAG Y học Chứng cứ (Evidence-Based QA) tự động đối chiếu và trích dẫn bài báo gốc NCBI PubMed.</div>',
    unsafe_allow_html=True
)

# Preset Quick-Start Buttons
st.markdown("##### 💡 Câu hỏi lâm sàng mẫu (Nhấp để tra cứu nhanh):")
col1, col2, col3, col4 = st.columns(4)

preset_prompts = [
    ("💊 Metformin & CKD", "Bệnh nhân đái tháo đường type 2 có suy thận mạn eGFR 35 mL/phút có dùng được Metformin không?"),
    ("⚠️ Tương tác Paxlovid", "Paxlovid (Nirmatrelvir/Ritonavir) có tương tác thuốc nguy hiểm với các nhóm Statin không?"),
    ("⚖️ SGLT2i vs GLP-1 RA", "So sánh hiệu quả giảm biến cố tim mạch và thận giữa nhóm SGLT2 inhibitors và GLP-1 receptor agonists?"),
    ("🧬 Đột biến EGFR", "Cơ chế kháng thuốc Osimertinib phổ biến nhất trong ung thư phổi không tế bào nhỏ (NSCLC) đột biến EGFR là gì?")
]

with col1:
    if st.button(preset_prompts[0][0], use_container_width=True):
        st.session_state.pending_query = preset_prompts[0][1]
        st.rerun()

with col2:
    if st.button(preset_prompts[1][0], use_container_width=True):
        st.session_state.pending_query = preset_prompts[1][1]
        st.rerun()

with col3:
    if st.button(preset_prompts[2][0], use_container_width=True):
        st.session_state.pending_query = preset_prompts[2][1]
        st.rerun()

with col4:
    if st.button(preset_prompts[3][0], use_container_width=True):
        st.session_state.pending_query = preset_prompts[3][1]
        st.rerun()

st.markdown("<hr style='margin: 16px 0 24px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Helper: Render an Assistant Message with Citations & Contexts
# -----------------------------------------------------------------------------
def render_assistant_content(msg: Dict[str, Any]):
    """Renders the synthesized answer, citation pills, evidence accordion, and badges."""
    answer = msg.get("content", "")
    citations = msg.get("citations", [])
    contexts = msg.get("contexts", [])
    latency = msg.get("processing_time", None)
    is_grounded = msg.get("is_grounded", True)

    # 1. Main Answer Text
    st.markdown(answer)

    # 2. PubMed Citation Chips
    if citations:
        st.markdown("###### 📚 Bằng chứng khoa học trích dẫn (PubMed NCBI):")
        chips_html = '<div style="margin-bottom: 12px;">'
        for cit in citations:
            pmid = cit.get("pmid", "")
            title = cit.get("title", f"PubMed Article {pmid}")
            journal = cit.get("journal", "")
            year = cit.get("pub_year", "")
            url = cit.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

            hover_text = f"{title}"
            if journal:
                hover_text += f" - {journal}"
            if year:
                hover_text += f" ({year})"

            chips_html += (
                f'<a href="{url}" target="_blank" class="citation-chip" title="{hover_text}">'
                f'📄 PMID: {pmid}</a>'
            )
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    # 3. Evidence Passage Inspector (Accordion)
    if contexts and show_raw_contexts:
        with st.expander(f"🔬 Xem chi tiết {len(contexts)} đoạn ngữ cảnh bằng chứng gốc (Cross-Encoder Reranked)"):
            for idx, ctx in enumerate(contexts, 1):
                ctx_pmid = ctx.get("pmid", "N/A")
                score = ctx.get("score")
                score_str = f"Rerank Score: {score:.4f}" if score is not None else "Evidence Chunk"
                text = ctx.get("text", "")
                chunk_id = ctx.get("chunk_id", f"chunk_{idx}")

                st.markdown(
                    f"""
                    <div class="passage-card">
                        <div class="passage-meta">
                            <b>#{idx}</b> | PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{ctx_pmid}/" target="_blank">{ctx_pmid}</a> | {score_str} | ID: <code>{chunk_id}</code>
                        </div>
                        <div>{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 4. Footer Metadata (Grounding Badge & Latency)
    meta_cols = st.columns([1, 1])
    with meta_cols[0]:
        if is_grounded and citations:
            st.markdown(
                '<span class="grounding-badge-ok">🟢 Bằng chứng xác thực (Grounded)</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="grounding-badge-warn">🟡 Không đủ bằng chứng rõ ràng</span>',
                unsafe_allow_html=True
            )
    with meta_cols[1]:
        if latency is not None:
            st.markdown(
                f'<div style="text-align: right;"><span class="meta-tag">⚡ Thời gian xử lý: {latency:.2f}s</span></div>',
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# 8. Render Existing Chat Conversation History
# -----------------------------------------------------------------------------
for message in st.session_state.messages:
    role = message.get("role")
    if role == "user":
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(message.get("content", ""))
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🩺"):
            render_assistant_content(message)


# -----------------------------------------------------------------------------
# 9. Handle User Query Input & Response Generation
# -----------------------------------------------------------------------------
user_query = st.chat_input("Nhập câu hỏi lâm sàng, triệu chứng, hoặc cơ chế dược lý...")

# Handle pending query from preset buttons if user clicked one
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    # 1. Render and record User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(user_query)

    # 2. Render and compute Assistant Message
    with st.chat_message("assistant", avatar="🩺"):
        full_answer = ""
        citations_result = []
        contexts_result = []
        is_grounded_result = True
        latency_val = 0.0
        start_t = time.perf_counter()

        # Decide whether to stream or use sync endpoint
        if use_streaming and api_online:
            message_placeholder = st.empty()
            with st.spinner("Đang truy xuất kho tri thức NCBI PubMed & Reranking..."):
                stream_gen = query_backend_stream(
                    query=user_query,
                    top_k=top_k_slider,
                    include_raw_contexts=show_raw_contexts
                )

            token_stream_started = False
            for event_type, payload in stream_gen:
                if event_type == "token":
                    full_answer += payload
                    message_placeholder.markdown(full_answer + "▌")
                    token_stream_started = True
                elif event_type == "done":
                    citations_result = payload.get("citations", [])
                    contexts_result = payload.get("contexts", [])
                    is_grounded_result = payload.get("is_grounded", len(citations_result) > 0)
                elif event_type == "error":
                    st.error(f"⚠️ Lỗi streaming API: {payload}")
                    full_answer = f"Không thể hoàn thành truy vấn: {payload}"

            latency_val = time.perf_counter() - start_t
            message_placeholder.markdown(full_answer)

            # If done event did not deliver raw contexts due to SSE size, fetch or keep citations
            if not citations_result and "[" in full_answer and "PMID:" in full_answer:
                import re
                pmids = list(set(re.findall(r"PMID:\s*(\d+)", full_answer)))
                citations_result = [{"pmid": p, "url": f"https://pubmed.ncbi.nlm.nih.gov/{p}/"} for p in pmids]

        else:
            # Sync mode: either FastAPI sync or direct in-process pipeline
            with st.spinner("Đang phân tích câu hỏi, truy xuất ngữ cảnh và tổng hợp câu trả lời..."):
                try:
                    if api_online:
                        resp_data = query_backend_sync(
                            query=user_query,
                            top_k=top_k_slider,
                            include_raw_contexts=show_raw_contexts
                        )
                    else:
                        resp_data = query_pipeline_direct(
                            query=user_query,
                            top_k=top_k_slider,
                            include_raw_contexts=show_raw_contexts
                        )

                    full_answer = resp_data.get("answer", "")
                    citations_result = resp_data.get("citations", [])
                    contexts_result = resp_data.get("contexts", []) or []
                    latency_val = resp_data.get("processing_time_sec", time.perf_counter() - start_t)
                    is_grounded_result = resp_data.get("is_grounded", len(citations_result) > 0)

                except Exception as exc:
                    full_answer = f"⚠️ Lỗi xử lý truy vấn: {exc}"
                    latency_val = time.perf_counter() - start_t

        # Package message record
        assistant_record = {
            "role": "assistant",
            "content": full_answer,
            "citations": citations_result,
            "contexts": contexts_result,
            "processing_time": latency_val,
            "is_grounded": is_grounded_result
        }

        # Save to session history
        st.session_state.messages.append(assistant_record)

        # Re-render complete assistant card with citations & contexts
        st.rerun()
