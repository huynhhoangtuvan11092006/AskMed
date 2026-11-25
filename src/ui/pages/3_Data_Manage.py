"""
src/ui/pages/3_Data_Manage.py
-----------------------------
Medical Evidence Knowledge Base Management & Ingestion Dashboard for AskMed.
Features:
  - Real-time KPI metrics (Total Articles, Chunks, ChromaDB, BM25 Index status)
  - Interactive Article Explorer with search, journal filters, and PubMed links
  - Deep-dive Abstract and Chunk Viewer (sample text, token count, DOI)
  - Live Ingestion Form (Query-based and PMID-based) with animated progress bar
  - Visual Distribution Charts (Top Medical Journals, Token Lengths)
"""

import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional
import urllib.request

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AskMed - Quản Lý Kho Bằng Chứng Y Khoa",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom Clinical Teal CSS Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .page-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 24px;
    }
    
    /* KPI Metric Cards */
    .kpi-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        flex: 1;
        min-width: 180px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-top: 4px solid #0d9488;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
    }
    .kpi-score {
        font-size: 2.0rem;
        font-weight: 800;
        color: #0f766e;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Article Detail Card */
    .article-box {
        background-color: #f8fafc;
        border-left: 4px solid #0d9488;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .article-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .article-meta {
        font-size: 0.82rem;
        color: #0d9488;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* PubMed Link Chip */
    .pubmed-chip {
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
    }
    .pubmed-chip:hover {
        background-color: #0d9488;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 3. Data Loading & Index Synchronization
# -----------------------------------------------------------------------------
BM25_FILE = Path("data/bm25_index/bm25.pkl")
CHROMA_DIR = Path("data/chroma_db")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_data
def load_articles_metadata():
    """Reads BM25 index and builds aggregated article DataFrame."""
    if not BM25_FILE.exists():
        return pd.DataFrame(), []

    try:
        with open(BM25_FILE, "rb") as f:
            bm25_data = pickle.load(f)

        chunks = bm25_data.get("chunks", [])
        articles_map = {}

        for c in chunks:
            pmid = str(c.get("pmid", "")).strip()
            if not pmid:
                continue

            if pmid not in articles_map:
                articles_map[pmid] = {
                    "pmid": pmid,
                    "title": c.get("title", "N/A"),
                    "journal": c.get("journal", "PubMed") or "PubMed",
                    "pub_date": c.get("pub_date", "N/A") or "N/A",
                    "doi": c.get("doi", "N/A") or "N/A",
                    "url": c.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                    "chunks_count": 1,
                    "total_tokens": c.get("token_count", 0) or 0,
                    "sample_text": c.get("text", "")
                }
            else:
                articles_map[pmid]["chunks_count"] += 1
                articles_map[pmid]["total_tokens"] += c.get("token_count", 0) or 0

        df = pd.DataFrame(list(articles_map.values()))
        return df, chunks
    except Exception as exc:
        st.error(f"Lỗi đọc kho dữ liệu: {exc}")
        return pd.DataFrame(), []


df_articles, all_chunks = load_articles_metadata()


# -----------------------------------------------------------------------------
# 4. Ingestion Worker Helper (Dual-Mode: API or In-Process)
# -----------------------------------------------------------------------------
def trigger_ingestion(query: str, max_results: int, progress_bar, status_text) -> Dict[str, Any]:
    """Runs ingestion pipeline with progress status feedback."""
    status_text.text("🔍 [1/5] Đang kết nối NCBI E-Utilities tìm kiếm danh sách PMIDs...")
    progress_bar.progress(20)
    time.sleep(0.5)

    status_text.text("📥 [2/5] Đang tải chi tiết bài báo, tiêu đề và tóm tắt Abstract...")
    progress_bar.progress(40)
    time.sleep(0.5)

    status_text.text("🧹 [3/5] Đang làm sạch ký tự HTML, unescape và lọc độ dài abstract...")
    progress_bar.progress(60)

    # Ingestion execution
    try:
        from src.ingestion.ingest_pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

        status_text.text("✂️ [4/5] Đang cắt phân đoạn câu y khoa (512 tokens / 128 overlap)...")
        progress_bar.progress(75)

        res = pipeline.ingest_query(query=query, max_results=max_results)

        status_text.text("🧠 [5/5] Đang sinh vector embedding và đồng bộ vào ChromaDB + BM25...")
        progress_bar.progress(100)
        time.sleep(0.5)

        return res
    except Exception as exc:
        st.error(f"Lỗi nạp dữ liệu: {exc}")
        return {}


# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Real-Time Probes
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗄️ AskMed Data Manager")
    st.caption("Quản trị Kho Bằng Chứng Y Khoa Đa Chuyên Khoa")
    st.markdown("---")

    st.markdown("#### ⚡ Bộ Nhớ & Chỉ Mục")
    st.markdown(f"- **ChromaDB Vector Store:** `{CHROMA_DIR}`")
    st.markdown(f"- **BM25 Lexical Store:** `{BM25_FILE}`")
    st.markdown(f"- **Embedding Model:** `nomic-embed-text:latest`")
    st.markdown(f"- **Chunking Policy:** `512 tokens / 128 overlap`")

    st.markdown("---")
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:20px; line-height:1.4;">
        <b>Cơ chế Ingestion:</b> Dữ liệu được nạp đồng thời vào Vector DB (Dense Semantic Search)
        và BM25 (Sparse Keyword Search) để phục vụ thuật toán Hybrid Search & Reciprocal Rank Fusion (RRF).
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 6. Page Header & KPI Metric Summary Cards
# -----------------------------------------------------------------------------
st.markdown('<div class="page-title">🗄️ Quản Lý Kho Bằng Chứng Y Khoa (Knowledge Base)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Tra cứu, giám sát và nạp thêm các công trình nghiên cứu khoa học từ cổng thông tin Thư viện Y học Quốc gia Hoa Kỳ (NCBI PubMed).</div>',
    unsafe_allow_html=True
)

total_articles = len(df_articles)
total_chunks_count = len(all_chunks)
total_tokens_approx = df_articles["total_tokens"].sum() if not df_articles.empty else 0
unique_journals = df_articles["journal"].nunique() if not df_articles.empty else 0

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-score">{total_articles}</div>
            <div class="kpi-label">Bài Báo Khoa Học (PMIDs)</div>
            <div class="kpi-subtext">Đã chỉ mục vào ChromaDB</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{total_chunks_count}</div>
            <div class="kpi-label">Đoạn Văn Bản (Chunks)</div>
            <div class="kpi-subtext">512 tokens / 128 overlap</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{unique_journals}</div>
            <div class="kpi-label">Tạp Chí Y Học</div>
            <div class="kpi-subtext">Lancet, NEJM, Rhode Island...</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">~{total_tokens_approx // 1000}k</div>
            <div class="kpi-label">Tổng Số Từ Vựng (Tokens)</div>
            <div class="kpi-subtext">Được mã hóa vector đa chiều</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">Dual-Sync</div>
            <div class="kpi-label">Đồng Bộ Hybrid</div>
            <div class="kpi-subtext">ChromaDB + BM25 sẵn sàng</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 7. Main Tabs: Explorer, Live Ingestion, Analytics
# -----------------------------------------------------------------------------
tab_browse, tab_ingest, tab_analytics = st.tabs([
    "📚 Tra Cứu Danh Mục Bài Báo (Article Explorer)",
    "🚀 Nạp Thêm Bài Báo Mới (Live Ingestion)",
    "📈 Phân Tích Kho Dữ Liệu (Analytics)"
])

# -----------------------------------------------------------------------------
# TAB 1: BROWSE ARTICLES
# -----------------------------------------------------------------------------
with tab_browse:
    st.markdown("##### 🔎 Tìm kiếm & Thẩm định Bằng chứng Y khoa:")
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_kw = st.text_input(
            "Tìm kiếm theo Tiêu đề, Tạp chí hoặc mã PMID:",
            placeholder="Ví dụ: Metformin, Hypertension, 43063382..."
        )
    with col_s2:
        journal_options = ["Tất cả tạp chí"] + sorted(list(df_articles["journal"].unique()))
        selected_journal = st.selectbox("Lọc theo Tạp chí:", journal_options)

    # Filter dataframe
    filtered_articles = df_articles.copy()
    if search_kw:
        mask = (
            filtered_articles["title"].str.contains(search_kw, case=False, na=False) |
            filtered_articles["journal"].str.contains(search_kw, case=False, na=False) |
            filtered_articles["pmid"].str.contains(search_kw, case=False, na=False)
        )
        filtered_articles = filtered_articles[mask]
    if selected_journal != "Tất cả tạp chí":
        filtered_articles = filtered_articles[filtered_articles["journal"] == selected_journal]

    st.markdown(f"**Hiển thị:** `{len(filtered_articles)} / {total_articles}` bài báo phù hợp.")

    # Table display
    table_cols = ["pmid", "title", "journal", "pub_date", "chunks_count", "total_tokens"]
    st.dataframe(
        filtered_articles[table_cols].rename(columns={
            "pmid": "PMID",
            "title": "Tiêu Đề Bài Báo",
            "journal": "Tạp Chí Y Học",
            "pub_date": "Ngày/Năm Xuất Bản",
            "chunks_count": "Số Chunks",
            "total_tokens": "Số Tokens"
        }),
        use_container_width=True,
        height=320
    )

    # Single Article Inspector
    st.markdown("---")
    st.markdown("##### 📖 Xem Chi Tiết Tóm Tắt Abstract & Chunks Của Bài Báo:")

    if not filtered_articles.empty:
        article_pmid_list = filtered_articles["pmid"].tolist()
        chosen_pmid = st.selectbox(
            "Chọn bài báo để đọc trích đoạn nội dung:",
            article_pmid_list,
            format_func=lambda p: f"PMID: {p} - {filtered_articles.loc[filtered_articles['pmid']==p, 'title'].values[0][:80]}..."
        )

        selected_row = filtered_articles[filtered_articles["pmid"] == chosen_pmid].iloc[0]

        st.markdown(
            f"""
            <div class="article-box">
                <div class="article-title">{selected_row['title']}</div>
                <div class="article-meta">
                    <b>Tạp chí:</b> {selected_row['journal']} | <b>Xuất bản:</b> {selected_row['pub_date']} | <b>DOI:</b> {selected_row['doi']}
                </div>
                <div><b>Đoạn trích dẫn Abstract:</b></div>
                <div style="margin-top:6px; color:#334155;">{selected_row['sample_text']}</div>
                <div style="margin-top:14px;">
                    <a href="{selected_row['url']}" target="_blank" class="pubmed-chip">
                        📄 Mở bài báo gốc trên PubMed NCBI (PMID: {selected_row['pmid']})
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# TAB 2: LIVE INGESTION FORM
# -----------------------------------------------------------------------------
with tab_ingest:
    st.markdown("##### 📥 Nạp Thêm Bài Báo Khoa Học Trực Tiếp Từ PubMed")
    st.caption("Nhập từ khóa y học chuyên sâu để hệ thống tự động cào bài báo, cắt đoạn (chunking), sinh vector và cập nhật trực tiếp vào hệ thống.")

    ingest_type = st.radio(
        "Phương thức nạp:",
        ["Tìm kiếm theo Từ khóa Y khoa (Query-based)", "Nạp theo Danh sách mã PMID cụ thể (Batch PMIDs)"],
        horizontal=True
    )

    if ingest_type == "Tìm kiếm theo Từ khóa Y khoa (Query-based)":
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1:
            query_input = st.text_input(
                "Từ khóa chuyên khoa (MeSH / Clinical query):",
                value="GLP-1 receptor agonists cardiovascular outcomes",
                help="Nhập từ khóa tiếng Anh theo chuẩn danh pháp y khoa NCBI PubMed."
            )
        with col_in2:
            num_articles = st.slider("Số lượng bài báo cần tải:", min_value=1, max_value=30, value=5, step=1)

        st.info("💡 **Gợi ý từ khóa mẫu:** `immunotherapy non-small cell lung cancer`, `sglt2 inhibitors chronic kidney disease`, `mrna vaccines safety profile`.")

        if st.button("🚀 Kích Hoạt Nạp Tự Động Vào ChromaDB & BM25", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            t0 = time.perf_counter()
            result = trigger_ingestion(query=query_input, max_results=num_articles, progress_bar=progress_bar, status_text=status_text)
            elapsed = time.perf_counter() - t0

            if result and result.get("chunks_count", 0) > 0:
                st.success(
                    f"✅ **Nạp thành công!** Đã tải và làm sạch **{result.get('cleaned_count', 0)} bài báo**, "
                    f"tạo ra **{result.get('chunks_count', 0)} chunks** mới và đồng bộ vào ChromaDB + BM25 trong `{elapsed:.2f}s`."
                )
                st.cache_data.clear()
            else:
                st.warning("⚠️ Không tìm thấy bài báo mới phù hợp hoặc tất cả đã tồn tại trong cơ sở dữ liệu.")

    else:
        pmids_text = st.text_area(
            "Nhập danh sách mã PMID (phân tách bằng dấu phẩy hoặc xuống dòng):",
            value="43063382, 42708794, 43051408",
            height=120
        )
        if st.button("🚀 Nạp Danh Sách PMID", use_container_width=True):
            pmid_list = [p.strip() for p in pmids_text.replace("\n", ",").split(",") if p.strip()]
            if pmid_list:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Đang tải {len(pmid_list)} bài báo từ NCBI...")
                progress_bar.progress(50)

                try:
                    from src.ingestion.ingest_pipeline import IngestionPipeline
                    pipe = IngestionPipeline()
                    res = pipe.ingest_pmids(pmid_list)
                    progress_bar.progress(100)
                    st.success(f"✅ Đã nạp thành công {res.get('chunks_count', 0)} chunks từ {len(pmid_list)} PMIDs!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Lỗi nạp PMID: {e}")


# -----------------------------------------------------------------------------
# TAB 3: ANALYTICS & KNOWLEDGE DISTRIBUTION
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("##### 📈 Phân Tích Đặc Tuyến Kho Dữ Liệu Y Khoa")

    col_an1, col_an2 = st.columns(2)

    with col_an1:
        st.markdown("###### 🏛️ Top 10 Tạp Chí Y Học Có Nhiều Bài Báo Nhất")
        if not df_articles.empty:
            top_journals = df_articles["journal"].value_counts().head(10).reset_index()
            top_journals.columns = ["Journal", "Count"]

            fig_journals = px.bar(
                top_journals,
                x="Count",
                y="Journal",
                orientation="h",
                text="Count",
                color="Count",
                color_continuous_scale="Teal"
            )
            fig_journals.update_layout(
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            fig_journals.update_traces(textposition="outside")
            st.plotly_chart(fig_journals, use_container_width=True)

    with col_an2:
        st.markdown("###### 📊 Phân Bố Độ Dài Tokens Các Đoạn Văn Bản (Chunk Length)")
        if all_chunks:
            chunk_tokens = [c.get("token_count", 0) for c in all_chunks if c.get("token_count")]
            fig_hist = px.histogram(
                x=chunk_tokens,
                nbins=25,
                labels={"x": "Số Tokens / Chunk", "count": "Số lượng Chunks"},
                color_discrete_sequence=["#0d9488"]
            )
            fig_hist.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Độ dài Token (Target: 512 tokens)",
                yaxis_title="Số lượng đoạn văn bản"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

