"""
src/ui/pages/2_Evaluation.py
----------------------------
RAGAs Benchmark Evaluation & Quality Dashboard for AskMed.
Features:
  - KPI Dashboard displaying 4 core RAGAs metrics + Grounding Rate & Latency
  - Interactive Radar Chart (Spider Chart) comparing RAG metrics against clinical targets
  - Category & Topic Breakdown (Grouped Bar Chart across medical specialties)
  - Interactive Single-Query Inspector (Question, Generated Answer, Ground Truth, PMIDs, Scores)
  - Full Benchmark Data Table with CSV export capabilities
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AskMed - Đánh Giá Chất Lượng RAGAs",
    page_icon="📊",
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
        min-width: 170px;
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

    /* Comparison Box */
    .qa-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .qa-box-title {
        font-weight: 700;
        margin-bottom: 6px;
        color: #0f766e;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Metric Score Pill */
    .score-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-right: 8px;
    }
    .score-high {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #10b981;
    }
    .score-med {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #f59e0b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 3. Data Loading & Preprocessing
# -----------------------------------------------------------------------------
DATA_DIR = Path("data/eval/results")
CSV_PATH = DATA_DIR / "benchmark_report.csv"
SUMMARY_PATH = DATA_DIR / "benchmark_summary.json"


@st.cache_data
def load_benchmark_data():
    """Loads CSV and Summary JSON files with caching."""
    if not CSV_PATH.exists():
        st.error(f"Không tìm thấy file kết quả đánh giá tại `{CSV_PATH}`.")
        return None, None
    df = pd.read_csv(CSV_PATH)
    summary = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return df, summary


df_benchmark, summary_data = load_benchmark_data()

if df_benchmark is None or df_benchmark.empty:
    st.warning("Chưa có dữ liệu đánh giá RAGAs. Hãy chạy pipeline đánh giá trước.")
    st.stop()


# -----------------------------------------------------------------------------
# 4. Sidebar Filters & Summary Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Bộ Lọc Đánh Giá")
    
    # Topic filter
    all_topics = ["Tất cả"] + sorted(list(df_benchmark["topic"].dropna().unique()))
    selected_topic = st.selectbox("Chọn Chuyên Khoa (Topic):", all_topics, index=0)

    # Category filter
    all_cats = ["Tất cả"] + sorted(list(df_benchmark["category"].dropna().unique()))
    selected_cat = st.selectbox("Chọn Phân Loại Y Học (Category):", all_cats, index=0)

    st.markdown("---")
    st.markdown("#### 🎯 Mục Tiêu Chuẩn (Targets)")
    st.markdown(
        """
        - **Context Precision:** $\ge 75\%$
        - **Context Recall:** $\ge 80\%$
        - **Faithfulness:** $\ge 85\%$
        - **Answer Relevancy:** $\ge 85\%$
        - **Grounding Rate:** $100\%$
        """
    )
    st.markdown("---")
    st.info("💡 **Gợi ý:** Sử dụng trang này để kiểm tra độ tin cậy và chứng thực không ảo giác y khoa trước Hội đồng bảo vệ.")


# Apply filters
filtered_df = df_benchmark.copy()
if selected_topic != "Tất cả":
    filtered_df = filtered_df[filtered_df["topic"] == selected_topic]
if selected_cat != "Tất cả":
    filtered_df = filtered_df[filtered_df["category"] == selected_cat]


# -----------------------------------------------------------------------------
# 5. Header & KPI Metric Summary Cards
# -----------------------------------------------------------------------------
st.markdown('<div class="page-title">📊 Báo Cáo Đánh Giá Chất Lượng RAGAs</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">Đánh giá thực nghiệm toàn diện trên bộ dữ liệu <b>{len(df_benchmark)} câu hỏi chuẩn y khoa</b> '
    '(Tiểu đường, Tăng huyết áp, Ung bướu, COVID-19) kết hợp truy xuất kho 620 bài báo NCBI PubMed.</div>',
    unsafe_allow_html=True
)

# Compute metrics
avg_precision = filtered_df["context_precision"].mean() * 100
avg_recall = filtered_df["context_recall"].mean() * 100
avg_faithfulness = filtered_df["faithfulness"].mean() * 100
avg_relevancy = filtered_df["answer_relevancy"].mean() * 100
avg_composite = filtered_df["ragas_composite"].mean() * 100
grounding_rate = (filtered_df["is_grounded"].astype(bool).sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 100

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-score">{avg_composite:.2f}%</div>
            <div class="kpi-label">RAGAs Composite</div>
            <div class="kpi-subtext">Điểm chất lượng tổng hợp</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{avg_precision:.2f}%</div>
            <div class="kpi-label">Context Precision</div>
            <div class="kpi-subtext">Độ chính xác ngữ cảnh (&ge;75%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{avg_recall:.2f}%</div>
            <div class="kpi-label">Context Recall</div>
            <div class="kpi-subtext">Độ bao phủ bằng chứng (&ge;80%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{avg_faithfulness:.2f}%</div>
            <div class="kpi-label">Faithfulness</div>
            <div class="kpi-subtext">Độ trung thực, chống bịa đặt (&ge;85%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{avg_relevancy:.2f}%</div>
            <div class="kpi-label">Answer Relevancy</div>
            <div class="kpi-subtext">Độ liên quan câu trả lời (&ge;85%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-score">{grounding_rate:.1f}%</div>
            <div class="kpi-label">Grounding Rate</div>
            <div class="kpi-subtext">Tỷ lệ có trích dẫn NCBI ({len(filtered_df)} câu)</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 6. Interactive Visualizations (Tabs)
# -----------------------------------------------------------------------------
tab_radar, tab_topics, tab_dist = st.tabs([
    "🕸️ Biểu đồ Radar (Spider Chart)",
    "📊 Phân tích theo Chuyên Khoa (Topic Analysis)",
    "📈 Phân bố Điểm số (Score Distributions)"
])

with tab_radar:
    st.markdown("##### 🎯 Đối Sánh 4 Chiều RAGAs Với Ngưỡng Chuẩn Đề Xuất")
    
    categories_radar = [
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevancy",
        "RAGAs Composite"
    ]
    actual_values = [
        filtered_df["context_precision"].mean(),
        filtered_df["context_recall"].mean(),
        filtered_df["faithfulness"].mean(),
        filtered_df["answer_relevancy"].mean(),
        filtered_df["ragas_composite"].mean()
    ]
    target_values = [0.75, 0.80, 0.85, 0.85, 0.80]

    # Close the radar loop
    radar_cats = categories_radar + [categories_radar[0]]
    actual_vals = actual_values + [actual_values[0]]
    target_vals = target_values + [target_values[0]]

    fig_radar = go.Figure()

    # Actual AskMed performance
    fig_radar.add_trace(go.Scatterpolar(
        r=actual_vals,
        theta=radar_cats,
        fill="toself",
        name="AskMed RAG Pipeline",
        line=dict(color="#0d9488", width=3),
        fillcolor="rgba(13, 148, 136, 0.25)"
    ))

    # Target Benchmark line
    fig_radar.add_trace(go.Scatterpolar(
        r=target_vals,
        theta=radar_cats,
        name="Ngưỡng Chuẩn Y Khoa (Target)",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        fill=None
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.0, 1.0],
                tickformat=".0%",
                gridcolor="#e2e8f0"
            ),
            angularaxis=dict(
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1"
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=40),
        height=480
    )

    st.plotly_chart(fig_radar, use_container_width=True)


with tab_topics:
    st.markdown("##### 🩺 So Sánh Chất Lượng Giữa Các Chuyên Khoa Y Tế")
    topic_summary = df_benchmark.groupby("topic")[
        ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "ragas_composite"]
    ].mean().reset_index()

    topic_summary_melted = topic_summary.melt(
        id_vars=["topic"],
        var_name="Metric",
        value_name="Score"
    )

    metric_name_map = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "ragas_composite": "RAGAs Composite"
    }
    topic_summary_melted["Metric"] = topic_summary_melted["Metric"].map(metric_name_map)

    fig_bar = px.bar(
        topic_summary_melted,
        x="topic",
        y="Score",
        color="Metric",
        barmode="group",
        text=topic_summary_melted["Score"].apply(lambda v: f"{v*100:.1f}%"),
        labels={"topic": "Chuyên Khoa (Topic)", "Score": "Điểm số"},
        color_discrete_sequence=["#0d9488", "#028090", "#00A896", "#3b82f6", "#6366f1"]
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_bar.update_traces(textposition="outside", textfont_size=10)
    st.plotly_chart(fig_bar, use_container_width=True)


with tab_dist:
    st.markdown("##### 📈 Phân Bố Điểm Số 25 Câu Hỏi Kiểm Thử")
    dist_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    fig_box = px.box(
        df_benchmark[dist_cols].rename(columns={
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall"
        }),
        points="all",
        color_discrete_sequence=["#0d9488"]
    )
    fig_box.update_layout(
        yaxis=dict(range=[0, 1.05], tickformat=".0%", title="Điểm số"),
        xaxis_title="Chỉ số RAGAs",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_box, use_container_width=True)


st.markdown("<hr style='margin: 28px 0 24px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Single Test Query Deep-Dive Inspector
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Kiểm Tra Chi Tiết Từng Câu Hỏi Kiểm Thử")
st.markdown("Chọn một câu hỏi trong bộ kiểm thử để đối chiếu giữa câu hỏi, đáp án chuẩn, câu trả lời sinh bởi RAG và các chỉ số:")

query_options = [
    f"[{row['id']}] ({row['topic']}) {row['question'][:90]}..."
    for _, row in filtered_df.iterrows()
]

if query_options:
    selected_idx = st.selectbox("Chọn câu hỏi cần thẩm định:", range(len(query_options)), format_func=lambda i: query_options[i])
    selected_item = filtered_df.iloc[selected_idx]

    # Metrics score pills
    f_score = selected_item["faithfulness"]
    r_score = selected_item["answer_relevancy"]
    p_score = selected_item["context_precision"]
    rec_score = selected_item["context_recall"]
    comp_score = selected_item["ragas_composite"]

    st.markdown(
        f"""
        <div style="margin: 12px 0 16px 0;">
            <span class="score-pill score-high">🏆 Composite: {comp_score*100:.1f}%</span>
            <span class="score-pill {'score-high' if p_score >= 0.75 else 'score-med'}">Precision: {p_score*100:.1f}%</span>
            <span class="score-pill {'score-high' if rec_score >= 0.8 else 'score-med'}">Recall: {rec_score*100:.1f}%</span>
            <span class="score-pill {'score-high' if f_score >= 0.85 else 'score-med'}">Faithfulness: {f_score*100:.1f}%</span>
            <span class="score-pill {'score-high' if r_score >= 0.85 else 'score-med'}">Relevancy: {r_score*100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown(
            f"""
            <div class="qa-box">
                <div class="qa-box-title">❓ Câu hỏi lâm sàng ({selected_item['id']})</div>
                <div><b>Chuyên khoa:</b> <code>{selected_item['topic']}</code> | <b>Phân loại:</b> <code>{selected_item['category']}</code></div>
                <div style="margin-top:8px; font-weight:600; color:#0f172a;">{selected_item['question']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="qa-box" style="border-left: 4px solid #10b981;">
                <div class="qa-box-title" style="color: #047857;">🎯 Đáp án chuẩn chuyên gia (Ground Truth)</div>
                <div>{selected_item['ground_truth']}</div>
                <div style="margin-top:8px; font-size:0.8rem; color:#64748b;">
                    <b>PMID kỳ vọng:</b> <code>{selected_item['expected_pmids']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_q2:
        st.markdown(
            f"""
            <div class="qa-box" style="border-left: 4px solid #0d9488;">
                <div class="qa-box-title">🤖 Câu trả lời sinh bởi AskMed RAG</div>
                <div>{selected_item['generated_answer']}</div>
                <div style="margin-top:8px; font-size:0.8rem; color:#64748b;">
                    <b>PMID trích xuất:</b> <code>{selected_item['retrieved_pmids']}</code> | <b>Thời gian:</b> {selected_item['latency_total_sec']:.2f}s
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("Không có câu hỏi nào khớp với bộ lọc hiện tại.")


st.markdown("<hr style='margin: 28px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Complete Benchmark Data Table & Export
# -----------------------------------------------------------------------------
st.markdown("### 📋 Bảng Dữ Liệu Kiểm Thử Toàn Bộ (25 Câu Hỏi)")

display_cols = [
    "id", "topic", "category", "question",
    "faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_composite",
    "is_grounded", "latency_total_sec"
]

st.dataframe(
    filtered_df[display_cols].style.format({
        "faithfulness": "{:.2%}",
        "answer_relevancy": "{:.2%}",
        "context_precision": "{:.2%}",
        "context_recall": "{:.2%}",
        "ragas_composite": "{:.2%}",
        "latency_total_sec": "{:.2f}s"
    }),
    use_container_width=True,
    height=320
)

# Export CSV button
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Tải xuống Báo cáo Đánh giá (CSV)",
    data=csv_data,
    file_name="askmed_ragas_benchmark_report.csv",
    mime="text/csv",
    help="Tải toàn bộ kết quả benchmark để đưa vào phụ lục báo cáo đồ án."
)
