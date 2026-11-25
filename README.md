# 🩺 Ask_Med: Evidence-Based Medical Question Answering (MedRAG)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.164.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![RAGAs Benchmark](https://img.shields.io/badge/RAGAs_Composite-91.56%25-brightgreen.svg?style=flat)](data/eval/results/benchmark_report.csv)
[![Grounding Rate](https://img.shields.io/badge/Grounding_Rate-100%25-success.svg?style=flat)](data/eval/results/benchmark_summary.json)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker&logoColor=white)](docker-compose.yml)

> **Ask_Med** là hệ thống Hỏi - Đáp Y Khoa theo chuẩn **Y học Chứng cứ (Evidence-Based Medicine)**. Hệ thống kết hợp RAG đa tầng (ChromaDB + BM25 + Cross-Encoder Reranker) cùng mô hình Local LLM (`llama3.1:8b`), ép buộc câu trả lời bám sát 620+ bài báo khoa học **NCBI PubMed** và trích dẫn mã `[PMID: ...]` có thể nhấp chuột trực tiếp, triệt tiêu ảo giác (Hallucination) trong y tế.

---

## ⚡ Khởi Động Nhanh 1-Click (Docker Compose)

Khởi chạy đồng thời 3 dịch vụ (**Ollama**, **FastAPI Backend**, **Streamlit Frontend**):

```bash
# 1. Khởi động hệ thống
docker compose up -d --build

# (Tùy chọn) Kích hoạt GPU NVIDIA:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

**Địa chỉ truy cập:**
* 🩺 **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
* 📡 **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* 🩺 **Health Probe:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🏗️ Kiến Trúc Hệ Thống (Pipeline)

```
[Câu hỏi lâm sàng]
       │
       ▼
[1. Hybrid Search] ──► ChromaDB (Semantic) + BM25 (Từ khóa) ──► 60 Chunks
       │
       ▼
[2. RRF Fusion]    ──► Reciprocal Rank Fusion (k=60)        ──► 20 Chunks
       │
       ▼
[3. Reranker]      ──► Cross-Encoder (ms-marco-MiniLM-L6)   ──► Top 5 Chunks
       │
       ▼
[4. Local LLM]     ──► Ollama (llama3.1:8b, temp=0.0)       ──► Trả lời nghiêm ngặt
       │
       ▼
[5. Trích dẫn]     ──► Bóc tách [PMID: ...], Audit Grounding ──► Link NCBI PubMed
```

---

## 📊 Kết Quả Đánh Giá RAGAs (25 Câu Hỏi Chuẩn)

| Chỉ Số | Mục Tiêu | AskMed Đạt | Đánh Giá |
| :--- | :---: | :---: | :---: |
| **Context Recall** | $\ge 80.0\%$ | **96.33%** | 🟢 Bao phủ đầy đủ bằng chứng y khoa |
| **Answer Relevancy** | $\ge 85.0\%$ | **96.10%** | 🟢 Trả lời trúng trọng tâm câu hỏi |
| **RAGAs Composite** | $\ge 80.0\%$ | **91.56%** | 🏆 Điểm chất lượng tổng thể xuất sắc |
| **Faithfulness** | $\ge 85.0\%$ | **88.32%** | 🟢 Trung thực, chống bịa đặt |
| **Context Precision** | $\ge 75.0\%$ | **85.48%** | 🟢 Đoạn sau Rerank có độ liên quan cao |
| **Grounding Rate** | $100.0\%$ | **100.0% (25/25)** | 🟢 100% câu trả lời có trích dẫn PubMed |

---

## 💻 Chạy Cục Bộ (Local Python)

```bash
# 1. Kích hoạt môi trường và cài đặt dependencies
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Chạy Backend & Frontend
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
python -m streamlit run src/ui/app.py

# Hoặc qua lệnh tắt Makefile / PowerShell:
make api    # hoặc .\scripts\manage.ps1 api
make ui     # hoặc .\scripts\manage.ps1 ui
```

---

## ⚖️ Tuyên Bố Y Tế (Medical Disclaimer)

> **Lưu ý:** Ask_Med phục vụ mục đích **học thuật và nghiên cứu Y học Chứng cứ**. Hệ thống **KHÔNG** thay thế chẩn đoán chuyên môn hay chỉ định điều trị của bác sĩ chuyên khoa.
