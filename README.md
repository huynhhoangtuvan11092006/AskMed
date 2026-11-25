# Ask_Med: Hệ Thống Hỏi - Đáp Y Học Chứng Cứ Dựa Trên PubMed

Hệ thống Hỏi - Đáp Y Khoa theo chuẩn Y học Chứng cứ (Evidence-Based Medicine). Mô hình kết hợp truy xuất đa tầng (ChromaDB + BM25 + Cross-Encoder Reranker) và mô hình ngôn ngữ `llama3.1:8b` chạy cục bộ trên Ollama, trả lời dựa trên kho dữ liệu 620 bài báo khoa học từ PubMed NCBI và tự động gắn trích dẫn `[PMID: ...]`.

---

## 1. Kiến Trúc Hệ Thống

```text
Câu hỏi lâm sàng
      │
      ▼
+---------------------+     +--------------------+
|  ChromaDB (Vector)  |     |   BM25 (Từ khóa)   |
|  nomic-embed-text   |     |     rank_bm25      |
+----------+----------+     +---------+----------+
           │                          │
           +------------+-------------+
                        │
                        ▼
           +--------------------------+
           |  Reciprocal Rank Fusion  |
           |     (RRF Fusion k=60)    |
           +------------+-------------+
                        │
                        ▼
           +--------------------------+
           |   Cross-Encoder Rerank   |
           | ms-marco-MiniLM-L-6-v2   |
           +------------+-------------+
                        │
                        ▼
           +--------------------------+
           |     Tổng hợp câu trả lời |
           |   llama3.1:8b (Ollama)   |
           +------------+-------------+
                        │
                        ▼
           Câu trả lời trích dẫn [PMID: xxxxx]
```

---

## 2. Các Tính Năng Cốt Lõi

* **Tìm kiếm kết hợp (Hybrid Search):** Kết hợp tìm kiếm ngữ nghĩa ChromaDB (`nomic-embed-text`) và tìm kiếm từ khóa chính xác BM25 qua thuật toán Reciprocal Rank Fusion (RRF, $k=60$).
* **Thẩm định tương quan (Cross-Encoder Reranking):** Mô hình `ms-marco-MiniLM-L-6-v2` chấm điểm tương quan từng cặp câu hỏi - tài liệu, giữ lại 5 đoạn văn bản phù hợp nhất.
* **Trích dẫn minh bạch (PMID Citations):** Mọi tuyên bố y tế đều được liên kết trực tiếp với bài báo gốc trên PubMed (`https://pubmed.ncbi.nlm.nih.gov/<pmid>/`).
* **Cơ chế chống ảo giác (Strict Grounding Guard):** Hệ thống chỉ trả lời dựa trên tài liệu được cung cấp, tự động từ chối nếu không có bằng chứng khoa học trong kho dữ liệu.
* **Xử lý thời gian thực (FastAPI & Streaming):** Backend hỗ trợ Server-Sent Events (SSE) giúp phản hồi từng từ mượt mà trên giao diện người dùng.
* **Giao diện trực quan (Streamlit UI):** Khung chat lâm sàng, bảng phân tích hiệu năng RAGAs và trang quản trị kho dữ liệu 620 bài báo.
* **Vận hành cục bộ (Local & Private):** Chạy hoàn toàn trên máy chủ nội bộ qua Ollama, bảo mật dữ liệu y tế.

---

## 3. Các Kịch Bản Kiểm Thử

![Kết quả thực nghiệm](docs/images/demo_results.png)

### Kịch bản 1: Tra cứu cơ chế tác động dược lý (Pharmacology Mechanism)
* **Tình huống:** Bác sĩ cần tra cứu cơ chế phân tử của Metformin trong ức chế tân tạo glucose ở gan.
* **Diễn biến:**
  * Lượt 1: Truy xuất tài liệu từ ChromaDB và BM25, Cross-Encoder chọn lọc đoạn văn bản về chuỗi hô hấp ty thể và enzym AMPK.
  * Lượt 2: Hệ thống phân tích và giải thích: Metformin ức chế phức hợp I ty thể -> tăng tỷ lệ AMP/ATP -> hoạt hóa AMPK -> ức chế enzym tân tạo glucose (gluconeogenesis).
* **Kết quả:** Trích dẫn thành công bài báo `[PMID: 43063382]` và `[PMID: 42976769]`.

### Kịch bản 2: Cảnh báo an toàn & Chống chỉ định suy thận (Safety & Contraindications)
* **Tình huống:** Bệnh nhân suy thận mạn với mức lọc cầu thận giảm sâu (`eGFR < 30 mL/phút`).
* **Diễn biến:**
  * Lượt 1: Hệ thống nhận diện thông số định lượng lâm sàng eGFR < 30 mL/phút từ câu hỏi.
  * Lượt 2: Đưa ra cảnh báo chống chỉ định tuyệt đối do thuốc thải trừ qua thận; biến chứng nghiêm trọng nhất là Nhiễm toan Lactic (MALA - Metformin-Associated Lactic Acidosis).
* **Kết quả:** Khuyến cáo ngừng thuốc và trích dẫn bài báo ca bệnh thực tế `[PMID: 43063382]`.

### Kịch bản 3: Xử lý bế tắc & Chống ảo giác với câu hỏi bẫy (Anti-Hallucination Stress-Test)
* **Tình huống:** Nhập câu hỏi bẫy về phác đồ chưa được kiểm chứng khoa học (dùng kết hợp liều cao Hydroxychloroquine và Ivermectin chữa ung thư tụy giai đoạn 4).
* **Diễn biến:**
  * Lượt 1: Hệ thống đối chiếu toàn bộ kho 620 bài báo, xác định không có dữ liệu chứng thực.
  * Lượt 2: Thay vì tự suy đoán hoặc bịa đặt liều dùng, hệ thống từ chối xác nhận: *"Không tìm thấy bằng chứng khoa học nào trong kho dữ liệu PubMed NCBI chứng minh phác đồ này có hiệu quả."*
* **Kết quả:** Đèn trạng thái chuyển sang cảnh báo vàng (Ungrounded/Refused), ngăn chặn phát tán thông tin sai lệch.

---

## 4. Hướng Dẫn Cài Đặt và Chạy Ứng Dụng

### Cách 1: Khởi động bằng Docker Compose (Khuyên dùng)

Yêu cầu máy tính đã cài đặt Docker Desktop.

```bash
# Khởi động 3 dịch vụ (Ollama, Backend API, Streamlit UI)
docker compose up -d --build

# Hoặc kích hoạt GPU NVIDIA:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

* Giao diện Web: http://localhost:8501
* Tài liệu API Swagger: http://localhost:8000/docs
* Kiểm tra sức khỏe hệ thống: http://localhost:8000/health

Dừng hệ thống:
```bash
docker compose down
```

### Cách 2: Chạy Cục Bộ (Local Python)

```bash
# 1. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 2. Khởi động Backend và Frontend
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
python -m streamlit run src/ui/app.py
```

Chạy kiểm thử tự động toàn bộ kịch bản:
```bash
pytest -v
python scripts/demo_runner.py
```

---

## 5. Cấu Trúc Dự Án

```plaintext
Ask_Med/
├── configs/
│   └── config.yaml              # Cấu hình trung tâm: Models, Paths, Ingestion, RRF
├── data/
│   ├── bm25_index/bm25.pkl      # Chỉ mục từ khóa BM25 (703 chunks)
│   ├── chroma_db/               # Cơ sở dữ liệu vector ChromaDB (703 items)
│   └── eval/
│       ├── eval_dataset.json    # Bộ 25 câu hỏi kiểm thử chuẩn
│       └── results/             # Báo cáo đánh giá RAGAs (CSV, JSON)
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI REST & SSE Streaming Endpoints
│   │   └── schemas.py           # Pydantic v2 Schemas (Data Contracts)
│   ├── generation/
│   │   ├── citation_parser.py   # Bóc tách và kiểm thực mã PMID
│   │   ├── llm_client.py        # Client Ollama LLM
│   │   ├── prompts.py           # Clinical Prompts chống ảo giác
│   │   └── rag_pipeline.py      # Bộ điều phối tổng thể AskMedPipeline
│   ├── ingestion/
│   │   ├── chunker.py           # Cắt đoạn câu (512 tokens / 128 overlap)
│   │   ├── cleaner.py           # Làm sạch và lọc chất lượng Abstract
│   │   ├── ingest_pipeline.py   # Pipeline cào và lập chỉ mục PubMed
│   │   └── pubmed_fetcher.py    # Giao tiếp API NCBI E-Utilities
│   ├── retrieval/
│   │   ├── bm25_search.py       # Tìm kiếm từ khóa BM25
│   │   ├── hybrid_retriever.py  # Thuật toán Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py          # Cross-Encoder Reranker
│   │   └── vector_search.py     # Tìm kiếm ngữ nghĩa ChromaVectorStore
│   └── ui/
│       ├── app.py               # Trang chủ Streamlit & Tổng quan
│       └── pages/
│           ├── 1_Chat.py        # Giao diện Chat lâm sàng & Passage Inspector
│           ├── 2_Evaluation.py  # Dashboard phân tích RAGAs & Radar Chart
│           └── 3_Data_Manage.py # Quản lý kho 620 bài báo & Nạp tự động
├── docs/images/                 # Hình ảnh minh họa kịch bản demo và phân tích
├── scripts/
│   ├── demo_runner.py           # Diễn tập tự động các kịch bản lâm sàng
│   └── manage.ps1               # Script tiện ích quản trị Windows PowerShell
├── tests/                       # 37+ Unit tests & Integration tests
├── Dockerfile                   # Image Backend FastAPI
├── Dockerfile.ui                # Image Frontend Streamlit
├── docker-compose.yml           # Điều phối 3 containers (Ollama, API, UI)
├── Makefile                     # Quản lý 12 lệnh vận hành tự động
├── requirements.txt             # Danh mục thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn dự án
```
