# Ask_Med: Production-Grade Medical Literature RAG System

Evidence-based medical Q&A grounded in PubMed scientific papers. Hybrid retrieval (vector + BM25), cross-encoder reranking, RAGAs evaluation, and local inference via Ollama.

> **Disclaimer:** This system is for educational and research purposes only. It does NOT constitute medical advice.

---

## Architecture

```text
User Question
      │
      ▼
+---------------------+     +--------------------+
|  ChromaDB (vector)  |     |   BM25 (keyword)   |
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
           |      LLM Generation      |
           |   llama3.1:8b (Ollama)   |
           +------------+-------------+
                        │
                        ▼
           Cited Answer [PMID: xxxxx]
```

---

## Features

* **Hybrid Search** -- vector similarity (ChromaDB `nomic-embed-text`) + BM25 keyword matching fused with Reciprocal Rank Fusion ($k=60$).
* **Cross-Encoder Reranking** -- `ms-marco-MiniLM-L-6-v2` reranks top candidates for high clinical precision.
* **PMID Citations** -- every claim in the answer links back to a specific PubMed paper (`https://pubmed.ncbi.nlm.nih.gov/<pmid>/`).
* **Strict Grounding Guard** -- system refuses unverified claims when evidence is absent in the knowledge base, preventing medical hallucination.
* **RAGAs Evaluation** -- faithfulness, answer relevancy, context precision, and context recall evaluated on 25 clinical test cases.
* **Custom Retrieval Metrics** -- hit rate, MRR, average rerank score, and grounding verification audit.
* **FastAPI Backend** -- `/api/v1/query` (REST and SSE streaming), `/api/v1/health`, `/api/v1/ingest`, `/api/v1/paper/{pmid}`, `/api/v1/stats`.
* **Streamlit UI** -- multi-page clinical application with chat interface, evaluation dashboard, and data management.
* **Fully Local & Private** -- runs entirely on local hardware via Ollama (no patient-adjacent data leaves your machine).
* **Docker Containerized** -- complete 3-tier microservice architecture with GPU passthrough support.

---

## Quick Start

### Prerequisites

| Requirement | Purpose |
| :--- | :--- |
| **Python 3.11+** | Core runtime environment |
| **[Ollama](https://ollama.ai/)** | Local LLM (`llama3.1:8b`) & Embeddings (`nomic-embed-text`) |
| **Docker & Compose** | Containerized deployment (optional) |
| **Git** | Source code management |

### 1. Clone and Install

```bash
git clone https://github.com/huynhhoangtuvan11092006/AskMed.git
cd AskMed

# Option A: using Makefile
make install

# Option B: manual virtualenv setup
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull Ollama Models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Start the Application

```bash
# Terminal 1: FastAPI Backend
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Streamlit Frontend
python -m streamlit run src/ui/app.py
```

* **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

### 4. Query via API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the molecular mechanism of action of metformin?", "top_k": 3}'
```

---

## Screenshots

| Page | Description |
| :--- | :--- |
| **Query (`1_Chat.py`)** | Clinical chat interface with cited answers, clickable PubMed badges, and expandable passage cards. |
| **Evaluation (`2_Evaluation.py`)** | Interactive RAGAs benchmark dashboard with radar chart, topic breakdowns, and per-question inspector. |
| **Data Management (`3_Data_Manage.py`)** | Article catalog (620 PMIDs), chunk token distribution, and live NCBI PubMed ingestion controls. |
| **Overview (`app.py`)** | Multi-stage pipeline architecture visualizer, subsystem status probes, and legal guidance. |

![Evaluation Dashboard](docs/images/demo_results.png)

---

## Evaluation Results

Evaluated across **25 medical benchmark test cases** (Diabetes, Hypertension, Oncology, COVID-19) stored in [data/eval/results/benchmark_report.csv](data/eval/results/benchmark_report.csv):

| Metric | Target | Ask_Med Score | Status |
| :--- | :---: | :---: | :---: |
| **Context Recall** | $\ge 80.0\%$ | **96.33%** (0.96) | Excellent coverage |
| **Answer Relevancy** | $\ge 85.0\%$ | **96.10%** (0.96) | High clinical relevance |
| **RAGAs Composite** | $\ge 80.0\%$ | **91.56%** (0.92) | Outstanding overall quality |
| **Faithfulness** | $\ge 85.0\%$ | **88.32%** (0.88) | Factually grounded |
| **Context Precision** | $\ge 75.0\%$ | **85.48%** (0.85) | Exact match with CV target |
| **Grounding Rate** | $100.0\%$ | **100.0% (25/25)** | Full PubMed citation linkage |
| **Hit Rate@5** | -- | **60.00%** | Multi-passage retrieval |
| **MRR** | -- | **0.4500** | Mean Reciprocal Rank |
| **Avg Rerank Score** | -- | **-0.0155** | Cross-encoder logits |

Run the evaluation suite:
```bash
python -m src.evaluation.eval_pipeline --benchmark data/eval/eval_dataset.json
# or via Make:
make eval
```

Results are saved to `data/eval/results/benchmark_report.csv` and viewable in the Streamlit Evaluation Dashboard.

---

## Tech Stack

| Layer | Technology | Why |
| :--- | :--- | :--- |
| **LLM (primary)** | Ollama `llama3.1:8b` | Runs locally, zero API cost, medical data stays private |
| **Embeddings** | Ollama `nomic-embed-text` (768d) | High-quality open embeddings, fast CPU/GPU inference |
| **Vector Store** | ChromaDB (persistent) | Embedded, persistent SQLite storage, native HNSW indexing |
| **Sparse Retrieval** | `rank_bm25` | Lightweight BM25 catching exact lexical terms that embeddings miss |
| **Fusion** | Reciprocal Rank Fusion ($k=60$) | Parameter-free merging of dense and sparse ranked lists |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight (~80 MB) model, joint cross-attention, fast on CPU |
| **Chunking** | Sentence-aware Token Chunker | 512 tokens / 128 overlap, respects clinical sentence boundaries |
| **Data Source** | PubMed via Biopython Entrez | Authoritative biomedical literature, official NCBI API |
| **API** | FastAPI + Uvicorn | Async, auto Swagger docs, Pydantic v2 validation, SSE streaming |
| **UI** | Streamlit | Rapid clinical prototyping, interactive Plotly charts, session state |
| **Evaluation** | RAGAs + Custom Metrics | Established RAG framework (Precision, Recall, Faithfulness, Relevancy) |
| **Config** | YAML + Pydantic | Strongly typed configuration, environment variable overrides |

---

## Design Decisions

### Why hybrid search (vector + BM25)?
Vector search excels at semantic similarity (*"heart attack"* matches *"myocardial infarction"*) but can miss exact terms that matter in medicine -- drug names (Metformin, Paxlovid, Osimertinib), gene symbols (EGFR, T790M, C797S), and quantitative lab thresholds (`eGFR < 30`). BM25 catches these exact lexical matches. Combining both with Reciprocal Rank Fusion consistently outperforms either retriever alone on medical benchmarks without requiring weight tuning (RRF is parameter-free at $k=60$).

### Why cross-encoder reranking?
Bi-encoder retrieval (ChromaDB) is fast but approximate -- it computes query and document embeddings independently. A cross-encoder jointly attends to the query-document pair, giving significantly more accurate relevance scores. We retrieve a broad set (top-60 hybrid candidates, RRF down to top-20) and rerank down to top-5, getting the precision of a cross-encoder with the speed of bi-encoder retrieval. `ms-marco-MiniLM-L-6-v2` was chosen because it is only ~80 MB, runs on CPU in <100 ms per query, and performs well on passage reranking.

### Why 512-token chunks with 128-token overlap?
PubMed abstracts are typically 200-400 words. A 512-token chunk comfortably fits an entire abstract or a substantial section. The 128-token overlap ensures that sentences split across chunk boundaries are captured in at least one chunk, preventing information loss at boundaries. This is especially important for medical text where a clinical finding and its statistical significance may span a chunk boundary.

### Why Ollama-first?
Medical literature and clinical queries can be sensitive. Running the LLM locally via Ollama means no user queries ever leave the user's machine or local intranet. All inferences execute on local hardware, keeping everything private and offline-capable.

### Why dict-based pipeline (not LangChain Documents)?
The retrieval pipeline uses plain `list[dict]` with a text key throughout, rather than LangChain Document objects. This gives full control over metadata fields (PMID, title, journal, year, rerank scores, RRF scores) without fighting framework abstractions. It also simplifies serialization, testing (dicts are easy to construct in fixtures), and debugging.

---

## Configuration

All settings are configured in [configs/config.yaml](configs/config.yaml). Settings can be overridden via environment variables.

| Setting | Default | Description |
| :--- | :--- | :--- |
| `models.ollama.llm_model` | `llama3.1:8b` | Primary Ollama generative model |
| `models.ollama.embed_model` | `nomic-embed-text` | Ollama embedding model (768 dimensions) |
| `models.ollama.base_url` | `http://127.0.0.1:11434` | Ollama daemon endpoint (overridable via `OLLAMA_HOST`) |
| `models.ollama.temperature` | `0.0` | Deterministic sampling for clinical facts |
| `models.reranker.model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `models.reranker.top_n` | `5` | Final evidence chunks passed to LLM |
| `ingestion.chunk_size` | `512` | Token chunk size |
| `ingestion.chunk_overlap` | `128` | Overlap between consecutive chunks |
| `ingestion.min_abstract_words` | `30` | Minimum abstract word count filter |
| `paths.chroma_db_dir` | `data/chroma_db` | Persistent ChromaDB storage path |
| `paths.bm25_index_file` | `data/bm25_index/bm25.pkl` | Sparse BM25 serialized index file |

---

## Project Structure

```plaintext
Ask_Med/
├── configs/
│   └── config.yaml              # Central system configuration
├── data/
│   ├── bm25_index/bm25.pkl      # Sparse BM25 keyword index (703 chunks)
│   ├── chroma_db/               # ChromaDB persistent vector database (703 items)
│   └── eval/
│       ├── eval_dataset.json    # 25 medical benchmark test cases
│       └── results/
│           ├── benchmark_report.csv      # Detailed benchmark report
│           └── benchmark_summary.json    # Aggregated benchmark metrics
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI application with REST & SSE endpoints
│   │   └── schemas.py           # Pydantic v2 data models
│   ├── evaluation/
│   │   ├── eval_pipeline.py     # End-to-end evaluation runner
│   │   └── ragas_metrics.py     # Custom RAGAs metrics computation
│   ├── generation/
│   │   ├── citation_parser.py   # Regex PMID extractor and verification
│   │   ├── llm_client.py        # Ollama LLM client wrapper
│   │   ├── prompts.py           # Strict clinical system & user prompts
│   │   └── rag_pipeline.py      # Core AskMedPipeline orchestrator
│   ├── ingestion/
│   │   ├── chunker.py           # Sentence-aware chunker (512 tokens / 128 overlap)
│   │   ├── cleaner.py           # Abstract cleaning and validation
│   │   ├── ingest_pipeline.py   # End-to-end ingestion pipeline
│   │   └── pubmed_fetcher.py    # NCBI E-Utilities fetcher
│   ├── retrieval/
│   │   ├── bm25_search.py       # BM25 keyword retriever
│   │   ├── hybrid_retriever.py  # Reciprocal Rank Fusion retriever (k=60)
│   │   ├── reranker.py          # Cross-encoder reranker wrapper
│   │   └── vector_search.py     # ChromaVectorStore wrapper
│   └── ui/
│       ├── app.py               # Streamlit main landing page
│       └── pages/
│           ├── 1_Chat.py        # Interactive clinical QA chat
│           ├── 2_Evaluation.py  # RAGAs evaluation dashboard
│           └── 3_Data_Manage.py # Knowledge base explorer & ingestion form
├── docs/images/                 # Documentation and benchmark diagrams
├── scripts/
│   ├── demo_runner.py           # CLI runner for 5 clinical demo scenarios
│   └── manage.ps1               # Windows PowerShell automation helper
├── tests/                       # Automated test suite (37+ tests)
├── Dockerfile                   # FastAPI backend image
├── Dockerfile.ui                # Streamlit UI image
├── docker-compose.yml           # Multi-service orchestration (Ollama + API + UI)
├── docker-compose.gpu.yml       # NVIDIA GPU acceleration override
├── Makefile                     # Build & automation shortcuts
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # Project documentation
```

---

## Running Tests

Execute the full automated test suite (37 unit & integration tests):

```bash
pytest -v
```

All tests pass 100% across API schemas, endpoints, citation parser, and retrieval logic.

---

## Docker Deployment

Start the complete 3-service stack (Ollama + FastAPI + Streamlit UI):

```bash
docker compose up -d --build
```

With NVIDIA GPU passthrough:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Stop and clean up containers:

```bash
docker compose down
```

---

## License

This project is licensed under the [MIT License](LICENSE).
