# ==============================================================================
# Ask_Med: Project Operations & Automation Makefile
# Evidence-Based Medical Question Answering (MedRAG)
# ==============================================================================

.PHONY: help install docker-build docker-up docker-down docker-logs docker-gpu ingest eval test api ui clean

# Default Python binary (can be overridden via environment or CLI)
PYTHON ?= python
PIP ?= pip

help:
	@echo "=============================================================================="
	@echo "                        ASK_MED AUTOMATION COMMANDS                           "
	@echo "=============================================================================="
	@echo "  make install       : Install all required Python dependencies locally"
	@echo "  make docker-build  : Build Docker images for Backend and UI"
	@echo "  make docker-up     : Start all 3 containers (Ollama, API, UI) in background"
	@echo "  make docker-down   : Stop and remove all running containers"
	@echo "  make docker-logs   : Stream live logs from all Docker containers"
	@echo "  make docker-gpu    : Launch with NVIDIA GPU acceleration override"
	@echo "  make api           : Run FastAPI backend server locally (port 8000)"
	@echo "  make ui            : Run Streamlit clinical UI locally (port 8501)"
	@echo "  make ingest        : Ingest PubMed articles into ChromaDB & BM25"
	@echo "  make eval          : Run RAGAs benchmark evaluation pipeline"
	@echo "  make test          : Execute all unit and integration tests via pytest"
	@echo "  make clean         : Clean up bytecode, cache files, and temporary logs"
	@echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. Local Environment Management
# ------------------------------------------------------------------------------
install:
	@echo "--> Installing project dependencies from requirements.txt..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "--> Installation completed successfully."

# ------------------------------------------------------------------------------
# 2. Docker Orchestration
# ------------------------------------------------------------------------------
docker-build:
	@echo "--> Building Ask_Med Docker images (Backend & Streamlit UI)..."
	docker compose build

docker-up:
	@echo "--> Starting all Ask_Med containers (Ollama, FastAPI, Streamlit)..."
	docker compose up -d
	@echo "--> Services running: API on http://localhost:8000 | UI on http://localhost:8501"

docker-down:
	@echo "--> Stopping and removing all Ask_Med containers..."
	docker compose down

docker-logs:
	@echo "--> Streaming live logs from containers (Ctrl+C to stop)..."
	docker compose logs -f

docker-gpu:
	@echo "--> Launching Ask_Med with NVIDIA GPU acceleration..."
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
	@echo "--> GPU Acceleration active for Ollama & Cross-Encoder."

# ------------------------------------------------------------------------------
# 3. Local Development Runners
# ------------------------------------------------------------------------------
api:
	@echo "--> Starting FastAPI server on http://127.0.0.1:8000..."
	$(PYTHON) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

ui:
	@echo "--> Starting Streamlit UI on http://localhost:8501..."
	$(PYTHON) -m streamlit run src/ui/app.py

# ------------------------------------------------------------------------------
# 4. Ingestion, Evaluation & Automated Testing
# ------------------------------------------------------------------------------
ingest:
	@echo "--> Executing PubMed Ingestion Pipeline..."
	$(PYTHON) -m src.ingestion.ingest_pipeline --query "diabetes mellitus treatment" --limit 10

eval:
	@echo "--> Running 25-Question RAGAs Benchmark Evaluation..."
	$(PYTHON) -m src.evaluation.eval_pipeline --benchmark data/eval/eval_dataset.json

test:
	@echo "--> Executing full test suite via pytest..."
	$(PYTHON) -m pytest -v

# ------------------------------------------------------------------------------
# 5. Cleanup Utilities
# ------------------------------------------------------------------------------
clean:
	@echo "--> Cleaning up Python cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "--> Clean up complete."
