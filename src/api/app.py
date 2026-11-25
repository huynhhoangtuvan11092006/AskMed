"""
src/api/app.py
--------------
Production-grade FastAPI application for AskMed Evidence-Based Medical RAG.
Exposes RESTful endpoints:
  - POST /api/v1/query       : Evidence-based QA with citations & streaming support
  - GET  /api/v1/paper/{pmid}: Retrieve metadata and abstract for a specific PubMed article
  - GET  /api/v1/stats       : Knowledge base statistics (article counts, model info)
  - POST /api/v1/ingest      : Asynchronous background ingestion of new PubMed articles
  - GET  /health             : System liveness and dependency readiness probe
  - GET  /api/v1/health      : Versioned health check
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Dict, Iterator, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import (
    CitationItem,
    ContextItem,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.config import get_config
from src.generation.llm_client import OllamaLLMClient
from src.generation.rag_pipeline import AskMedPipeline, get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cached reference to the singleton pipeline
_pipeline_instance: Optional[AskMedPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes pipeline components and verifies Ollama daemon on startup.
    """
    global _pipeline_instance
    logger.info("Initializing AskMed FastAPI Application...")
    try:
        # Pre-warm singleton pipeline
        _pipeline_instance = get_pipeline()
        logger.info("AskMedPipeline pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm pipeline during startup: {e}")

    yield

    logger.info("Shutting down AskMed FastAPI Application...")


# Initialize FastAPI app with comprehensive OpenAPI metadata
app = FastAPI(
    title="AskMed - Evidence-Based Medical QA API",
    description="""
**AskMed** is an open-source, evidence-grounded clinical question answering API powered by 
local LLMs (`llama3.1:8b`), Hybrid Search (Dense ChromaDB + Sparse BM25), and FlashRank Cross-Encoder Reranking.
Every claim in the answer is strictly attributed to PubMed NCBI scientific literature.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware (permits local Streamlit and frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Step 6.3: HTTP Request Logging & Latency Tracking Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    """
    Measures per-request execution duration, injects 'X-Process-Time' response header,
    and logs request method, route path, client IP, and HTTP status code.
    """
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        logger.info(
            f"API {method} {path} -> {response.status_code} ({process_time:.4f}s) [client={client_ip}]"
        )
        return response
    except Exception as exc:
        process_time = time.perf_counter() - start_time
        logger.error(
            f"API {method} {path} FAILED after {process_time:.4f}s: {exc}",
            exc_info=True
        )
        raise exc


# -----------------------------------------------------------------------------
# Step 6.3: Centralized Exception Handlers (Standardized JSON error formatting)
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Formats Pydantic 422 input validation errors into a clean, developer-friendly JSON format.
    """
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field, "message": msg})

    logger.warning(f"Validation failed on {request.method} {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Request Validation Failed",
            "message": "The input payload did not meet validation constraints.",
            "details": errors,
            "status_code": 422,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Standardizes explicit HTTPException responses across all routes.
    """
    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catches unhandled runtime exceptions. Specifically detects Ollama daemon connectivity
    outages and converts them to a dedicated 503 Service Unavailable response.
    """
    err_str = str(exc)
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)

    # Detect Ollama connectivity failure (daemon stopped or port unreachable)
    if "Connection refused" in err_str or "Failed to connect to Ollama" in err_str or "11434" in err_str:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Ollama LLM Service Unavailable",
                "detail": (
                    "Could not connect to the local Ollama LLM server (http://127.0.0.1:11434). "
                    "Please ensure the Ollama daemon is running ('ollama serve')."
                ),
                "status_code": 503,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred while processing the medical request.",
            "status_code": 500,
        },
    )


# Helper: Load PMID doc cache from BM25 index file
def _get_pmid_cache() -> Dict[str, Dict[str, Any]]:
    bm25_path = Path("data/bm25_index/bm25.pkl")
    if not bm25_path.exists():
        return {}
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        return data.get("pmid_to_doc", {})
    except Exception as e:
        logger.warning(f"Could not load pmid_to_doc from bm25.pkl: {e}")
        return {}


# -----------------------------------------------------------------------------
# 1. Health & Readiness Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health & Monitoring"],
    summary="Root health and readiness check"
)
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health & Monitoring"],
    summary="Versioned health and readiness check"
)
async def check_health() -> HealthResponse:
    """
    Checks operational readiness:
    - Service uptime
    - Ollama LLM server connectivity
    - ChromaDB vector database accessibility
    - BM25 lexical index availability
    """
    ollama_ok = False
    try:
        llm = OllamaLLMClient()
        ollama_ok = llm.is_healthy()
    except Exception as e:
        logger.warning(f"Ollama health probe failed: {e}")

    bm25_ok = Path("data/bm25_index/bm25.pkl").exists()
    chroma_ok = Path("data/chroma_db").exists()

    overall_status = "ok" if (ollama_ok and bm25_ok and chroma_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        ollama_connected=ollama_ok,
        vector_db_ready=chroma_ok,
        bm25_ready=bm25_ok,
        version="1.0.0"
    )


# -----------------------------------------------------------------------------
# 2. Query QA Endpoint (Supports both JSON and Streaming)
# -----------------------------------------------------------------------------
@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Clinical QA"],
    summary="Submit a clinical medical question",
    responses={
        200: {"description": "Evidence-based answer with verified PMID citations."},
        500: {"description": "Internal pipeline or inference failure."}
    }
)
async def ask_medical_query(request: QueryRequest):
    """
    Executes the complete multi-stage RAG pipeline:
    1. **Hybrid Retrieval:** Dense Semantic Search (ChromaDB) + Sparse Keyword Search (BM25)
    2. **RRF Fusion:** Merges ranked lists into top candidate passages
    3. **Cross-Encoder Reranker:** Reranks passages using `ms-marco-MiniLM-L-6-v2`
    4. **LLM Synthesis:** Generates answer strictly grounded in retrieved evidence
    5. **Citation Verification:** Extracts, hyper-links, and validates `[PMID: <id>]` citations
    """
    global _pipeline_instance
    pipeline = _pipeline_instance or get_pipeline()

    # Handle streaming response if requested
    if request.stream:
        try:
            token_iter, meta = pipeline.ask_stream(
                query=request.query,
                rerank_top_n=request.top_k,
            )

            def event_generator() -> Iterator[str]:
                for token in token_iter:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                
                # Emit completion event with metadata
                citations = meta.get("citations", [])
                yield f"data: {json.dumps({'type': 'done', 'citations': citations})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        except Exception as e:
            logger.error(f"Streaming query failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Streaming query processing failed: {str(e)}"
            )

    # Standard non-streaming JSON response
    try:
        t0 = time.perf_counter()
        result = pipeline.ask(
            query=request.query,
            rerank_top_n=request.top_k,
        )
        latency = round(time.perf_counter() - t0, 4)

        # Convert citations
        citation_items = []
        for c in result.get("citations", []):
            citation_items.append(
                CitationItem(
                    pmid=str(c.get("pmid", "")),
                    title=c.get("title", ""),
                    url=c.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{c.get('pmid', '')}/"),
                    journal=c.get("journal"),
                    pub_year=str(c.get("pub_year", "")) if c.get("pub_year") else None
                )
            )

        # Convert raw contexts if requested
        raw_contexts = None
        if request.include_raw_contexts:
            raw_contexts = []
            for p in result.get("retrieved_passages", []):
                pmid_val = str(p.get("pmid") or p.get("metadata", {}).get("pmid", ""))
                chunk_id = str(p.get("chunk_id") or p.get("metadata", {}).get("chunk_id", ""))
                txt = p.get("text", "") or p.get("metadata", {}).get("text", "")
                score = p.get("rerank_score") or p.get("score")
                raw_contexts.append(
                    ContextItem(
                        pmid=pmid_val,
                        chunk_id=chunk_id,
                        text=txt,
                        score=float(score) if score is not None else None
                    )
                )

        grounding_audit = result.get("grounding", {})
        is_grounded = bool(grounding_audit.get("is_grounded", len(citation_items) > 0))

        return QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            citations=citation_items,
            raw_contexts=raw_contexts,
            is_grounded=is_grounded,
            latency_seconds=latency
        )
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Medical query failed: {e}", exc_info=True)
        if "Connection refused" in err_msg or "Failed to connect to Ollama" in err_msg or "11434" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama LLM Service is unavailable (http://127.0.0.1:11434). Please check that 'ollama serve' is running."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )


# -----------------------------------------------------------------------------
# 3. Specific Paper Retrieval Endpoint
# -----------------------------------------------------------------------------
@app.get(
    "/api/v1/paper/{pmid}",
    tags=["Knowledge Base"],
    summary="Retrieve article details by PMID"
)
async def get_paper_details(pmid: str):
    """
    Fetches full metadata and abstract for a specific PubMed article stored in the local index.
    """
    clean_pmid = pmid.strip()
    cache = _get_pmid_cache()

    if clean_pmid in cache:
        doc = cache[clean_pmid]
        return {
            "pmid": clean_pmid,
            "title": doc.get("title", ""),
            "journal": doc.get("journal", ""),
            "pub_year": doc.get("pub_year", ""),
            "authors": doc.get("authors", []),
            "abstract": doc.get("text", "") or doc.get("abstract", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{clean_pmid}/"
        }

    # Fallback to inspecting raw JSON articles
    raw_path = Path(f"data/raw/pubmed_{clean_pmid}.json")
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "pmid": clean_pmid,
                "title": data.get("title", ""),
                "journal": data.get("journal", ""),
                "pub_year": data.get("pub_year", ""),
                "authors": data.get("authors", []),
                "abstract": data.get("abstract", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{clean_pmid}/"
            }
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Article with PMID '{clean_pmid}' not found in the local database."
    )


# -----------------------------------------------------------------------------
# 4. Knowledge Base Statistics Endpoint
# -----------------------------------------------------------------------------
@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["Knowledge Base"],
    summary="Get repository and pipeline statistics"
)
async def get_system_stats() -> StatsResponse:
    """
    Returns aggregated metrics on index size and active machine learning models.
    """
    cfg = get_config()
    total_articles = 0
    total_chunks = 0
    bm25_status_str = "not_found"

    bm25_path = Path("data/bm25_index/bm25.pkl")
    if bm25_path.exists():
        try:
            with open(bm25_path, "rb") as f:
                b_data = pickle.load(f)
            chunks = b_data.get("chunks", [])
            total_chunks = len(chunks)
            unique_pmids = set(str(c.get("pmid", "")).strip() for c in chunks if c.get("pmid"))
            total_articles = len(unique_pmids)
            bm25_status_str = "loaded"
        except Exception as e:
            logger.warning(f"Could not parse bm25.pkl: {e}")

    vector_db_path = Path("data/chroma_db")
    vdb_status = "connected" if vector_db_path.exists() else "not_initialized"

    return StatsResponse(
        total_articles=total_articles,
        total_chunks=total_chunks,
        vector_db_status=vdb_status,
        bm25_status=bm25_status_str,
        embedding_model=cfg.models.ollama.embed_model,
        reranker_model=cfg.models.reranker.model_name,
        llm_model=cfg.models.ollama.llm_model
    )


# -----------------------------------------------------------------------------
# 5. Background Ingestion Endpoint
# -----------------------------------------------------------------------------
def _background_ingest_task(keywords: List[str], max_per_kw: int):
    """
    Worker function executed in the background to avoid blocking API requests.
    """
    logger.info(f"Starting background ingestion for keywords: {keywords}")
    try:
        from src.ingestion.ingest_pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        for kw in keywords:
            pipeline.ingest_query(query=kw, max_results=max_per_kw)
        logger.info(f"Background ingestion completed successfully for {keywords}.")
    except Exception as e:
        logger.error(f"Background ingestion encountered an error: {e}", exc_info=True)


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Knowledge Base"],
    summary="Trigger asynchronous PubMed ingestion"
)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Launches an asynchronous background worker task to search, download, clean,
    chunk, and index new PubMed medical articles into ChromaDB and BM25.
    """
    background_tasks.add_task(
        _background_ingest_task,
        keywords=request.keywords,
        max_per_kw=request.max_articles_per_keyword
    )

    return IngestResponse(
        status="in_progress",
        message=f"Ingestion task queued for {len(request.keywords)} topic(s): {', '.join(request.keywords)}.",
        articles_fetched=0,
        chunks_created=0
    )


# -----------------------------------------------------------------------------
# Root index redirect to interactive docs
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root_redirect():
    return {
        "project": "AskMed API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health"
    }

