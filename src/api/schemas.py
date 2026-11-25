"""
src/api/schemas.py
------------------
Pydantic v2 data models & request/response schemas for the AskMed RESTful API.
Provides type validation, serialization, and OpenAPI / Swagger documentation examples.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CitationItem(BaseModel):
    """
    Metadata for a scientific citation extracted from PubMed NCBI.
    """
    model_config = ConfigDict(extra="ignore")

    pmid: str = Field(
        ...,
        description="Unique PubMed Identifier (e.g., '37111272')",
        examples=["37111272"]
    )
    title: str = Field(
        default="",
        description="Title of the scientific paper",
        examples=["Metformin in type 2 diabetes mellitus: mechanisms and clinical guidelines"]
    )
    url: str = Field(
        ...,
        description="Direct link to the official NCBI PubMed article",
        examples=["https://pubmed.ncbi.nlm.nih.gov/37111272/"]
    )
    journal: Optional[str] = Field(
        default=None,
        description="Journal name where the study was published",
        examples=["Lancet Diabetes Endocrinol"]
    )
    pub_year: Optional[str] = Field(
        default=None,
        description="Year of publication",
        examples=["2023"]
    )


class ContextItem(BaseModel):
    """
    Individual text chunk retrieved from the hybrid search and reranking pipeline.
    """
    model_config = ConfigDict(extra="ignore")

    pmid: str = Field(
        ...,
        description="PubMed ID associated with this chunk",
        examples=["37111272"]
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Internal chunk identifier",
        examples=["37111272_chunk_0"]
    )
    text: str = Field(
        ...,
        description="Extracted text excerpt from the study abstract or full text",
        examples=["Metformin is universally recommended as foundational first-line therapy..."]
    )
    score: Optional[float] = Field(
        default=None,
        description="Cross-encoder reranking relevance score",
        examples=[2.45]
    )


class QueryRequest(BaseModel):
    """
    Incoming request payload for clinical question answering.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="The medical question or clinical case description",
        examples=["What is the recommended first-line treatment for type 2 diabetes?"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top retrieved and reranked context chunks to use (1-20)",
        examples=[5]
    )
    include_raw_contexts: bool = Field(
        default=False,
        description="Whether to return raw retrieved context chunks alongside the answer",
        examples=[False]
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream response tokens (used for real-time frontend chat)",
        examples=[False]
    )


class QueryResponse(BaseModel):
    """
    Outbound response payload containing the evidence-based answer and verified citations.
    """
    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        description="The original user query",
        examples=["What is the recommended first-line treatment for type 2 diabetes?"]
    )
    answer: str = Field(
        ...,
        description="Synthesized medical answer with embedded [PMID: <id>] citation tags",
        examples=[
            "Metformin is universally recommended as the foundational first-line therapy [PMID: 37111272]..."
        ]
    )
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="List of verified PubMed citations backing up the claims in the answer"
    )
    raw_contexts: Optional[List[ContextItem]] = Field(
        default=None,
        description="Raw context text chunks if requested via include_raw_contexts=True"
    )
    is_grounded: bool = Field(
        ...,
        description="True if the answer contains at least one verified PMID citation, False otherwise",
        examples=[True]
    )
    latency_seconds: float = Field(
        ...,
        description="Total round-trip inference latency in seconds",
        examples=[1.85]
    )


class IngestRequest(BaseModel):
    """
    Request payload to trigger background PubMed ingestion for new topics.
    """
    model_config = ConfigDict(extra="forbid")

    keywords: List[str] = Field(
        ...,
        min_length=1,
        description="List of MeSH medical keywords or search terms to crawl from PubMed",
        examples=[["hypertension resistant", "spironolactone"]]
    )
    max_articles_per_keyword: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of articles to download per keyword (1-100)",
        examples=[10]
    )


class IngestResponse(BaseModel):
    """
    Response payload acknowledging the ingestion task status.
    """
    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        ...,
        description="Status of the ingestion operation ('success', 'in_progress', 'error')",
        examples=["success"]
    )
    message: str = Field(
        ...,
        description="Human-readable status summary message",
        examples=["Successfully indexed 20 new articles and created 48 chunks."]
    )
    articles_fetched: int = Field(
        default=0,
        description="Number of new PubMed articles downloaded",
        examples=[20]
    )
    chunks_created: int = Field(
        default=0,
        description="Number of text chunks embedded and indexed into ChromaDB and BM25",
        examples=[48]
    )


class StatsResponse(BaseModel):
    """
    Current statistics of the knowledge base and pipeline components.
    """
    model_config = ConfigDict(extra="ignore")

    total_articles: int = Field(
        ...,
        description="Total number of unique PubMed articles in the local repository",
        examples=[620]
    )
    total_chunks: int = Field(
        ...,
        description="Total number of text chunks indexed in ChromaDB and BM25",
        examples=[703]
    )
    vector_db_status: str = Field(
        ...,
        description="Status of ChromaDB vector store ('connected', 'error')",
        examples=["connected"]
    )
    bm25_status: str = Field(
        ...,
        description="Status of BM25 lexical index ('loaded', 'not_found')",
        examples=["loaded"]
    )
    embedding_model: str = Field(
        ...,
        description="Name of the dense embedding model in use",
        examples=["nomic-embed-text"]
    )
    reranker_model: str = Field(
        ...,
        description="Name of the cross-encoder reranker model in use",
        examples=["ms-marco-MiniLM-L-6-v2"]
    )
    llm_model: str = Field(
        ...,
        description="Name of the local generator LLM",
        examples=["llama3.1:8b"]
    )


class HealthResponse(BaseModel):
    """
    Health check payload verifying service uptime and subsystem connectivity.
    """
    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        ...,
        description="Overall service health ('ok', 'degraded', 'error')",
        examples=["ok"]
    )
    ollama_connected: bool = Field(
        ...,
        description="Whether the local Ollama instance is accessible",
        examples=[True]
    )
    vector_db_ready: bool = Field(
        ...,
        description="Whether ChromaDB is loaded and queryable",
        examples=[True]
    )
    bm25_ready: bool = Field(
        ...,
        description="Whether the BM25 index is present in memory",
        examples=[True]
    )
    version: str = Field(
        default="1.0.0",
        description="AskMed API software version",
        examples=["1.0.0"]
    )

