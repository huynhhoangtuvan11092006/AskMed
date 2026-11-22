"""
src/config.py
Centralized configuration manager for the Ask_Med system.
Parses, validates, and serves configs defined in configs/config.yaml using Pydantic.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


# ------------------------------------------------------------------------------
# Pydantic Schemas for Structured Validation
# ------------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "Ask_Med"
    version: str = "1.0.0"
    description: str = "Evidence-Based Medical Question Answering with PubMed Grounding"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    chroma_db_dir: Path = Path("data/chroma_db")
    bm25_index_file: Path = Path("data/bm25_index/bm25.pkl")
    eval_dataset_file: Path = Path("data/eval/eval_dataset.json")
    eval_results_dir: Path = Path("data/eval/results")

    def ensure_directories_exist(self) -> None:
        """Helper to ensure all storage directories exist on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_file.parent.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)


class OllamaConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="llama3.1:8b")
    embed_model: str = Field(default="nomic-embed-text")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=5, gt=0)


class ModelsConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class IngestionConfig(BaseModel):
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=128, ge=0)
    entrez_email: str = "ask_med_team@medrag.local"
    entrez_tool: str = "AskMed_EvidenceRAG"
    min_abstract_words: int = Field(default=30, ge=10)
    default_search_limit: int = Field(default=50, gt=0)


class RetrievalConfig(BaseModel):
    collection_name: str = "medical_knowledge"
    dense_top_k: int = Field(default=20, gt=0)
    sparse_top_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"


class UIConfig(BaseModel):
    port: int = 8501


class AppConfig(BaseModel):
    """Root configuration object representing the entire system state."""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


# ------------------------------------------------------------------------------
# Config Loader & Environment Overrides
# ------------------------------------------------------------------------------

def _find_default_config_path() -> Path:
    """Find configs/config.yaml relative to current file or current working directory."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # Try project root configs/config.yaml
    candidate = project_root / "configs" / "config.yaml"
    if candidate.exists():
        return candidate

    # Try relative path from cwd
    cwd_candidate = Path.cwd() / "configs" / "config.yaml"
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(
        f"Cannot locate configs/config.yaml. Searched in: '{candidate}' and '{cwd_candidate}'"
    )


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Loads config from YAML, applies environment variable overrides,
    and validates using Pydantic.
    """
    resolved_path = Path(config_path) if config_path else _find_default_config_path()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Allow overriding Ollama Host from Environment Variables (for Docker Compose)
    env_ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
    if env_ollama_host:
        # Standardize URL schema
        if not env_ollama_host.startswith("http://") and not env_ollama_host.startswith("https://"):
            env_ollama_host = f"http://{env_ollama_host}"
        config.models.ollama.base_url = env_ollama_host

    # Ensure storage paths exist
    config.paths.ensure_directories_exist()

    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Returns a cached singleton instance of AppConfig for the application."""
    return load_config()


if __name__ == "__main__":
    # Self-test when executed directly
    cfg = get_config()
    print("=" * 60)
    print(f"Loaded Config for: {cfg.project.name} (v{cfg.project.version})")
    print(f"Ollama Base URL   : {cfg.models.ollama.base_url}")
    print(f"LLM Model         : {cfg.models.ollama.llm_model}")
    print(f"Embedding Model   : {cfg.models.ollama.embed_model}")
    print(f"Reranker Model    : {cfg.models.reranker.model_name} (Top {cfg.models.reranker.top_n})")
    print(f"Chunk Size        : {cfg.ingestion.chunk_size} tokens (overlap: {cfg.ingestion.chunk_overlap})")
    print(f"Hybrid Fusion RRF : k = {cfg.retrieval.rrf_k}")
    print("=" * 60)
    print("Config validation status: OK")

