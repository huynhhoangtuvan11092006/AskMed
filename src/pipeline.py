"""
src/pipeline.py
Top-level entry point exposing the Ask_Med end-to-end medical QA pipeline.
"""

from src.generation.rag_pipeline import AskMedPipeline, ask_med, get_pipeline

__all__ = ["AskMedPipeline", "ask_med", "get_pipeline"]
