"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_completion_model: str = "gpt-4o-mini"

    # Database
    database_url: str = "postgresql+asyncpg://rag:changeme@localhost:5432/rag_engine"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_file_size_mb: int = 50

    # Query
    default_top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_tokens: int = 4096

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Embedding dimensions (text-embedding-3-small = 1536)
    embedding_dimensions: int = 1536

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
