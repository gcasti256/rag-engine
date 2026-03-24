"""Request/response schemas for the API (JSON body variants)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_engine.config import settings


class QueryRequest(BaseModel):
    """JSON request body for /query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=50)
    namespace: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    search_method: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")


class IngestRequest(BaseModel):
    """Metadata for document ingestion (used with multipart upload)."""

    namespace: str = "default"
    title: str = ""
