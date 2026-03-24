"""Pydantic models for the RAG pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"


class ChunkMetadata(BaseModel):
    """Metadata attached to each text chunk."""

    source: str
    document_id: str
    title: str = ""
    page_number: int | None = None
    section: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TextChunk(BaseModel):
    """A chunk of text with metadata ready for embedding."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: ChunkMetadata
    token_count: int = 0


class Document(BaseModel):
    """An ingested document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    document_type: DocumentType
    title: str = ""
    namespace: str = "default"
    chunk_count: int = 0
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    """A source citation for a generated answer."""

    chunk_id: str
    source: str
    content: str
    page_number: int | None = None
    section: str = ""
    relevance_score: float = 0.0


class QueryResult(BaseModel):
    """The result of a RAG query."""

    answer: str
    citations: list[Citation]
    confidence: float = 0.0
    query: str
    model: str = ""
    total_tokens_used: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector search."""

    id: str
    content: str
    metadata: ChunkMetadata
    score: float = 0.0
    search_method: str = "vector"


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    document_id: str
    filename: str
    chunk_count: int
    total_tokens: int
    namespace: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    database: str = "connected"
    embedding_model: str = ""
    documents_count: int = 0
