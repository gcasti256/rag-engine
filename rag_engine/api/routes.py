"""API route definitions."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from rag_engine.api.schemas import QueryRequest  # noqa: TC001 — FastAPI needs runtime access
from rag_engine.config import settings
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.models import (
    Document,
    HealthResponse,
    IngestResponse,
    QueryResult,
)
from rag_engine.query.pipeline import QueryPipeline
from rag_engine.storage import EmbeddingService, VectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Namespace must be alphanumeric with hyphens/underscores, max 64 chars
_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_VALID_SEARCH_METHODS = {"hybrid", "vector", "keyword"}

logger = structlog.get_logger()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """Verify API key if one is configured. Skip auth if no key is set."""
    if not settings.api_key:
        return
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


router = APIRouter(dependencies=[Depends(verify_api_key)])

MAX_UPLOAD_SIZE = settings.max_file_size_mb * 1024 * 1024  # 50MB default
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".csv"}


def _validate_namespace(namespace: str) -> str:
    """Validate and return a sanitized namespace string."""
    if not _NAMESPACE_RE.match(namespace):
        raise HTTPException(
            status_code=400,
            detail="Invalid namespace. Must be 1-64 alphanumeric characters, "
            "hyphens, or underscores.",
        )
    return namespace

# Shared instances
_embedding_service = EmbeddingService()
_vector_store = VectorStore(embedding_service=_embedding_service)
_ingestion_pipeline = IngestionPipeline(vector_store=_vector_store)
_query_pipeline = QueryPipeline(vector_store=_vector_store)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Check API health and database connectivity."""
    try:
        doc_count = await _vector_store.get_document_count()
        db_status = "connected"
    except Exception:
        doc_count = 0
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        embedding_model=settings.openai_embedding_model,
        documents_count=doc_count,
    )


# ------------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    namespace: str = Form(default="default"),
    title: str = Form(default=""),
) -> IngestResponse:
    """Upload and ingest a document into the RAG pipeline.

    Supported formats: PDF, Markdown, plain text, CSV.
    """
    # Check Content-Length header if present
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitize filename: strip path components to prevent directory traversal
    safe_filename = PurePosixPath(file.filename).name
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check file extension
    suffix = PurePosixPath(safe_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    _validate_namespace(namespace)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    # Double-check actual content size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    try:
        result = await _ingestion_pipeline.ingest(
            content=content,
            filename=safe_filename,
            namespace=namespace,
            title=title or None,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("ingest.error", error=str(e), filename=safe_filename)
        raise HTTPException(status_code=500, detail="Ingestion failed") from e


# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------


@router.post("/query", response_model=QueryResult, tags=["query"])
async def query_documents(body: QueryRequest) -> QueryResult:
    """Query the RAG pipeline with a natural language question.

    Returns an answer with source citations and confidence score.
    """
    try:
        result = await _query_pipeline.query(
            question=body.question,
            top_k=body.top_k,
            namespace=body.namespace,
            search_method=body.search_method,
        )
        return result
    except Exception as e:
        logger.error("query.error", error=str(e), question=body.question)
        raise HTTPException(status_code=500, detail="Query processing failed") from e


@router.post("/query/stream", tags=["query"])
async def query_stream(
    question: str = Form(...),
    top_k: int = Form(default=settings.default_top_k),
    namespace: str = Form(default="default"),
    search_method: str = Form(default="hybrid"),
) -> StreamingResponse:
    """Stream a RAG query response token by token."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question must be 2000 characters or fewer")
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    if search_method not in _VALID_SEARCH_METHODS:
        valid = ", ".join(sorted(_VALID_SEARCH_METHODS))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_method. Must be one of: {valid}",
        )
    _validate_namespace(namespace)

    async def generate() -> AsyncIterator[str]:
        async for token in _query_pipeline.query_stream(
            question=question,
            top_k=top_k,
            namespace=namespace,
            search_method=search_method,
        ):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ------------------------------------------------------------------
# Documents
# ------------------------------------------------------------------


@router.get("/documents", response_model=list[Document], tags=["documents"])
async def list_documents(
    namespace: str = Query(default=None),
) -> list[Document]:
    """List all ingested documents, optionally filtered by namespace."""
    return await _vector_store.list_documents(namespace=namespace)


@router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    deleted = await _vector_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}
