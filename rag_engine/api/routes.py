"""API route definitions."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

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

logger = structlog.get_logger()

router = APIRouter()

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
    file: UploadFile = File(...),  # noqa: B008
    namespace: str = Form(default="default"),
    title: str = Form(default=""),
) -> IngestResponse:
    """Upload and ingest a document into the RAG pipeline.

    Supported formats: PDF, Markdown, plain text, CSV.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        result = await _ingestion_pipeline.ingest(
            content=content,
            filename=file.filename,
            namespace=namespace,
            title=title or None,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("ingest.error", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed") from e


# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------


@router.post("/query", response_model=QueryResult, tags=["query"])
async def query_documents(
    question: str = Form(...),
    top_k: int = Form(default=settings.default_top_k),
    namespace: str = Form(default="default"),
    search_method: str = Form(default="hybrid"),
) -> QueryResult:
    """Query the RAG pipeline with a natural language question.

    Returns an answer with source citations and confidence score.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    if search_method not in ("hybrid", "vector", "keyword"):
        raise HTTPException(
            status_code=400,
            detail="search_method must be one of: hybrid, vector, keyword",
        )

    try:
        result = await _query_pipeline.query(
            question=question,
            top_k=top_k,
            namespace=namespace,
            search_method=search_method,
        )
        return result
    except Exception as e:
        logger.error("query.error", error=str(e), question=question)
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

    async def generate():
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
