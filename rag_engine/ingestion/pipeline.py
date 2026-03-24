"""End-to-end document ingestion pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from rag_engine.config import settings
from rag_engine.ingestion.chunker import RecursiveChunker
from rag_engine.ingestion.extractors import extract_text
from rag_engine.models import Document, DocumentType, IngestResponse

if TYPE_CHECKING:
    from rag_engine.storage import VectorStore

logger = structlog.get_logger()

# Map file extensions to DocumentType
_EXT_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".txt": DocumentType.TEXT,
    ".text": DocumentType.TEXT,
    ".csv": DocumentType.CSV,
}


class IngestionPipeline:
    """Orchestrates document extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: RecursiveChunker | None = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker or RecursiveChunker()

    async def ingest(
        self,
        content: bytes,
        filename: str,
        namespace: str = "default",
        title: str | None = None,
    ) -> IngestResponse:
        """Ingest a document: extract → chunk → embed → store."""
        from pathlib import Path

        ext = Path(filename).suffix.lower()
        doc_type = _EXT_TO_TYPE.get(ext)
        if not doc_type:
            raise ValueError(f"Unsupported file type: {ext}")

        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max {settings.max_file_size_mb}MB)"
            )

        logger.info("ingestion.start", filename=filename, type=doc_type, size_mb=round(size_mb, 2))

        # Extract text
        extraction = extract_text(content, filename)
        doc_title = title or extraction.title or Path(filename).stem

        # Create document record
        doc = Document(
            filename=filename,
            document_type=doc_type,
            title=doc_title,
            namespace=namespace,
        )

        # Chunk
        chunks = self.chunker.chunk_pages(
            pages=extraction.pages,
            document_id=doc.id,
            source=filename,
            title=doc_title,
            namespace=namespace,
        )

        if not chunks:
            raise ValueError("No text content extracted from document")

        # Store document and chunks
        total_tokens = sum(c.token_count for c in chunks)
        doc.chunk_count = len(chunks)
        doc.total_tokens = total_tokens

        await self.vector_store.store_document(doc)
        await self.vector_store.store_chunks(chunks, namespace=namespace)

        logger.info(
            "ingestion.complete",
            document_id=doc.id,
            chunks=len(chunks),
            tokens=total_tokens,
        )

        return IngestResponse(
            document_id=doc.id,
            filename=filename,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            namespace=namespace,
        )
