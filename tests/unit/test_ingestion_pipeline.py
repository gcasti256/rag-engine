"""Tests for the ingestion pipeline (mocked vector store)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_engine.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.store_document = AsyncMock()
    store.store_chunks = AsyncMock()
    return store


@pytest.fixture
def pipeline(mock_store: AsyncMock) -> IngestionPipeline:
    return IngestionPipeline(vector_store=mock_store)


class TestIngestionPipeline:
    async def test_ingest_markdown(
        self, pipeline: IngestionPipeline, mock_store: AsyncMock
    ) -> None:
        """Should extract, chunk, and store a markdown document."""
        content = b"# Test\n\nThis is test content for ingestion."
        result = await pipeline.ingest(content, "test.md", namespace="test")
        assert result.filename == "test.md"
        assert result.chunk_count > 0
        assert result.namespace == "test"
        mock_store.store_document.assert_awaited_once()
        mock_store.store_chunks.assert_awaited_once()

    async def test_ingest_text(
        self, pipeline: IngestionPipeline, mock_store: AsyncMock
    ) -> None:
        """Should ingest plain text files."""
        content = b"Plain text content for testing."
        result = await pipeline.ingest(content, "notes.txt")
        assert result.filename == "notes.txt"
        assert result.chunk_count > 0

    async def test_ingest_csv(
        self, pipeline: IngestionPipeline, mock_store: AsyncMock
    ) -> None:
        """Should ingest CSV files."""
        content = b"name,value\nAlpha,100\nBravo,200\n"
        result = await pipeline.ingest(content, "data.csv")
        assert result.filename == "data.csv"
        assert result.chunk_count > 0

    async def test_unsupported_type_raises(self, pipeline: IngestionPipeline) -> None:
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            await pipeline.ingest(b"data", "file.xlsx")

    async def test_empty_content_raises(self, pipeline: IngestionPipeline) -> None:
        """Should raise ValueError when no text is extracted."""
        with pytest.raises(ValueError, match="No text content"):
            await pipeline.ingest(b"   ", "empty.txt")

    async def test_file_too_large_raises(self, pipeline: IngestionPipeline) -> None:
        """Should raise ValueError for files exceeding size limit."""
        # Default max is 50MB
        huge = b"x" * (51 * 1024 * 1024)
        with pytest.raises(ValueError, match="File too large"):
            await pipeline.ingest(huge, "big.txt")

    async def test_custom_title(
        self, pipeline: IngestionPipeline, mock_store: AsyncMock
    ) -> None:
        """Should use custom title when provided."""
        content = b"Some content."
        result = await pipeline.ingest(content, "doc.txt", title="Custom Title")
        assert result.chunk_count > 0
        # Verify the document was stored (title is set on the Document object)
        mock_store.store_document.assert_awaited_once()
        stored_doc = mock_store.store_document.call_args[0][0]
        assert stored_doc.title == "Custom Title"
