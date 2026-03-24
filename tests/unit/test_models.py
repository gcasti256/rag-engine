"""Tests for Pydantic models."""

from __future__ import annotations

from rag_engine.models import (
    ChunkMetadata,
    Citation,
    Document,
    DocumentType,
    QueryResult,
    TextChunk,
)


class TestTextChunk:
    def test_auto_id(self) -> None:
        """TextChunk should auto-generate a UUID."""
        chunk = TextChunk(
            content="test",
            metadata=ChunkMetadata(source="test.txt", document_id="doc1"),
        )
        assert chunk.id
        assert len(chunk.id) == 36  # UUID format

    def test_token_count_default(self) -> None:
        """Token count should default to 0."""
        chunk = TextChunk(
            content="test",
            metadata=ChunkMetadata(source="test.txt", document_id="doc1"),
        )
        assert chunk.token_count == 0


class TestDocument:
    def test_document_creation(self) -> None:
        """Document should be creatable with required fields."""
        doc = Document(filename="test.pdf", document_type=DocumentType.PDF)
        assert doc.id
        assert doc.filename == "test.pdf"
        assert doc.namespace == "default"

    def test_document_type_enum(self) -> None:
        """DocumentType enum should have expected values."""
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.CSV.value == "csv"
        assert DocumentType.MARKDOWN.value == "markdown"


class TestQueryResult:
    def test_query_result(self) -> None:
        """QueryResult should contain all expected fields."""
        result = QueryResult(
            answer="The revenue was $4.2B",
            citations=[
                Citation(
                    chunk_id="c1",
                    source="report.pdf",
                    content="Revenue: $4.2B",
                    page_number=5,
                    relevance_score=0.92,
                )
            ],
            confidence=0.85,
            query="What was the revenue?",
            model="gpt-4o-mini",
        )
        assert result.confidence == 0.85
        assert len(result.citations) == 1
        assert result.citations[0].page_number == 5

    def test_empty_citations(self) -> None:
        """QueryResult should work with empty citations."""
        result = QueryResult(answer="No info.", citations=[], query="test?")
        assert result.citations == []
        assert result.confidence == 0.0
