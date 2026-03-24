"""Tests for the query pipeline context building and citation logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rag_engine.models import ChunkMetadata, RetrievedChunk
from rag_engine.query.pipeline import QueryPipeline


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def pipeline(mock_vector_store: AsyncMock) -> QueryPipeline:
    with patch("rag_engine.query.pipeline.AsyncOpenAI"):
        return QueryPipeline(vector_store=mock_vector_store, api_key="sk-test")


def _make_chunk(
    chunk_id: str = "c1",
    content: str = "Test content",
    source: str = "doc.pdf",
    page_number: int | None = 1,
    section: str = "Intro",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            source=source,
            document_id="doc-1",
            page_number=page_number,
            section=section,
        ),
        score=score,
        search_method="hybrid",
    )


class TestBuildContext:
    def test_single_chunk(self, pipeline: QueryPipeline) -> None:
        """Single chunk should produce a context block."""
        chunks = [_make_chunk()]
        context, used = pipeline._build_context(chunks)
        assert "Context Document 1" in context
        assert "Test content" in context
        assert len(used) == 1

    def test_respects_token_limit(self, pipeline: QueryPipeline) -> None:
        """Should stop adding chunks when token limit is exceeded."""
        # Create chunks with enough content to exceed default max_context_tokens
        chunks = [
            _make_chunk(chunk_id=f"c{i}", content="word " * 500)
            for i in range(20)
        ]
        _context, used = pipeline._build_context(chunks)
        assert len(used) < len(chunks)

    def test_includes_section_and_page(self, pipeline: QueryPipeline) -> None:
        """Context block should include section and page info."""
        chunks = [_make_chunk(section="Revenue", page_number=5)]
        context, _used = pipeline._build_context(chunks)
        assert "Section: Revenue" in context
        assert "Page: 5" in context

    def test_no_section_info_when_empty(self, pipeline: QueryPipeline) -> None:
        """Context block should omit section when empty."""
        chunks = [_make_chunk(section="", page_number=None)]
        context, _used = pipeline._build_context(chunks)
        assert "Section:" not in context
        assert "Page:" not in context

    def test_empty_chunks(self, pipeline: QueryPipeline) -> None:
        """Empty chunk list should return empty context."""
        context, used = pipeline._build_context([])
        assert context == ""
        assert used == []


class TestBuildCitations:
    def test_builds_citations(self, pipeline: QueryPipeline) -> None:
        """Should convert chunks to citation objects."""
        chunks = [_make_chunk(content="Short content", score=0.92)]
        citations = pipeline._build_citations(chunks)
        assert len(citations) == 1
        assert citations[0].source == "doc.pdf"
        assert citations[0].relevance_score == 0.92
        assert citations[0].content == "Short content"

    def test_truncates_long_content(self, pipeline: QueryPipeline) -> None:
        """Citations should truncate content over 200 chars."""
        long = "x" * 300
        chunks = [_make_chunk(content=long)]
        citations = pipeline._build_citations(chunks)
        assert citations[0].content.endswith("...")
        assert len(citations[0].content) == 203  # 200 + "..."

    def test_multiple_citations(self, pipeline: QueryPipeline) -> None:
        """Should build citations for multiple chunks."""
        chunks = [
            _make_chunk(chunk_id="c1", source="a.pdf"),
            _make_chunk(chunk_id="c2", source="b.pdf"),
        ]
        citations = pipeline._build_citations(chunks)
        assert len(citations) == 2
        assert citations[0].source == "a.pdf"
        assert citations[1].source == "b.pdf"


class TestRetrieve:
    async def test_vector_method(
        self, pipeline: QueryPipeline, mock_vector_store: AsyncMock
    ) -> None:
        """Should call vector_search for 'vector' method."""
        mock_vector_store.vector_search = AsyncMock(return_value=[])
        await pipeline._retrieve("test?", 5, "default", "vector")
        mock_vector_store.vector_search.assert_awaited_once()

    async def test_keyword_method(
        self, pipeline: QueryPipeline, mock_vector_store: AsyncMock
    ) -> None:
        """Should call keyword_search for 'keyword' method."""
        mock_vector_store.keyword_search = AsyncMock(return_value=[])
        await pipeline._retrieve("test?", 5, "default", "keyword")
        mock_vector_store.keyword_search.assert_awaited_once()

    async def test_hybrid_default(
        self, pipeline: QueryPipeline, mock_vector_store: AsyncMock
    ) -> None:
        """Should call hybrid_search for 'hybrid' method."""
        mock_vector_store.hybrid_search = AsyncMock(return_value=[])
        await pipeline._retrieve("test?", 5, "default", "hybrid")
        mock_vector_store.hybrid_search.assert_awaited_once()


class TestQuery:
    async def test_no_results_returns_no_info(
        self, pipeline: QueryPipeline, mock_vector_store: AsyncMock
    ) -> None:
        """Should return no-info message when no chunks found."""
        mock_vector_store.hybrid_search = AsyncMock(return_value=[])
        result = await pipeline.query("What is revenue?")
        assert "don't have enough information" in result.answer
        assert result.confidence == 0.0
        assert result.citations == []

    async def test_successful_query(
        self, pipeline: QueryPipeline, mock_vector_store: AsyncMock
    ) -> None:
        """Should return generated answer with citations."""
        chunks = [_make_chunk(content="Revenue was $4.2B", score=0.9)]
        mock_vector_store.hybrid_search = AsyncMock(return_value=chunks)

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Revenue was $4.2B [Source 1]"
        mock_response.usage.total_tokens = 150
        pipeline.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await pipeline.query("What was revenue?")
        assert result.answer == "Revenue was $4.2B [Source 1]"
        assert len(result.citations) == 1
        assert result.confidence > 0
        assert result.retrieval_time_ms >= 0
        assert result.generation_time_ms >= 0
