"""Tests for the recursive text chunker."""

from __future__ import annotations

from rag_engine.ingestion.chunker import RecursiveChunker
from rag_engine.ingestion.extractors import ExtractedPage


class TestRecursiveChunker:
    def test_short_text_single_chunk(self, chunker: RecursiveChunker) -> None:
        """Short text should produce a single chunk."""
        pages = [ExtractedPage(content="Short text.", page_number=1)]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        long_text = " ".join(["word"] * 100)
        pages = [ExtractedPage(content=long_text)]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")
        assert len(chunks) > 1

    def test_metadata_propagation(self, chunker: RecursiveChunker) -> None:
        """Chunk metadata should include document and page info."""
        pages = [
            ExtractedPage(
                content="Test content for chunking.",
                page_number=5,
                section="Introduction",
            )
        ]
        chunks = chunker.chunk_pages(
            pages, document_id="doc-123", source="report.pdf", title="Q3 Report"
        )
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta.document_id == "doc-123"
        assert meta.source == "report.pdf"
        assert meta.title == "Q3 Report"
        assert meta.page_number == 5
        assert meta.section == "Introduction"

    def test_total_chunks_updated(self, sample_pages: list[ExtractedPage]) -> None:
        """All chunks should have correct total_chunks count."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_pages(sample_pages, document_id="doc1", source="test.txt")
        total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata.total_chunks == total

    def test_chunk_index_sequential(self, sample_pages: list[ExtractedPage]) -> None:
        """Chunk indices should be sequential starting from 0."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_pages(sample_pages, document_id="doc1", source="test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_token_count_positive(self, chunker: RecursiveChunker) -> None:
        """Each chunk should have a positive token count."""
        pages = [ExtractedPage(content="This is a test sentence with several words.")]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_empty_input(self, chunker: RecursiveChunker) -> None:
        """Empty pages should produce no chunks."""
        chunks = chunker.chunk_pages([], document_id="doc1", source="test.txt")
        assert len(chunks) == 0

    def test_whitespace_only_content(self, chunker: RecursiveChunker) -> None:
        """Whitespace-only content should produce no chunks."""
        pages = [ExtractedPage(content="   \n\n  ")]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")
        assert len(chunks) == 0

    def test_overlap_present(self) -> None:
        """Chunks should have overlapping content when overlap > 0."""
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=5)
        text = (
            "Alpha bravo charlie delta echo foxtrot golf hotel "
            "india juliet kilo lima mike november."
        )
        pages = [ExtractedPage(content=text)]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")

        if len(chunks) >= 2:
            # At least some content should appear in adjacent chunks
            first_words = set(chunks[0].content.lower().split())
            second_words = set(chunks[1].content.lower().split())
            assert len(first_words & second_words) > 0

    def test_paragraph_boundary_splitting(self) -> None:
        """Chunker should prefer splitting at paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
        text = (
            "First paragraph with several words of content here.\n\n"
            "Second paragraph with more words of content here.\n\n"
            "Third paragraph with even more words of content."
        )
        pages = [ExtractedPage(content=text)]
        chunks = chunker.chunk_pages(pages, document_id="doc1", source="test.txt")
        assert len(chunks) >= 2

    def test_unique_chunk_ids(self, sample_pages: list[ExtractedPage]) -> None:
        """Each chunk should have a unique ID."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_pages(sample_pages, document_id="doc1", source="test.txt")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
