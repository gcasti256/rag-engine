"""Tests for document extractors."""

from __future__ import annotations

import pytest

from rag_engine.ingestion.extractors import (
    CSVExtractor,
    MarkdownExtractor,
    TextExtractor,
    extract_text,
)


class TestMarkdownExtractor:
    def test_section_splitting(self, sample_markdown: bytes) -> None:
        """Markdown should be split into sections by headings."""
        extractor = MarkdownExtractor()
        result = extractor.extract(sample_markdown, "report.md")
        assert len(result.pages) >= 3
        assert result.title == "Financial Report Q3 2024"

    def test_section_names(self, sample_markdown: bytes) -> None:
        """Sections should have correct heading names."""
        extractor = MarkdownExtractor()
        result = extractor.extract(sample_markdown, "report.md")
        sections = [p.section for p in result.pages]
        assert "Executive Summary" in sections
        assert "Revenue Breakdown" in sections

    def test_content_preserved(self, sample_markdown: bytes) -> None:
        """All content should be preserved after extraction."""
        extractor = MarkdownExtractor()
        result = extractor.extract(sample_markdown, "report.md")
        full = result.full_text
        assert "$4.2 billion" in full
        assert "22.3%" in full

    def test_empty_markdown(self) -> None:
        """Empty markdown should produce minimal result."""
        extractor = MarkdownExtractor()
        result = extractor.extract(b"", "empty.md")
        assert len(result.pages) <= 1


class TestTextExtractor:
    def test_plain_text(self) -> None:
        """Plain text should be extracted as a single page."""
        extractor = TextExtractor()
        result = extractor.extract(b"Hello world", "test.txt")
        assert len(result.pages) == 1
        assert result.pages[0].content == "Hello world"
        assert result.title == "test"

    def test_unicode_handling(self) -> None:
        """Unicode characters should be handled correctly."""
        extractor = TextExtractor()
        text = "Revenue: €1.5M • Growth: 12% → 15%".encode()
        result = extractor.extract(text, "unicode.txt")
        assert "€1.5M" in result.pages[0].content


class TestCSVExtractor:
    def test_csv_extraction(self, sample_csv: bytes) -> None:
        """CSV should be converted to readable text records."""
        extractor = CSVExtractor()
        result = extractor.extract(sample_csv, "data.csv")
        assert len(result.pages) >= 1
        full = result.full_text
        assert "Card processing fees" in full
        assert "1250000" in full

    def test_csv_title(self, sample_csv: bytes) -> None:
        """CSV title should be derived from filename."""
        extractor = CSVExtractor()
        result = extractor.extract(sample_csv, "financial_data.csv")
        assert result.title == "financial_data"


class TestExtractText:
    def test_supported_extensions(self) -> None:
        """Should accept supported file extensions."""
        result = extract_text(b"test content", "file.txt")
        assert result.pages[0].content == "test content"

    def test_unsupported_extension(self) -> None:
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(b"data", "file.xlsx")

    def test_markdown_extension(self, sample_markdown: bytes) -> None:
        """Should route .md files to MarkdownExtractor."""
        result = extract_text(sample_markdown, "doc.md")
        assert len(result.pages) >= 3
