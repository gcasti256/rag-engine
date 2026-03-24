"""Text extraction from various document formats."""

from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class ExtractedPage:
    """A single page/section of extracted text."""

    content: str
    page_number: int | None = None
    section: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from text extraction."""

    pages: list[ExtractedPage]
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.content for p in self.pages)


class BaseExtractor(ABC):
    """Base class for document extractors."""

    @abstractmethod
    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        """Extract text from document bytes."""
        ...


class PDFExtractor(BaseExtractor):
    """Extract text from PDF documents using PyMuPDF."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        doc = fitz.open(stream=content, filetype="pdf")
        pages: list[ExtractedPage] = []
        title = doc.metadata.get("title", "") or Path(filename).stem

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    ExtractedPage(
                        content=text,
                        page_number=page_num + 1,
                    )
                )

        metadata = {
            k: str(v) for k, v in (doc.metadata or {}).items() if v
        }
        doc.close()

        return ExtractionResult(pages=pages, title=title, metadata=metadata)


class MarkdownExtractor(BaseExtractor):
    """Extract text from Markdown files, preserving section structure."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        text = content.decode("utf-8", errors="replace")
        sections = self._split_sections(text)
        title = Path(filename).stem

        # Try to extract title from first heading
        for section in sections:
            if section.section:
                title = section.section
                break

        return ExtractionResult(pages=sections, title=title)

    def _split_sections(self, text: str) -> list[ExtractedPage]:
        """Split markdown into sections by headings."""
        lines = text.split("\n")
        sections: list[ExtractedPage] = []
        current_section = ""
        current_content: list[str] = []

        for line in lines:
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections.append(
                            ExtractedPage(content=content, section=current_section)
                        )
                current_section = line.lstrip("#").strip()
                current_content = [line]
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections.append(
                    ExtractedPage(content=content, section=current_section)
                )

        return sections if sections else [ExtractedPage(content=text)]


class TextExtractor(BaseExtractor):
    """Extract plain text files."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        text = content.decode("utf-8", errors="replace").strip()
        title = Path(filename).stem
        return ExtractionResult(
            pages=[ExtractedPage(content=text)],
            title=title,
        )


class CSVExtractor(BaseExtractor):
    """Extract text from CSV files, converting rows to readable text."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows: list[str] = []

        for i, row in enumerate(reader):
            # Convert each row to a readable text block
            parts = [f"{key}: {value}" for key, value in row.items() if value]
            rows.append(f"Record {i + 1}: {'; '.join(parts)}")

        title = Path(filename).stem
        # Group rows into pages of 50 for chunking
        page_size = 50
        pages: list[ExtractedPage] = []
        for i in range(0, len(rows), page_size):
            batch = rows[i : i + page_size]
            pages.append(
                ExtractedPage(
                    content="\n".join(batch),
                    section=f"Rows {i + 1}-{i + len(batch)}",
                )
            )

        return ExtractionResult(pages=pages, title=title)


# Registry of extractors by file extension
_EXTRACTORS: dict[str, BaseExtractor] = {
    ".pdf": PDFExtractor(),
    ".md": MarkdownExtractor(),
    ".markdown": MarkdownExtractor(),
    ".txt": TextExtractor(),
    ".text": TextExtractor(),
    ".csv": CSVExtractor(),
}


def extract_text(content: bytes, filename: str) -> ExtractionResult:
    """Extract text from a document based on its file extension."""
    ext = Path(filename).suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if not extractor:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(_EXTRACTORS.keys())}")
    return extractor.extract(content, filename)
