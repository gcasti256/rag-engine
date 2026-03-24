"""Document ingestion pipeline — extraction, chunking, and metadata."""

from rag_engine.ingestion.chunker import RecursiveChunker
from rag_engine.ingestion.extractors import (
    CSVExtractor,
    MarkdownExtractor,
    PDFExtractor,
    TextExtractor,
    extract_text,
)
from rag_engine.ingestion.pipeline import IngestionPipeline

__all__ = [
    "CSVExtractor",
    "IngestionPipeline",
    "MarkdownExtractor",
    "PDFExtractor",
    "RecursiveChunker",
    "TextExtractor",
    "extract_text",
]
