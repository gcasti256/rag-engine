"""Recursive text chunking with configurable size and overlap."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import tiktoken

from rag_engine.config import settings
from rag_engine.models import ChunkMetadata, TextChunk

if TYPE_CHECKING:
    from rag_engine.ingestion.extractors import ExtractedPage


class RecursiveChunker:
    """Split text into overlapping chunks using recursive character splitting.

    Tries to split on paragraph boundaries first, then sentences, then words,
    preserving semantic coherence within chunks.
    """

    SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        document_id: str,
        source: str,
        title: str = "",
        namespace: str = "default",
    ) -> list[TextChunk]:
        """Chunk a list of extracted pages into TextChunks with metadata."""
        all_chunks: list[TextChunk] = []

        for page in pages:
            raw_chunks = self._split_text(page.content)
            for _i, chunk_text in enumerate(raw_chunks):
                token_count = self.count_tokens(chunk_text)
                metadata = ChunkMetadata(
                    source=source,
                    document_id=document_id,
                    title=title,
                    page_number=page.page_number,
                    section=page.section,
                    chunk_index=len(all_chunks),
                    total_chunks=0,  # Updated after all chunks are created
                )
                all_chunks.append(
                    TextChunk(
                        content=chunk_text,
                        metadata=metadata,
                        token_count=token_count,
                    )
                )

        # Update total_chunks count
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata.total_chunks = total

        return all_chunks

    def _split_text(self, text: str, separators: list[str] | None = None) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if separators is None:
            separators = self.SEPARATORS

        if not text.strip():
            return []

        # If text fits in a single chunk, return it
        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Try each separator
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: split by character count
                return self._split_by_tokens(text)

            if sep not in text:
                continue

            parts = text.split(sep)
            chunks: list[str] = []
            current: list[str] = []
            current_tokens = 0

            for part in parts:
                part_tokens = self.count_tokens(part)

                if current_tokens + part_tokens > self.chunk_size and current:
                    chunk_text = sep.join(current).strip()
                    if chunk_text:
                        # If this chunk is still too large, recurse with finer separators
                        if self.count_tokens(chunk_text) > self.chunk_size:
                            chunks.extend(
                                self._split_text(chunk_text, separators[i + 1 :])
                            )
                        else:
                            chunks.append(chunk_text)

                    # Overlap: keep last parts that fit in overlap window
                    overlap_parts: list[str] = []
                    overlap_tokens = 0
                    for p in reversed(current):
                        pt = self.count_tokens(p)
                        if overlap_tokens + pt > self.chunk_overlap:
                            break
                        overlap_parts.insert(0, p)
                        overlap_tokens += pt
                    current = overlap_parts
                    current_tokens = overlap_tokens

                current.append(part)
                current_tokens += part_tokens

            # Don't forget the last chunk
            if current:
                chunk_text = sep.join(current).strip()
                if chunk_text:
                    if self.count_tokens(chunk_text) > self.chunk_size:
                        chunks.extend(
                            self._split_text(chunk_text, separators[i + 1 :])
                        )
                    else:
                        chunks.append(chunk_text)

            if chunks:
                return chunks

        return [text.strip()] if text.strip() else []

    def _split_by_tokens(self, text: str) -> list[str]:
        """Split text into chunks by token count (last resort)."""
        tokens = self.encoder.encode(text)
        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = end - self.chunk_overlap if end < len(tokens) else end

        return chunks
