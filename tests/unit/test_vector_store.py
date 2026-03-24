"""Tests for vector store (mocked database and embeddings)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_engine.models import ChunkMetadata, Document, DocumentType, TextChunk
from rag_engine.storage.vector_store import VectorStore


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    service = AsyncMock()
    service.embed_texts = AsyncMock(return_value=[[0.1] * 1536])
    service.embed_query = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def store(mock_embedding_service: AsyncMock) -> VectorStore:
    return VectorStore(embedding_service=mock_embedding_service)


class TestStoreDocument:
    async def test_store_document(self, store: VectorStore) -> None:
        """Should store a document record in the database."""
        doc = Document(
            filename="test.pdf",
            document_type=DocumentType.PDF,
            title="Test",
            chunk_count=5,
            total_tokens=100,
        )
        with patch("rag_engine.storage.vector_store.async_session") as mock_session_fn:
            mock_session = AsyncMock()
            mock_session.add = MagicMock()  # add() is sync
            mock_session_fn.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_fn.return_value.__aexit__ = AsyncMock(return_value=False)
            await store.store_document(doc)
            mock_session.add.assert_called_once()
            mock_session.commit.assert_awaited_once()


class TestStoreChunks:
    async def test_store_chunks_embeds_and_saves(
        self, store: VectorStore, mock_embedding_service: AsyncMock
    ) -> None:
        """Should embed texts and store chunk records."""
        chunks = [
            TextChunk(
                content="Test content",
                metadata=ChunkMetadata(source="test.txt", document_id="doc-1"),
                token_count=3,
            )
        ]
        with patch("rag_engine.storage.vector_store.async_session") as mock_session_fn:
            mock_session = AsyncMock()
            mock_session.add = MagicMock()  # add() is sync
            mock_session_fn.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_fn.return_value.__aexit__ = AsyncMock(return_value=False)
            await store.store_chunks(chunks)
            mock_embedding_service.embed_texts.assert_awaited_once_with(["Test content"])
            mock_session.add.assert_called_once()
            mock_session.commit.assert_awaited_once()

    async def test_store_empty_chunks(
        self, store: VectorStore, mock_embedding_service: AsyncMock
    ) -> None:
        """Should do nothing for empty chunk list."""
        await store.store_chunks([])
        mock_embedding_service.embed_texts.assert_not_awaited()


class TestDeleteDocument:
    async def test_delete_existing(self, store: VectorStore) -> None:
        """Should return True when document is deleted."""
        with patch("rag_engine.storage.vector_store.async_session") as mock_session_fn:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session_fn.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_fn.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await store.delete_document("doc-123")
            assert result is True

    async def test_delete_nonexistent(self, store: VectorStore) -> None:
        """Should return False when document doesn't exist."""
        with patch("rag_engine.storage.vector_store.async_session") as mock_session_fn:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 0
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session_fn.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_fn.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await store.delete_document("nonexistent")
            assert result is False


class TestGetDocumentCount:
    async def test_count(self, store: VectorStore) -> None:
        """Should return document count from database."""
        with patch("rag_engine.storage.vector_store.async_session") as mock_session_fn:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one.return_value = 42
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session_fn.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_fn.return_value.__aexit__ = AsyncMock(return_value=False)
            count = await store.get_document_count()
            assert count == 42
