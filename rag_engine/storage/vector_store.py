"""Vector store with pgvector for similarity search and BM25 for keyword search."""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import delete, func, select

from rag_engine.config import settings
from rag_engine.database import ChunkRecord, DocumentRecord, async_session
from rag_engine.models import ChunkMetadata, Document, DocumentType, RetrievedChunk, TextChunk
from rag_engine.storage.embeddings import EmbeddingService

logger = structlog.get_logger()


class VectorStore:
    """Hybrid vector + keyword search over document chunks."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embeddings = embedding_service or EmbeddingService()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def store_document(self, doc: Document) -> None:
        """Store document metadata."""
        async with async_session() as session:
            record = DocumentRecord(
                id=doc.id,
                filename=doc.filename,
                document_type=doc.document_type.value,
                title=doc.title,
                namespace=doc.namespace,
                chunk_count=doc.chunk_count,
                total_tokens=doc.total_tokens,
            )
            session.add(record)
            await session.commit()

    async def store_chunks(self, chunks: list[TextChunk], namespace: str = "default") -> None:
        """Embed and store text chunks."""
        if not chunks:
            return

        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = await self.embeddings.embed_texts(texts)

        async with async_session() as session:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                record = ChunkRecord(
                    id=chunk.id,
                    document_id=chunk.metadata.document_id,
                    content=chunk.content,
                    embedding=embedding,
                    source=chunk.metadata.source,
                    title=chunk.metadata.title,
                    page_number=chunk.metadata.page_number,
                    section=chunk.metadata.section,
                    chunk_index=chunk.metadata.chunk_index,
                    total_chunks=chunk.metadata.total_chunks,
                    token_count=chunk.token_count,
                    namespace=namespace,
                )
                session.add(record)
            await session.commit()

        logger.info("vector_store.stored", chunks=len(chunks), namespace=namespace)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        namespace: str = "default",
        threshold: float = settings.similarity_threshold,
    ) -> list[RetrievedChunk]:
        """Semantic similarity search using pgvector cosine distance."""
        query_embedding = await self.embeddings.embed_query(query)

        async with async_session() as session:
            # pgvector cosine distance: <=> operator (returns distance, lower = more similar)
            distance = ChunkRecord.embedding.cosine_distance(query_embedding).label("distance")
            stmt = (
                select(ChunkRecord, distance)
                .where(ChunkRecord.namespace == namespace)
                .order_by(distance)
                .limit(top_k)
            )
            results = await session.execute(stmt)
            rows = results.all()

        chunks: list[RetrievedChunk] = []
        for row in rows:
            record = row[0]
            dist = row[1]
            similarity = 1 - dist  # Convert distance to similarity

            if similarity < threshold:
                continue

            chunks.append(
                RetrievedChunk(
                    id=str(record.id),
                    content=str(record.content),
                    metadata=ChunkMetadata(
                        source=str(record.source),
                        document_id=str(record.document_id),
                        title=str(record.title or ""),
                        page_number=int(record.page_number) if record.page_number else None,
                        section=str(record.section or ""),
                        chunk_index=int(record.chunk_index or 0),
                        total_chunks=int(record.total_chunks or 0),
                    ),
                    score=round(float(similarity), 4),
                    search_method="vector",
                )
            )

        return chunks

    _BM25_MAX_CHUNKS = 1000

    async def keyword_search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        namespace: str = "default",
    ) -> list[RetrievedChunk]:
        """BM25 keyword search over chunk content."""
        async with async_session() as session:
            stmt = (
                select(ChunkRecord)
                .where(ChunkRecord.namespace == namespace)
                .order_by(ChunkRecord.created_at.desc())
                .limit(self._BM25_MAX_CHUNKS)
            )
            results = await session.execute(stmt)
            records = results.scalars().all()

        if not records:
            return []

        # BM25 scoring is CPU-bound; run in a thread to avoid blocking the event loop
        def _bm25_score() -> list[tuple[int, float]]:
            tokenized_corpus = [r.content.lower().split() for r in records]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            top_indices = np.argsort(scores)[::-1][:top_k]
            max_score = float(np.max(scores)) if float(np.max(scores)) > 0 else 1.0

            scored: list[tuple[int, float]] = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                scored.append((int(idx), float(scores[idx] / max_score)))
            return scored

        scored_indices = await asyncio.to_thread(_bm25_score)

        chunks: list[RetrievedChunk] = []
        for idx, normalized_score in scored_indices:
            record = records[idx]
            chunks.append(
                RetrievedChunk(
                    id=str(record.id),
                    content=str(record.content),
                    metadata=ChunkMetadata(
                        source=str(record.source),
                        document_id=str(record.document_id),
                        title=str(record.title or ""),
                        page_number=int(record.page_number) if record.page_number else None,
                        section=str(record.section or ""),
                        chunk_index=int(record.chunk_index or 0),
                        total_chunks=int(record.total_chunks or 0),
                    ),
                    score=round(normalized_score, 4),
                    search_method="bm25",
                )
            )

        return chunks

    async def hybrid_search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        namespace: str = "default",
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[RetrievedChunk]:
        """Combine vector and keyword search with reciprocal rank fusion."""
        hybrid_threshold = max(settings.similarity_threshold * 0.7, 0.3)
        vector_results = await self.vector_search(
            query, top_k=top_k * 2, namespace=namespace, threshold=hybrid_threshold
        )
        keyword_results = await self.keyword_search(
            query, top_k=top_k * 2, namespace=namespace
        )

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results):
            rrf_score = vector_weight / (k + rank + 1)
            scores[chunk.id] = scores.get(chunk.id, 0) + rrf_score
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(keyword_results):
            rrf_score = keyword_weight / (k + rank + 1)
            scores[chunk.id] = scores.get(chunk.id, 0) + rrf_score
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for chunk_id in sorted_ids:
            chunk = chunk_map[chunk_id]
            chunk.score = round(scores[chunk_id], 4)
            chunk.search_method = "hybrid"
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def list_documents(
        self, namespace: str | None = None
    ) -> list[Document]:
        """List all ingested documents."""
        async with async_session() as session:
            stmt = select(DocumentRecord)
            if namespace:
                stmt = stmt.where(DocumentRecord.namespace == namespace)
            stmt = stmt.order_by(DocumentRecord.created_at.desc())
            results = await session.execute(stmt)
            records = results.scalars().all()

        return [
            Document(
                id=str(r.id),
                filename=str(r.filename),
                document_type=DocumentType(str(r.document_type)),
                title=str(r.title or ""),
                namespace=str(r.namespace or "default"),
                chunk_count=int(r.chunk_count or 0),
                total_tokens=int(r.total_tokens or 0),
                created_at=r.created_at,  # type: ignore[arg-type]
            )
            for r in records
        ]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        async with async_session() as session:
            # Delete chunks
            await session.execute(
                delete(ChunkRecord).where(ChunkRecord.document_id == document_id)
            )
            # Delete document
            result = await session.execute(
                delete(DocumentRecord).where(DocumentRecord.id == document_id)
            )
            await session.commit()
            rowcount = getattr(result, "rowcount", 0) or 0
            return rowcount > 0

    async def get_document_count(self, namespace: str | None = None) -> int:
        """Get total document count."""
        async with async_session() as session:
            stmt = select(func.count()).select_from(DocumentRecord)
            if namespace:
                stmt = stmt.where(DocumentRecord.namespace == namespace)
            result = await session.execute(stmt)
            count: int = result.scalar_one()
            return count
