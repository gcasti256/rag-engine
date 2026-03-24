"""Database connection and schema management with SQLAlchemy + pgvector."""

from __future__ import annotations

from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from rag_engine.config import settings


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """Stores document metadata."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    title = Column(String, default="")
    namespace = Column(String, default="default", index=True)
    chunk_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


class ChunkRecord(Base):
    """Stores text chunks with vector embeddings."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimensions))
    source = Column(String, nullable=False)
    title = Column(String, default="")
    page_number = Column(Integer, nullable=True)
    section = Column(String, default="")
    chunk_index = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    namespace = Column(String, default="default", index=True)
    relevance_boost = Column(Float, default=1.0)
    created_at = Column(DateTime, server_default=func.now())


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Lazily create the async engine."""
    return create_async_engine(
        settings.database_url,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get async session factory."""
    return async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)


def async_session() -> AsyncSession:
    """Get an async session context manager."""
    factory = get_session_factory()
    return factory()


async def init_db() -> None:
    """Create tables and install pgvector extension."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            sa_text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get a database session."""
    session: AsyncSession
    async with get_session_factory()() as session:
        return session
