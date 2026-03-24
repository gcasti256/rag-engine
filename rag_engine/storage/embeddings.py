"""Embedding generation via OpenAI API with batching."""

from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from rag_engine.config import settings

logger = structlog.get_logger()

# OpenAI recommends max 2048 inputs per batch
_MAX_BATCH_SIZE = 2048


class EmbeddingService:
    """Generate embeddings using OpenAI's embedding API."""

    def __init__(
        self,
        model: str = settings.openai_embedding_model,
        api_key: str | None = None,
    ):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "embeddings.batch",
                batch_size=len(batch),
                total_tokens=response.usage.total_tokens,
            )

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]
