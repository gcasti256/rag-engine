"""RAG query pipeline: retrieve → augment → generate with citations."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog
import tiktoken
from openai import AsyncOpenAI

from rag_engine.config import settings
from rag_engine.models import Citation, QueryResult, RetrievedChunk
from rag_engine.storage import VectorStore

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context documents. Follow these rules strictly:

1. Base your answer exclusively on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, say "I don't have enough information to answer this question based on the available documents."
3. Cite your sources using [Source N] notation, where N corresponds to the context chunk number.
4. Be specific and factual. Prefer direct quotes when the context contains relevant passages.
5. Structure your answer clearly with paragraphs for complex topics.
6. If data involves numbers, percentages, or financial figures, quote them exactly as they appear in the sources."""

CONTEXT_TEMPLATE = """--- Context Document {index} ---
Source: {source}
{section_info}{page_info}
Content:
{content}
---"""


class QueryPipeline:
    """Orchestrates retrieval, context assembly, and LLM generation."""

    def __init__(
        self,
        vector_store: VectorStore,
        model: str = settings.openai_completion_model,
        api_key: str | None = None,
    ):
        self.vector_store = vector_store
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def query(
        self,
        question: str,
        top_k: int = settings.default_top_k,
        namespace: str = "default",
        search_method: str = "hybrid",
    ) -> QueryResult:
        """Execute a full RAG query: retrieve → augment → generate."""
        start = time.monotonic()

        # Retrieve relevant chunks
        chunks = await self._retrieve(question, top_k, namespace, search_method)
        retrieval_time = (time.monotonic() - start) * 1000

        if not chunks:
            return QueryResult(
                answer="I don't have enough information to answer this question. No relevant documents were found.",
                citations=[],
                confidence=0.0,
                query=question,
                model=self.model,
                retrieval_time_ms=retrieval_time,
            )

        # Build context
        context, used_chunks = self._build_context(chunks)

        # Generate answer
        gen_start = time.monotonic()
        answer, total_tokens = await self._generate(question, context)
        generation_time = (time.monotonic() - gen_start) * 1000

        # Build citations
        citations = self._build_citations(used_chunks)

        # Compute confidence from average relevance scores
        avg_score = sum(c.score for c in used_chunks) / len(used_chunks)
        confidence = min(avg_score, 1.0)

        return QueryResult(
            answer=answer,
            citations=citations,
            confidence=round(confidence, 3),
            query=question,
            model=self.model,
            total_tokens_used=total_tokens,
            retrieval_time_ms=round(retrieval_time, 1),
            generation_time_ms=round(generation_time, 1),
        )

    async def query_stream(
        self,
        question: str,
        top_k: int = settings.default_top_k,
        namespace: str = "default",
        search_method: str = "hybrid",
    ) -> AsyncIterator[str]:
        """Stream a RAG query response token by token."""
        chunks = await self._retrieve(question, top_k, namespace, search_method)

        if not chunks:
            yield "I don't have enough information to answer this question. No relevant documents were found."
            return

        context, used_chunks = self._build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
            stream=True,
        )

        async for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

        # Yield citations at the end
        yield "\n\n---\n**Sources:**\n"
        for i, chunk in enumerate(used_chunks, 1):
            source = chunk.metadata.source
            page = f", Page {chunk.metadata.page_number}" if chunk.metadata.page_number else ""
            section = f", Section: {chunk.metadata.section}" if chunk.metadata.section else ""
            yield f"- [Source {i}] {source}{page}{section}\n"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        namespace: str,
        search_method: str,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using the specified search method."""
        if search_method == "vector":
            return await self.vector_store.vector_search(query, top_k, namespace)
        elif search_method == "keyword":
            return await self.vector_store.keyword_search(query, top_k, namespace)
        else:
            return await self.vector_store.hybrid_search(query, top_k, namespace)

    def _build_context(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        """Build context string from chunks, respecting token limits."""
        context_parts: list[str] = []
        used_chunks: list[RetrievedChunk] = []
        total_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            section_info = f"Section: {chunk.metadata.section}\n" if chunk.metadata.section else ""
            page_info = f"Page: {chunk.metadata.page_number}\n" if chunk.metadata.page_number else ""

            block = CONTEXT_TEMPLATE.format(
                index=i,
                source=chunk.metadata.source,
                section_info=section_info,
                page_info=page_info,
                content=chunk.content,
            )

            block_tokens = len(self.encoder.encode(block))
            if total_tokens + block_tokens > settings.max_context_tokens:
                break

            context_parts.append(block)
            used_chunks.append(chunk)
            total_tokens += block_tokens

        return "\n".join(context_parts), used_chunks

    async def _generate(self, question: str, context: str) -> tuple[str, int]:
        """Generate answer using LLM with context."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
            max_tokens=2048,
        )

        answer = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0

        return answer, total_tokens

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Build citation objects from retrieved chunks."""
        return [
            Citation(
                chunk_id=chunk.id,
                source=chunk.metadata.source,
                content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                page_number=chunk.metadata.page_number,
                section=chunk.metadata.section,
                relevance_score=chunk.score,
            )
            for chunk in chunks
        ]
