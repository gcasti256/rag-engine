"""Integration tests for the full ingestion + query pipeline.

These require a running PostgreSQL with pgvector and valid OpenAI key.
Run with: pytest tests/integration/ -m integration
"""

from __future__ import annotations

import pytest

from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.query.pipeline import QueryPipeline
from rag_engine.storage import VectorStore


@pytest.mark.integration
async def test_ingest_and_query() -> None:
    """Full pipeline: ingest a document and query it."""
    store = VectorStore()
    ingestion = IngestionPipeline(vector_store=store)
    query_pipeline = QueryPipeline(vector_store=store)

    # Ingest a test document
    content = b"""# Q3 Financial Summary

American Express reported net revenue of $15.4 billion in Q3 2024,
representing a 12% increase year-over-year. Card member spending
grew 8% on an FX-adjusted basis.

## Key Metrics
- Net income: $2.5 billion
- Earnings per share: $3.49
- Card member loans: $134.5 billion
- Net write-off rate: 2.2%
"""

    result = await ingestion.ingest(
        content=content,
        filename="q3_report.md",
        namespace="test-integration",
    )

    assert result.chunk_count > 0
    assert result.total_tokens > 0

    # Query the ingested document
    query_result = await query_pipeline.query(
        question="What was American Express net revenue in Q3?",
        namespace="test-integration",
    )

    assert query_result.answer
    assert "$15.4 billion" in query_result.answer or "15.4" in query_result.answer
    assert len(query_result.citations) > 0

    # Cleanup
    await store.delete_document(result.document_id)
