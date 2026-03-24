"""Tests for API routes (mocked dependencies)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.models import Document, DocumentType, IngestResponse, QueryResult


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked lifespan."""
    app = create_app()
    # Override lifespan to skip DB init
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint should return status."""
        with patch("rag_engine.api.routes._vector_store") as mock_store:
            mock_store.get_document_count = AsyncMock(return_value=5)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "1.0.0"


class TestIngestEndpoint:
    def test_ingest_missing_file(self, client: TestClient) -> None:
        """Ingest should fail without a file."""
        response = client.post("/ingest")
        assert response.status_code == 422

    def test_ingest_success(self, client: TestClient) -> None:
        """Ingest should process a valid file."""
        mock_response = IngestResponse(
            document_id="doc-123",
            filename="test.md",
            chunk_count=5,
            total_tokens=250,
            namespace="default",
        )
        with patch("rag_engine.api.routes._ingestion_pipeline") as mock_pipeline:
            mock_pipeline.ingest = AsyncMock(return_value=mock_response)
            response = client.post(
                "/ingest",
                files={"file": ("test.md", b"# Test\nContent here", "text/markdown")},
                data={"namespace": "default"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == "doc-123"
            assert data["chunk_count"] == 5


class TestQueryEndpoint:
    def test_query_empty_question(self, client: TestClient) -> None:
        """Query should reject empty questions."""
        response = client.post("/query", data={"question": ""})
        assert response.status_code in (400, 422)

    def test_query_success(self, client: TestClient) -> None:
        """Query should return a result."""
        mock_result = QueryResult(
            answer="Revenue was $4.2B",
            citations=[],
            confidence=0.85,
            query="What was revenue?",
            model="gpt-4o-mini",
        )
        with patch("rag_engine.api.routes._query_pipeline") as mock_pipeline:
            mock_pipeline.query = AsyncMock(return_value=mock_result)
            response = client.post(
                "/query",
                data={"question": "What was revenue?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Revenue was $4.2B"
            assert data["confidence"] == 0.85

    def test_query_invalid_search_method(self, client: TestClient) -> None:
        """Query should reject invalid search methods."""
        response = client.post(
            "/query",
            data={"question": "test?", "search_method": "invalid"},
        )
        assert response.status_code == 400


class TestDocumentsEndpoint:
    def test_list_documents(self, client: TestClient) -> None:
        """Should list documents."""
        mock_docs = [
            Document(
                id="doc-1",
                filename="report.pdf",
                document_type=DocumentType.PDF,
                title="Q3 Report",
            )
        ]
        with patch("rag_engine.api.routes._vector_store") as mock_store:
            mock_store.list_documents = AsyncMock(return_value=mock_docs)
            response = client.get("/documents")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["filename"] == "report.pdf"

    def test_delete_document(self, client: TestClient) -> None:
        """Should delete a document."""
        with patch("rag_engine.api.routes._vector_store") as mock_store:
            mock_store.delete_document = AsyncMock(return_value=True)
            response = client.delete("/documents/doc-123")
            assert response.status_code == 200

    def test_delete_nonexistent(self, client: TestClient) -> None:
        """Should 404 on missing document."""
        with patch("rag_engine.api.routes._vector_store") as mock_store:
            mock_store.delete_document = AsyncMock(return_value=False)
            response = client.delete("/documents/nonexistent")
            assert response.status_code == 404
