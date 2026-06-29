"""
Integration tests for the FinRAG API
Run with: pytest tests/integration/ -v
"""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_vector_service():
    """Mock vector store to avoid needing ChromaDB in tests."""
    service = MagicMock()
    service.initialize = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.add_chunks = AsyncMock(return_value=5)
    service.delete_document_chunks = AsyncMock(return_value=5)
    service.get_collection_stats = MagicMock(return_value={
        "available": True,
        "collection": "test_collection",
        "document_count": 10,
    })
    return service


@pytest.fixture
def app(mock_vector_service):
    """Create test app with mocked services."""
    with patch("app.services.vector_store.VectorStoreService") as MockVS:
        MockVS.return_value = mock_vector_service
        with patch("app.core.database.init_db", AsyncMock()):
            from app.main import app as fastapi_app
            fastapi_app.state.vector_service = mock_vector_service
            yield fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/api/v1/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "FinRAG" in data["name"]

    def test_health_endpoint_exists(self, client):
        resp = client.get("/api/v1/health")
        # Should return 200 even if Ollama is down (degraded status)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")


class TestDocumentEndpoints:
    def test_list_documents_empty(self, client):
        resp = client.get("/api/v1/documents/")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_non_pdf_rejected(self, client):
        file_content = b"not a pdf"
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_get_nonexistent_document(self, client):
        resp = client.get("/api/v1/documents/99999")
        assert resp.status_code == 404

    def test_delete_nonexistent_document(self, client):
        resp = client.delete("/api/v1/documents/99999")
        assert resp.status_code == 404


class TestQueryEndpoint:
    def test_query_too_short_rejected(self, client):
        resp = client.post("/api/v1/query/", json={"question": "ok"})
        assert resp.status_code == 422  # Pydantic validation error

    def test_query_history_endpoint(self, client):
        resp = client.get("/api/v1/query/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "queries" in data
