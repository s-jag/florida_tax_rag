"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.dependencies import (
    get_agent_graph,
    get_neo4j_client,
    get_weaviate_client,
)


@pytest.fixture
def mock_neo4j():
    """Create mock Neo4j client."""
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.run_query.return_value = []
    return mock


@pytest.fixture
def mock_weaviate():
    """Create mock Weaviate client."""
    mock = MagicMock()
    mock.health_check.return_value = True
    return mock


@pytest.fixture
def mock_agent_graph():
    """Create mock agent graph."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "final_answer": "The Florida sales tax rate is 6%.",
            "citations": [
                {
                    "doc_id": "statute:212.05",
                    "citation": "Fla. Stat. § 212.05",
                    "doc_type": "statute",
                    "text_snippet": "There is hereby levied...",
                }
            ],
            "temporally_valid_chunks": [
                {
                    "chunk_id": "chunk:212.05:001",
                    "doc_id": "statute:212.05",
                    "doc_type": "statute",
                    "citation": "Fla. Stat. § 212.05",
                    "text": "Sales tax provisions...",
                    "score": 0.95,
                }
            ],
            "confidence": 0.85,
            "validation_passed": True,
            "errors": [],
            "reasoning_steps": [
                {"node": "decompose", "description": "Analyzed query"},
                {"node": "retrieve", "description": "Found 5 relevant chunks"},
            ],
        }
    )
    return mock


@pytest.fixture
def client(mock_neo4j, mock_weaviate, mock_agent_graph):
    """Create test client with mocked dependencies."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_neo4j_client] = lambda: mock_neo4j
    app.dependency_overrides[get_weaviate_client] = lambda: mock_weaviate
    app.dependency_overrides[get_agent_graph] = lambda: mock_agent_graph

    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Florida Tax RAG API"
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_all_healthy(self, client, mock_neo4j, mock_weaviate):
        """Test health check when all services healthy."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert len(data["services"]) == 2
        assert all(s["healthy"] for s in data["services"])

    def test_health_neo4j_down(self, client, mock_neo4j, mock_weaviate):
        """Test health check when Neo4j is down."""
        mock_neo4j.health_check.return_value = False

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    def test_health_all_down(self, client, mock_neo4j, mock_weaviate):
        """Test health check when all services are down."""
        mock_neo4j.health_check.return_value = False
        mock_weaviate.health_check.return_value = False

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_health_neo4j_error(self, client, mock_neo4j, mock_weaviate):
        """Test health check when Neo4j throws an error."""
        mock_neo4j.health_check.side_effect = Exception("Connection refused")

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        neo4j_service = next(s for s in data["services"] if s["name"] == "neo4j")
        assert neo4j_service["healthy"] is False
        assert "Connection refused" in neo4j_service["error"]


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_success(self, client, mock_agent_graph):
        """Test successful query execution."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the Florida sales tax rate?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "The Florida sales tax rate is 6%."
        assert data["confidence"] == 0.85
        assert data["validation_passed"] is True
        assert len(data["citations"]) == 1
        assert data["citations"][0]["doc_id"] == "statute:212.05"

    def test_query_with_options(self, client, mock_agent_graph):
        """Test query with custom options."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the Florida sales tax rate?",
                "options": {
                    "doc_types": ["statute", "rule"],
                    "tax_year": 2024,
                    "include_reasoning": True,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["reasoning_steps"] is not None
        assert len(data["reasoning_steps"]) >= 1

    def test_query_validation_error_short(self, client):
        """Test query with input too short."""
        response = client.post(
            "/api/v1/query",
            json={"query": "ab"},  # Too short
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"] == "VALIDATION_ERROR"

    def test_query_validation_error_missing(self, client):
        """Test query with missing required field."""
        response = client.post(
            "/api/v1/query",
            json={},  # Missing query
        )

        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_query_includes_sources(self, client, mock_agent_graph):
        """Test query response includes sources."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the Florida sales tax rate?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["chunk_id"] == "chunk:212.05:001"
        assert data["sources"][0]["relevance_score"] == 0.95

    def test_query_includes_processing_time(self, client, mock_agent_graph):
        """Test query response includes processing time."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the Florida sales tax rate?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_query_includes_request_id(self, client, mock_agent_graph):
        """Test query response includes request ID."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the Florida sales tax rate?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0


class TestSourceEndpoint:
    """Tests for /sources/{chunk_id} endpoint."""

    def test_get_chunk_success(self, client, mock_neo4j):
        """Test successful chunk retrieval."""
        mock_neo4j.run_query.return_value = [
            {
                "c": {
                    "id": "chunk:212.05:001",
                    "level": "parent",
                    "text": "Sales tax provisions...",
                    "ancestry": "Chapter 212 > § 212.05",
                    "citation": "Fla. Stat. § 212.05",
                },
                "d": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                },
                "parent_id": None,
                "child_ids": ["chunk:212.05:001:001"],
            }
        ]

        with patch("src.api.routes.get_citing_documents") as mock_citing, \
             patch("src.api.routes.get_cited_documents") as mock_cited:
            mock_citing.return_value = []
            mock_cited.return_value = []

            response = client.get("/api/v1/sources/chunk:212.05:001")

            assert response.status_code == 200
            data = response.json()
            assert data["chunk_id"] == "chunk:212.05:001"
            assert data["doc_type"] == "statute"

    def test_get_chunk_not_found(self, client, mock_neo4j):
        """Test chunk not found."""
        mock_neo4j.run_query.return_value = [{"c": None, "d": None}]

        response = client.get("/api/v1/sources/nonexistent")

        assert response.status_code == 404


class TestStatuteEndpoint:
    """Tests for /statute/{section} endpoint."""

    def test_get_statute_success(self, client, mock_neo4j):
        """Test successful statute retrieval."""
        # Mock the interpretation chain query
        with patch("src.api.routes.get_interpretation_chain") as mock_chain:
            mock_statute = MagicMock()
            mock_statute.model_dump.return_value = {
                "id": "statute:212.05",
                "doc_type": "statute",
                "title": "Sales Tax",
                "full_citation": "Fla. Stat. § 212.05",
                "section": "212.05",
            }

            mock_result = MagicMock()
            mock_result.statute = mock_statute
            mock_result.implementing_rules = []
            mock_result.interpreting_cases = []
            mock_result.interpreting_taas = []

            mock_chain.return_value = mock_result

            response = client.get("/api/v1/statute/212.05")

            assert response.status_code == 200
            data = response.json()
            assert data["statute"]["section"] == "212.05"

    def test_get_statute_not_found(self, client, mock_neo4j):
        """Test statute not found."""
        with patch("src.api.routes.get_interpretation_chain") as mock_chain:
            mock_chain.return_value = None

            response = client.get("/api/v1/statute/999.999")

            assert response.status_code == 404


class TestRelatedDocumentsEndpoint:
    """Tests for /graph/{doc_id}/related endpoint."""

    def test_get_related_documents(self, client, mock_neo4j):
        """Test getting related documents."""
        with patch("src.api.routes.get_citing_documents") as mock_citing, \
             patch("src.api.routes.get_cited_documents") as mock_cited:
            mock_citing.return_value = []
            mock_cited.return_value = []

            response = client.get("/api/v1/graph/case:123/related")

            assert response.status_code == 200
            data = response.json()
            assert data["doc_id"] == "case:123"
            assert "citing_documents" in data
            assert "cited_documents" in data

    def test_get_related_statute_includes_chain(self, client, mock_neo4j):
        """Test statute related documents include interpretation chain."""
        with patch("src.api.routes.get_citing_documents") as mock_citing, \
             patch("src.api.routes.get_cited_documents") as mock_cited, \
             patch("src.api.routes.get_interpretation_chain") as mock_chain:
            mock_citing.return_value = []
            mock_cited.return_value = []

            mock_result = MagicMock()
            mock_result.implementing_rules = []
            mock_result.interpreting_cases = []
            mock_result.interpreting_taas = []
            mock_chain.return_value = mock_result

            response = client.get("/api/v1/graph/statute:212.05/related")

            assert response.status_code == 200
            data = response.json()
            assert "interpretation_chain" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_validation_error_returns_details(self, client):
        """Test validation errors include field details."""
        response = client.post(
            "/api/v1/query",
            json={"query": "x"},  # Too short
        )

        assert response.status_code == 422
        data = response.json()
        assert len(data["details"]) >= 1
        assert any("query" in d.get("field", "") for d in data["details"])

    def test_invalid_timeout_value(self, client):
        """Test invalid timeout value is rejected."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the sales tax rate?",
                "options": {"timeout_seconds": 1000},  # Too high
            },
        )

        assert response.status_code == 422

    def test_invalid_tax_year(self, client):
        """Test invalid tax year is rejected."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the sales tax rate?",
                "options": {"tax_year": 1800},  # Too old
            },
        )

        assert response.status_code == 422
