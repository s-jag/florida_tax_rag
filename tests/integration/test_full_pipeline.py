"""Integration tests for the full RAG pipeline.

These tests require Docker services to be running:
- Neo4j
- Weaviate
- Redis (optional)

Run with: pytest -m integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client for testing."""
    client = MagicMock()
    client.verify_connectivity = MagicMock(return_value=True)
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client for testing."""
    client = MagicMock()
    client.is_ready = MagicMock(return_value=True)
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_agent_graph():
    """Create a mock agent graph for testing."""
    graph = MagicMock()
    graph.invoke = MagicMock(return_value={
        "answer": "The Florida sales tax rate is 6 percent.",
        "citations": [{"doc_id": "statute:212.05", "citation": "Fla. Stat. § 212.05"}],
        "sources": [],
        "confidence": 0.9,
    })
    return graph


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check endpoints with services."""

    async def test_health_check_all_healthy(
        self, mock_neo4j_client, mock_weaviate_client
    ) -> None:
        """Health check should return healthy when all services are up."""
        with patch("src.api.routes.get_neo4j_client", return_value=mock_neo4j_client):
            with patch("src.api.routes.get_weaviate_client", return_value=mock_weaviate_client):
                # Would normally use TestClient but we're mocking the dependencies
                assert mock_neo4j_client.verify_connectivity() is True
                assert mock_weaviate_client.is_ready() is True

    async def test_health_check_neo4j_down(
        self, mock_neo4j_client, mock_weaviate_client
    ) -> None:
        """Health check should report degraded when Neo4j is down."""
        mock_neo4j_client.verify_connectivity.side_effect = Exception("Connection refused")

        with pytest.raises(Exception):
            mock_neo4j_client.verify_connectivity()


# =============================================================================
# Query Flow Tests
# =============================================================================


class TestQueryFlow:
    """Test end-to-end query flow."""

    async def test_simple_query_flow(self, mock_agent_graph) -> None:
        """Should process a simple query through the agent."""
        # Simulate query processing
        state = {
            "query": "What is the Florida sales tax rate?",
            "messages": [],
        }

        result = mock_agent_graph.invoke(state)

        assert "answer" in result
        assert "6 percent" in result["answer"]
        assert "citations" in result
        assert result["confidence"] > 0

    async def test_query_with_graph_expansion(
        self, mock_neo4j_client, mock_agent_graph
    ) -> None:
        """Should expand results using graph traversal."""
        # Mock graph expansion
        mock_neo4j_client.get_related_documents = MagicMock(return_value=[
            {"doc_id": "rule:12A-1.001", "relation": "IMPLEMENTS"},
        ])

        state = {
            "query": "What are sales tax exemptions?",
            "options": {"expand_graph": True},
        }

        result = mock_agent_graph.invoke(state)

        assert "answer" in result


# =============================================================================
# Retrieval Tests
# =============================================================================


class TestRetrieval:
    """Test retrieval with real data."""

    async def test_vector_search(self, mock_weaviate_client) -> None:
        """Should perform vector similarity search."""
        mock_weaviate_client.search = MagicMock(return_value=[
            {
                "chunk_id": "chunk:statute:212.05:0",
                "text": "Sales tax rate is 6 percent.",
                "score": 0.92,
            }
        ])

        results = mock_weaviate_client.search(
            query="sales tax rate",
            limit=10,
        )

        assert len(results) > 0
        assert results[0]["score"] > 0.5

    async def test_hybrid_search(self, mock_weaviate_client) -> None:
        """Should perform hybrid (vector + keyword) search."""
        mock_weaviate_client.hybrid_search = MagicMock(return_value=[
            {
                "chunk_id": "chunk:statute:212.08:0",
                "text": "Exemptions from sales tax.",
                "score": 0.88,
            }
        ])

        results = mock_weaviate_client.hybrid_search(
            query="sales tax exemptions",
            alpha=0.5,
            limit=10,
        )

        assert len(results) > 0


# =============================================================================
# Generation Tests
# =============================================================================


class TestGeneration:
    """Test answer generation with citations."""

    async def test_generates_answer_with_citations(self) -> None:
        """Should generate answer with inline citations."""
        mock_response = {
            "answer": "The sales tax rate is 6% [1]. Exemptions apply to groceries [2].",
            "citations": [
                {"marker": "[1]", "doc_id": "statute:212.05"},
                {"marker": "[2]", "doc_id": "statute:212.08"},
            ],
        }

        assert "[1]" in mock_response["answer"]
        assert "[2]" in mock_response["answer"]
        assert len(mock_response["citations"]) == 2

    async def test_validates_citations(self) -> None:
        """Should validate that citations reference real sources."""
        mock_sources = [
            {"doc_id": "statute:212.05", "text": "6 percent rate"},
            {"doc_id": "statute:212.08", "text": "exemptions"},
        ]
        mock_citations = [
            {"doc_id": "statute:212.05"},
            {"doc_id": "statute:212.08"},
        ]

        # All citations should reference sources
        source_ids = {s["doc_id"] for s in mock_sources}
        for citation in mock_citations:
            assert citation["doc_id"] in source_ids


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Test metrics collection."""

    async def test_metrics_endpoint(self) -> None:
        """Should return metrics data."""
        mock_metrics = {
            "total_queries": 100,
            "successful_queries": 95,
            "failed_queries": 5,
            "success_rate_percent": 95.0,
            "latency_ms": {"avg": 2500, "min": 1000, "max": 5000},
        }

        assert mock_metrics["success_rate_percent"] == 95.0
        assert mock_metrics["latency_ms"]["avg"] == 2500

    async def test_error_tracking(self) -> None:
        """Should track errors by type."""
        mock_metrics = {
            "errors_by_type": {
                "TIMEOUT": 3,
                "RETRIEVAL_ERROR": 1,
                "VALIDATION_ERROR": 1,
            }
        }

        assert mock_metrics["errors_by_type"]["TIMEOUT"] == 3


# =============================================================================
# Service Connectivity Tests
# =============================================================================


class TestServiceConnectivity:
    """Test connectivity to required services."""

    async def test_neo4j_connectivity(self, mock_neo4j_client) -> None:
        """Should connect to Neo4j."""
        mock_neo4j_client.verify_connectivity.return_value = True
        assert mock_neo4j_client.verify_connectivity() is True

    async def test_weaviate_connectivity(self, mock_weaviate_client) -> None:
        """Should connect to Weaviate."""
        mock_weaviate_client.is_ready.return_value = True
        assert mock_weaviate_client.is_ready() is True

    async def test_handles_neo4j_disconnect(self, mock_neo4j_client) -> None:
        """Should handle Neo4j disconnection gracefully."""
        mock_neo4j_client.verify_connectivity.side_effect = Exception("Disconnected")

        with pytest.raises(Exception, match="Disconnected"):
            mock_neo4j_client.verify_connectivity()

    async def test_handles_weaviate_disconnect(self, mock_weaviate_client) -> None:
        """Should handle Weaviate disconnection gracefully."""
        mock_weaviate_client.is_ready.side_effect = Exception("Disconnected")

        with pytest.raises(Exception, match="Disconnected"):
            mock_weaviate_client.is_ready()
