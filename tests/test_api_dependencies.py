"""Tests for src/api/dependencies.py."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request

from src.api.dependencies import (
    cleanup_clients,
    generate_request_id,
    get_agent_graph,
    get_neo4j_client,
    get_request_id,
    get_weaviate_client,
    request_id_var,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_request() -> MagicMock:
    """Create a mock Starlette request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    return request


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear LRU caches before and after each test."""
    get_neo4j_client.cache_clear()
    get_weaviate_client.cache_clear()
    get_agent_graph.cache_clear()
    yield
    get_neo4j_client.cache_clear()
    get_weaviate_client.cache_clear()
    get_agent_graph.cache_clear()


# =============================================================================
# Generate Request ID Tests
# =============================================================================


class TestGenerateRequestId:
    """Test generate_request_id function."""

    def test_returns_uuid_string(self) -> None:
        """Should return a valid UUID string."""
        result = generate_request_id()

        # Should be a valid UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_returns_unique_ids(self) -> None:
        """Should return unique IDs on each call."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100


# =============================================================================
# Get Request ID Tests
# =============================================================================


class TestGetRequestId:
    """Test get_request_id function."""

    async def test_uses_header_if_present(self, mock_request: MagicMock) -> None:
        """Should use X-Request-ID from headers if present."""
        mock_request.headers = {"X-Request-ID": "test-id-from-header"}

        result = await get_request_id(mock_request)

        assert result == "test-id-from-header"

    async def test_generates_new_id_if_no_header(self, mock_request: MagicMock) -> None:
        """Should generate new ID if header not present."""
        mock_request.headers = {}

        result = await get_request_id(mock_request)

        # Should be a valid UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    async def test_sets_context_var(self, mock_request: MagicMock) -> None:
        """Should set the request_id context var."""
        mock_request.headers = {"X-Request-ID": "context-test-id"}

        await get_request_id(mock_request)

        assert request_id_var.get() == "context-test-id"


# =============================================================================
# Neo4j Client Singleton Tests
# =============================================================================


class TestGetNeo4jClient:
    """Test get_neo4j_client function."""

    def test_returns_neo4j_client(self) -> None:
        """Should return a Neo4jClient instance."""
        with patch("src.api.dependencies.Neo4jClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance

            result = get_neo4j_client()

            assert result == mock_instance
            mock_client_class.assert_called_once()

    def test_returns_singleton(self) -> None:
        """Should return same instance on multiple calls."""
        with patch("src.api.dependencies.Neo4jClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance

            result1 = get_neo4j_client()
            result2 = get_neo4j_client()

            assert result1 is result2
            # Should only create one instance
            mock_client_class.assert_called_once()


# =============================================================================
# Weaviate Client Singleton Tests
# =============================================================================


class TestGetWeaviateClient:
    """Test get_weaviate_client function."""

    def test_returns_weaviate_client(self) -> None:
        """Should return a WeaviateClient instance."""
        with patch("src.api.dependencies.WeaviateClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance

            result = get_weaviate_client()

            assert result == mock_instance
            mock_client_class.assert_called_once()

    def test_returns_singleton(self) -> None:
        """Should return same instance on multiple calls."""
        with patch("src.api.dependencies.WeaviateClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance

            result1 = get_weaviate_client()
            result2 = get_weaviate_client()

            assert result1 is result2
            mock_client_class.assert_called_once()


# =============================================================================
# Agent Graph Singleton Tests
# =============================================================================


class TestGetAgentGraph:
    """Test get_agent_graph function."""

    def test_returns_agent_graph(self) -> None:
        """Should return a compiled agent graph."""
        with patch("src.api.dependencies.create_tax_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            result = get_agent_graph()

            assert result == mock_graph
            mock_create.assert_called_once()

    def test_returns_singleton(self) -> None:
        """Should return same instance on multiple calls."""
        with patch("src.api.dependencies.create_tax_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            result1 = get_agent_graph()
            result2 = get_agent_graph()

            assert result1 is result2
            mock_create.assert_called_once()


# =============================================================================
# Cleanup Clients Tests
# =============================================================================


class TestCleanupClients:
    """Test cleanup_clients function."""

    async def test_clears_caches(self) -> None:
        """Should clear all LRU caches."""
        # Create cached instances first
        with patch("src.api.dependencies.Neo4jClient"):
            with patch("src.api.dependencies.WeaviateClient"):
                with patch("src.api.dependencies.create_tax_agent_graph"):
                    get_neo4j_client()
                    get_weaviate_client()
                    get_agent_graph()

        # Check caches have entries
        assert get_neo4j_client.cache_info().currsize > 0
        assert get_weaviate_client.cache_info().currsize > 0
        assert get_agent_graph.cache_info().currsize > 0

        # Cleanup
        await cleanup_clients()

        # Caches should be cleared
        assert get_neo4j_client.cache_info().currsize == 0
        assert get_weaviate_client.cache_info().currsize == 0
        assert get_agent_graph.cache_info().currsize == 0
