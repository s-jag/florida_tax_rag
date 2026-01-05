"""Tests for Neo4j client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.client import Neo4jClient, Neo4jConfig


class TestNeo4jConfig:
    """Tests for Neo4jConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Neo4jConfig(password="test")
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.database == "neo4j"
        assert config.max_connection_pool_size == 50

    def test_custom_values(self):
        """Test custom configuration."""
        config = Neo4jConfig(
            uri="bolt://custom:7687",
            user="admin",
            password="secret",
            database="florida_tax",
        )
        assert config.uri == "bolt://custom:7687"
        assert config.user == "admin"
        assert config.database == "florida_tax"


class TestNeo4jClient:
    """Tests for Neo4jClient."""

    def test_context_manager(self):
        """Test client as context manager."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        with patch.object(client, "close") as mock_close:
            with client:
                pass
            mock_close.assert_called_once()

    def test_health_check_failure(self):
        """Test health check when Neo4j is unavailable."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        # Mock the session to raise ServiceUnavailable
        mock_session = MagicMock()
        from neo4j.exceptions import ServiceUnavailable

        mock_session.run.side_effect = ServiceUnavailable("Connection refused")

        with patch.object(client, "session") as mock_session_cm:
            mock_session_cm.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_session_cm.return_value.__exit__ = MagicMock(return_value=False)

            assert client.health_check() is False

    def test_run_query_returns_list(self):
        """Test that run_query returns a list of dicts."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        mock_session = MagicMock()
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {"name": "Test1"}
        mock_record2 = MagicMock()
        mock_record2.data.return_value = {"name": "Test2"}
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([mock_record1, mock_record2])
        )
        mock_session.run.return_value = mock_result

        with patch.object(client, "session") as mock_session_cm:
            mock_session_cm.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_session_cm.return_value.__exit__ = MagicMock(return_value=False)

            results = client.run_query("MATCH (n) RETURN n.name AS name")

            assert len(results) == 2
            assert results[0] == {"name": "Test1"}
            assert results[1] == {"name": "Test2"}

    def test_run_write_returns_counters(self):
        """Test that run_write returns summary counters."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        mock_session = MagicMock()
        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 5
        mock_summary.counters.nodes_deleted = 0
        mock_summary.counters.relationships_created = 3
        mock_summary.counters.relationships_deleted = 0
        mock_summary.counters.properties_set = 10
        mock_result = MagicMock()
        mock_result.consume.return_value = mock_summary
        mock_session.run.return_value = mock_result

        with patch.object(client, "session") as mock_session_cm:
            mock_session_cm.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_session_cm.return_value.__exit__ = MagicMock(return_value=False)

            result = client.run_write("CREATE (n:Test)")

            assert result["nodes_created"] == 5
            assert result["relationships_created"] == 3
            assert result["properties_set"] == 10

    def test_batch_write_splits_correctly(self):
        """Test that batch_write splits items correctly."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        items = [{"id": i} for i in range(10)]

        call_count = 0
        batch_sizes = []

        def mock_run_write(query, params):
            nonlocal call_count
            call_count += 1
            batch_sizes.append(len(params["items"]))
            return {
                "nodes_created": len(params["items"]),
                "relationships_created": 0,
                "properties_set": 0,
            }

        with patch.object(client, "run_write", side_effect=mock_run_write):
            result = client.batch_write(
                "UNWIND $items AS item CREATE (n:Test {id: item.id})",
                "items",
                items,
                batch_size=3,
            )

        # 10 items / 3 per batch = 4 batches (3, 3, 3, 1)
        assert call_count == 4
        assert batch_sizes == [3, 3, 3, 1]
        assert result["nodes_created"] == 10

    def test_batch_write_empty_list(self):
        """Test batch_write with empty list."""
        config = Neo4jConfig(password="test")
        client = Neo4jClient(config)

        with patch.object(client, "run_write") as mock_run_write:
            result = client.batch_write(
                "UNWIND $items AS item CREATE (n:Test)",
                "items",
                [],
                batch_size=100,
            )

            # Should not call run_write for empty list
            mock_run_write.assert_not_called()
            assert result["nodes_created"] == 0


class TestNeo4jSchema:
    """Tests for schema module."""

    def test_get_schema_queries(self):
        """Test that get_schema_queries returns constraint and index queries."""
        from src.graph.schema import get_schema_queries

        queries = get_schema_queries()

        assert len(queries) > 0
        # Should include constraints
        constraint_queries = [q for q in queries if "CONSTRAINT" in q]
        assert len(constraint_queries) >= 2

        # Should include indexes
        index_queries = [q for q in queries if "INDEX" in q]
        assert len(index_queries) >= 4

    def test_node_labels(self):
        """Test NodeLabel enum."""
        from src.graph.schema import NodeLabel

        assert NodeLabel.DOCUMENT.value == "Document"
        assert NodeLabel.STATUTE.value == "Statute"
        assert NodeLabel.RULE.value == "Rule"
        assert NodeLabel.CASE.value == "Case"
        assert NodeLabel.TAA.value == "TAA"
        assert NodeLabel.CHUNK.value == "Chunk"

    def test_edge_types(self):
        """Test EdgeType enum."""
        from src.graph.schema import EdgeType

        assert EdgeType.CITES.value == "CITES"
        assert EdgeType.IMPLEMENTS.value == "IMPLEMENTS"
        assert EdgeType.AUTHORITY.value == "AUTHORITY"
        assert EdgeType.HAS_CHUNK.value == "HAS_CHUNK"
        assert EdgeType.CHILD_OF.value == "CHILD_OF"


class TestNeo4jClientIntegration:
    """Integration tests requiring a running Neo4j instance.

    These tests are marked with pytest.mark.integration and skipped
    if Neo4j is not available.
    """

    @pytest.fixture
    def client(self):
        """Create a test client."""
        try:
            client = Neo4jClient()
            if not client.health_check():
                pytest.skip("Neo4j not available")
            yield client
            client.close()
        except Exception:
            pytest.skip("Neo4j not available")

    @pytest.mark.integration
    def test_run_query(self, client):
        """Test running a simple query."""
        results = client.run_query("RETURN 1 AS num")
        assert len(results) == 1
        assert results[0]["num"] == 1

    @pytest.mark.integration
    def test_get_node_counts(self, client):
        """Test getting node counts."""
        counts = client.get_node_counts()
        assert isinstance(counts, dict)

    @pytest.mark.integration
    def test_get_edge_counts(self, client):
        """Test getting edge counts."""
        counts = client.get_edge_counts()
        assert isinstance(counts, dict)
