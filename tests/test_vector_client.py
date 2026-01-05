"""Tests for Weaviate client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.vector.client import SearchResult, WeaviateClient, WeaviateConfig
from src.vector.schema import (
    CollectionName,
    LEGAL_CHUNK_PROPERTIES,
    VOYAGE_LAW_2_DIMENSION,
    get_legal_chunk_collection_config,
)


class TestWeaviateConfig:
    """Tests for WeaviateConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WeaviateConfig()
        assert config.url == "http://localhost:8080"
        assert config.api_key is None
        assert config.grpc_port == 50051

    def test_custom_values(self):
        """Test custom configuration."""
        config = WeaviateConfig(
            url="http://weaviate.example.com:8080",
            api_key="test-key",
            grpc_port=50052,
        )
        assert config.url == "http://weaviate.example.com:8080"
        assert config.api_key == "test-key"
        assert config.grpc_port == 50052


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_required_fields(self):
        """Test required fields."""
        result = SearchResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Sample text",
            score=0.95,
        )
        assert result.chunk_id == "chunk:statute:212.05:0"
        assert result.score == 0.95
        assert result.distance is None

    def test_optional_fields(self):
        """Test optional fields."""
        result = SearchResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Sample text",
            score=0.95,
            ancestry="Florida Statutes > Chapter 212 > § 212.05",
            citation="Fla. Stat. § 212.05",
            distance=0.05,
        )
        assert result.ancestry == "Florida Statutes > Chapter 212 > § 212.05"
        assert result.distance == 0.05


class TestCollectionName:
    """Tests for CollectionName enum."""

    def test_legal_chunk_value(self):
        """Test LegalChunk collection name."""
        assert CollectionName.LEGAL_CHUNK.value == "LegalChunk"


class TestLegalChunkProperties:
    """Tests for schema properties."""

    def test_property_count(self):
        """Test that all expected properties are defined."""
        assert len(LEGAL_CHUNK_PROPERTIES) == 10

    def test_property_names(self):
        """Test property names."""
        names = {p.name for p in LEGAL_CHUNK_PROPERTIES}
        expected = {
            "chunk_id",
            "doc_id",
            "doc_type",
            "level",
            "ancestry",
            "citation",
            "text",
            "text_with_ancestry",
            "effective_date",
            "token_count",
        }
        assert names == expected


class TestVoyageDimension:
    """Tests for Voyage AI dimension constant."""

    def test_voyage_law_2_dimension(self):
        """Test voyage-law-2 embedding dimension."""
        assert VOYAGE_LAW_2_DIMENSION == 1024


class TestGetLegalChunkCollectionConfig:
    """Tests for collection config function."""

    def test_config_keys(self):
        """Test that config contains required keys."""
        config = get_legal_chunk_collection_config()
        assert "name" in config
        assert "description" in config
        assert "properties" in config
        assert "vector_config" in config
        assert "inverted_index_config" in config

    def test_config_name(self):
        """Test collection name."""
        config = get_legal_chunk_collection_config()
        assert config["name"] == "LegalChunk"


class TestWeaviateClient:
    """Tests for WeaviateClient."""

    def test_context_manager(self):
        """Test client as context manager."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        with patch.object(client, "close") as mock_close:
            with client:
                pass
            mock_close.assert_called_once()

    def test_parse_search_results_empty(self):
        """Test parsing empty search results."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        results = client._parse_search_results([])
        assert len(results) == 0

    def test_parse_search_results_with_data(self):
        """Test parsing search results with data."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        mock_obj = MagicMock()
        mock_obj.properties = {
            "chunk_id": "chunk:statute:212.05:0",
            "doc_id": "statute:212.05",
            "doc_type": "statute",
            "level": "parent",
            "text": "Tax on sales",
            "ancestry": "Florida Statutes > Chapter 212",
        }
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.score = 0.95
        mock_obj.metadata.distance = None

        results = client._parse_search_results([mock_obj])

        assert len(results) == 1
        assert results[0].chunk_id == "chunk:statute:212.05:0"
        assert results[0].score == 0.95

    def test_parse_search_results_distance_to_score(self):
        """Test converting distance to score when no score available."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        mock_obj = MagicMock()
        mock_obj.properties = {
            "chunk_id": "chunk:statute:212.05:0",
            "doc_id": "statute:212.05",
            "doc_type": "statute",
            "level": "parent",
            "text": "Tax on sales",
        }
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.score = None
        mock_obj.metadata.distance = 0.1

        results = client._parse_search_results([mock_obj])

        assert len(results) == 1
        assert results[0].score == 0.9  # 1.0 - 0.1
        assert results[0].distance == 0.1


class TestBatchInsert:
    """Tests for batch insert functionality."""

    def test_batch_insert_length_mismatch(self):
        """Test batch_insert raises error on length mismatch."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        chunks = [{"chunk_id": "1"}, {"chunk_id": "2"}]
        vectors = [[0.1, 0.2]]  # Only 1 vector for 2 chunks

        # The validation happens before any Weaviate calls
        with pytest.raises(ValueError, match="Mismatch"):
            client.batch_insert(chunks, vectors)

    def test_batch_insert_empty(self):
        """Test batch_insert with empty lists."""
        config = WeaviateConfig()
        client = WeaviateClient(config)

        result = client.batch_insert([], [])

        assert result["inserted"] == 0
        assert result["errors"] == 0


class TestWeaviateClientIntegration:
    """Integration tests requiring a running Weaviate instance.

    These tests are marked with pytest.mark.integration and skipped
    if Weaviate is not available.
    """

    @pytest.fixture
    def client(self):
        """Create a test client."""
        try:
            client = WeaviateClient()
            if not client.health_check():
                pytest.skip("Weaviate not available")
            yield client
            client.close()
        except Exception:
            pytest.skip("Weaviate not available")

    @pytest.mark.integration
    def test_health_check(self, client):
        """Test health check."""
        assert client.health_check() is True

    @pytest.mark.integration
    def test_collection_exists(self, client):
        """Test that LegalChunk collection exists."""
        info = client.get_collection_info()
        assert info is not None
        assert info["name"] == "LegalChunk"

    @pytest.mark.integration
    def test_collection_properties(self, client):
        """Test collection has expected properties."""
        info = client.get_collection_info()
        assert info is not None

        prop_names = {p["name"] for p in info["properties"]}
        expected = {
            "chunk_id",
            "doc_id",
            "doc_type",
            "level",
            "ancestry",
            "citation",
            "text",
            "text_with_ancestry",
            "effective_date",
            "token_count",
        }
        assert prop_names == expected
