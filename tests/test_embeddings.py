"""Tests for embedding generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vector.embeddings import (
    EmbeddingCache,
    VoyageEmbedder,
    verify_embeddings,
)
from src.vector.schema import VOYAGE_LAW_2_DIMENSION


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_hash_text_deterministic(self):
        """Test hash is deterministic for same input."""
        mock_redis = MagicMock()
        cache = EmbeddingCache(mock_redis)

        hash1 = cache._hash_text("test text")
        hash2 = cache._hash_text("test text")

        assert hash1 == hash2
        assert len(hash1) == 16  # 16 character hash

    def test_hash_text_different_for_different_input(self):
        """Test hash differs for different inputs."""
        mock_redis = MagicMock()
        cache = EmbeddingCache(mock_redis)

        hash1 = cache._hash_text("text one")
        hash2 = cache._hash_text("text two")

        assert hash1 != hash2

    def test_make_key_includes_prefix(self):
        """Test key includes prefix."""
        mock_redis = MagicMock()
        cache = EmbeddingCache(mock_redis, prefix="test:")

        key = cache._make_key("some text")

        assert key.startswith("test:")

    def test_get_returns_none_when_not_cached(self):
        """Test get returns None for cache miss."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        cache = EmbeddingCache(mock_redis)

        result = cache.get("uncached text")

        assert result is None
        mock_redis.get.assert_called_once()

    def test_get_returns_embedding_when_cached(self):
        """Test get returns embedding for cache hit."""
        mock_redis = MagicMock()
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = json.dumps(embedding)
        cache = EmbeddingCache(mock_redis)

        result = cache.get("cached text")

        assert result == embedding

    def test_set_stores_embedding(self):
        """Test set stores embedding with TTL."""
        mock_redis = MagicMock()
        cache = EmbeddingCache(mock_redis, ttl=3600)
        embedding = [0.1, 0.2, 0.3]

        cache.set("text", embedding)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600  # TTL
        assert json.loads(call_args[0][2]) == embedding

    def test_get_many_returns_cached_embeddings(self):
        """Test get_many returns only cached embeddings."""
        mock_redis = MagicMock()
        embedding1 = [0.1, 0.2, 0.3]
        # Simulate: first text cached, second not cached
        mock_redis.mget.return_value = [json.dumps(embedding1), None]
        cache = EmbeddingCache(mock_redis)

        results = cache.get_many(["text1", "text2"])

        assert 0 in results
        assert results[0] == embedding1
        assert 1 not in results  # Cache miss

    def test_set_many_stores_all_embeddings(self):
        """Test set_many stores all embeddings."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        cache = EmbeddingCache(mock_redis)
        texts = ["text1", "text2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        cache.set_many(texts, embeddings)

        assert mock_pipe.setex.call_count == 2
        mock_pipe.execute.assert_called_once()


class TestVoyageEmbedder:
    """Tests for VoyageEmbedder."""

    def test_init_sets_defaults(self):
        """Test initialization with defaults."""
        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder()

        assert embedder.model == "voyage-law-2"
        assert embedder.batch_size == 72  # Conservative default to stay under 120K token limit
        assert embedder.dimension == VOYAGE_LAW_2_DIMENSION
        assert embedder.cache is None

    def test_init_with_custom_batch_size(self):
        """Test batch size is capped at 128."""
        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder(batch_size=200)

        assert embedder.batch_size == 128  # Capped at Voyage limit

    def test_embed_texts_empty_list(self):
        """Test embed_texts with empty list."""
        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder()

        result = embedder.embed_texts([])

        assert result == []

    def test_embed_texts_uses_cache(self):
        """Test embed_texts checks cache first."""
        mock_cache = MagicMock()
        cached_embedding = [0.1] * VOYAGE_LAW_2_DIMENSION
        mock_cache.get_many.return_value = {0: cached_embedding}

        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder(cache=mock_cache)

        result = embedder.embed_texts(["cached text"], show_progress=False)

        assert result[0] == cached_embedding
        mock_cache.get_many.assert_called_once()
        assert embedder.stats["cache_hits"] == 1

    def test_embed_texts_batching(self):
        """Test embed_texts processes in batches."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * VOYAGE_LAW_2_DIMENSION] * 50

        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder(batch_size=50)
            embedder.client = mock_client
            mock_client.embed.return_value = mock_result

        texts = ["text"] * 150  # 3 batches of 50

        with patch.object(
            embedder, "_embed_batch", return_value=[[0.1] * VOYAGE_LAW_2_DIMENSION] * 50
        ):
            result = embedder.embed_texts(texts, show_progress=False)

        assert len(result) == 150

    def test_embed_query_uses_query_input_type(self):
        """Test embed_query uses 'query' input type."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * VOYAGE_LAW_2_DIMENSION]

        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder()
            embedder.client = mock_client
            mock_client.embed.return_value = mock_result

        result = embedder.embed_query("search query")

        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "query"
        assert len(result) == VOYAGE_LAW_2_DIMENSION

    def test_get_stats_returns_statistics(self):
        """Test get_stats returns all statistics."""
        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder()
            embedder.stats = {
                "cache_hits": 10,
                "api_calls": 5,
                "texts_embedded": 100,
                "total_time": 1.5,
            }

        stats = embedder.get_stats()

        assert stats["cache_hits"] == 10
        assert stats["api_calls"] == 5
        assert stats["texts_embedded"] == 100
        assert "cache_hit_rate" in stats

    def test_reset_stats_clears_statistics(self):
        """Test reset_stats clears all statistics."""
        with patch.object(VoyageEmbedder, "_get_api_key", return_value="test-key"):
            embedder = VoyageEmbedder()
            embedder.stats["cache_hits"] = 100

        embedder.reset_stats()

        assert embedder.stats["cache_hits"] == 0
        assert embedder.stats["api_calls"] == 0


class TestVerifyEmbeddings:
    """Tests for verify_embeddings function."""

    def test_empty_embeddings_invalid(self):
        """Test empty embeddings are invalid."""
        result = verify_embeddings([])

        assert result["valid"] is False

    def test_correct_dimension_valid(self):
        """Test correct dimension passes."""
        embeddings = [np.random.randn(VOYAGE_LAW_2_DIMENSION).tolist() for _ in range(5)]
        # Normalize
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            embeddings[i] = [x / norm for x in emb]

        result = verify_embeddings(embeddings)

        assert result["dimension_ok"] is True
        assert result["dimension"] == VOYAGE_LAW_2_DIMENSION

    def test_wrong_dimension_invalid(self):
        """Test wrong dimension fails."""
        embeddings = [[0.1] * 512]  # Wrong dimension

        result = verify_embeddings(embeddings)

        assert result["dimension_ok"] is False
        assert result["valid"] is False

    def test_normalized_embeddings_pass(self):
        """Test normalized embeddings pass normalization check."""
        # Create normalized embeddings (L2 norm = 1)
        embeddings = []
        for _ in range(5):
            vec = np.random.randn(VOYAGE_LAW_2_DIMENSION)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec.tolist())

        result = verify_embeddings(embeddings)

        assert result["normalized"] is True
        assert abs(result["avg_norm"] - 1.0) < 0.1


class TestVoyageEmbedderIntegration:
    """Integration tests requiring Voyage API key.

    These tests are marked with pytest.mark.integration and skipped
    if the API key is not available.
    """

    @pytest.fixture
    def embedder(self):
        """Create embedder with API key from settings."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            api_key = settings.voyage_api_key.get_secret_value()
            if api_key == "placeholder_for_later":
                pytest.skip("Voyage API key not configured")
            return VoyageEmbedder(api_key=api_key)
        except Exception as e:
            pytest.skip(f"Cannot create embedder: {e}")

    @pytest.mark.integration
    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        result = embedder.embed_texts(
            ["This is a test sentence about Florida tax law."], show_progress=False
        )

        assert len(result) == 1
        assert len(result[0]) == VOYAGE_LAW_2_DIMENSION

    @pytest.mark.integration
    def test_embed_query(self, embedder):
        """Test embedding a query."""
        result = embedder.embed_query("What is the sales tax rate in Florida?")

        assert len(result) == VOYAGE_LAW_2_DIMENSION

    @pytest.mark.integration
    def test_embeddings_are_normalized(self, embedder):
        """Test that embeddings are normalized (L2 norm close to 1)."""
        texts = [
            "Florida sales tax exemptions",
            "Chapter 212 tax regulations",
            "Administrative code rule 12A-1.001",
        ]
        embeddings = embedder.embed_texts(texts, show_progress=False)

        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.1, f"Embedding not normalized: norm={norm}"

    @pytest.mark.integration
    def test_different_texts_have_different_embeddings(self, embedder):
        """Test that different texts produce different embeddings."""
        embeddings = embedder.embed_texts(
            ["Sales tax in Florida", "Corporate income tax rates"],
            show_progress=False,
        )

        # Compute cosine similarity
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Should be similar (both about tax) but not identical
        assert similarity < 0.99, "Embeddings should differ for different texts"
        assert similarity > 0.5, "Tax-related texts should have some similarity"
