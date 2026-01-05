"""Voyage AI embedding generation with Redis caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

import numpy as np
import redis
import voyageai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.ingestion.chunking import LegalChunk

from .schema import VOYAGE_LAW_2_DIMENSION


class EmbeddingCache:
    """Redis-backed cache for embeddings."""

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "emb:",
        ttl: int = 86400 * 30,  # 30 days
    ):
        """Initialize embedding cache.

        Args:
            redis_client: Redis client instance
            prefix: Key prefix for cache entries
            ttl: Time-to-live in seconds (default 30 days)
        """
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl
        self._logger = logging.getLogger(__name__)

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text.

        Args:
            text: Text to hash

        Returns:
            16-character hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _make_key(self, text: str) -> str:
        """Create Redis key for text.

        Args:
            text: Text to create key for

        Returns:
            Redis key string
        """
        return f"{self.prefix}{self._hash_text(text)}"

    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding by text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None if not found
        """
        key = self._make_key(text)
        data = self.redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._make_key(text)
        self.redis.setex(key, self.ttl, json.dumps(embedding))

    def get_many(self, texts: list[str]) -> dict[int, list[float]]:
        """Get multiple cached embeddings.

        Args:
            texts: List of texts to look up

        Returns:
            Dictionary mapping index to embedding (only for cache hits)
        """
        if not texts:
            return {}

        keys = [self._make_key(text) for text in texts]
        values = self.redis.mget(keys)

        results = {}
        for i, value in enumerate(values):
            if value is not None:
                results[i] = json.loads(value)

        return results

    def set_many(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Cache multiple embeddings.

        Args:
            texts: List of texts that were embedded
            embeddings: Corresponding embedding vectors
        """
        if not texts:
            return

        pipe = self.redis.pipeline()
        for text, embedding in zip(texts, embeddings):
            key = self._make_key(text)
            pipe.setex(key, self.ttl, json.dumps(embedding))
        pipe.execute()

    def clear(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of keys deleted
        """
        pattern = f"{self.prefix}*"
        keys = list(self.redis.scan_iter(pattern))
        if keys:
            return self.redis.delete(*keys)
        return 0


class VoyageEmbedder:
    """Voyage AI embedder with batching, caching, and rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-law-2",
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 72,  # Conservative default to stay under 120K token limit
        rate_limit_delay: float = 0.1,
    ):
        """Initialize Voyage AI embedder.

        Args:
            api_key: Voyage AI API key (or loads from settings)
            model: Embedding model name
            cache: Optional Redis cache for embeddings
            batch_size: Number of texts per API batch (max 128)
            rate_limit_delay: Seconds to wait between batches
        """
        if api_key is None:
            api_key = self._get_api_key()

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.cache = cache
        self.batch_size = min(batch_size, 128)  # Voyage limit
        self.rate_limit_delay = rate_limit_delay
        self.dimension = VOYAGE_LAW_2_DIMENSION
        self._logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "texts_embedded": 0,
            "total_time": 0.0,
        }

    def _get_api_key(self) -> str:
        """Get API key from settings."""
        from config.settings import get_settings

        settings = get_settings()
        return settings.voyage_api_key.get_secret_value()

    @retry(
        retry=retry_if_exception_type((voyageai.error.RateLimitError, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Embed a single batch of texts with retry logic.

        Args:
            texts: Texts to embed (max 128)
            input_type: "document" or "query"

        Returns:
            List of embedding vectors
        """
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
        )
        self.stats["api_calls"] += 1
        return result.embeddings

    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "document",
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed texts with batching, caching, and rate limiting.

        Args:
            texts: List of texts to embed
            input_type: "document" for chunks, "query" for search queries
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        start_time = time.time()
        embeddings: list[Optional[list[float]]] = [None] * len(texts)

        # Check cache first
        texts_to_embed: list[tuple[int, str]] = []

        if self.cache:
            cached = self.cache.get_many(texts)
            self.stats["cache_hits"] += len(cached)

            for i, text in enumerate(texts):
                if i in cached:
                    embeddings[i] = cached[i]
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = [(i, text) for i, text in enumerate(texts)]

        # Embed uncached texts in batches
        if texts_to_embed:
            batches = [
                texts_to_embed[i : i + self.batch_size]
                for i in range(0, len(texts_to_embed), self.batch_size)
            ]

            iterator = tqdm(batches, desc="Embedding", disable=not show_progress)

            for batch_idx, batch in enumerate(iterator):
                batch_indices = [idx for idx, _ in batch]
                batch_texts = [text for _, text in batch]

                try:
                    batch_embeddings = self._embed_batch(batch_texts, input_type)

                    # Store results
                    for i, (orig_idx, text) in enumerate(batch):
                        embeddings[orig_idx] = batch_embeddings[i]

                    # Cache results
                    if self.cache:
                        self.cache.set_many(batch_texts, batch_embeddings)

                    self.stats["texts_embedded"] += len(batch_texts)

                except Exception as e:
                    self._logger.error(f"Failed to embed batch {batch_idx}: {e}")
                    # Fill with None for failed batch
                    for orig_idx, _ in batch:
                        if embeddings[orig_idx] is None:
                            embeddings[orig_idx] = [0.0] * self.dimension

                # Rate limiting between batches
                if batch_idx < len(batches) - 1:
                    time.sleep(self.rate_limit_delay)

        self.stats["total_time"] += time.time() - start_time

        # Return embeddings, using zero vectors for any failures
        return [
            emb if emb is not None else [0.0] * self.dimension
            for emb in embeddings
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query for retrieval.

        Uses "query" input_type for better retrieval performance.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
        )
        self.stats["api_calls"] += 1
        self.stats["texts_embedded"] += 1
        return result.embeddings[0]

    def embed_chunks(
        self,
        chunks: list[LegalChunk],
        show_progress: bool = True,
    ) -> list[tuple[str, list[float]]]:
        """Embed chunks using text_with_ancestry field.

        Args:
            chunks: List of LegalChunk objects
            show_progress: Whether to show progress bar

        Returns:
            List of (chunk_id, embedding) tuples
        """
        if not chunks:
            return []

        # Extract texts for embedding
        texts = [chunk.text_with_ancestry for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]

        # Embed all texts
        embeddings = self.embed_texts(texts, input_type="document", show_progress=show_progress)

        return list(zip(chunk_ids, embeddings))

    def get_stats(self) -> dict[str, Any]:
        """Get embedding statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["texts_embedded"])
                if (self.stats["cache_hits"] + self.stats["texts_embedded"]) > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "texts_embedded": 0,
            "total_time": 0.0,
        }


def create_redis_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
) -> redis.Redis:
    """Create a Redis client.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database number

    Returns:
        Redis client instance
    """
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def create_embedder_with_cache(
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> VoyageEmbedder:
    """Create a VoyageEmbedder with optional Redis cache.

    Args:
        api_key: Voyage AI API key (or loads from settings)
        use_cache: Whether to use Redis caching

    Returns:
        Configured VoyageEmbedder instance
    """
    cache = None

    if use_cache:
        from config.settings import get_settings

        settings = get_settings()
        redis_client = create_redis_client(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
        cache = EmbeddingCache(redis_client)

    return VoyageEmbedder(api_key=api_key, cache=cache)


def verify_embeddings(embeddings: list[list[float]]) -> dict[str, Any]:
    """Verify embedding quality.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Verification results including dimension check and normalization stats
    """
    if not embeddings:
        return {"valid": False, "error": "No embeddings provided"}

    arr = np.array(embeddings)

    # Check dimension
    dimension_ok = arr.shape[1] == VOYAGE_LAW_2_DIMENSION

    # Check normalization (L2 norms should be close to 1.0)
    norms = np.linalg.norm(arr, axis=1)
    avg_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    normalized = abs(avg_norm - 1.0) < 0.1  # Within 10% of 1.0

    return {
        "valid": dimension_ok and normalized,
        "count": len(embeddings),
        "dimension": arr.shape[1],
        "expected_dimension": VOYAGE_LAW_2_DIMENSION,
        "dimension_ok": dimension_ok,
        "avg_norm": avg_norm,
        "std_norm": std_norm,
        "normalized": normalized,
    }
