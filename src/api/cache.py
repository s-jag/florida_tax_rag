"""Query result caching for the API."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)


class QueryCache:
    """Redis-backed cache for query results.

    Caches complete query responses to avoid re-executing the full pipeline
    for identical queries. Uses a 1-hour TTL by default.

    Example:
        cache = QueryCache(redis_client)

        # Check cache before processing
        cached = await cache.get(query, options)
        if cached:
            return cached

        # Process query...
        response = await process_query(query)

        # Cache result
        await cache.set(query, options, response)
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "query:",
        ttl: int = 3600,  # 1 hour
    ):
        """Initialize query cache.

        Args:
            redis_client: Redis client instance
            prefix: Key prefix for cache entries
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl

        # Statistics
        self._hits = 0
        self._misses = 0

    def _cache_key(self, query: str, options: Optional[dict[str, Any]] = None) -> str:
        """Generate cache key from query and options.

        Args:
            query: The query text
            options: Optional query options (tax_year, etc.)

        Returns:
            Cache key string
        """
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()

        # Include relevant options in the key
        key_parts = [normalized_query]

        if options:
            # Only include options that affect the query result
            relevant_options = {
                k: v for k, v in sorted(options.items())
                if k in ("tax_year", "include_reasoning") and v is not None
            }
            if relevant_options:
                key_parts.append(json.dumps(relevant_options, sort_keys=True))

        content = ":".join(key_parts)
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.prefix}{hash_value}"

    async def get(
        self,
        query: str,
        options: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Get cached response if exists.

        Args:
            query: The query text
            options: Optional query options

        Returns:
            Cached response dict or None if not found
        """
        key = self._cache_key(query, options)

        try:
            data = self.redis.get(key)
            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            logger.debug("cache_hit", key=key)
            return json.loads(data)

        except Exception as e:
            logger.warning("cache_get_error", key=key, error=str(e))
            self._misses += 1
            return None

    async def set(
        self,
        query: str,
        options: Optional[dict[str, Any]],
        response: dict[str, Any],
    ) -> bool:
        """Cache response with TTL.

        Args:
            query: The query text
            options: Optional query options
            response: Response dict to cache

        Returns:
            True if cached successfully, False otherwise
        """
        key = self._cache_key(query, options)

        try:
            # Don't cache failed responses
            if not response.get("answer") or response.get("answer") == "Unable to generate answer":
                return False

            # Don't cache low-confidence responses
            if response.get("confidence", 0) < 0.3:
                return False

            # Serialize and cache
            data = json.dumps(response)
            self.redis.setex(key, self.ttl, data)
            logger.debug("cache_set", key=key, ttl=self.ttl)
            return True

        except Exception as e:
            logger.warning("cache_set_error", key=key, error=str(e))
            return False

    async def invalidate(self, query: str, options: Optional[dict[str, Any]] = None) -> bool:
        """Invalidate a cached entry.

        Args:
            query: The query text
            options: Optional query options

        Returns:
            True if entry was deleted, False if not found
        """
        key = self._cache_key(query, options)
        try:
            return self.redis.delete(key) > 0
        except Exception as e:
            logger.warning("cache_invalidate_error", key=key, error=str(e))
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit/miss counts and hit rate
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(hit_rate, 3),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._hits = 0
        self._misses = 0

    async def clear(self) -> int:
        """Clear all cached queries.

        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{self.prefix}*"
            keys = list(self.redis.scan_iter(pattern))
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("cache_clear_error", error=str(e))
            return 0


# Singleton instance
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> Optional[QueryCache]:
    """Get the singleton query cache instance.

    Returns:
        QueryCache instance or None if not configured
    """
    global _query_cache

    if _query_cache is not None:
        return _query_cache

    try:
        from config.settings import get_settings

        settings = get_settings()

        # Create Redis client
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

        # Verify connection
        redis_client.ping()

        _query_cache = QueryCache(redis_client)
        logger.info("query_cache_initialized", ttl=_query_cache.ttl)
        return _query_cache

    except Exception as e:
        logger.warning("query_cache_unavailable", error=str(e))
        return None


def clear_query_cache_singleton() -> None:
    """Clear the singleton instance (for testing)."""
    global _query_cache
    _query_cache = None
