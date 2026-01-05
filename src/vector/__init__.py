"""Weaviate vector store for semantic search."""

from .client import SearchResult, WeaviateClient, WeaviateConfig
from .embeddings import (
    EmbeddingCache,
    VoyageEmbedder,
    create_embedder_with_cache,
    create_redis_client,
    verify_embeddings,
)
from .schema import (
    LEGAL_CHUNK_PROPERTIES,
    VOYAGE_LAW_2_DIMENSION,
    CollectionName,
    get_legal_chunk_collection_config,
)

__all__ = [
    # Client
    "WeaviateClient",
    "WeaviateConfig",
    "SearchResult",
    # Embeddings
    "VoyageEmbedder",
    "EmbeddingCache",
    "create_embedder_with_cache",
    "create_redis_client",
    "verify_embeddings",
    # Schema
    "CollectionName",
    "LEGAL_CHUNK_PROPERTIES",
    "VOYAGE_LAW_2_DIMENSION",
    "get_legal_chunk_collection_config",
]
