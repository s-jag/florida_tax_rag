"""Weaviate client for the Florida Tax vector store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import weaviate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.exceptions import WeaviateConnectionError

from .schema import CollectionName, get_legal_chunk_collection_config


class WeaviateConfig(BaseModel):
    """Weaviate connection configuration."""

    url: str = Field(
        default="http://localhost:8080", description="Weaviate HTTP URL"
    )
    api_key: Optional[str] = Field(
        default=None, description="Optional API key for authentication"
    )
    grpc_port: int = Field(default=50051, description="gRPC port")


class SearchResult(BaseModel):
    """A single search result from Weaviate."""

    chunk_id: str
    doc_id: str
    doc_type: str
    level: str
    ancestry: Optional[str] = None
    citation: Optional[str] = None
    text: str
    text_with_ancestry: Optional[str] = None
    effective_date: Optional[datetime] = None
    token_count: Optional[int] = None
    score: float = Field(description="Search relevance score")
    distance: Optional[float] = Field(
        default=None, description="Vector distance (for vector search)"
    )


class WeaviateClient:
    """Weaviate database client for hybrid search."""

    def __init__(self, config: Optional[WeaviateConfig] = None):
        """Initialize Weaviate client.

        Args:
            config: Optional configuration. If not provided, loads from settings.
        """
        if config is None:
            from config.settings import get_settings

            settings = get_settings()
            api_key = None
            if settings.weaviate_api_key:
                api_key = settings.weaviate_api_key.get_secret_value()
            config = WeaviateConfig(
                url=settings.weaviate_url,
                api_key=api_key,
            )

        self._config = config
        self._client: Optional[weaviate.WeaviateClient] = None
        self._logger = logging.getLogger(__name__)

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get or create the Weaviate client."""
        if self._client is None:
            # Parse URL to get host and port
            url = self._config.url
            if url.startswith("http://"):
                host = url[7:]
            elif url.startswith("https://"):
                host = url[8:]
            else:
                host = url

            # Remove port if present
            if ":" in host:
                host, port_str = host.rsplit(":", 1)
                http_port = int(port_str)
            else:
                http_port = 8080

            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=self._config.url.startswith("https"),
                grpc_host=host,
                grpc_port=self._config.grpc_port,
                grpc_secure=False,
            )

        return self._client

    def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "WeaviateClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()

    def health_check(self) -> bool:
        """Check if Weaviate is reachable.

        Returns:
            True if Weaviate is healthy, False otherwise
        """
        try:
            return self.client.is_ready()
        except WeaviateConnectionError as e:
            self._logger.error(f"Weaviate health check failed: {e}")
            return False

    def init_schema(self) -> bool:
        """Initialize the LegalChunk collection schema.

        Creates the collection if it doesn't exist.

        Returns:
            True if collection was created, False if it already exists
        """
        collection_name = CollectionName.LEGAL_CHUNK.value

        if self.client.collections.exists(collection_name):
            self._logger.info(f"Collection '{collection_name}' already exists")
            return False

        config = get_legal_chunk_collection_config()
        self.client.collections.create(**config)
        self._logger.info(f"Created collection '{collection_name}'")
        return True

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of collection to delete. Defaults to LegalChunk.

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        name = collection_name or CollectionName.LEGAL_CHUNK.value

        if not self.client.collections.exists(name):
            self._logger.info(f"Collection '{name}' does not exist")
            return False

        self.client.collections.delete(name)
        self._logger.info(f"Deleted collection '{name}'")
        return True

    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Get collection schema and object count.

        Args:
            collection_name: Name of collection. Defaults to LegalChunk.

        Returns:
            Dictionary with collection info or None if collection doesn't exist
        """
        name = collection_name or CollectionName.LEGAL_CHUNK.value

        if not self.client.collections.exists(name):
            return None

        collection = self.client.collections.get(name)
        config = collection.config.get()

        # Get object count
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count

        return {
            "name": name,
            "description": config.description,
            "properties": [
                {"name": p.name, "data_type": str(p.data_type)}
                for p in config.properties
            ],
            "object_count": count,
        }

    def insert_chunk(
        self,
        chunk_data: dict[str, Any],
        vector: list[float],
    ) -> str:
        """Insert a single chunk with its vector.

        Args:
            chunk_data: Chunk properties
            vector: Embedding vector from Voyage AI

        Returns:
            UUID of the inserted object
        """
        collection = self.client.collections.get(CollectionName.LEGAL_CHUNK.value)
        uuid = collection.data.insert(properties=chunk_data, vector=vector)
        return str(uuid)

    def batch_insert(
        self,
        chunks_data: list[dict[str, Any]],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Batch insert chunks with their vectors.

        Args:
            chunks_data: List of chunk properties
            vectors: List of embedding vectors (same order as chunks)
            batch_size: Number of items per batch

        Returns:
            Dictionary with insert statistics
        """
        # Validation (not retried)
        if len(chunks_data) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks_data)} chunks vs {len(vectors)} vectors"
            )

        if not chunks_data:
            return {"inserted": 0, "errors": 0}

        return self._batch_insert_with_retry(chunks_data, vectors, batch_size)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _batch_insert_with_retry(
        self,
        chunks_data: list[dict[str, Any]],
        vectors: list[list[float]],
        batch_size: int,
    ) -> dict[str, int]:
        """Internal method that handles the actual batch insert with retry."""

        collection = self.client.collections.get(CollectionName.LEGAL_CHUNK.value)
        inserted = 0
        errors = 0

        # Process in batches
        for i in range(0, len(chunks_data), batch_size):
            batch_chunks = chunks_data[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            with collection.batch.dynamic() as batch:
                for chunk, vector in zip(batch_chunks, batch_vectors):
                    batch.add_object(properties=chunk, vector=vector)

            # Check for failed objects
            if collection.batch.failed_objects:
                errors += len(collection.batch.failed_objects)
                for failed in collection.batch.failed_objects:
                    self._logger.error(f"Failed to insert: {failed.message}")
            else:
                inserted += len(batch_chunks)

        return {"inserted": inserted, "errors": errors}

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        alpha: float = 0.5,
        limit: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Text query for BM25 search
            query_vector: Query embedding vector
            alpha: Weight between vector (1.0) and keyword (0.0) search
            limit: Maximum number of results
            filters: Optional filters (e.g., {"doc_type": "statute"})

        Returns:
            List of SearchResult objects
        """
        collection = self.client.collections.get(CollectionName.LEGAL_CHUNK.value)

        # Build filter if provided
        weaviate_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(Filter.by_property(key).contains_any(value))
                else:
                    conditions.append(Filter.by_property(key).equal(value))

            if len(conditions) == 1:
                weaviate_filter = conditions[0]
            else:
                weaviate_filter = Filter.all_of(conditions)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            filters=weaviate_filter,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True, distance=True),
        )

        return self._parse_search_results(response.objects)

    def keyword_search(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Perform BM25 keyword search only.

        Args:
            query: Text query for BM25 search
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        collection = self.client.collections.get(CollectionName.LEGAL_CHUNK.value)

        # Build filter if provided
        weaviate_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(Filter.by_property(key).contains_any(value))
                else:
                    conditions.append(Filter.by_property(key).equal(value))

            if len(conditions) == 1:
                weaviate_filter = conditions[0]
            else:
                weaviate_filter = Filter.all_of(conditions)

        response = collection.query.bm25(
            query=query,
            limit=limit,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
        )

        return self._parse_search_results(response.objects)

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Perform vector similarity search only.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        collection = self.client.collections.get(CollectionName.LEGAL_CHUNK.value)

        # Build filter if provided
        weaviate_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(Filter.by_property(key).contains_any(value))
                else:
                    conditions.append(Filter.by_property(key).equal(value))

            if len(conditions) == 1:
                weaviate_filter = conditions[0]
            else:
                weaviate_filter = Filter.all_of(conditions)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(distance=True),
        )

        return self._parse_search_results(response.objects)

    def _parse_search_results(self, objects: list[Any]) -> list[SearchResult]:
        """Parse Weaviate response objects into SearchResult models.

        Args:
            objects: List of Weaviate response objects

        Returns:
            List of SearchResult objects
        """
        results = []
        for obj in objects:
            props = obj.properties

            # Get score/distance from metadata
            score = 0.0
            distance = None
            if obj.metadata:
                if obj.metadata.score is not None:
                    score = obj.metadata.score
                if obj.metadata.distance is not None:
                    distance = obj.metadata.distance
                    # Convert distance to score if no score available
                    if score == 0.0:
                        score = 1.0 - distance

            results.append(
                SearchResult(
                    chunk_id=props.get("chunk_id", ""),
                    doc_id=props.get("doc_id", ""),
                    doc_type=props.get("doc_type", ""),
                    level=props.get("level", ""),
                    ancestry=props.get("ancestry"),
                    citation=props.get("citation"),
                    text=props.get("text", ""),
                    text_with_ancestry=props.get("text_with_ancestry"),
                    effective_date=props.get("effective_date"),
                    token_count=props.get("token_count"),
                    score=score,
                    distance=distance,
                )
            )

        return results
