"""Neo4j client for the Florida Tax knowledge graph."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(..., description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_pool_size: int = Field(default=50, description="Maximum connection pool size")


class Neo4jClient:
    """Neo4j database client with connection pooling and query helpers."""

    def __init__(self, config: Neo4jConfig | None = None):
        """Initialize Neo4j client.

        Args:
            config: Optional configuration. If not provided, loads from settings.
        """
        if config is None:
            from config.settings import get_settings

            settings = get_settings()
            config = Neo4jConfig(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password.get_secret_value(),
            )

        self._config = config
        self._driver: Driver | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
                max_connection_pool_size=self._config.max_connection_pool_size,
            )
        return self._driver

    def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    @contextmanager
    def session(self, database: str | None = None) -> Generator[Any, None, None]:
        """Get a session context manager.

        Args:
            database: Optional database name override

        Yields:
            Neo4j session
        """
        db = database or self._config.database
        session = self.driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def health_check(self) -> bool:
        """Check if Neo4j is reachable and authenticated.

        Returns:
            True if Neo4j is healthy, False otherwise
        """
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS health")
                record = result.single()
                return record is not None and record["health"] == 1
        except (ServiceUnavailable, AuthError) as e:
            self._logger.error(f"Neo4j health check failed: {e}")
            return False

    def run_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            database: Optional database name override

        Returns:
            List of result records as dictionaries
        """
        with self.session(database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> dict[str, int]:
        """Execute a write query and return summary counters.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            database: Optional database name override

        Returns:
            Dictionary of operation counters
        """
        with self.session(database) as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def batch_write(
        self,
        query: str,
        batch_param_name: str,
        items: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Execute a batched write using UNWIND.

        Args:
            query: Cypher query with UNWIND $batch_param_name AS item
            batch_param_name: Name of the batch parameter in the query
            items: List of items to process
            batch_size: Number of items per batch

        Returns:
            Aggregated counters from all batches
        """
        totals: dict[str, int] = {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_set": 0,
        }

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(items) + batch_size - 1) // batch_size
            self._logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch)} items")

            result = self.run_write(query, {batch_param_name: batch})
            totals["nodes_created"] += result["nodes_created"]
            totals["relationships_created"] += result["relationships_created"]
            totals["properties_set"] += result["properties_set"]

        return totals

    def get_node_counts(self) -> dict[str, int]:
        """Get counts of all node types.

        Returns:
            Dictionary mapping label to count
        """
        query = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n)
            WHERE label IN labels(n)
            RETURN count(n) AS count
        }
        RETURN label, count
        """
        results = self.run_query(query)
        return {r["label"]: r["count"] for r in results}

    def get_edge_counts(self) -> dict[str, int]:
        """Get counts of all relationship types.

        Returns:
            Dictionary mapping relationship type to count
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL {
            WITH relationshipType
            MATCH ()-[r]->()
            WHERE type(r) = relationshipType
            RETURN count(r) AS count
        }
        RETURN relationshipType, count
        """
        results = self.run_query(query)
        return {r["relationshipType"]: r["count"] for r in results}

    def clear_database(self) -> None:
        """Delete all nodes and relationships. Use with caution!"""
        self._logger.warning("Clearing all data from Neo4j database")
        # Delete in batches to avoid memory issues with large graphs
        while True:
            self.run_write(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            # Check if any nodes were deleted by running a count query
            count_result = self.run_query("MATCH (n) RETURN count(n) AS count")
            if count_result[0]["count"] == 0:
                break
