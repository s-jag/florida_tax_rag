"""FastAPI dependency injection for the RAG API."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from src.agent import create_tax_agent_graph
from src.graph.client import Neo4jClient
from src.vector.client import WeaviateClient

# Context variable for request ID (useful for logging)
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# =============================================================================
# Client Singletons
# =============================================================================


@lru_cache
def get_neo4j_client() -> Neo4jClient:
    """Get singleton Neo4j client with connection pool.

    The Neo4jClient already uses connection pooling via the driver.
    Using lru_cache ensures we reuse the same client/pool across requests.
    """
    return Neo4jClient()


@lru_cache
def get_weaviate_client() -> WeaviateClient:
    """Get singleton Weaviate client.

    The WeaviateClient maintains a single connection that is reused.
    """
    return WeaviateClient()


@lru_cache
def get_agent_graph():
    """Get singleton compiled agent graph.

    The compiled graph is stateless and thread-safe.
    State is passed per-invocation, not stored in the graph.
    """
    return create_tax_agent_graph()


# =============================================================================
# Request-scoped Dependencies
# =============================================================================


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


async def get_request_id(request: Request) -> str:
    """Get or generate request ID for the current request.

    Checks for X-Request-ID header first, generates if not present.
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = generate_request_id()

    # Store in context var for logging
    request_id_var.set(request_id)
    return request_id


# =============================================================================
# Typed Dependencies (for FastAPI injection)
# =============================================================================

Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j_client)]
WeaviateDep = Annotated[WeaviateClient, Depends(get_weaviate_client)]
AgentGraphDep = Annotated[object, Depends(get_agent_graph)]  # StateGraph type
RequestIdDep = Annotated[str, Depends(get_request_id)]


# =============================================================================
# Lifecycle Management
# =============================================================================


async def cleanup_clients():
    """Cleanup client connections on shutdown.

    Called from the lifespan context manager.
    """
    # Clear cached clients to trigger cleanup
    get_neo4j_client.cache_clear()
    get_weaviate_client.cache_clear()
    get_agent_graph.cache_clear()
