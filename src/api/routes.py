"""API routes for the Florida Tax RAG system."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.agent import TaxAgentState
from src.graph.queries import (
    get_cited_documents,
    get_citing_documents,
    get_interpretation_chain,
)

from .dependencies import (
    AgentGraphDep,
    Neo4jDep,
    RequestIdDep,
    WeaviateDep,
)
from .models import (
    ChunkDetailResponse,
    CitationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
    RelatedDocumentsResponse,
    ServiceHealth,
    SourceResponse,
    StatuteWithRulesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Query Endpoints
# =============================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        408: {"description": "Request timeout"},
        500: {"description": "Internal error"},
    },
)
async def query(
    request: QueryRequest,
    request_id: RequestIdDep,
    graph: AgentGraphDep,
) -> QueryResponse:
    """Execute a tax law query through the RAG agent.

    This endpoint invokes the full agent graph including:
    - Query decomposition
    - Hybrid retrieval (vector + keyword + graph)
    - Relevance scoring and filtering
    - Temporal validation
    - Answer synthesis with citations
    - Hallucination detection and correction
    """
    start_time = time.perf_counter()

    # Build initial state
    initial_state: TaxAgentState = {
        "original_query": request.query,
        "retrieved_chunks": [],
        "errors": [],
        "reasoning_steps": [],
    }

    # Add tax year if specified
    if request.options.tax_year:
        initial_state["query_tax_year"] = request.options.tax_year

    try:
        # Execute with timeout
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=request.options.timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "request_id": request_id,
                "error": "Request timed out",
                "details": [
                    {
                        "code": "TIMEOUT",
                        "message": f"Query exceeded {request.options.timeout_seconds}s timeout",
                    }
                ],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "request_id": request_id,
                "error": "Query execution failed",
                "details": [{"code": "EXECUTION_ERROR", "message": str(e)}],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # Calculate processing time
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Build response
    citations = [
        CitationResponse(
            doc_id=c.get("doc_id", ""),
            citation=c.get("citation", ""),
            doc_type=c.get("doc_type", ""),
            text_snippet=c.get("text_snippet", ""),
        )
        for c in result.get("citations", [])
    ]

    sources = [
        SourceResponse(
            chunk_id=chunk.get("chunk_id", ""),
            doc_id=chunk.get("doc_id", ""),
            doc_type=chunk.get("doc_type", ""),
            citation=chunk.get("citation"),
            text=chunk.get("text", "")[:500],  # Truncate for response
            effective_date=chunk.get("effective_date"),
            relevance_score=chunk.get("score", 0.0),
        )
        for chunk in result.get("temporally_valid_chunks", [])[:10]
    ]

    reasoning_steps = None
    if request.options.include_reasoning:
        reasoning_steps = [
            ReasoningStep(
                step_number=i + 1,
                node=step.get("node", "unknown") if isinstance(step, dict) else "unknown",
                description=step.get("description", str(step)) if isinstance(step, dict) else str(step),
            )
            for i, step in enumerate(result.get("reasoning_steps", []))
        ]

    # Extract warnings from validation result if present
    warnings = list(result.get("errors", []))
    if result.get("validation_result"):
        validation = result["validation_result"]
        if validation.get("hallucinations"):
            warnings.append(
                f"Found {len(validation['hallucinations'])} potential issues in response"
            )

    return QueryResponse(
        request_id=request_id,
        answer=result.get("final_answer") or "Unable to generate answer",
        citations=citations,
        sources=sources,
        confidence=result.get("confidence", 0.0),
        warnings=warnings,
        reasoning_steps=reasoning_steps,
        validation_passed=result.get("validation_passed", False),
        processing_time_ms=processing_time_ms,
    )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    request_id: RequestIdDep,
    graph: AgentGraphDep,
) -> StreamingResponse:
    """Stream query results via Server-Sent Events.

    Events:
    - `status`: Status updates (node entered, etc.)
    - `reasoning`: Reasoning step completed
    - `chunk`: Source chunk retrieved
    - `answer`: Final answer
    - `complete`: Final result
    - `error`: Error occurred
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()

        initial_state: TaxAgentState = {
            "original_query": request.query,
            "retrieved_chunks": [],
            "errors": [],
            "reasoning_steps": [],
        }

        if request.options.tax_year:
            initial_state["query_tax_year"] = request.options.tax_year

        try:
            # Use astream to get node-by-node updates
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name, state_update in event.items():
                    # Send status update
                    yield f"event: status\ndata: {json.dumps({'node': node_name, 'timestamp': datetime.utcnow().isoformat()})}\n\n"

                    # Send reasoning steps if any
                    if "reasoning_steps" in state_update:
                        for step in state_update["reasoning_steps"]:
                            yield f"event: reasoning\ndata: {json.dumps(step if isinstance(step, dict) else {'description': str(step)})}\n\n"

                    # Send retrieved chunks
                    if "current_retrieval_results" in state_update:
                        for chunk in state_update["current_retrieval_results"][:5]:
                            yield f"event: chunk\ndata: {json.dumps({'chunk_id': chunk.get('chunk_id'), 'citation': chunk.get('citation'), 'doc_type': chunk.get('doc_type')})}\n\n"

                    # Send final answer
                    if "final_answer" in state_update and state_update["final_answer"]:
                        yield f"event: answer\ndata: {json.dumps({'text': state_update['final_answer']})}\n\n"

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            yield f"event: complete\ndata: {json.dumps({'request_id': request_id, 'processing_time_ms': processing_time_ms})}\n\n"

        except asyncio.TimeoutError:
            yield f"event: error\ndata: {json.dumps({'code': 'TIMEOUT', 'message': 'Request timed out'})}\n\n"
        except Exception as e:
            logger.exception(f"Streaming query failed: {e}")
            yield f"event: error\ndata: {json.dumps({'code': 'EXECUTION_ERROR', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


# =============================================================================
# Source/Chunk Endpoints
# =============================================================================


@router.get(
    "/sources/{chunk_id}",
    response_model=ChunkDetailResponse,
    responses={404: {"description": "Chunk not found"}},
)
async def get_chunk(
    chunk_id: str,
    neo4j: Neo4jDep,
    request_id: RequestIdDep,
) -> ChunkDetailResponse:
    """Get full details for a specific chunk by ID.

    Retrieves chunk content, metadata, and relationships from Neo4j.
    """
    query = """
    MATCH (c:Chunk {id: $chunk_id})
    OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
    OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Chunk)
    OPTIONAL MATCH (child:Chunk)-[:CHILD_OF]->(c)
    RETURN c, d, parent.id AS parent_id, collect(DISTINCT child.id) AS child_ids
    """

    results = neo4j.run_query(query, {"chunk_id": chunk_id})

    if not results or results[0].get("c") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "request_id": request_id,
                "error": f"Chunk not found: {chunk_id}",
                "details": [],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    row = results[0]
    chunk_data = dict(row["c"]) if row["c"] else {}
    doc_data = dict(row["d"]) if row.get("d") else {}

    # Get related documents via graph
    related_docs = []
    if doc_data.get("id"):
        citing = get_citing_documents(neo4j, doc_data["id"])
        cited = get_cited_documents(neo4j, doc_data["id"])
        related_docs = [d.id for d in (citing or []) + (cited or [])][:10]

    return ChunkDetailResponse(
        chunk_id=chunk_data.get("id", chunk_id),
        doc_id=doc_data.get("id", ""),
        doc_type=doc_data.get("doc_type", chunk_data.get("doc_type", "")),
        level=chunk_data.get("level", ""),
        text=chunk_data.get("text", ""),
        text_with_ancestry=chunk_data.get("text_with_ancestry"),
        ancestry=chunk_data.get("ancestry"),
        citation=chunk_data.get("citation"),
        effective_date=chunk_data.get("effective_date"),
        token_count=chunk_data.get("token_count"),
        parent_chunk_id=row.get("parent_id"),
        child_chunk_ids=row.get("child_ids", []),
        related_doc_ids=related_docs,
    )


# =============================================================================
# Statute/Graph Endpoints
# =============================================================================


@router.get(
    "/statute/{section}",
    response_model=StatuteWithRulesResponse,
    responses={404: {"description": "Statute not found"}},
)
async def get_statute(
    section: str,
    neo4j: Neo4jDep,
    request_id: RequestIdDep,
) -> StatuteWithRulesResponse:
    """Get a statute with its implementing rules and interpreting documents.

    Args:
        section: Statute section number (e.g., '212.05')
    """
    result = get_interpretation_chain(neo4j, section)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "request_id": request_id,
                "error": f"Statute not found: {section}",
                "details": [],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    return StatuteWithRulesResponse(
        statute=result.statute.model_dump() if hasattr(result.statute, 'model_dump') else dict(result.statute),
        implementing_rules=[
            r.model_dump() if hasattr(r, 'model_dump') else dict(r)
            for r in (result.implementing_rules or [])
        ],
        interpreting_cases=[
            c.model_dump() if hasattr(c, 'model_dump') else dict(c)
            for c in (result.interpreting_cases or [])
        ],
        interpreting_taas=[
            t.model_dump() if hasattr(t, 'model_dump') else dict(t)
            for t in (result.interpreting_taas or [])
        ],
    )


@router.get(
    "/graph/{doc_id:path}/related",
    response_model=RelatedDocumentsResponse,
)
async def get_related_documents(
    doc_id: str,
    neo4j: Neo4jDep,
) -> RelatedDocumentsResponse:
    """Get documents related to a given document via citations.

    Returns both documents that cite this one and documents cited by this one.
    """
    citing = get_citing_documents(neo4j, doc_id) or []
    cited = get_cited_documents(neo4j, doc_id) or []

    # If it's a statute, also get interpretation chain
    interpretation = None
    if doc_id.startswith("statute:"):
        section = doc_id.replace("statute:", "")
        chain = get_interpretation_chain(neo4j, section)
        if chain:
            interpretation = {
                "implementing_rules": [
                    r.model_dump() if hasattr(r, 'model_dump') else dict(r)
                    for r in (chain.implementing_rules or [])
                ],
                "interpreting_cases": [
                    c.model_dump() if hasattr(c, 'model_dump') else dict(c)
                    for c in (chain.interpreting_cases or [])
                ],
                "interpreting_taas": [
                    t.model_dump() if hasattr(t, 'model_dump') else dict(t)
                    for t in (chain.interpreting_taas or [])
                ],
            }

    return RelatedDocumentsResponse(
        doc_id=doc_id,
        citing_documents=[
            d.model_dump() if hasattr(d, 'model_dump') else dict(d) for d in citing
        ],
        cited_documents=[
            d.model_dump() if hasattr(d, 'model_dump') else dict(d) for d in cited
        ],
        interpretation_chain=interpretation,
    )


# =============================================================================
# Health Endpoint
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check(
    neo4j: Neo4jDep,
    weaviate: WeaviateDep,
) -> HealthResponse:
    """Check health of all backend services.

    Returns individual service status and overall system health.
    """
    services = []

    # Check Neo4j
    neo4j_start = time.perf_counter()
    try:
        neo4j_healthy = neo4j.health_check()
        neo4j_latency = (time.perf_counter() - neo4j_start) * 1000
        services.append(
            ServiceHealth(
                name="neo4j",
                healthy=neo4j_healthy,
                latency_ms=neo4j_latency,
                error=None if neo4j_healthy else "Health check returned False",
            )
        )
    except Exception as e:
        services.append(
            ServiceHealth(
                name="neo4j",
                healthy=False,
                latency_ms=None,
                error=str(e),
            )
        )

    # Check Weaviate
    weaviate_start = time.perf_counter()
    try:
        weaviate_healthy = weaviate.health_check()
        weaviate_latency = (time.perf_counter() - weaviate_start) * 1000
        services.append(
            ServiceHealth(
                name="weaviate",
                healthy=weaviate_healthy,
                latency_ms=weaviate_latency,
                error=None if weaviate_healthy else "Health check returned False",
            )
        )
    except Exception as e:
        services.append(
            ServiceHealth(
                name="weaviate",
                healthy=False,
                latency_ms=None,
                error=str(e),
            )
        )

    # Determine overall status
    all_healthy = all(s.healthy for s in services)
    any_healthy = any(s.healthy for s in services)

    if all_healthy:
        overall_status = "healthy"
    elif any_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.utcnow(),
    )
