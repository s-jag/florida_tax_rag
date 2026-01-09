"""API routes for the Florida Tax RAG system."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.agent import TaxAgentState
from src.graph.queries import (
    get_cited_documents,
    get_citing_documents,
    get_interpretation_chain,
)
from src.observability.logging import get_logger
from src.observability.metrics import get_metrics_collector
from src.observability.profiler import profile_request

from .cache import get_query_cache
from .dependencies import (
    AgentGraphDep,
    Neo4jDep,
    RequestIdDep,
    WeaviateDep,
)
from .errors import NotFoundError, QueryTimeoutError, RetrievalError
from .models import (
    ChunkDetailResponse,
    CitationResponse,
    HealthResponse,
    LatencyStats,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
    RelatedDocumentsResponse,
    ServiceHealth,
    SourceResponse,
    StatuteWithRulesResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Query Endpoints
# =============================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute Tax Law Query",
    description="""
Execute a Florida tax law query through the RAG agent.

The agent performs:
1. **Query decomposition** - Breaks complex queries into sub-queries
2. **Hybrid retrieval** - Combines vector, keyword, and graph search
3. **Relevance scoring** - Filters and ranks results
4. **Temporal validation** - Ensures documents are current
5. **Answer synthesis** - Generates answer with citations
6. **Validation** - Detects and corrects hallucinations
    """,
    responses={
        200: {
            "description": "Successful query response",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "abc-123-def",
                        "answer": "The Florida state sales tax rate is 6%...",
                        "citations": [
                            {"doc_id": "statute:212.05", "citation": "Fla. Stat. § 212.05"}
                        ],
                        "confidence": 0.92,
                        "validation_passed": True,
                        "processing_time_ms": 3250,
                    }
                }
            },
        },
        408: {"description": "Request timeout - query exceeded time limit"},
        500: {"description": "Internal error during query processing"},
    },
    tags=["Query"],
)
async def query(
    request: QueryRequest,
    request_id: RequestIdDep,
    graph: AgentGraphDep,
) -> QueryResponse:
    """Execute a tax law query through the RAG agent."""
    start_time = time.perf_counter()

    # Check cache first
    cache = get_query_cache()
    cache_options = {
        "tax_year": request.options.tax_year,
        "include_reasoning": request.options.include_reasoning,
    }

    if cache:
        cached = await cache.get(request.query, cache_options)
        if cached:
            logger.info("cache_hit", query_preview=request.query[:50])
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return QueryResponse(
                request_id=request_id,
                answer=cached.get("answer", ""),
                citations=[CitationResponse(**c) for c in cached.get("citations", [])],
                sources=[SourceResponse(**s) for s in cached.get("sources", [])],
                confidence=cached.get("confidence", 0.0),
                warnings=cached.get("warnings", []) + ["Response from cache"],
                reasoning_steps=None,  # Don't return reasoning from cache
                validation_passed=cached.get("validation_passed", False),
                processing_time_ms=processing_time_ms,
                stage_timings={"cache_hit": float(processing_time_ms)},
            )

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

    stage_timings: dict[str, float] = {}

    try:
        logger.info(
            "query_execution_started",
            query_preview=request.query[:100],
            timeout_seconds=request.options.timeout_seconds,
        )

        # Execute with timeout and profiling
        with profile_request(request_id) as profiler:
            result = await asyncio.wait_for(
                graph.ainvoke(initial_state),
                timeout=request.options.timeout_seconds,
            )
            stage_timings = profiler.get_stage_timings()

    except TimeoutError:
        logger.warning(
            "query_timeout",
            query_preview=request.query[:50],
            timeout_seconds=request.options.timeout_seconds,
        )
        raise QueryTimeoutError(
            message=f"Query exceeded {request.options.timeout_seconds}s timeout",
            details={
                "timeout_seconds": request.options.timeout_seconds,
                "query_preview": request.query[:50],
            },
        )
    except Exception as e:
        logger.exception(
            "query_execution_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise RetrievalError(
            message="Query execution failed",
            details={"error_type": type(e).__name__, "error_message": str(e)},
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
                description=step.get("description", str(step))
                if isinstance(step, dict)
                else str(step),
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

    response = QueryResponse(
        request_id=request_id,
        answer=result.get("final_answer") or "Unable to generate answer",
        citations=citations,
        sources=sources,
        confidence=result.get("confidence", 0.0),
        warnings=warnings,
        reasoning_steps=reasoning_steps,
        validation_passed=result.get("validation_passed", False),
        processing_time_ms=processing_time_ms,
        stage_timings=stage_timings if stage_timings else None,
    )

    # Cache successful response
    if cache and response.answer and response.answer != "Unable to generate answer":
        cache_data = {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "sources": [s.model_dump() for s in response.sources],
            "confidence": response.confidence,
            "warnings": response.warnings,
            "validation_passed": response.validation_passed,
        }
        await cache.set(request.query, cache_options, cache_data)

    return response


@router.post(
    "/query/stream",
    summary="Stream Tax Law Query",
    description="""
Stream query results via Server-Sent Events (SSE).

**Event Types:**
- `status` - Node execution updates (e.g., entering decompose, retrieve)
- `reasoning` - Reasoning step completed with description
- `chunk` - Source chunk retrieved with citation info
- `answer` - Final answer text (may stream in parts)
- `complete` - Query completed with request_id and timing
- `error` - Error occurred with code and message

**Example SSE Stream:**
```
event: status
data: {"node": "decompose", "timestamp": "2024-01-15T10:30:00Z"}

event: chunk
data: {"chunk_id": "statute:212.05:001", "citation": "Fla. Stat. § 212.05"}

event: answer
data: {"text": "The Florida sales tax rate is 6%..."}

event: complete
data: {"request_id": "abc-123", "processing_time_ms": 3250}
```
    """,
    tags=["Query"],
)
async def query_stream(
    request: QueryRequest,
    request_id: RequestIdDep,
    graph: AgentGraphDep,
) -> StreamingResponse:
    """Stream query results via Server-Sent Events."""

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

        except TimeoutError:
            logger.warning("streaming_query_timeout", query_preview=request.query[:50])
            yield f"event: error\ndata: {json.dumps({'code': 'TIMEOUT', 'message': 'Request timed out'})}\n\n"
        except Exception as e:
            logger.exception(
                "streaming_query_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
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
    summary="Get Chunk Details",
    description="""
Get full details for a specific chunk by its ID.

Returns:
- Full text content
- Document metadata (type, citation, effective date)
- Hierarchical relationships (parent/child chunks)
- Related documents via citation graph
    """,
    responses={404: {"description": "Chunk not found"}},
    tags=["Sources"],
)
async def get_chunk(
    chunk_id: str,
    neo4j: Neo4jDep,
    request_id: RequestIdDep,
) -> ChunkDetailResponse:
    """Get full details for a specific chunk by ID."""
    query = """
    MATCH (c:Chunk {id: $chunk_id})
    OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
    OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Chunk)
    OPTIONAL MATCH (child:Chunk)-[:CHILD_OF]->(c)
    RETURN c, d, parent.id AS parent_id, collect(DISTINCT child.id) AS child_ids
    """

    results = neo4j.run_query(query, {"chunk_id": chunk_id})

    if not results or results[0].get("c") is None:
        logger.info("chunk_not_found", chunk_id=chunk_id)
        raise NotFoundError(
            message=f"Chunk not found: {chunk_id}",
            details={"chunk_id": chunk_id},
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
    summary="Get Statute with Related Documents",
    description="""
Get a Florida statute with its full interpretation chain.

Returns:
- **Statute** - The primary statute document
- **Implementing Rules** - F.A.C. rules that implement this statute
- **Interpreting Cases** - Court cases that interpret this statute
- **Interpreting TAAs** - Technical Assistance Advisements

**Example:** `/api/v1/statute/212.05` returns the sales tax rate statute
with all related administrative rules and case law.
    """,
    responses={404: {"description": "Statute not found"}},
    tags=["Graph"],
)
async def get_statute(
    section: str,
    neo4j: Neo4jDep,
    request_id: RequestIdDep,
) -> StatuteWithRulesResponse:
    """Get a statute with its implementing rules and interpreting documents."""
    result = get_interpretation_chain(neo4j, section)

    if result is None:
        logger.info("statute_not_found", section=section)
        raise NotFoundError(
            message=f"Statute not found: {section}",
            details={"section": section},
        )

    return StatuteWithRulesResponse(
        statute=result.statute.model_dump()
        if hasattr(result.statute, "model_dump")
        else dict(result.statute),
        implementing_rules=[
            r.model_dump() if hasattr(r, "model_dump") else dict(r)
            for r in (result.implementing_rules or [])
        ],
        interpreting_cases=[
            c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in (result.interpreting_cases or [])
        ],
        interpreting_taas=[
            t.model_dump() if hasattr(t, "model_dump") else dict(t)
            for t in (result.interpreting_taas or [])
        ],
    )


@router.get(
    "/graph/{doc_id:path}/related",
    response_model=RelatedDocumentsResponse,
    summary="Get Related Documents",
    description="""
Get documents related to a given document via the citation graph.

Returns:
- **Citing Documents** - Documents that cite this document
- **Cited Documents** - Documents cited by this document
- **Interpretation Chain** - For statutes, includes implementing rules and cases

The `doc_id` format is `{type}:{identifier}`:
- `statute:212.05` - Florida Statute section
- `rule:12A-1.001` - F.A.C. rule
- `case:123456` - Court case
- `taa:24B01-001` - Technical Assistance Advisement
    """,
    tags=["Graph"],
)
async def get_related_documents(
    doc_id: str,
    neo4j: Neo4jDep,
) -> RelatedDocumentsResponse:
    """Get documents related to a given document via citations."""
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
                    r.model_dump() if hasattr(r, "model_dump") else dict(r)
                    for r in (chain.implementing_rules or [])
                ],
                "interpreting_cases": [
                    c.model_dump() if hasattr(c, "model_dump") else dict(c)
                    for c in (chain.interpreting_cases or [])
                ],
                "interpreting_taas": [
                    t.model_dump() if hasattr(t, "model_dump") else dict(t)
                    for t in (chain.interpreting_taas or [])
                ],
            }

    return RelatedDocumentsResponse(
        doc_id=doc_id,
        citing_documents=[d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in citing],
        cited_documents=[d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in cited],
        interpretation_chain=interpretation,
    )


# =============================================================================
# Metrics Endpoint
# =============================================================================


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get API Metrics",
    description="""
Get API performance metrics and statistics.

Returns:
- **Query counts** - Total, successful, and failed queries
- **Success rate** - Percentage of successful queries
- **Latency stats** - Average, min, max response times
- **Error breakdown** - Counts by error type
- **Uptime** - Time since server start
    """,
    tags=["Monitoring"],
)
async def get_metrics() -> MetricsResponse:
    """Get API metrics."""
    metrics = get_metrics_collector()
    stats = metrics.get_stats()

    return MetricsResponse(
        total_queries=stats["total_queries"],
        successful_queries=stats["successful_queries"],
        failed_queries=stats["failed_queries"],
        success_rate_percent=stats["success_rate_percent"],
        latency_ms=LatencyStats(
            avg=stats["latency_ms"]["avg"],
            min=stats["latency_ms"]["min"],
            max=stats["latency_ms"]["max"],
        ),
        errors_by_type=stats["errors_by_type"],
        uptime_seconds=stats["uptime_seconds"],
        started_at=stats["started_at"],
    )


# =============================================================================
# Health Endpoint
# =============================================================================


@router.get(
    "/ping",
    summary="Ping",
    description="Simple liveness check endpoint. Returns immediately without checking external services.",
    tags=["Monitoring"],
)
async def ping():
    """Simple ping endpoint for basic liveness check."""
    return {"status": "ok", "message": "pong"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="""
Check health of all backend services.

Returns:
- **Overall status** - `healthy`, `degraded`, or `unhealthy`
- **Service details** - Per-service health and latency

Status meanings:
- `healthy` - All services operational
- `degraded` - Some services unhealthy but system functional
- `unhealthy` - Critical services unavailable
    """,
    tags=["Monitoring"],
)
async def health_check(
    neo4j: Neo4jDep,
    weaviate: WeaviateDep,
) -> HealthResponse:
    """Check health of all backend services."""
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
