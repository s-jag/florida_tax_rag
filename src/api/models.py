"""Pydantic models for FastAPI request/response handling."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class DocType(str, Enum):
    """Document types for filtering."""

    STATUTE = "statute"
    RULE = "rule"
    CASE = "case"
    TAA = "taa"


# =============================================================================
# Request Models
# =============================================================================


class QueryOptions(BaseModel):
    """Options for query execution."""

    doc_types: list[DocType] | None = Field(
        default=None,
        description="Filter by document types",
    )
    tax_year: int | None = Field(
        default=None,
        ge=1990,
        le=2030,
        description="Specific tax year to focus on",
    )
    expand_graph: bool = Field(
        default=True,
        description="Whether to expand results using graph traversal",
    )
    include_reasoning: bool = Field(
        default=False,
        description="Include reasoning steps in response",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Request timeout in seconds",
    )


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The tax law question to answer",
    )
    options: QueryOptions = Field(
        default_factory=QueryOptions,
        description="Query options",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the Florida state sales tax rate?",
                    "options": {"include_reasoning": False, "timeout_seconds": 60},
                },
                {
                    "query": "Are groceries exempt from sales tax in Florida?",
                    "options": {"doc_types": ["statute", "rule"], "tax_year": 2024},
                },
            ]
        }
    }


# =============================================================================
# Response Models
# =============================================================================


class CitationResponse(BaseModel):
    """A citation in the response."""

    doc_id: str = Field(..., description="Document identifier")
    citation: str = Field(..., description="Legal citation text")
    doc_type: str = Field(..., description="Document type")
    text_snippet: str = Field(..., description="Relevant text snippet")


class SourceResponse(BaseModel):
    """A source chunk used in generating the answer."""

    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    doc_type: str = Field(..., description="Document type")
    citation: str | None = Field(default=None, description="Legal citation")
    text: str = Field(..., description="Chunk text content")
    effective_date: datetime | None = Field(default=None, description="Document effective date")
    relevance_score: float = Field(..., description="Relevance score 0-1")


class ReasoningStep(BaseModel):
    """A step in the agent's reasoning process."""

    step_number: int = Field(..., description="Step number")
    node: str = Field(..., description="Graph node that produced this step")
    description: str = Field(..., description="Description of what happened")


class QueryResponse(BaseModel):
    """Response from /query endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    answer: str = Field(..., description="Generated answer with inline citations")
    citations: list[CitationResponse] = Field(
        default_factory=list, description="Citations referenced in the answer"
    )
    sources: list[SourceResponse] = Field(default_factory=list, description="Source chunks used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    warnings: list[str] = Field(default_factory=list, description="Warnings about the response")
    reasoning_steps: list[ReasoningStep] | None = Field(
        default=None, description="Reasoning steps (if requested)"
    )
    validation_passed: bool = Field(..., description="Whether validation passed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    stage_timings: dict[str, float] | None = Field(
        default=None, description="Per-stage timing breakdown in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "abc-123-def",
                "answer": "The Florida state sales tax rate is 6% pursuant to Fla. Stat. § 212.05. This rate applies to most retail sales of tangible personal property...",
                "citations": [
                    {
                        "doc_id": "statute:212.05",
                        "citation": "Fla. Stat. § 212.05",
                        "doc_type": "statute",
                        "text_snippet": "There is levied on each taxable transaction...",
                    }
                ],
                "sources": [
                    {
                        "chunk_id": "statute:212.05:chunk_001",
                        "doc_id": "statute:212.05",
                        "doc_type": "statute",
                        "citation": "Fla. Stat. § 212.05",
                        "text": "There is levied on each taxable transaction...",
                        "relevance_score": 0.92,
                    }
                ],
                "confidence": 0.92,
                "warnings": [],
                "validation_passed": True,
                "processing_time_ms": 3250,
            }
        }
    }


# =============================================================================
# Chunk/Source Detail Models
# =============================================================================


class ChunkDetailResponse(BaseModel):
    """Full chunk details for /sources/{chunk_id}."""

    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    doc_type: str = Field(..., description="Document type")
    level: str = Field(..., description="Chunk level (parent or child)")
    text: str = Field(..., description="Chunk text content")
    text_with_ancestry: str | None = Field(default=None, description="Text with ancestry context")
    ancestry: str | None = Field(default=None, description="Ancestry path")
    citation: str | None = Field(default=None, description="Legal citation")
    effective_date: datetime | None = Field(default=None, description="Document effective date")
    token_count: int | None = Field(default=None, description="Token count")
    parent_chunk_id: str | None = Field(
        default=None, description="Parent chunk ID if this is a child"
    )
    child_chunk_ids: list[str] = Field(default_factory=list, description="Child chunk IDs")
    related_doc_ids: list[str] = Field(
        default_factory=list, description="Related document IDs via citations"
    )


class StatuteWithRulesResponse(BaseModel):
    """Statute with implementing rules for /statute/{section}."""

    statute: dict = Field(..., description="Statute document data")
    implementing_rules: list[dict] = Field(
        default_factory=list, description="Rules that implement this statute"
    )
    interpreting_cases: list[dict] = Field(
        default_factory=list, description="Cases that interpret this statute"
    )
    interpreting_taas: list[dict] = Field(
        default_factory=list, description="TAAs that interpret this statute"
    )


class RelatedDocumentsResponse(BaseModel):
    """Related documents for /graph/{doc_id}/related."""

    doc_id: str = Field(..., description="Document identifier")
    citing_documents: list[dict] = Field(
        default_factory=list, description="Documents that cite this one"
    )
    cited_documents: list[dict] = Field(
        default_factory=list, description="Documents cited by this one"
    )
    interpretation_chain: dict | None = Field(
        default=None, description="Full interpretation chain (for statutes)"
    )


# =============================================================================
# Health/Status Models
# =============================================================================


class ServiceHealth(BaseModel):
    """Health status of a single service."""

    name: str = Field(..., description="Service name")
    healthy: bool = Field(..., description="Whether service is healthy")
    latency_ms: float | None = Field(default=None, description="Health check latency in ms")
    error: str | None = Field(default=None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    services: list[ServiceHealth] = Field(
        default_factory=list, description="Individual service health"
    )
    timestamp: datetime = Field(..., description="Health check timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "services": [
                    {"name": "neo4j", "healthy": True, "latency_ms": 12.5},
                    {"name": "weaviate", "healthy": True, "latency_ms": 8.3},
                ],
                "timestamp": "2024-01-15T10:30:00.000Z",
            }
        }
    }


class ErrorDetail(BaseModel):
    """Error detail for ErrorResponse."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: str | None = Field(default=None, description="Field that caused error")


class ErrorResponse(BaseModel):
    """Standard error response."""

    request_id: str = Field(..., description="Request identifier")
    error: str = Field(..., description="Error summary")
    details: list[ErrorDetail] = Field(
        default_factory=list, description="Detailed error information"
    )
    timestamp: datetime = Field(..., description="Error timestamp")


# =============================================================================
# Metrics Models
# =============================================================================


class LatencyStats(BaseModel):
    """Latency statistics for metrics."""

    avg: float = Field(..., description="Average latency in ms")
    min: float | None = Field(default=None, description="Minimum latency in ms")
    max: float | None = Field(default=None, description="Maximum latency in ms")


class MetricsResponse(BaseModel):
    """Response from /metrics endpoint."""

    total_queries: int = Field(..., description="Total queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    success_rate_percent: float = Field(..., description="Success rate percentage")
    latency_ms: LatencyStats = Field(..., description="Latency statistics")
    errors_by_type: dict[str, int] = Field(default_factory=dict, description="Error counts by type")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    started_at: str = Field(..., description="Start time ISO format")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_queries": 1500,
                "successful_queries": 1425,
                "failed_queries": 75,
                "success_rate_percent": 95.0,
                "latency_ms": {"avg": 3250.5, "min": 1200.0, "max": 8500.0},
                "errors_by_type": {"TIMEOUT": 50, "RETRIEVAL_ERROR": 15, "VALIDATION_ERROR": 10},
                "uptime_seconds": 86400.0,
                "started_at": "2024-01-14T10:30:00.000Z",
            }
        }
    }
