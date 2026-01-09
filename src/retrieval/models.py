"""Data models for the retrieval system."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class CitationContext(BaseModel):
    """Citation relationship context for a chunk."""

    target_doc_id: str = Field(..., description="ID of the cited document")
    target_citation: str = Field(..., description="Citation string of the target")
    relation_type: str = Field(
        ..., description="Type of relation (CITES, IMPLEMENTS, AUTHORITY, INTERPRETS)"
    )
    context_snippet: str | None = Field(
        default=None, description="Surrounding text where citation appears"
    )


class RetrievalResult(BaseModel):
    """A single retrieval result with scores and graph context."""

    # Identity
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    doc_type: str = Field(..., description="Document type (statute, rule, case, taa)")
    level: str = Field(..., description="Chunk level (parent or child)")

    # Content
    text: str = Field(..., description="Chunk text content")
    text_with_ancestry: str | None = Field(default=None, description="Text with ancestry prefix")
    ancestry: str | None = Field(
        default=None, description="Hierarchical path (e.g., Chapter 212 > § 212.05)"
    )
    citation: str | None = Field(
        default=None, description="Legal citation (e.g., Fla. Stat. § 212.05)"
    )
    effective_date: date | None = Field(default=None, description="Effective date of the document")
    token_count: int | None = Field(default=None, description="Number of tokens in the chunk")

    # Scores
    score: float = Field(..., description="Combined relevance score")
    vector_score: float | None = Field(
        default=None, description="Score from vector similarity search"
    )
    keyword_score: float | None = Field(default=None, description="Score from BM25 keyword search")
    graph_boost: float = Field(default=0.0, description="Score boost from graph relationships")

    # Graph enrichment
    parent_chunk_id: str | None = Field(
        default=None, description="ID of the parent chunk (for child chunks)"
    )
    related_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of related chunks from graph expansion"
    )
    citation_context: list[CitationContext] = Field(
        default_factory=list, description="Citation relationships for this chunk"
    )

    # Source tracking
    source: str = Field(
        default="hybrid",
        description="Source of retrieval (hybrid, vector, keyword, graph)",
    )
