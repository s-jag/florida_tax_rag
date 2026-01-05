"""Pydantic models for the generation layer."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedCitation(BaseModel):
    """A citation extracted from generated text.

    This represents a citation found in the LLM's response before
    it has been validated against the source chunks.
    """

    citation_text: str = Field(
        ...,
        description="The citation text, e.g., '§ 212.05(1)' or 'Rule 12A-1.005'",
    )
    position: int = Field(
        ...,
        description="Character position in the response where citation appears",
    )
    citation_type: str = Field(
        default="unknown",
        description="Type of citation: 'statute', 'rule', 'case', 'taa', 'unknown'",
    )


class ValidatedCitation(BaseModel):
    """A citation that has been verified against source chunks.

    After extraction, citations are validated to ensure they reference
    documents that were actually provided in the context.
    """

    citation_text: str = Field(
        ...,
        description="The citation text from the response",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the source chunk if found, None if hallucinated",
    )
    verified: bool = Field(
        ...,
        description="True if citation matches a chunk, False if hallucinated",
    )
    raw_text: str = Field(
        default="",
        description="Actual text from the source chunk (empty if not verified)",
    )
    doc_type: str = Field(
        default="unknown",
        description="Document type: 'statute', 'rule', 'case', 'taa', 'unknown'",
    )


class GeneratedResponse(BaseModel):
    """Complete response from the generation layer.

    Contains the generated answer, validated citations, confidence score,
    and metadata about the generation process.
    """

    answer: str = Field(
        ...,
        description="The generated answer text with inline citations",
    )
    citations: list[ValidatedCitation] = Field(
        default_factory=list,
        description="List of validated citations extracted from the answer",
    )
    chunks_used: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that were included in the context",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) based on source quality and verification",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or caveats about the response",
    )
    generation_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about generation (model, tokens, timing, etc.)",
    )
