"""Pydantic models for the generation layer."""

from __future__ import annotations

from enum import Enum
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


# ============================================================================
# Hallucination Detection Models
# ============================================================================


class HallucinationType(str, Enum):
    """Types of hallucinations that can be detected in generated responses."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    """Claim not backed by any source document."""

    MISQUOTED_TEXT = "misquoted_text"
    """Quote or paraphrase doesn't match source meaning."""

    FABRICATED_CITATION = "fabricated_citation"
    """Citation references non-existent law or document."""

    OUTDATED_INFO = "outdated_info"
    """Information from superseded or repealed law."""

    MISATTRIBUTED = "misattributed"
    """Claim attributed to wrong source."""

    OVERGENERALIZATION = "overgeneralization"
    """Single case or exception presented as general rule."""


class DetectedHallucination(BaseModel):
    """A single detected hallucination in the response.

    Contains details about the hallucinated claim, its type, severity,
    and suggested correction if available.
    """

    claim_text: str = Field(
        ...,
        description="The exact text from the response containing the hallucination",
    )
    hallucination_type: HallucinationType = Field(
        ...,
        description="Classification of the hallucination type",
    )
    cited_source: Optional[str] = Field(
        default=None,
        description="The citation referenced for this claim, if any",
    )
    actual_source_text: Optional[str] = Field(
        default=None,
        description="What the source actually says (for misquoted/misattributed)",
    )
    severity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Severity score: 0.0=minor, 1.0=critical legal error",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this is considered a hallucination",
    )
    suggested_correction: Optional[str] = Field(
        default=None,
        description="Suggested corrected text, if correction is possible",
    )


class ValidationResult(BaseModel):
    """Complete result of response validation.

    Contains all detected hallucinations, verified claims, and
    flags indicating whether correction or regeneration is needed.
    """

    hallucinations: list[DetectedHallucination] = Field(
        default_factory=list,
        description="List of detected hallucinations",
    )
    verified_claims: list[str] = Field(
        default_factory=list,
        description="Claims that were verified against sources",
    )
    overall_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall accuracy score (1.0 = fully verified)",
    )
    needs_regeneration: bool = Field(
        default=False,
        description="True if hallucinations are severe enough to require regeneration",
    )
    needs_correction: bool = Field(
        default=False,
        description="True if response can be corrected without full regeneration",
    )
    validation_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about the validation process",
    )


class CorrectionResult(BaseModel):
    """Result of applying corrections to a response.

    Contains the corrected answer and details about what changes were made.
    """

    corrected_answer: str = Field(
        ...,
        description="The corrected response text",
    )
    corrections_made: list[str] = Field(
        default_factory=list,
        description="List of corrections applied",
    )
    disclaimers_added: list[str] = Field(
        default_factory=list,
        description="Disclaimers added to the response",
    )
    confidence_adjustment: float = Field(
        default=0.0,
        ge=-1.0,
        le=0.0,
        description="Negative adjustment to apply to confidence score",
    )
