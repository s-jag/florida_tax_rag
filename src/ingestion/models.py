"""Unified document models for the consolidated legal corpus."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of legal documents in the corpus."""

    STATUTE = "statute"
    RULE = "rule"
    TAA = "taa"
    CASE = "case"


class LegalDocument(BaseModel):
    """Unified model for all legal documents in the corpus.

    This model provides a consistent interface for all document types,
    making it easier to process, embed, and query the entire corpus.
    """

    # Core identity
    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'statute:212.05', 'case:1093614')",
    )
    doc_type: DocumentType = Field(..., description="Type of legal document")
    title: str = Field(..., description="Document title or name")
    full_citation: str = Field(..., description="Canonical legal citation")

    # Content
    text: str = Field(..., description="Plain text content of the document")

    # Temporal
    effective_date: Optional[date] = Field(
        default=None,
        description="Effective date (for statutes/rules) or filing date (for cases)",
    )

    # Source
    source_url: str = Field(..., description="Original source URL")

    # Hierarchy (for statutes/rules)
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent document ID (e.g., chapter for section)",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="Child document IDs (e.g., subsections)",
    )

    # Cross-references (raw citations, resolved in graph building phase)
    cites_statutes: list[str] = Field(
        default_factory=list,
        description="Statute citations (e.g., ['212.05', '212.02(10)(i)'])",
    )
    cites_rules: list[str] = Field(
        default_factory=list,
        description="Rule citations (e.g., ['12A-1.073'])",
    )
    cites_cases: list[str] = Field(
        default_factory=list,
        description="Case document IDs (e.g., ['case:1093614'])",
    )

    # Metadata
    scraped_at: datetime = Field(..., description="When the document was scraped")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific extra fields that don't fit unified schema",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    @classmethod
    def generate_id(cls, doc_type: DocumentType, identifier: str) -> str:
        """Generate a consistent document ID.

        Args:
            doc_type: The document type
            identifier: The type-specific identifier (section, rule_number, cluster_id, etc.)

        Returns:
            A prefixed ID string like 'statute:212.05'
        """
        return f"{doc_type.value}:{identifier}"


class CorpusMetadata(BaseModel):
    """Metadata for the consolidated corpus."""

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the corpus was created",
    )
    total_documents: int = Field(..., description="Total number of documents")
    by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Document count by type",
    )
    version: str = Field(default="1.0", description="Corpus schema version")


class Corpus(BaseModel):
    """The consolidated legal document corpus."""

    metadata: CorpusMetadata
    documents: list[LegalDocument] = Field(
        default_factory=list,
        description="All legal documents in the corpus",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
