"""Pydantic models for scraped legal documents."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class StatuteMetadata(BaseModel):
    """Metadata for a Florida Statute section."""

    title: str = Field(..., description="Title name (e.g., 'TAXATION AND FINANCE')")
    title_number: int = Field(..., description="Title number (e.g., 14 for Title XIV)")
    chapter: int = Field(..., description="Chapter number (e.g., 212)")
    chapter_name: str = Field(default="", description="Chapter name (e.g., 'Tax on Sales')")
    section: str = Field(..., description="Section number (e.g., '212.05')")
    section_name: str = Field(default="", description="Section name/title")
    subsection: Optional[str] = Field(default=None, description="Subsection (e.g., '(1)(a)')")
    effective_date: Optional[date] = Field(
        default=None, description="Effective date of this version"
    )
    history: list[str] = Field(default_factory=list, description="Amendment history/years")

    @computed_field
    @property
    def full_citation(self) -> str:
        """Generate the full legal citation."""
        base = f"Fla. Stat. § {self.section}"
        if self.subsection:
            base += self.subsection
        return base

    @computed_field
    @property
    def hierarchy_path(self) -> str:
        """Generate the hierarchical path for context enrichment."""
        parts = [
            "Florida Statutes",
            f"Title {self.title_number} ({self.title})",
            f"Chapter {self.chapter}",
            f"Section {self.section}",
        ]
        if self.subsection:
            parts.append(f"Subsection {self.subsection}")
        return " > ".join(parts)


class RawStatute(BaseModel):
    """A raw scraped Florida Statute."""

    metadata: StatuteMetadata
    text: str = Field(..., description="Plain text content of the statute")
    html: str = Field(..., description="Original HTML content (preserved for re-parsing)")
    source_url: str = Field(..., description="URL where this was scraped from")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of when this was scraped",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class RuleMetadata(BaseModel):
    """Metadata for a Florida Administrative Code rule."""

    chapter: str = Field(..., description="Chapter (e.g., '12A-1')")
    rule_number: str = Field(..., description="Full rule number (e.g., '12A-1.005')")
    title: str = Field(..., description="Rule title")
    effective_date: Optional[date] = Field(default=None, description="Effective date")
    references_statutes: list[str] = Field(
        default_factory=list,
        description="Statute citations referenced by this rule",
    )
    rulemaking_authority: list[str] = Field(
        default_factory=list,
        description="Statutes that provide rulemaking authority",
    )
    law_implemented: list[str] = Field(
        default_factory=list,
        description="Statutes that this rule implements",
    )

    @computed_field
    @property
    def full_citation(self) -> str:
        """Generate the full legal citation."""
        return f"Fla. Admin. Code R. {self.rule_number}"

    @computed_field
    @property
    def hierarchy_path(self) -> str:
        """Generate the hierarchical path for context enrichment."""
        return f"Florida Administrative Code > Chapter {self.chapter} > Rule {self.rule_number}"


class RawRule(BaseModel):
    """A raw scraped Florida Administrative Code rule."""

    metadata: RuleMetadata
    text: str = Field(..., description="Plain text content of the rule")
    html: str = Field(..., description="Original HTML content")
    source_url: str = Field(..., description="URL where this was scraped from")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of when this was scraped",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TAAMetadata(BaseModel):
    """Metadata for a Technical Assistance Advisement."""

    taa_number: str = Field(..., description="TAA number (e.g., '23A-001')")
    title: str = Field(..., description="TAA title/subject")
    issue_date: Optional[date] = Field(default=None, description="Date issued")
    tax_type: str = Field(default="", description="Type of tax addressed")
    topics: list[str] = Field(default_factory=list, description="Topics covered")
    statutes_cited: list[str] = Field(default_factory=list, description="Statutes cited")
    rules_cited: list[str] = Field(default_factory=list, description="Rules cited")

    @computed_field
    @property
    def full_citation(self) -> str:
        """Generate the full citation."""
        return f"Fla. DOR TAA {self.taa_number}"


class RawTAA(BaseModel):
    """A raw scraped Technical Assistance Advisement."""

    metadata: TAAMetadata
    text: str = Field(..., description="Plain text content")
    html: str = Field(..., description="Original HTML content")
    source_url: str = Field(..., description="URL where this was scraped from")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of when this was scraped",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ScrapedDocument(BaseModel):
    """Union type for any scraped document."""

    doc_type: str = Field(..., description="Type: 'statute', 'rule', or 'taa'")
    statute: Optional[RawStatute] = None
    rule: Optional[RawRule] = None
    taa: Optional[RawTAA] = None

    @classmethod
    def from_statute(cls, statute: RawStatute) -> ScrapedDocument:
        return cls(doc_type="statute", statute=statute)

    @classmethod
    def from_rule(cls, rule: RawRule) -> ScrapedDocument:
        return cls(doc_type="rule", rule=rule)

    @classmethod
    def from_taa(cls, taa: RawTAA) -> ScrapedDocument:
        return cls(doc_type="taa", taa=taa)
