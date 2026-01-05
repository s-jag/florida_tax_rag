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

    taa_number: str = Field(..., description="TAA number (e.g., 'TAA 23A-001')")
    title: str = Field(..., description="TAA title/subject from Re: line")
    issue_date: Optional[date] = Field(default=None, description="Date issued")
    tax_type: str = Field(default="", description="Type of tax addressed (Sales, Corporate, etc.)")
    tax_type_code: str = Field(default="", description="Single letter tax type code (A, B, C, etc.)")
    topics: list[str] = Field(default_factory=list, description="Topics covered")
    question: str = Field(default="", description="Question/issue posed in the TAA")
    answer: str = Field(default="", description="Answer/response from DOR")
    statutes_cited: list[str] = Field(default_factory=list, description="Statutes cited")
    rules_cited: list[str] = Field(default_factory=list, description="Rules cited")

    @computed_field
    @property
    def full_citation(self) -> str:
        """Generate the full citation."""
        return f"Fla. DOR {self.taa_number}"

    @computed_field
    @property
    def hierarchy_path(self) -> str:
        """Generate the hierarchical path for context enrichment."""
        return f"Florida DOR Technical Assistance Advisements > {self.tax_type} > {self.taa_number}"


class RawTAA(BaseModel):
    """A raw scraped Technical Assistance Advisement."""

    metadata: TAAMetadata
    text: str = Field(..., description="Plain text content")
    pdf_path: Optional[str] = Field(default=None, description="Path to downloaded PDF")
    source_url: str = Field(..., description="URL where this was scraped from")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of when this was scraped",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class CaseMetadata(BaseModel):
    """Metadata for a Florida court case."""

    case_name: str = Field(..., description="Short case name")
    case_name_full: str = Field(default="", description="Full case name with parties")
    citations: list[str] = Field(default_factory=list, description="Case citations (e.g., '215 So. 3d 46')")
    court: str = Field(..., description="Court name (e.g., 'Supreme Court of Florida')")
    court_id: str = Field(..., description="CourtListener court ID (e.g., 'fla')")
    date_filed: Optional[date] = Field(default=None, description="Date the opinion was filed")
    docket_number: str = Field(default="", description="Court docket number")
    judges: str = Field(default="", description="Judges on the panel")
    statutes_cited: list[str] = Field(default_factory=list, description="Statute citations in opinion")
    cases_cited: list[int] = Field(default_factory=list, description="CourtListener IDs of cited cases")
    cluster_id: int = Field(..., description="CourtListener cluster ID for this case")

    @computed_field
    @property
    def full_citation(self) -> str:
        """Generate the full citation."""
        if self.citations:
            return f"{self.case_name}, {self.citations[0]}"
        return self.case_name

    @computed_field
    @property
    def hierarchy_path(self) -> str:
        """Generate the hierarchical path for context enrichment."""
        return f"Florida Case Law > {self.court} > {self.case_name}"


class RawCase(BaseModel):
    """A raw scraped court case from CourtListener."""

    metadata: CaseMetadata
    opinion_text: str = Field(default="", description="Full opinion text or snippet")
    opinion_html: Optional[str] = Field(default=None, description="HTML version of opinion if available")
    source_url: str = Field(..., description="CourtListener URL for this case")
    pdf_url: Optional[str] = Field(default=None, description="URL to opinion PDF if available")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of when this was scraped",
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ScrapedDocument(BaseModel):
    """Union type for any scraped document."""

    doc_type: str = Field(..., description="Type: 'statute', 'rule', 'taa', or 'case'")
    statute: Optional[RawStatute] = None
    rule: Optional[RawRule] = None
    taa: Optional[RawTAA] = None
    case: Optional[RawCase] = None

    @classmethod
    def from_statute(cls, statute: RawStatute) -> ScrapedDocument:
        return cls(doc_type="statute", statute=statute)

    @classmethod
    def from_rule(cls, rule: RawRule) -> ScrapedDocument:
        return cls(doc_type="rule", rule=rule)

    @classmethod
    def from_taa(cls, taa: RawTAA) -> ScrapedDocument:
        return cls(doc_type="taa", taa=taa)

    @classmethod
    def from_case(cls, case: RawCase) -> ScrapedDocument:
        return cls(doc_type="case", case=case)
