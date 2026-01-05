"""Citation extraction from legal text.

This module provides comprehensive citation extraction for Florida legal
documents, including statutes, administrative rules, court cases, and TAAs.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Import existing patterns from scrapers
from src.scrapers.utils import STATUTE_PATTERN, RULE_PATTERN


class CitationType(str, Enum):
    """Types of legal citations."""

    STATUTE = "statute"
    RULE = "rule"
    CASE = "case"
    TAA = "taa"
    CHAPTER = "chapter"


class RelationType(str, Enum):
    """Types of citation relationships."""

    CITES = "cites"  # General citation
    IMPLEMENTS = "implements"  # Rule implements statute
    AUTHORITY = "authority"  # Rulemaking authority
    INTERPRETS = "interprets"  # Case interprets statute/rule
    AMENDS = "amends"  # Amendment reference
    SUPERSEDES = "supersedes"  # Supersedes prior authority


class Citation(BaseModel):
    """A parsed legal citation."""

    # Core fields
    raw_text: str = Field(..., description="Original text matched")
    normalized: str = Field(..., description="Normalized citation string")
    citation_type: CitationType = Field(..., description="Type of citation")

    # Parsed components (depend on type)
    chapter: Optional[str] = Field(default=None, description="Chapter number (e.g., '212')")
    section: Optional[str] = Field(default=None, description="Section number (e.g., '212.05')")
    subsection: Optional[str] = Field(default=None, description="Subsection (e.g., '(1)(a)')")

    # For case citations
    volume: Optional[int] = Field(default=None, description="Reporter volume")
    reporter: Optional[str] = Field(default=None, description="Reporter name (e.g., 'So. 2d')")
    page: Optional[int] = Field(default=None, description="Starting page")
    year: Optional[int] = Field(default=None, description="Decision year")
    court: Optional[str] = Field(default=None, description="Court name")

    # Match metadata
    start_pos: int = Field(default=0, description="Start position in text")
    end_pos: int = Field(default=0, description="End position in text")


class CitationRelation(BaseModel):
    """A relationship between a source chunk and a target citation."""

    source_chunk_id: str = Field(..., description="ID of chunk containing citation")
    source_doc_id: str = Field(..., description="ID of source document")
    target_citation: Citation = Field(..., description="The extracted citation")
    relation_type: RelationType = Field(..., description="Type of relationship")
    context: str = Field(default="", description="Surrounding text for context")
    target_doc_id: Optional[str] = Field(default=None, description="Resolved target document ID")


# Florida case citation patterns
FLORIDA_CASE_PATTERN = re.compile(
    r"""
    (\d{1,4})                                           # Volume number
    \s+
    (So\.\s*[23]d|Fla\.\s*L\.\s*Weekly(?:\s*Supp\.)?|Fla\.)  # Reporter
    \s+
    (\d{1,5})                                           # Starting page
    (?:\s*,\s*(\d+))?                                   # Optional pinpoint page
    (?:\s*\(([^)]+?)\s*(\d{4})\))?                      # Court and year in parens
    """,
    re.VERBOSE | re.IGNORECASE,
)

# LEXIS/Westlaw patterns
LEXIS_PATTERN = re.compile(
    r"""
    (\d{4})                                             # Year
    \s+
    (Fla\.?\s*LEXIS|WL)                                 # Reporter type
    \s+
    (\d+)                                               # Document number
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Chapter reference pattern
CHAPTER_PATTERN = re.compile(
    r"""
    (?:chapters?|ch\.)\s+                               # Chapter keyword
    (\d{1,3})                                           # Chapter number
    (?:\s*[-–]\s*(\d{1,3}))?                            # Optional range end
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_statute_citations(text: str) -> list[Citation]:
    """Extract Florida Statute citations from text.

    Wraps the existing STATUTE_PATTERN and returns Citation objects
    with parsed components.

    Args:
        text: The text to search for citations.

    Returns:
        List of unique Citation objects.
    """
    citations = []

    # Fix HTML artifacts (newlines after "s.")
    text_cleaned = re.sub(r"s\.\s*\n\s*", "s. ", text)

    for match in STATUTE_PATTERN.finditer(text_cleaned):
        # Groups 1,2 are from the first alternative (§ prefix)
        # Groups 3,4 are from the second alternative (F.S. suffix)
        section = match.group(1) or match.group(3)
        subsection = (match.group(2) or match.group(4) or "").lower().replace(" ", "")

        if not section:
            continue

        # Parse chapter from section (e.g., "212" from "212.05")
        chapter = section.split(".")[0] if "." in section else section

        normalized = section + subsection

        citations.append(
            Citation(
                raw_text=match.group(0),
                normalized=normalized,
                citation_type=CitationType.STATUTE,
                chapter=chapter,
                section=section,
                subsection=subsection if subsection else None,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Deduplicate by normalized form
    seen = set()
    unique = []
    for c in citations:
        if c.normalized not in seen:
            seen.add(c.normalized)
            unique.append(c)

    return unique


def extract_rule_citations(text: str) -> list[Citation]:
    """Extract Florida Administrative Code rule citations from text.

    Args:
        text: The text to search for citations.

    Returns:
        List of unique Citation objects.
    """
    citations = []

    for match in RULE_PATTERN.finditer(text):
        # Groups 1,2 are from the first alternative (Rule prefix)
        # Groups 3,4 are from the second alternative (F.A.C. suffix)
        rule_num = match.group(1) or match.group(3)
        subsection = (match.group(2) or match.group(4) or "").lower().replace(" ", "")

        if not rule_num:
            continue

        # Normalize: uppercase the letter in rule number
        normalized = re.sub(
            r"(\d+)([a-z])-",
            lambda m: f"{m.group(1)}{m.group(2).upper()}-",
            rule_num,
            flags=re.IGNORECASE,
        )
        normalized += subsection

        # Extract chapter (e.g., "12A-1" from "12A-1.005")
        chapter = normalized.rsplit(".", 1)[0] if "." in normalized else normalized

        # Section without subsection
        section = normalized.split("(")[0] if "(" in normalized else normalized

        citations.append(
            Citation(
                raw_text=match.group(0),
                normalized=normalized,
                citation_type=CitationType.RULE,
                chapter=chapter,
                section=section,
                subsection=subsection if subsection else None,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in citations:
        if c.normalized not in seen:
            seen.add(c.normalized)
            unique.append(c)

    return unique


def extract_case_citations(text: str) -> list[Citation]:
    """Extract Florida court case citations from text.

    Handles Southern Reporter (So. 2d, So. 3d), Florida Law Weekly,
    LEXIS, and Westlaw citations.

    Args:
        text: The text to search for citations.

    Returns:
        List of unique Citation objects.
    """
    citations = []

    # Standard reporter citations
    for match in FLORIDA_CASE_PATTERN.finditer(text):
        volume = int(match.group(1))
        reporter = match.group(2)
        page = int(match.group(3))
        court = match.group(5)
        year = int(match.group(6)) if match.group(6) else None

        # Normalize reporter (remove extra spaces)
        reporter_norm = re.sub(r"\s+", " ", reporter).strip()
        normalized = f"{volume} {reporter_norm} {page}"

        citations.append(
            Citation(
                raw_text=match.group(0),
                normalized=normalized,
                citation_type=CitationType.CASE,
                volume=volume,
                reporter=reporter_norm,
                page=page,
                year=year,
                court=court,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # LEXIS/WL citations
    for match in LEXIS_PATTERN.finditer(text):
        year = int(match.group(1))
        reporter = match.group(2)
        doc_num = match.group(3)

        # Normalize
        reporter_norm = re.sub(r"\s+", " ", reporter).strip()
        normalized = f"{year} {reporter_norm} {doc_num}"

        citations.append(
            Citation(
                raw_text=match.group(0),
                normalized=normalized,
                citation_type=CitationType.CASE,
                reporter=reporter_norm,
                year=year,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in citations:
        if c.normalized not in seen:
            seen.add(c.normalized)
            unique.append(c)

    return unique


def extract_chapter_citations(text: str) -> list[Citation]:
    """Extract chapter references (e.g., 'chapter 196', 'chapters 192-197').

    Args:
        text: The text to search for citations.

    Returns:
        List of Citation objects.
    """
    citations = []

    for match in CHAPTER_PATTERN.finditer(text):
        start_chapter = match.group(1)
        end_chapter = match.group(2)

        if end_chapter:
            normalized = f"chapter:{start_chapter}-{end_chapter}"
        else:
            normalized = f"chapter:{start_chapter}"

        citations.append(
            Citation(
                raw_text=match.group(0),
                normalized=normalized,
                citation_type=CitationType.CHAPTER,
                chapter=start_chapter,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in citations:
        if c.normalized not in seen:
            seen.add(c.normalized)
            unique.append(c)

    return unique


def extract_all_citations(text: str) -> list[Citation]:
    """Extract all types of citations from text.

    Args:
        text: The text to search for citations.

    Returns:
        List of all Citation objects found.
    """
    citations = []
    citations.extend(extract_statute_citations(text))
    citations.extend(extract_rule_citations(text))
    citations.extend(extract_case_citations(text))
    citations.extend(extract_chapter_citations(text))
    return citations


def detect_relation_type(
    citation: Citation,
    context: str,
    source_doc_type: str,
) -> RelationType:
    """Detect the relationship type based on context and document type.

    Args:
        citation: The extracted citation
        context: Text surrounding the citation (50-100 chars each side)
        source_doc_type: Type of source document (statute, rule, taa, case)

    Returns:
        The detected relationship type
    """
    context_lower = context.lower()

    # Check for authority keywords (typically in rules)
    if source_doc_type == "rule":
        if any(
            kw in context_lower
            for kw in ["rulemaking authority", "authority:", "specific authority"]
        ):
            return RelationType.AUTHORITY
        if any(
            kw in context_lower
            for kw in ["law implemented", "implements", "implementing"]
        ):
            return RelationType.IMPLEMENTS

    # Check for interpretation (cases)
    if source_doc_type == "case":
        if any(
            kw in context_lower
            for kw in ["interpret", "constru", "held that", "holding", "pursuant to"]
        ):
            return RelationType.INTERPRETS

    # Check for amendment
    if any(kw in context_lower for kw in ["amend", "amended by", "as amended"]):
        return RelationType.AMENDS

    # Check for supersession
    if any(kw in context_lower for kw in ["supersed", "replac", "repeal"]):
        return RelationType.SUPERSEDES

    # Default to general citation
    return RelationType.CITES


def get_context(text: str, start: int, end: int, window: int = 75) -> str:
    """Extract context window around a citation match.

    Args:
        text: The full text
        start: Start position of the citation
        end: End position of the citation
        window: Number of characters on each side

    Returns:
        The context string
    """
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    return text[context_start:context_end]
