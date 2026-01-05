"""Hierarchical chunking for legal documents.

This module implements proposition-based hierarchical chunking that preserves
legal context by splitting documents into parent chunks (full sections) and
child chunks (subsections).
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .models import Corpus, DocumentType, LegalDocument
from .tokenizer import count_tokens


class ChunkLevel(str, Enum):
    """Level of a chunk in the document hierarchy."""

    PARENT = "parent"  # Full section (e.g., § 212.08)
    CHILD = "child"  # Subsection (e.g., § 212.08(7)(a))


class LegalChunk(BaseModel):
    """A chunk of a legal document for embedding and retrieval."""

    # Identity
    id: str = Field(..., description="Unique chunk ID (e.g., 'chunk:statute:212.08:0')")
    doc_id: str = Field(..., description="Parent document ID (e.g., 'statute:212.08')")
    level: ChunkLevel = Field(..., description="Parent or child chunk")

    # Hierarchy context
    ancestry: str = Field(
        ..., description="Hierarchy path (e.g., 'Florida Statutes > Title XIV > Chapter 212 > § 212.08')"
    )
    subsection_path: str = Field(
        default="", description="Subsection marker (e.g., '(7)(a)') or empty for parent"
    )

    # Content
    text: str = Field(..., description="Raw chunk text")
    text_with_ancestry: str = Field(..., description="Text with ancestry prepended for embedding")

    # Parent-child linking
    parent_chunk_id: Optional[str] = Field(default=None, description="ID of parent chunk (for child chunks)")
    child_chunk_ids: list[str] = Field(default_factory=list, description="IDs of child chunks (for parent chunks)")

    # Preserved metadata
    citation: str = Field(..., description="Full legal citation")
    effective_date: Optional[date] = Field(default=None, description="Effective/filing date")
    doc_type: str = Field(..., description="Document type: statute, rule, taa, case")

    # Token stats
    token_count: int = Field(..., description="Number of tokens in text_with_ancestry")

    model_config = {"json_encoders": {date: lambda v: v.isoformat() if v else None}}


# Regex patterns for subsection markers
# These patterns match markers at the start of a line (after newline)
SUBSECTION_PATTERN = re.compile(
    r'\n(\((\d+)\)|\(([a-z])\)|(\d+)\.|([a-z])\.)',
    re.MULTILINE
)

# Pattern to identify top-level subsections only: (1), (2), etc.
TOP_LEVEL_PATTERN = re.compile(r'^\((\d+)\)\s*$', re.MULTILINE)


def parse_top_level_subsections(text: str) -> list[tuple[str, str, int, int]]:
    """Parse text into top-level subsections.

    Only splits on (1), (2), (3) etc. patterns to create meaningful chunks.
    Nested subsections ((a), (b), 1., 2.) are kept within their parent.

    Args:
        text: The document text to parse

    Returns:
        List of (marker, content, start_pos, end_pos) tuples
    """
    subsections = []

    # Find all top-level markers like \n(1)\n, \n(2)\n, etc.
    # The marker should be on its own line
    pattern = re.compile(r'\n\((\d+)\)\n', re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return subsections

    for i, match in enumerate(matches):
        marker = f"({match.group(1)})"
        start = match.end()

        # End is either the next marker or end of text
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()
        if content:
            subsections.append((marker, content, start, end))

    return subsections


def build_ancestry(doc: LegalDocument) -> str:
    """Build an ancestry string from document metadata.

    Args:
        doc: The legal document

    Returns:
        Ancestry string like "Florida Statutes > Title XIV > Chapter 212 > § 212.05"
    """
    if doc.doc_type == DocumentType.STATUTE:
        metadata = doc.metadata
        parts = ["Florida Statutes"]

        title_num = metadata.get("title_number")
        title_name = metadata.get("title_name", "")
        if title_num:
            parts.append(f"Title {title_num}" + (f" ({title_name})" if title_name else ""))

        chapter = metadata.get("chapter")
        if chapter:
            parts.append(f"Chapter {chapter}")

        # Add section from citation
        parts.append(doc.full_citation.replace("Fla. Stat. ", ""))

        return " > ".join(parts)

    elif doc.doc_type == DocumentType.RULE:
        metadata = doc.metadata
        parts = ["Florida Administrative Code"]

        chapter = metadata.get("chapter")
        if chapter:
            parts.append(f"Chapter {chapter}")

        parts.append(doc.full_citation.replace("Fla. Admin. Code R. ", "Rule "))

        return " > ".join(parts)

    elif doc.doc_type == DocumentType.TAA:
        metadata = doc.metadata
        parts = ["Florida DOR Technical Assistance Advisements"]

        tax_type = metadata.get("tax_type")
        if tax_type:
            parts.append(tax_type)

        parts.append(doc.full_citation.replace("Fla. DOR ", ""))

        return " > ".join(parts)

    elif doc.doc_type == DocumentType.CASE:
        metadata = doc.metadata
        parts = ["Florida Case Law"]

        court = metadata.get("court", "")
        if court:
            parts.append(court)

        # Use short case name
        parts.append(doc.title)

        return " > ".join(parts)

    return doc.full_citation


def create_chunk(
    doc: LegalDocument,
    chunk_index: int,
    level: ChunkLevel,
    text: str,
    ancestry: str,
    subsection_path: str = "",
    parent_chunk_id: Optional[str] = None,
) -> LegalChunk:
    """Create a LegalChunk with computed fields.

    Args:
        doc: Source document
        chunk_index: Index of this chunk within the document
        level: Parent or child level
        text: Raw chunk text
        ancestry: Hierarchy path
        subsection_path: Subsection marker if applicable
        parent_chunk_id: ID of parent chunk for child chunks

    Returns:
        A LegalChunk instance
    """
    chunk_id = f"chunk:{doc.id}:{chunk_index}"

    # Build text with ancestry prepended
    text_with_ancestry = f"{ancestry}\n\n{text}"

    return LegalChunk(
        id=chunk_id,
        doc_id=doc.id,
        level=level,
        ancestry=ancestry,
        subsection_path=subsection_path,
        text=text,
        text_with_ancestry=text_with_ancestry,
        parent_chunk_id=parent_chunk_id,
        child_chunk_ids=[],
        citation=doc.full_citation,
        effective_date=doc.effective_date,
        doc_type=doc.doc_type.value,
        token_count=count_tokens(text_with_ancestry),
    )


def chunk_statute(doc: LegalDocument) -> list[LegalChunk]:
    """Chunk a statute document.

    Strategy:
    1. Create parent chunk with full section text
    2. Parse top-level subsection markers: (1), (2), (3), etc.
    3. Create child chunks for each top-level subsection
    4. Link parent <-> children
    5. Prepend ancestry to each chunk

    Args:
        doc: A statute LegalDocument

    Returns:
        List of LegalChunk objects
    """
    chunks = []
    ancestry = build_ancestry(doc)

    # Create parent chunk (full section)
    parent_chunk = create_chunk(
        doc=doc,
        chunk_index=0,
        level=ChunkLevel.PARENT,
        text=doc.text,
        ancestry=ancestry,
    )
    chunks.append(parent_chunk)

    # Parse subsections
    subsections = parse_top_level_subsections(doc.text)

    if not subsections:
        # No subsections found, just return parent
        return chunks

    # Create child chunks
    child_ids = []
    for i, (marker, content, _, _) in enumerate(subsections):
        child_index = i + 1  # Parent is 0
        child_chunk = create_chunk(
            doc=doc,
            chunk_index=child_index,
            level=ChunkLevel.CHILD,
            text=content,
            ancestry=ancestry,
            subsection_path=marker,
            parent_chunk_id=parent_chunk.id,
        )
        chunks.append(child_chunk)
        child_ids.append(child_chunk.id)

    # Update parent with child IDs
    parent_chunk.child_chunk_ids = child_ids

    return chunks


def chunk_rule(doc: LegalDocument) -> list[LegalChunk]:
    """Chunk an administrative rule document.

    Similar to statutes but rules often have lots of administrative
    metadata that we keep in the parent chunk.

    Args:
        doc: A rule LegalDocument

    Returns:
        List of LegalChunk objects
    """
    chunks = []
    ancestry = build_ancestry(doc)

    # Create parent chunk (full rule)
    parent_chunk = create_chunk(
        doc=doc,
        chunk_index=0,
        level=ChunkLevel.PARENT,
        text=doc.text,
        ancestry=ancestry,
    )
    chunks.append(parent_chunk)

    # Parse subsections (same pattern as statutes)
    subsections = parse_top_level_subsections(doc.text)

    if not subsections:
        return chunks

    # Create child chunks
    child_ids = []
    for i, (marker, content, _, _) in enumerate(subsections):
        child_index = i + 1
        child_chunk = create_chunk(
            doc=doc,
            chunk_index=child_index,
            level=ChunkLevel.CHILD,
            text=content,
            ancestry=ancestry,
            subsection_path=marker,
            parent_chunk_id=parent_chunk.id,
        )
        chunks.append(child_chunk)
        child_ids.append(child_chunk.id)

    parent_chunk.child_chunk_ids = child_ids

    return chunks


# TAA section headers to split on
TAA_SECTIONS = [
    "REQUESTED ADVISEMENT",
    "FACTS",
    "TAXPAYER'S POSITION",
    "LAW AND DISCUSSION",
    "CONCLUSION",
]


def chunk_taa(doc: LegalDocument) -> list[LegalChunk]:
    """Chunk a Technical Assistance Advisement.

    TAAs have logical sections that we can split on:
    - Question/Requested Advisement
    - Facts
    - Taxpayer's Position
    - Law and Discussion
    - Conclusion

    If the TAA is short (<2000 tokens), keep as single chunk.

    Args:
        doc: A TAA LegalDocument

    Returns:
        List of LegalChunk objects
    """
    chunks = []
    ancestry = build_ancestry(doc)

    # Create parent chunk (full TAA)
    parent_chunk = create_chunk(
        doc=doc,
        chunk_index=0,
        level=ChunkLevel.PARENT,
        text=doc.text,
        ancestry=ancestry,
    )
    chunks.append(parent_chunk)

    # If short enough, don't split further
    if parent_chunk.token_count < 2000:
        return chunks

    # Try to split on section headers
    text = doc.text
    sections_found = []

    for header in TAA_SECTIONS:
        # Look for header as a standalone line or at start of line
        pattern = re.compile(rf'\n{re.escape(header)}\n', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            sections_found.append((header, match.start(), match.end()))

    if len(sections_found) < 2:
        # Not enough sections to split meaningfully
        return chunks

    # Sort by position
    sections_found.sort(key=lambda x: x[1])

    # Create child chunks for each section
    child_ids = []
    for i, (header, start, end) in enumerate(sections_found):
        # Content goes from end of header to start of next section (or end of doc)
        if i + 1 < len(sections_found):
            content_end = sections_found[i + 1][1]
        else:
            content_end = len(text)

        content = text[end:content_end].strip()
        if not content:
            continue

        child_index = len(chunks)
        child_chunk = create_chunk(
            doc=doc,
            chunk_index=child_index,
            level=ChunkLevel.CHILD,
            text=content,
            ancestry=ancestry,
            subsection_path=header,
            parent_chunk_id=parent_chunk.id,
        )
        chunks.append(child_chunk)
        child_ids.append(child_chunk.id)

    parent_chunk.child_chunk_ids = child_ids

    return chunks


def chunk_case(doc: LegalDocument) -> list[LegalChunk]:
    """Chunk a court case document.

    Cases in our corpus are typically truncated (~356 chars avg).
    For short cases, create a single chunk.
    For longer cases, use paragraph-based chunking with overlap.

    Args:
        doc: A case LegalDocument

    Returns:
        List of LegalChunk objects
    """
    chunks = []
    ancestry = build_ancestry(doc)
    text = doc.text.strip()

    # Create parent chunk
    parent_chunk = create_chunk(
        doc=doc,
        chunk_index=0,
        level=ChunkLevel.PARENT,
        text=text,
        ancestry=ancestry,
    )
    chunks.append(parent_chunk)

    # If short (most cases in our corpus), just return parent
    if parent_chunk.token_count < 600:
        return chunks

    # For longer cases, split into paragraph-based chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) <= 1:
        return chunks

    # Build chunks of ~500 tokens with 50 token overlap
    target_tokens = 500
    overlap_tokens = 50

    current_text = ""
    current_tokens = 0
    child_ids = []
    chunk_index = 1

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > target_tokens and current_text:
            # Save current chunk
            child_chunk = create_chunk(
                doc=doc,
                chunk_index=chunk_index,
                level=ChunkLevel.CHILD,
                text=current_text.strip(),
                ancestry=ancestry,
                subsection_path=f"part-{chunk_index}",
                parent_chunk_id=parent_chunk.id,
            )
            chunks.append(child_chunk)
            child_ids.append(child_chunk.id)
            chunk_index += 1

            # Keep overlap from end of current chunk
            # Simple approach: keep last paragraph if it's short enough
            if para_tokens < overlap_tokens * 2:
                current_text = para + "\n\n"
                current_tokens = para_tokens
            else:
                current_text = para + "\n\n"
                current_tokens = para_tokens
        else:
            current_text += para + "\n\n"
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_text.strip():
        child_chunk = create_chunk(
            doc=doc,
            chunk_index=chunk_index,
            level=ChunkLevel.CHILD,
            text=current_text.strip(),
            ancestry=ancestry,
            subsection_path=f"part-{chunk_index}",
            parent_chunk_id=parent_chunk.id,
        )
        chunks.append(child_chunk)
        child_ids.append(child_chunk.id)

    parent_chunk.child_chunk_ids = child_ids

    return chunks


def chunk_document(doc: LegalDocument) -> list[LegalChunk]:
    """Route document to appropriate chunking function.

    Args:
        doc: A LegalDocument of any type

    Returns:
        List of LegalChunk objects
    """
    if doc.doc_type == DocumentType.STATUTE:
        return chunk_statute(doc)
    elif doc.doc_type == DocumentType.RULE:
        return chunk_rule(doc)
    elif doc.doc_type == DocumentType.TAA:
        return chunk_taa(doc)
    elif doc.doc_type == DocumentType.CASE:
        return chunk_case(doc)
    else:
        # Fallback: create single parent chunk
        ancestry = build_ancestry(doc)
        return [
            create_chunk(
                doc=doc,
                chunk_index=0,
                level=ChunkLevel.PARENT,
                text=doc.text,
                ancestry=ancestry,
            )
        ]


def chunk_corpus(corpus: Corpus) -> list[LegalChunk]:
    """Chunk all documents in a corpus.

    Args:
        corpus: A Corpus object with documents

    Returns:
        List of all LegalChunk objects
    """
    all_chunks = []

    for doc in corpus.documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks
