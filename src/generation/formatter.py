"""Chunk formatting utilities for LLM context."""

from __future__ import annotations


def format_chunks_for_context(chunks: list[dict]) -> str:
    """Format retrieval chunks with clear document type labels and metadata.

    Each chunk is formatted with:
    - Document number and type (uppercase)
    - Citation reference
    - Effective date (if available)
    - Ancestry/hierarchy path
    - Full text content

    Args:
        chunks: List of chunk dictionaries from retrieval results.
            Expected keys: doc_type, citation, effective_date, ancestry, text

    Returns:
        Formatted string with all chunks labeled and separated.
    """
    if not chunks:
        return "No legal documents provided."

    formatted = []
    for i, chunk in enumerate(chunks):
        doc_type = chunk.get("doc_type", "unknown").upper()
        citation = chunk.get("citation", f"Source {i+1}")
        effective_date = chunk.get("effective_date", "Not specified")
        ancestry = chunk.get("ancestry", "")
        text = chunk.get("text", "")

        # Handle date objects
        if hasattr(effective_date, "isoformat"):
            effective_date = effective_date.isoformat()

        formatted.append(
            f"""
--- Document {i+1} ({doc_type}) ---
Citation: {citation}
Effective Date: {effective_date}
Ancestry: {ancestry}

{text}
""".strip()
        )

    return "\n\n".join(formatted)


def format_chunk_for_citation(chunk: dict, index: int) -> str:
    """Format a single chunk for citation reference.

    Args:
        chunk: Chunk dictionary with citation info.
        index: 1-based index for referencing.

    Returns:
        Short citation reference string.
    """
    citation = chunk.get("citation", f"Source {index}")
    doc_type = chunk.get("doc_type", "unknown")
    return f"[{index}] {citation} ({doc_type})"
