"""LLM generation layer for Florida Tax RAG.

This module provides:
- TaxLawGenerator: Generates legally-accurate responses with citations
- format_chunks_for_context: Formats retrieval chunks for LLM context
- Response models: GeneratedResponse, ValidatedCitation, ExtractedCitation
"""

from src.generation.formatter import format_chunks_for_context
from src.generation.generator import TaxLawGenerator
from src.generation.models import (
    ExtractedCitation,
    GeneratedResponse,
    ValidatedCitation,
)

__all__ = [
    "TaxLawGenerator",
    "format_chunks_for_context",
    "GeneratedResponse",
    "ValidatedCitation",
    "ExtractedCitation",
]
