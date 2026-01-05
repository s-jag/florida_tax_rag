"""LLM generation layer for Florida Tax RAG.

This module provides:
- TaxLawGenerator: Generates legally-accurate responses with citations
- ResponseValidator: Validates responses for hallucinations
- ResponseCorrector: Corrects hallucinated content
- format_chunks_for_context: Formats retrieval chunks for LLM context
- Response models: GeneratedResponse, ValidatedCitation, ValidationResult, etc.
"""

from src.generation.corrector import ResponseCorrector
from src.generation.formatter import format_chunks_for_context
from src.generation.generator import TaxLawGenerator
from src.generation.models import (
    CorrectionResult,
    DetectedHallucination,
    ExtractedCitation,
    GeneratedResponse,
    HallucinationType,
    ValidatedCitation,
    ValidationResult,
)
from src.generation.validator import ResponseValidator

__all__ = [
    # Generator
    "TaxLawGenerator",
    "format_chunks_for_context",
    # Validator
    "ResponseValidator",
    # Corrector
    "ResponseCorrector",
    # Models
    "GeneratedResponse",
    "ValidatedCitation",
    "ExtractedCitation",
    "ValidationResult",
    "DetectedHallucination",
    "HallucinationType",
    "CorrectionResult",
]
