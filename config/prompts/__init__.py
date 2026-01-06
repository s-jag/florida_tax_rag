"""Centralized prompt management for the Florida Tax RAG system.

This module provides access to all LLM prompts used across the system.
Prompts are organized by functional area:
- retrieval: Query decomposition, classification, relevance scoring
- generation: Response generation, hallucination detection, correction
- evaluation: LLM judge prompts for evaluation

Usage:
    from config.prompts import SYSTEM_PROMPT, DECOMPOSITION_PROMPT
    from config.prompts.retrieval import CLASSIFICATION_PROMPT
"""

from __future__ import annotations

# Re-export all prompts for convenient access
from .retrieval import (
    DECOMPOSITION_PROMPT,
    CLASSIFICATION_PROMPT,
    RETRIEVAL_SYSTEM_MESSAGE,
    RELEVANCE_PROMPT,
)
from .generation import (
    GENERATION_SYSTEM_PROMPT,
    CONTEXT_TEMPLATE,
    HALLUCINATION_DETECTION_PROMPT,
    CORRECTION_PROMPT,
)
from .evaluation import JUDGE_PROMPT

# Legacy aliases for backward compatibility
SYSTEM_PROMPT = GENERATION_SYSTEM_PROMPT
SYSTEM_MESSAGE = RETRIEVAL_SYSTEM_MESSAGE

__all__ = [
    # Retrieval prompts
    "DECOMPOSITION_PROMPT",
    "CLASSIFICATION_PROMPT",
    "RETRIEVAL_SYSTEM_MESSAGE",
    "RELEVANCE_PROMPT",
    # Generation prompts
    "GENERATION_SYSTEM_PROMPT",
    "CONTEXT_TEMPLATE",
    "HALLUCINATION_DETECTION_PROMPT",
    "CORRECTION_PROMPT",
    # Evaluation prompts
    "JUDGE_PROMPT",
    # Legacy aliases
    "SYSTEM_PROMPT",
    "SYSTEM_MESSAGE",
]
