"""State definition for the Tax Agent."""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional, TypedDict


class Citation(TypedDict):
    """A citation in the final answer."""

    doc_id: str
    citation: str
    doc_type: str
    text_snippet: str


class TaxAgentState(TypedDict, total=False):
    """State for the Tax Agent workflow.

    Uses TypedDict for LangGraph compatibility.
    Fields marked with Annotated[..., add] will accumulate across nodes.
    """

    # Input
    original_query: str

    # Decomposition
    sub_queries: list  # list[SubQuery] - using list for TypedDict compat
    current_sub_query_idx: int
    decomposition_reasoning: str
    is_simple_query: bool

    # Retrieval
    retrieved_chunks: Annotated[list, add]  # Accumulates across sub-queries
    current_retrieval_results: list  # Results from current sub-query

    # Graph context
    graph_context: list  # list[CitationContext] from graph expansion
    interpretation_chains: dict  # statute_id -> chain info

    # Reasoning
    reasoning_steps: Annotated[list, add]  # Accumulates reasoning
    needs_more_info: bool

    # Self-correction (relevance filtering)
    relevance_scores: dict  # chunk_id -> relevance score (0-1)
    filtered_chunks: list  # Chunks that passed relevance threshold
    relevance_threshold: float  # Default 0.5

    # Temporal validation
    query_tax_year: Optional[int]  # Inferred from query
    temporally_valid_chunks: list  # Chunks valid for tax year

    # Output
    final_answer: Optional[str]
    citations: list  # list[Citation]
    confidence: float  # 0-1 confidence score

    # Error handling
    errors: Annotated[list, add]  # Accumulates errors

    # Iteration control
    max_iterations: int
    current_iteration: int

    # Internal state for synthesis (used between prompts)
    _synthesis_context: str  # Formatted context for LLM answer generation

    # Validation and correction
    validation_result: Optional[dict]  # ValidationResult as dict
    correction_result: Optional[dict]  # CorrectionResult as dict
    regeneration_count: int  # Number of regeneration attempts
    max_regenerations: int  # Max regeneration attempts (default 1)
    validation_passed: bool  # Whether validation passed
    original_answer: Optional[str]  # Original answer before corrections
