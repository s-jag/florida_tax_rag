"""Routing logic for the Tax Agent graph."""

from __future__ import annotations

from typing import Literal

from .state import TaxAgentState


def should_continue_retrieval(
    state: TaxAgentState,
) -> Literal["retrieve_next", "decompose_more", "synthesize"]:
    """Decide if we need more sub-queries or can synthesize.

    Decision logic:
    1. If more sub-queries remain -> "retrieve_next"
    2. If needs_more_info=True and under max_iterations -> "decompose_more"
    3. Otherwise -> "synthesize"

    Args:
        state: Current agent state

    Returns:
        Next node to route to
    """
    # Check if more sub-queries remain
    current_idx = state.get("current_sub_query_idx", 0)
    sub_queries = state.get("sub_queries", [])

    if current_idx < len(sub_queries):
        return "retrieve_next"

    # Check if we need more information
    if state.get("needs_more_info", False):
        # Limit iterations to prevent infinite loops
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if current_iteration < max_iterations:
            return "decompose_more"

    return "synthesize"


def should_expand_graph(state: TaxAgentState) -> Literal["expand", "score"]:
    """Decide if graph expansion is needed.

    Expand graph if:
    - Found statutes (look for implementing rules)
    - Found rules (look for parent statutes)
    - Have few results and need more context

    Args:
        state: Current agent state

    Returns:
        "expand" to run graph expansion, "score" to skip to relevance scoring
    """
    results = state.get("current_retrieval_results", [])

    if not results:
        return "score"

    # Check if we have statute or rule results that need expansion
    doc_types = set()
    for r in results:
        if isinstance(r, dict):
            doc_types.add(r.get("doc_type"))
        elif hasattr(r, "doc_type"):
            doc_types.add(r.doc_type)

    # Expand if we have primary authority that might have interpretations
    if "statute" in doc_types or "rule" in doc_types:
        return "expand"

    # Also expand if we have few results
    if len(results) < 3:
        return "expand"

    return "score"


def check_for_errors(state: TaxAgentState) -> Literal["continue", "error"]:
    """Check if there are critical errors that should halt execution.

    Args:
        state: Current agent state

    Returns:
        "continue" if no critical errors, "error" if execution should halt
    """
    errors = state.get("errors", [])

    # Check for critical errors
    critical_keywords = ["authentication", "connection", "timeout"]
    for error in errors:
        if any(kw in str(error).lower() for kw in critical_keywords):
            return "error"

    return "continue"


def route_after_validation(
    state: TaxAgentState,
) -> Literal["regenerate", "correct", "accept"]:
    """Route based on validation results.

    Decision logic:
    1. If validation passed -> "accept" (go to END)
    2. If needs_correction but not regeneration -> "correct"
    3. If needs_regeneration and under limit -> "regenerate" (back to synthesize)
    4. If needs_regeneration but at limit -> "correct" (best effort)

    Args:
        state: Current agent state with validation_result

    Returns:
        Next action: "regenerate", "correct", or "accept"
    """
    validation_passed = state.get("validation_passed", True)

    if validation_passed:
        return "accept"

    validation_data = state.get("validation_result", {})
    needs_regeneration = validation_data.get("needs_regeneration", False)
    needs_correction = validation_data.get("needs_correction", False)

    regeneration_count = state.get("regeneration_count", 0)
    max_regenerations = state.get("max_regenerations", 2)

    if needs_regeneration and regeneration_count < max_regenerations:
        return "regenerate"
    elif needs_correction or needs_regeneration:
        # If we can't regenerate anymore, try to correct
        return "correct"
    else:
        return "accept"
