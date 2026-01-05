"""LangGraph agentic workflow for tax law queries."""

from .edges import check_for_errors, should_continue_retrieval, should_expand_graph
from .graph import create_tax_agent_graph, get_graph_visualization
from .nodes import (
    check_temporal_validity,
    decompose_query,
    expand_with_graph,
    filter_irrelevant,
    retrieve_for_subquery,
    score_relevance,
    synthesize_answer,
)
from .state import Citation, TaxAgentState

__all__ = [
    # State
    "TaxAgentState",
    "Citation",
    # Graph
    "create_tax_agent_graph",
    "get_graph_visualization",
    # Nodes
    "decompose_query",
    "retrieve_for_subquery",
    "expand_with_graph",
    "score_relevance",
    "filter_irrelevant",
    "check_temporal_validity",
    "synthesize_answer",
    # Edges
    "should_continue_retrieval",
    "should_expand_graph",
    "check_for_errors",
]
