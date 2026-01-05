"""LangGraph definition for the Tax Agent."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .edges import should_continue_retrieval, should_expand_graph
from .nodes import (
    check_temporal_validity,
    decompose_query,
    expand_with_graph,
    filter_irrelevant,
    retrieve_for_subquery,
    score_relevance,
    synthesize_answer,
)
from .state import TaxAgentState


def create_tax_agent_graph() -> StateGraph:
    """Create the Tax Agent LangGraph workflow.

    Graph Structure:
        START -> decompose -> retrieve -> [expand_graph | score_relevance]
              -> filter -> check_temporal -> [retrieve_next | decompose_more | synthesize]
              -> END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph with state schema
    graph = StateGraph(TaxAgentState)

    # Add nodes
    graph.add_node("decompose", decompose_query)
    graph.add_node("retrieve", retrieve_for_subquery)
    graph.add_node("expand_graph", expand_with_graph)
    graph.add_node("score_relevance", score_relevance)
    graph.add_node("filter", filter_irrelevant)
    graph.add_node("check_temporal", check_temporal_validity)
    graph.add_node("synthesize", synthesize_answer)

    # Set entry point
    graph.set_entry_point("decompose")

    # Add edges
    graph.add_edge("decompose", "retrieve")

    # Conditional: after retrieve, decide expand vs score
    graph.add_conditional_edges(
        "retrieve",
        should_expand_graph,
        {
            "expand": "expand_graph",
            "score": "score_relevance",
        },
    )

    # After expansion, always score relevance
    graph.add_edge("expand_graph", "score_relevance")

    # Linear flow through filtering
    graph.add_edge("score_relevance", "filter")
    graph.add_edge("filter", "check_temporal")

    # Conditional: after temporal check, decide next step
    graph.add_conditional_edges(
        "check_temporal",
        should_continue_retrieval,
        {
            "retrieve_next": "retrieve",
            "decompose_more": "decompose",
            "synthesize": "synthesize",
        },
    )

    # Synthesize leads to END
    graph.add_edge("synthesize", END)

    return graph.compile()


def get_graph_visualization() -> str:
    """Get ASCII visualization of the graph.

    Returns:
        ASCII representation of the graph structure
    """
    graph = create_tax_agent_graph()
    return graph.get_graph().draw_ascii()
