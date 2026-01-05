"""Tests for the Tax Agent graph structure."""

from __future__ import annotations

import pytest

from src.agent import (
    Citation,
    TaxAgentState,
    create_tax_agent_graph,
    get_graph_visualization,
    should_continue_retrieval,
    should_expand_graph,
    check_for_errors,
)


class TestTaxAgentState:
    """Tests for TaxAgentState."""

    def test_state_creation(self):
        """Test state can be created with required fields."""
        state: TaxAgentState = {
            "original_query": "What is the sales tax rate?",
            "sub_queries": [],
            "current_sub_query_idx": 0,
            "retrieved_chunks": [],
            "errors": [],
        }
        assert state["original_query"] == "What is the sales tax rate?"

    def test_state_optional_fields(self):
        """Test state with optional fields."""
        state: TaxAgentState = {
            "original_query": "Test query",
            "is_simple_query": True,
            "confidence": 0.85,
            "final_answer": "The answer is...",
        }
        assert state["is_simple_query"] is True
        assert state["confidence"] == 0.85


class TestCitation:
    """Tests for Citation TypedDict."""

    def test_citation_creation(self):
        """Test citation can be created."""
        citation: Citation = {
            "doc_id": "statute:212.05",
            "citation": "Fla. Stat. § 212.05",
            "doc_type": "statute",
            "text_snippet": "There is levied and imposed...",
        }
        assert citation["doc_id"] == "statute:212.05"
        assert citation["doc_type"] == "statute"


class TestEdgeLogic:
    """Tests for routing edge functions."""

    def test_should_continue_retrieval_more_subqueries(self):
        """Test routing when more sub-queries remain."""
        state: TaxAgentState = {
            "original_query": "test",
            "sub_queries": [{"text": "q1"}, {"text": "q2"}],
            "current_sub_query_idx": 0,
        }
        assert should_continue_retrieval(state) == "retrieve_next"

    def test_should_continue_retrieval_done(self):
        """Test routing when all sub-queries processed."""
        state: TaxAgentState = {
            "original_query": "test",
            "sub_queries": [{"text": "q1"}],
            "current_sub_query_idx": 1,
            "needs_more_info": False,
        }
        assert should_continue_retrieval(state) == "synthesize"

    def test_should_continue_retrieval_needs_more_info(self):
        """Test routing when needs more info and under max iterations."""
        state: TaxAgentState = {
            "original_query": "test",
            "sub_queries": [],
            "current_sub_query_idx": 0,
            "needs_more_info": True,
            "current_iteration": 1,
            "max_iterations": 3,
        }
        assert should_continue_retrieval(state) == "decompose_more"

    def test_should_continue_retrieval_max_iterations_reached(self):
        """Test routing when max iterations reached."""
        state: TaxAgentState = {
            "original_query": "test",
            "sub_queries": [],
            "current_sub_query_idx": 0,
            "needs_more_info": True,
            "current_iteration": 3,
            "max_iterations": 3,
        }
        assert should_continue_retrieval(state) == "synthesize"

    def test_should_expand_graph_with_statutes(self):
        """Test graph expansion triggered for statutes."""
        state: TaxAgentState = {
            "original_query": "test",
            "current_retrieval_results": [
                {"doc_type": "statute", "doc_id": "statute:212.05"}
            ],
        }
        assert should_expand_graph(state) == "expand"

    def test_should_expand_graph_with_rules(self):
        """Test graph expansion triggered for rules."""
        state: TaxAgentState = {
            "original_query": "test",
            "current_retrieval_results": [
                {"doc_type": "rule", "doc_id": "rule:12A-1.001"}
            ],
        }
        assert should_expand_graph(state) == "expand"

    def test_should_expand_graph_few_results(self):
        """Test graph expansion triggered for few results."""
        state: TaxAgentState = {
            "original_query": "test",
            "current_retrieval_results": [
                {"doc_type": "case", "doc_id": "case:123"}
            ],
        }
        assert should_expand_graph(state) == "expand"

    def test_should_skip_expansion_many_results(self):
        """Test graph expansion skipped for many non-statute results."""
        state: TaxAgentState = {
            "original_query": "test",
            "current_retrieval_results": [
                {"doc_type": "case", "doc_id": "case:1"},
                {"doc_type": "case", "doc_id": "case:2"},
                {"doc_type": "case", "doc_id": "case:3"},
                {"doc_type": "taa", "doc_id": "taa:1"},
            ],
        }
        assert should_expand_graph(state) == "score"

    def test_should_skip_expansion_no_results(self):
        """Test graph expansion skipped when no results."""
        state: TaxAgentState = {
            "original_query": "test",
            "current_retrieval_results": [],
        }
        assert should_expand_graph(state) == "score"

    def test_check_for_errors_no_errors(self):
        """Test error check with no errors."""
        state: TaxAgentState = {
            "original_query": "test",
            "errors": [],
        }
        assert check_for_errors(state) == "continue"

    def test_check_for_errors_non_critical(self):
        """Test error check with non-critical errors."""
        state: TaxAgentState = {
            "original_query": "test",
            "errors": ["Minor parsing issue", "Warning: low confidence"],
        }
        assert check_for_errors(state) == "continue"

    def test_check_for_errors_critical(self):
        """Test error check with critical errors."""
        state: TaxAgentState = {
            "original_query": "test",
            "errors": ["Connection refused: database unavailable"],
        }
        assert check_for_errors(state) == "error"

    def test_check_for_errors_authentication(self):
        """Test error check with authentication error."""
        state: TaxAgentState = {
            "original_query": "test",
            "errors": ["Authentication failed for Neo4j"],
        }
        assert check_for_errors(state) == "error"


class TestGraphStructure:
    """Tests for graph structure."""

    def test_graph_creation(self):
        """Test graph can be created."""
        graph = create_tax_agent_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test graph has expected nodes."""
        graph = create_tax_agent_graph()
        graph_def = graph.get_graph()

        expected_nodes = [
            "decompose",
            "retrieve",
            "expand_graph",
            "score_relevance",
            "filter",
            "check_temporal",
            "synthesize",
        ]

        # LangGraph returns nodes as dict keys (strings)
        node_ids = list(graph_def.nodes.keys())
        for expected in expected_nodes:
            assert expected in node_ids, f"Missing node: {expected}"

    def test_graph_has_entry_point(self):
        """Test graph has decompose as entry point."""
        graph = create_tax_agent_graph()
        graph_def = graph.get_graph()

        # Find edges from __start__
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "decompose"

    def test_graph_has_end(self):
        """Test graph has synthesize leading to end."""
        graph = create_tax_agent_graph()
        graph_def = graph.get_graph()

        # Find edges from synthesize
        synth_edges = [e for e in graph_def.edges if e.source == "synthesize"]
        assert len(synth_edges) == 1
        assert synth_edges[0].target == "__end__"

    def test_graph_has_conditional_edges(self):
        """Test graph has conditional edges from retrieve and check_temporal."""
        graph = create_tax_agent_graph()
        graph_def = graph.get_graph()

        # Check edges from retrieve (should branch to expand_graph or score_relevance)
        retrieve_edges = [e for e in graph_def.edges if e.source == "retrieve"]
        retrieve_targets = {e.target for e in retrieve_edges}
        assert "expand_graph" in retrieve_targets or "score_relevance" in retrieve_targets

        # Check edges from check_temporal (should branch to retrieve, decompose, or synthesize)
        temporal_edges = [e for e in graph_def.edges if e.source == "check_temporal"]
        temporal_targets = {e.target for e in temporal_edges}
        assert len(temporal_targets) >= 1  # At least one target

    def test_graph_visualization(self):
        """Test graph can be visualized."""
        viz = get_graph_visualization()
        assert viz is not None
        assert "decompose" in viz
        assert "synthesize" in viz


class TestGraphExecution:
    """Tests for graph execution (with mocked nodes)."""

    @pytest.mark.asyncio
    async def test_simple_query_flow(self):
        """Test execution flow for a simple query.

        This will be implemented when nodes are implemented.
        """
        # Placeholder for future implementation
        pass
