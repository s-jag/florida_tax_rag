"""Integration tests for Tax Agent graph.

These tests require:
- Weaviate running (for vector search)
- Neo4j running (for graph expansion)
- Anthropic API key (for LLM calls)

Run with: pytest tests/test_agent_integration.py -v -s
"""

from __future__ import annotations

import pytest

from src.agent import create_tax_agent_graph


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_agent_flow():
    """Test complete agent flow with real query.

    This test requires all services (Weaviate, Neo4j, Anthropic) to be running.
    """
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "Is software consulting taxable in Miami?",
        }
    )

    # Verify decomposition happened
    assert "sub_queries" in result
    assert len(result.get("sub_queries", [])) > 0

    # Verify retrieval happened
    assert "retrieved_chunks" in result
    assert len(result.get("retrieved_chunks", [])) > 0

    # Verify filtering happened
    assert "filtered_chunks" in result

    # Verify reasoning was accumulated
    assert "reasoning_steps" in result
    assert len(result.get("reasoning_steps", [])) > 0

    # Verify temporal check
    assert "query_tax_year" in result
    assert result.get("temporally_valid_chunks") is not None

    # Verify citations prepared
    assert "citations" in result
    assert "confidence" in result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_simple_query_flow():
    """Test flow with simple query (no decomposition).

    This test requires all services (Weaviate, Neo4j, Anthropic) to be running.
    """
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "What is the Florida sales tax rate?",
        }
    )

    assert result.get("is_simple_query") is True
    assert len(result.get("retrieved_chunks", [])) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_homestead_exemption_query():
    """Test with homestead exemption query."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "What are the requirements for homestead exemption?",
        }
    )

    assert "sub_queries" in result
    assert "retrieved_chunks" in result
    assert "reasoning_steps" in result

    # Check that we got some relevant content
    if result.get("temporally_valid_chunks"):
        doc_types = [c.get("doc_type") for c in result["temporally_valid_chunks"]]
        # Should include some statutes or rules
        assert any(dt in ["statute", "rule"] for dt in doc_types)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_temporal_query_flow():
    """Test with query mentioning specific tax year."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "What was the sales tax rate in 2023?",
        }
    )

    # Should extract 2023 as the tax year
    assert result.get("query_tax_year") == 2023


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complex_multi_aspect_query():
    """Test with complex query that should decompose into multiple sub-queries."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "Are software as a service (SaaS) products taxable in Florida, and are there any exemptions for resale or manufacturing?",
        }
    )

    # Complex query should decompose
    assert "is_simple_query" in result

    # Should have multiple reasoning steps
    assert len(result.get("reasoning_steps", [])) >= 3

    # Should have citations prepared
    assert len(result.get("citations", [])) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_context_populated():
    """Test that graph context is populated when statutes are found."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "What does Florida Statute 212.05 say about sales tax?",
        }
    )

    # If we found statutes, graph_context should be populated
    retrieved = result.get("retrieved_chunks", [])
    has_statutes = any(c.get("doc_type") == "statute" for c in retrieved)

    if has_statutes:
        # Graph expansion should have run
        assert "graph_context" in result
        assert "interpretation_chains" in result


@pytest.mark.asyncio
async def test_confidence_score_range():
    """Test that confidence score is in valid range."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "Is food taxable?",
        }
    )

    confidence = result.get("confidence", -1)
    assert 0 <= confidence <= 1, f"Confidence {confidence} out of range [0, 1]"


@pytest.mark.asyncio
async def test_citations_have_required_fields():
    """Test that citations have all required fields."""
    graph = create_tax_agent_graph()

    result = await graph.ainvoke(
        {
            "original_query": "What is tangible personal property?",
        }
    )

    citations = result.get("citations", [])

    for citation in citations:
        assert "doc_id" in citation
        assert "citation" in citation
        assert "doc_type" in citation
        assert "text_snippet" in citation
