"""Unit tests for Tax Agent node functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes import (
    check_temporal_validity,
    decompose_query,
    expand_with_graph,
    filter_irrelevant,
    retrieve_for_subquery,
    score_relevance,
    synthesize_answer,
)
from src.agent.state import TaxAgentState


class TestDecomposeQuery:
    """Tests for decompose_query node."""

    @pytest.mark.asyncio
    async def test_simple_query_decomposition(self):
        """Test that simple queries are handled correctly."""
        state: TaxAgentState = {
            "original_query": "What is the sales tax rate?",
        }

        # Mock the decomposer
        mock_result = MagicMock()
        mock_result.is_simple = True
        mock_result.sub_queries = []
        mock_result.reasoning = "Simple rate question"
        mock_result.query_count = 0

        with patch("src.retrieval.create_decomposer") as mock_create:
            mock_decomposer = MagicMock()
            mock_decomposer.decompose = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_decomposer

            result = await decompose_query(state)

        assert result["is_simple_query"] is True
        assert len(result["sub_queries"]) == 1
        assert result["sub_queries"][0]["text"] == "What is the sales tax rate?"
        assert "Decomposition:" in result["reasoning_steps"][0]

    @pytest.mark.asyncio
    async def test_complex_query_decomposition(self):
        """Test that complex queries are decomposed."""
        state: TaxAgentState = {
            "original_query": "Is software consulting taxable in Miami-Dade?",
        }

        # Mock sub-queries
        mock_sq1 = MagicMock()
        mock_sq1.model_dump.return_value = {
            "text": "software consulting taxability Florida",
            "type": "exemption",
            "priority": 1,
        }
        mock_sq2 = MagicMock()
        mock_sq2.model_dump.return_value = {
            "text": "Miami-Dade county surtax",
            "type": "local",
            "priority": 2,
        }

        mock_result = MagicMock()
        mock_result.is_simple = False
        mock_result.sub_queries = [mock_sq1, mock_sq2]
        mock_result.reasoning = "Complex query with multiple aspects"
        mock_result.query_count = 2

        with patch("src.retrieval.create_decomposer") as mock_create:
            mock_decomposer = MagicMock()
            mock_decomposer.decompose = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_decomposer

            result = await decompose_query(state)

        assert result["is_simple_query"] is False
        assert len(result["sub_queries"]) == 2
        assert result["current_sub_query_idx"] == 0

    @pytest.mark.asyncio
    async def test_decomposition_error_fallback(self):
        """Test graceful fallback on decomposition error."""
        state: TaxAgentState = {
            "original_query": "Test query",
        }

        with patch("src.retrieval.create_decomposer") as mock_create:
            mock_decomposer = MagicMock()
            mock_decomposer.decompose = AsyncMock(side_effect=Exception("API error"))
            mock_create.return_value = mock_decomposer

            result = await decompose_query(state)

        # Should fall back to simple query behavior
        assert result["is_simple_query"] is True
        assert len(result["sub_queries"]) == 1
        assert "errors" in result
        assert any("Decomposition error" in e for e in result["errors"])


class TestRetrieveForSubquery:
    """Tests for retrieve_for_subquery node."""

    @pytest.mark.asyncio
    async def test_retrieve_first_subquery(self):
        """Test retrieval for first sub-query."""
        state: TaxAgentState = {
            "original_query": "Test query",
            "sub_queries": [
                {"text": "sub query 1", "type": "general", "priority": 1},
                {"text": "sub query 2", "type": "general", "priority": 2},
            ],
            "current_sub_query_idx": 0,
        }

        # Mock retrieval results
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "chunk_id": "chunk:1",
            "doc_id": "doc:1",
            "doc_type": "statute",
            "text": "Sample text",
            "score": 0.9,
        }

        with patch("src.retrieval.create_retriever") as mock_create:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_result]
            mock_create.return_value = mock_retriever

            result = await retrieve_for_subquery(state)

        # With parallel retrieval, all sub-queries are processed at once
        assert result["current_sub_query_idx"] == 2  # All sub-queries processed
        # Results are deduplicated by chunk_id
        assert len(result["current_retrieval_results"]) >= 1
        assert len(result["retrieved_chunks"]) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_fallback_to_original(self):
        """Test that original query is used when sub_query_idx exceeds list."""
        state: TaxAgentState = {
            "original_query": "Original query",
            "sub_queries": [],
            "current_sub_query_idx": 0,
        }

        with patch("src.retrieval.create_retriever") as mock_create:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_create.return_value = mock_retriever

            await retrieve_for_subquery(state)

        # Should have used original query
        mock_retriever.retrieve.assert_called_once()
        call_args = mock_retriever.retrieve.call_args
        assert call_args[0][0] == "Original query"


class TestExpandWithGraph:
    """Tests for expand_with_graph node."""

    @pytest.mark.asyncio
    async def test_expand_statute(self):
        """Test graph expansion for statute documents."""
        state: TaxAgentState = {
            "original_query": "Test",
            "current_retrieval_results": [
                {"doc_id": "statute:212.05", "doc_type": "statute"},
            ],
        }

        # Mock interpretation chain
        mock_rule = MagicMock()
        mock_rule.id = "rule:12A-1.001"
        mock_rule.full_citation = "Fla. Admin. Code R. 12A-1.001"

        mock_chain = MagicMock()
        mock_chain.implementing_rules = [mock_rule]
        mock_chain.interpreting_cases = []
        mock_chain.interpreting_taas = []
        mock_chain.model_dump.return_value = {"statute": {}, "rules": []}

        with patch("src.graph.Neo4jClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            with patch("src.graph.queries.get_interpretation_chain") as mock_get_chain:
                mock_get_chain.return_value = mock_chain

                result = await expand_with_graph(state)

        assert len(result["graph_context"]) == 1
        assert result["graph_context"][0]["relation_type"] == "IMPLEMENTS"
        assert "statute:212.05" in result["interpretation_chains"]

    @pytest.mark.asyncio
    async def test_expand_no_results(self):
        """Test graph expansion with no results."""
        state: TaxAgentState = {
            "original_query": "Test",
            "current_retrieval_results": [],
        }

        with patch("src.graph.Neo4jClient"):
            result = await expand_with_graph(state)

        assert result["graph_context"] == []
        assert result["interpretation_chains"] == {}


class TestScoreRelevance:
    """Tests for score_relevance node."""

    @pytest.mark.asyncio
    async def test_score_chunks(self):
        """Test relevance scoring of chunks."""
        state: TaxAgentState = {
            "original_query": "What is sales tax?",
            "current_retrieval_results": [
                {
                    "chunk_id": "chunk:1",
                    "doc_type": "statute",
                    "citation": "Fla. Stat. 212.05",
                    "text": "Sales tax...",
                },
                {
                    "chunk_id": "chunk:2",
                    "doc_type": "rule",
                    "citation": "Rule 12A-1",
                    "text": "Rules...",
                },
            ],
        }

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"score": 8, "reasoning": "Relevant"}')]

        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "test-key"

            with patch.object(
                __import__("anthropic", fromlist=["Anthropic"]),
                "Anthropic",
                return_value=MagicMock(
                    messages=MagicMock(create=MagicMock(return_value=mock_response))
                ),
            ):
                result = await score_relevance(state)

        assert "relevance_scores" in result
        assert len(result["relevance_scores"]) == 2
        # Scores should be normalized to 0-1
        for score in result["relevance_scores"].values():
            assert 0 <= score <= 1


class TestFilterIrrelevant:
    """Tests for filter_irrelevant node."""

    @pytest.mark.asyncio
    async def test_filter_below_threshold(self):
        """Test filtering chunks below threshold.

        Note: Filter keeps at least 10 chunks, so we need more than 10 to see filtering.
        """
        # Create 15 chunks - some above threshold, some below
        chunks = [{"chunk_id": f"chunk:{i}", "text": f"Text {i}"} for i in range(15)]

        # Scores: first 5 high (0.8), next 5 medium (0.6), last 5 low (0.3)
        scores = {}
        for i in range(5):
            scores[f"chunk:{i}"] = 0.8  # High
        for i in range(5, 10):
            scores[f"chunk:{i}"] = 0.6  # Medium (above threshold)
        for i in range(10, 15):
            scores[f"chunk:{i}"] = 0.3  # Low (below threshold)

        state: TaxAgentState = {
            "original_query": "Test",
            "current_retrieval_results": chunks,
            "relevance_scores": scores,
            "relevance_threshold": 0.5,
        }

        result = await filter_irrelevant(state)

        # Should keep 10 chunks (5 high + 5 medium) and filter 5 low ones
        assert len(result["filtered_chunks"]) == 10

        # Check that high/medium scores are kept
        filtered_ids = [c["chunk_id"] for c in result["filtered_chunks"]]
        for i in range(10):
            assert f"chunk:{i}" in filtered_ids

        # Check that low scores are filtered
        for i in range(10, 15):
            assert f"chunk:{i}" not in filtered_ids

    @pytest.mark.asyncio
    async def test_keep_minimum_10(self):
        """Test that at least 10 chunks are kept even if below threshold."""
        # Create 15 chunks with low scores
        chunks = [{"chunk_id": f"chunk:{i}", "text": f"Text {i}"} for i in range(15)]
        scores = {f"chunk:{i}": 0.2 for i in range(15)}  # All below 0.5

        state: TaxAgentState = {
            "original_query": "Test",
            "current_retrieval_results": chunks,
            "relevance_scores": scores,
            "relevance_threshold": 0.5,
        }

        result = await filter_irrelevant(state)

        # Should keep at least 10 even though all are below threshold
        assert len(result["filtered_chunks"]) == 10


class TestCheckTemporalValidity:
    """Tests for check_temporal_validity node."""

    @pytest.mark.asyncio
    async def test_extract_year_from_query(self):
        """Test tax year extraction from query."""
        state: TaxAgentState = {
            "original_query": "What was the tax rate in 2023?",
            "filtered_chunks": [
                {"chunk_id": "chunk:1", "effective_date": None},
            ],
        }

        result = await check_temporal_validity(state)

        assert result["query_tax_year"] == 2023

    @pytest.mark.asyncio
    async def test_filter_future_documents(self):
        """Test that documents from future years are filtered."""
        state: TaxAgentState = {
            "original_query": "Tax rules for 2022",
            "filtered_chunks": [
                {"chunk_id": "chunk:1", "citation": "Old doc", "effective_date": "2020-01-01"},
                {"chunk_id": "chunk:2", "citation": "Future doc", "effective_date": "2024-01-01"},
            ],
        }

        result = await check_temporal_validity(state)

        # Future doc should be filtered out for 2022 query
        valid_ids = [c["chunk_id"] for c in result["temporally_valid_chunks"]]
        assert "chunk:1" in valid_ids
        assert "chunk:2" not in valid_ids
        assert len(result["reasoning_steps"]) > 1  # Should have warning

    @pytest.mark.asyncio
    async def test_needs_more_info_flag(self):
        """Test needs_more_info is set when too few valid chunks."""
        state: TaxAgentState = {
            "original_query": "Test query",
            "filtered_chunks": [
                {"chunk_id": "chunk:1", "effective_date": None},
            ],
        }

        result = await check_temporal_validity(state)

        # With only 1 chunk, needs_more_info should be True
        assert result["needs_more_info"] is True


class TestSynthesizeAnswer:
    """Tests for synthesize_answer node."""

    @pytest.mark.asyncio
    async def test_prepare_citations(self):
        """Test citation preparation with mocked generator."""
        from src.generation.models import GeneratedResponse, ValidatedCitation

        state: TaxAgentState = {
            "original_query": "Test",
            "temporally_valid_chunks": [
                {
                    "chunk_id": "chunk:1",
                    "doc_id": "statute:212.05",
                    "doc_type": "statute",
                    "citation": "Fla. Stat. 212.05",
                    "text": "Sales tax is imposed on...",
                },
            ],
            "graph_context": [],
        }

        # Mock response with a citation
        mock_response = GeneratedResponse(
            answer="Tax is imposed [Source: § 212.05].",
            citations=[
                ValidatedCitation(
                    citation_text="§ 212.05",
                    chunk_id="chunk:1",
                    verified=True,
                    raw_text="Sales tax is imposed on...",
                    doc_type="statute",
                )
            ],
            chunks_used=["chunk:1"],
            confidence=0.9,
        )

        with patch("src.generation.TaxLawGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_response)
            mock_gen_cls.return_value = mock_gen

            result = await synthesize_answer(state)

        assert len(result["citations"]) == 1
        assert "212.05" in result["citations"][0]["citation"]
        assert result["citations"][0]["doc_type"] == "statute"
        assert result["final_answer"] is not None

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence from generator response.

        Confidence is now calculated by TaxLawGenerator based on:
        - Source type weights (statute=1.0, rule=0.9, case=0.7, taa=0.6)
        - Citation verification rate
        """
        from src.generation.models import GeneratedResponse

        # High confidence mock (statutes, verified citations)
        state_high: TaxAgentState = {
            "original_query": "Test",
            "temporally_valid_chunks": [
                {"doc_type": "statute", "citation": "Stat 1", "text": "...", "doc_id": "s1"},
            ],
            "graph_context": [],
        }

        mock_response_high = GeneratedResponse(
            answer="Answer with citations.",
            citations=[],
            chunks_used=["s1"],
            confidence=0.95,
        )

        with patch("src.generation.TaxLawGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_response_high)
            mock_gen_cls.return_value = mock_gen

            result_high = await synthesize_answer(state_high)

        assert result_high["confidence"] == 0.95

        # Low confidence mock (TAAs, unverified)
        state_low: TaxAgentState = {
            "original_query": "Test",
            "temporally_valid_chunks": [
                {"doc_type": "taa", "citation": "TAA 1", "text": "...", "doc_id": "t1"},
            ],
            "graph_context": [],
        }

        mock_response_low = GeneratedResponse(
            answer="Answer with low confidence.",
            citations=[],
            chunks_used=["t1"],
            confidence=0.4,
        )

        with patch("src.generation.TaxLawGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_response_low)
            mock_gen_cls.return_value = mock_gen

            result_low = await synthesize_answer(state_low)

        assert result_low["confidence"] == 0.4
        assert result_low["confidence"] < result_high["confidence"]

    @pytest.mark.asyncio
    async def test_synthesis_context_created(self):
        """Test that synthesis context is prepared using format_chunks_for_context."""
        from src.generation.models import GeneratedResponse

        state: TaxAgentState = {
            "original_query": "Test",
            "temporally_valid_chunks": [
                {
                    "citation": "Source 1",
                    "text": "Content 1",
                    "doc_type": "statute",
                    "doc_id": "s1",
                },
                {"citation": "Source 2", "text": "Content 2", "doc_type": "rule", "doc_id": "r1"},
            ],
            "graph_context": [],
        }

        mock_response = GeneratedResponse(
            answer="Generated answer.",
            citations=[],
            chunks_used=["s1", "r1"],
            confidence=0.8,
        )

        with patch("src.generation.TaxLawGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_response)
            mock_gen_cls.return_value = mock_gen

            result = await synthesize_answer(state)

        assert "_synthesis_context" in result
        # New format uses format_chunks_for_context
        assert "Document 1" in result["_synthesis_context"]
        assert "Source 1" in result["_synthesis_context"]
        assert "STATUTE" in result["_synthesis_context"]
        assert result["final_answer"] == "Generated answer."
