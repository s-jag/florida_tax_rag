"""Tests for query decomposition."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.models import RetrievalResult
from src.retrieval.multi_retriever import (
    MultiQueryRetriever,
    MultiRetrievalResult,
    SubQueryResult,
)
from src.retrieval.query_decomposer import (
    DecompositionResult,
    QueryDecomposer,
    QueryType,
    SubQuery,
)


class TestSubQuery:
    """Tests for SubQuery model."""

    def test_model_creation(self):
        """Test SubQuery can be created with required fields."""
        sq = SubQuery(
            text="Florida sales tax rate",
            type=QueryType.RATE,
            priority=1,
        )
        assert sq.text == "Florida sales tax rate"
        assert sq.type == QueryType.RATE
        assert sq.priority == 1

    def test_priority_default(self):
        """Test priority defaults to 3."""
        sq = SubQuery(text="test", type=QueryType.GENERAL)
        assert sq.priority == 3

    def test_priority_validation_min(self):
        """Test priority must be >= 1."""
        with pytest.raises(ValueError):
            SubQuery(text="test", type=QueryType.GENERAL, priority=0)

    def test_priority_validation_max(self):
        """Test priority must be <= 5."""
        with pytest.raises(ValueError):
            SubQuery(text="test", type=QueryType.GENERAL, priority=6)

    def test_type_normalization(self):
        """Test type string is normalized."""
        sq = SubQuery(text="test", type="DEFINITION")
        assert sq.type == QueryType.DEFINITION

        sq2 = SubQuery(text="test", type="def")
        assert sq2.type == QueryType.DEFINITION


class TestDecompositionResult:
    """Tests for DecompositionResult model."""

    def test_model_creation(self):
        """Test DecompositionResult can be created."""
        result = DecompositionResult(
            original_query="What is the sales tax rate?",
            sub_queries=[],
            reasoning="Simple query",
            is_simple=True,
        )
        assert result.original_query == "What is the sales tax rate?"
        assert result.is_simple is True
        assert result.query_count == 0

    def test_query_count_property(self):
        """Test query_count property."""
        sq = SubQuery(text="test", type=QueryType.RATE)
        result = DecompositionResult(
            original_query="test",
            sub_queries=[sq, sq],
            reasoning="test",
            is_simple=False,
        )
        assert result.query_count == 2


class TestQueryDecomposer:
    """Tests for QueryDecomposer."""

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create a mock Anthropic response."""
        response = MagicMock()
        response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "sub_queries": [
                            {
                                "text": "Florida sales tax software consulting",
                                "type": "definition",
                                "priority": 1,
                            },
                            {
                                "text": "professional services exemption Florida",
                                "type": "exemption",
                                "priority": 2,
                            },
                            {
                                "text": "Miami-Dade county surtax rate",
                                "type": "local",
                                "priority": 3,
                            },
                        ],
                        "reasoning": "Complex query about software consulting in Miami",
                        "is_simple": False,
                    }
                )
            )
        ]
        return response

    @pytest.fixture
    def mock_anthropic_simple_response(self):
        """Create a mock Anthropic response for simple query."""
        response = MagicMock()
        response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "sub_queries": [],
                        "reasoning": "Simple direct question about sales tax rate",
                        "is_simple": True,
                    }
                )
            )
        ]
        return response

    @pytest.fixture
    def mock_anthropic(self, mock_anthropic_response):
        """Create a mock Anthropic client."""
        client = MagicMock()
        client.messages.create.return_value = mock_anthropic_response
        return client

    def test_should_decompose_short_query(self):
        """Test short queries are not decomposed."""
        decomposer = QueryDecomposer(client=MagicMock())
        assert decomposer._should_decompose("What is sales tax?") is False

    def test_should_decompose_complex_keywords(self):
        """Test queries with complexity keywords are decomposed."""
        decomposer = QueryDecomposer(client=MagicMock())
        assert (
            decomposer._should_decompose(
                "Is software and consulting both taxable in Florida?"
            )
            is True
        )

    def test_should_decompose_county_reference(self):
        """Test queries with county names are decomposed."""
        decomposer = QueryDecomposer(client=MagicMock())
        assert (
            decomposer._should_decompose(
                "What is the sales tax rate in Miami-Dade county?"
            )
            is True
        )

    def test_should_decompose_multiple_questions(self):
        """Test queries with multiple question marks are decomposed."""
        decomposer = QueryDecomposer(client=MagicMock())
        assert (
            decomposer._should_decompose(
                "What is sales tax? And what are the exemptions?"
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_decompose_simple_query_heuristic(self):
        """Test simple query skips LLM via heuristic."""
        client = MagicMock()
        decomposer = QueryDecomposer(client=client)

        result = await decomposer.decompose("What is sales tax?")

        assert result.is_simple is True
        assert result.query_count == 0
        # LLM should not be called for simple queries
        client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_decompose_complex_query(self, mock_anthropic, mock_anthropic_response):
        """Test complex query gets decomposed via LLM."""
        mock_anthropic.messages.create.return_value = mock_anthropic_response
        decomposer = QueryDecomposer(client=mock_anthropic)

        result = await decomposer.decompose(
            "Do I owe sales tax on software consulting services in Miami?"
        )

        assert result.is_simple is False
        assert result.query_count == 3
        assert result.sub_queries[0].type == QueryType.DEFINITION
        assert result.sub_queries[1].type == QueryType.EXEMPTION
        assert result.sub_queries[2].type == QueryType.LOCAL

    @pytest.mark.asyncio
    async def test_decompose_handles_llm_error(self):
        """Test decomposition handles LLM errors gracefully."""
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")
        decomposer = QueryDecomposer(client=client)

        result = await decomposer.decompose(
            "Do I owe sales tax on software consulting in Miami?"
        )

        # Should fallback to simple query
        assert result.is_simple is True
        assert "failed" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_decompose_handles_invalid_json(self):
        """Test decomposition handles invalid JSON response."""
        response = MagicMock()
        response.content = [MagicMock(text="Not valid JSON")]

        client = MagicMock()
        client.messages.create.return_value = response
        decomposer = QueryDecomposer(client=client)

        result = await decomposer.decompose(
            "Do I owe sales tax on software consulting in Miami?"
        )

        assert result.is_simple is True

    @pytest.mark.asyncio
    async def test_classify_query(self):
        """Test query classification."""
        response = MagicMock()
        response.content = [MagicMock(text="exemption")]

        client = MagicMock()
        client.messages.create.return_value = response
        decomposer = QueryDecomposer(client=client)

        result = await decomposer.classify_query("Is food exempt from sales tax?")

        assert result == QueryType.EXEMPTION

    @pytest.mark.asyncio
    async def test_classify_query_unknown_type(self):
        """Test classification falls back to GENERAL for unknown types."""
        response = MagicMock()
        response.content = [MagicMock(text="unknown_type")]

        client = MagicMock()
        client.messages.create.return_value = response
        decomposer = QueryDecomposer(client=client)

        result = await decomposer.classify_query("Some weird query")

        assert result == QueryType.GENERAL

    def test_parse_json_with_code_blocks(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        decomposer = QueryDecomposer(client=MagicMock())

        response = """```json
        {
            "sub_queries": [{"text": "test", "type": "rate", "priority": 1}],
            "reasoning": "test",
            "is_simple": false
        }
        ```"""

        result = decomposer._parse_decomposition_response("original", response)

        assert result.query_count == 1
        assert result.sub_queries[0].text == "test"


class TestMultiQueryRetriever:
    """Tests for MultiQueryRetriever."""

    @pytest.fixture
    def mock_decomposer(self):
        """Create a mock decomposer."""
        decomposer = MagicMock(spec=QueryDecomposer)
        return decomposer

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock hybrid retriever."""
        retriever = MagicMock()
        return retriever

    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(
                chunk_id="chunk:statute:212.05:0",
                doc_id="statute:212.05",
                doc_type="statute",
                level="parent",
                text="Sales tax rate text",
                score=0.90,
            ),
            RetrievalResult(
                chunk_id="chunk:statute:212.08:0",
                doc_id="statute:212.08",
                doc_type="statute",
                level="parent",
                text="Exemptions text",
                score=0.85,
            ),
        ]

    @pytest.mark.asyncio
    async def test_simple_query_skips_parallel(
        self, mock_decomposer, mock_retriever, sample_results
    ):
        """Test simple queries skip parallel retrieval."""
        # Setup simple decomposition result
        mock_decomposer.decompose = AsyncMock(
            return_value=DecompositionResult(
                original_query="What is sales tax?",
                sub_queries=[],
                reasoning="Simple",
                is_simple=True,
            )
        )
        mock_retriever.retrieve.return_value = sample_results

        multi_retriever = MultiQueryRetriever(
            decomposer=mock_decomposer,
            retriever=mock_retriever,
        )

        result = await multi_retriever.retrieve("What is sales tax?")

        assert len(result.sub_query_results) == 0
        assert len(result.merged_results) == 2
        assert result.decomposition.is_simple is True

    @pytest.mark.asyncio
    async def test_complex_query_runs_parallel(
        self, mock_decomposer, mock_retriever, sample_results
    ):
        """Test complex queries run parallel sub-queries."""
        sub_queries = [
            SubQuery(text="software consulting definition", type=QueryType.DEFINITION),
            SubQuery(text="services exemption Florida", type=QueryType.EXEMPTION),
        ]

        mock_decomposer.decompose = AsyncMock(
            return_value=DecompositionResult(
                original_query="Is software consulting taxable?",
                sub_queries=sub_queries,
                reasoning="Complex query",
                is_simple=False,
            )
        )
        mock_retriever.retrieve.return_value = sample_results

        multi_retriever = MultiQueryRetriever(
            decomposer=mock_decomposer,
            retriever=mock_retriever,
        )

        result = await multi_retriever.retrieve("Is software consulting taxable?")

        # Should have results from both sub-queries
        assert len(result.sub_query_results) == 2
        assert result.decomposition.is_simple is False

    @pytest.mark.asyncio
    async def test_merge_deduplicates_by_chunk_id(self, mock_decomposer, mock_retriever):
        """Test results are deduplicated by chunk_id."""
        # Same chunk returned by both sub-queries
        result1 = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Same chunk",
            score=0.85,
        )
        result2 = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Same chunk",
            score=0.90,
        )

        sub_queries = [
            SubQuery(text="query1", type=QueryType.DEFINITION),
            SubQuery(text="query2", type=QueryType.EXEMPTION),
        ]

        mock_decomposer.decompose = AsyncMock(
            return_value=DecompositionResult(
                original_query="test",
                sub_queries=sub_queries,
                reasoning="test",
                is_simple=False,
            )
        )

        # First call returns result1, second returns result2
        mock_retriever.retrieve.side_effect = [[result1], [result2]]

        multi_retriever = MultiQueryRetriever(
            decomposer=mock_decomposer,
            retriever=mock_retriever,
        )

        result = await multi_retriever.retrieve("test query")

        # Should have only 1 result (deduplicated)
        assert len(result.merged_results) == 1
        # Should have the higher score + boost
        assert result.merged_results[0].score > 0.90

    @pytest.mark.asyncio
    async def test_multi_match_boost(self, mock_decomposer, mock_retriever):
        """Test chunks matching multiple sub-queries get boosted."""
        # Same chunk with same score
        result1 = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Matched twice",
            score=0.80,
        )
        result2 = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Matched twice",
            score=0.80,
        )

        sub_queries = [
            SubQuery(text="query1", type=QueryType.DEFINITION),
            SubQuery(text="query2", type=QueryType.EXEMPTION),
        ]

        mock_decomposer.decompose = AsyncMock(
            return_value=DecompositionResult(
                original_query="test",
                sub_queries=sub_queries,
                reasoning="test",
                is_simple=False,
            )
        )
        mock_retriever.retrieve.side_effect = [[result1], [result2]]

        multi_retriever = MultiQueryRetriever(
            decomposer=mock_decomposer,
            retriever=mock_retriever,
            multi_match_boost=0.1,
        )

        result = await multi_retriever.retrieve("test")

        # Score should be 0.80 + 0.1 (one boost for second match)
        assert result.merged_results[0].score == pytest.approx(0.90, abs=0.01)

    @pytest.mark.asyncio
    async def test_retrieve_simple_method(self, mock_decomposer, mock_retriever):
        """Test retrieve_simple bypasses decomposition."""
        sample_result = RetrievalResult(
            chunk_id="chunk:test:0",
            doc_id="test",
            doc_type="statute",
            level="parent",
            text="Test",
            score=0.9,
        )
        mock_retriever.retrieve.return_value = [sample_result]

        multi_retriever = MultiQueryRetriever(
            decomposer=mock_decomposer,
            retriever=mock_retriever,
        )

        results = await multi_retriever.retrieve_simple("simple query")

        assert len(results) == 1
        # Decomposer should not be called
        mock_decomposer.decompose.assert_not_called()


class TestSubQueryResult:
    """Tests for SubQueryResult model."""

    def test_result_count_property(self):
        """Test result_count property."""
        sq = SubQuery(text="test", type=QueryType.RATE)
        results = [
            RetrievalResult(
                chunk_id=f"chunk:{i}",
                doc_id=f"doc:{i}",
                doc_type="statute",
                level="parent",
                text="test",
                score=0.9,
            )
            for i in range(3)
        ]

        sqr = SubQueryResult(sub_query=sq, results=results)
        assert sqr.result_count == 3


class TestMultiRetrievalResult:
    """Tests for MultiRetrievalResult model."""

    def test_unique_doc_ids_property(self):
        """Test unique_doc_ids counts unique documents."""
        results = [
            RetrievalResult(
                chunk_id="chunk:0",
                doc_id="doc:A",
                doc_type="statute",
                level="parent",
                text="test",
                score=0.9,
            ),
            RetrievalResult(
                chunk_id="chunk:1",
                doc_id="doc:A",
                doc_type="statute",
                level="child",
                text="test",
                score=0.8,
            ),
            RetrievalResult(
                chunk_id="chunk:2",
                doc_id="doc:B",
                doc_type="statute",
                level="parent",
                text="test",
                score=0.7,
            ),
        ]

        mr = MultiRetrievalResult(
            original_query="test",
            decomposition=DecompositionResult(
                original_query="test",
                sub_queries=[],
                reasoning="test",
                is_simple=True,
            ),
            merged_results=results,
        )

        assert mr.unique_doc_ids == 2
        assert mr.unique_chunk_ids == 3


class TestQueryDecomposerIntegration:
    """Integration tests (require API key)."""

    @pytest.fixture
    def decomposer(self):
        """Create a real decomposer."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            api_key = settings.anthropic_api_key.get_secret_value()

            # Skip if placeholder key
            if "placeholder" in api_key.lower() or api_key.startswith("sk-ant-"):
                if len(api_key) < 50:  # Real keys are longer
                    pytest.skip("Anthropic API key is a placeholder")

            from src.retrieval import create_decomposer

            return create_decomposer()
        except Exception as e:
            pytest.skip(f"Could not create decomposer: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_decomposition(self, decomposer):
        """Test real query decomposition."""
        try:
            result = await decomposer.decompose(
                "Do I owe sales tax on software consulting services in Miami?"
            )

            # Should decompose into multiple sub-queries
            assert not result.is_simple or result.query_count > 0
            assert result.reasoning
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                pytest.skip(f"Invalid API key: {e}")
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_query_real(self, decomposer):
        """Test simple query handling."""
        result = await decomposer.decompose("What is the Florida sales tax rate?")

        # Should be simple or have minimal sub-queries
        assert result.is_simple or result.query_count <= 1
