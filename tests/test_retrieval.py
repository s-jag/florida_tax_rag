"""Tests for the hybrid retrieval system."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.graph_expander import GraphExpander
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import CitationContext, RetrievalResult
from src.retrieval.reranker import LegalReranker


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_model_creation(self):
        """Test RetrievalResult can be created with required fields."""
        result = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Sample text",
            score=0.85,
        )

        assert result.chunk_id == "chunk:statute:212.05:0"
        assert result.doc_type == "statute"
        assert result.score == 0.85

    def test_optional_fields_default(self):
        """Test optional fields default correctly."""
        result = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Sample text",
            score=0.85,
        )

        assert result.ancestry is None
        assert result.citation is None
        assert result.effective_date is None
        assert result.graph_boost == 0.0
        assert result.related_chunk_ids == []
        assert result.citation_context == []
        assert result.source == "hybrid"

    def test_citation_context_list(self):
        """Test citation context can be added."""
        citation = CitationContext(
            target_doc_id="statute:212.08",
            target_citation="Fla. Stat. § 212.08",
            relation_type="CITES",
        )

        result = RetrievalResult(
            chunk_id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            doc_type="statute",
            level="parent",
            text="Sample text",
            score=0.85,
            citation_context=[citation],
        )

        assert len(result.citation_context) == 1
        assert result.citation_context[0].target_doc_id == "statute:212.08"


class TestCitationContext:
    """Tests for CitationContext model."""

    def test_model_creation(self):
        """Test CitationContext can be created."""
        ctx = CitationContext(
            target_doc_id="statute:212.08",
            target_citation="Fla. Stat. § 212.08",
            relation_type="CITES",
            context_snippet="as provided in § 212.08",
        )

        assert ctx.target_doc_id == "statute:212.08"
        assert ctx.relation_type == "CITES"
        assert ctx.context_snippet == "as provided in § 212.08"


class TestLegalReranker:
    """Tests for LegalReranker."""

    @pytest.fixture
    def reranker(self):
        """Create a reranker instance."""
        return LegalReranker()

    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(
                chunk_id="chunk:case:123:0",
                doc_id="case:123",
                doc_type="case",
                level="parent",
                text="Case text",
                score=0.90,
            ),
            RetrievalResult(
                chunk_id="chunk:statute:212.05:0",
                doc_id="statute:212.05",
                doc_type="statute",
                level="parent",
                text="Statute text",
                score=0.85,
            ),
            RetrievalResult(
                chunk_id="chunk:taa:25A-009:0",
                doc_id="taa:25A-009",
                doc_type="taa",
                level="parent",
                text="TAA text",
                score=0.95,
            ),
        ]

    def test_primary_authority_boost(self, reranker, sample_results):
        """Test statutes rank higher than cases with primary authority boost."""
        ranked = reranker.rerank(
            sample_results,
            prefer_primary=True,
            prefer_recent=False,
        )

        # Statute should rank higher due to primary authority boost
        # even though TAA had higher original score
        statute_idx = next(i for i, r in enumerate(ranked) if r.doc_type == "statute")
        taa_idx = next(i for i, r in enumerate(ranked) if r.doc_type == "taa")

        # After reranking: statute * 1.0 = 0.85, taa * 0.7 = 0.665
        assert statute_idx < taa_idx

    def test_no_primary_boost(self, reranker, sample_results):
        """Test ranking without primary authority boost."""
        ranked = reranker.rerank(
            sample_results,
            prefer_primary=False,
            prefer_recent=False,
        )

        # Without primary boost, TAA with 0.95 should be first
        assert ranked[0].doc_type == "taa"

    def test_recency_boost(self, reranker):
        """Test recent docs rank higher with recency boost."""
        results = [
            RetrievalResult(
                chunk_id="chunk:statute:old:0",
                doc_id="statute:old",
                doc_type="statute",
                level="parent",
                text="Old statute",
                score=0.90,
                effective_date=date(2000, 1, 1),
            ),
            RetrievalResult(
                chunk_id="chunk:statute:new:0",
                doc_id="statute:new",
                doc_type="statute",
                level="parent",
                text="New statute",
                score=0.85,
                effective_date=date(2024, 1, 1),
            ),
        ]

        ranked = reranker.rerank(
            results,
            prefer_primary=False,
            prefer_recent=True,
        )

        # Newer statute should rank higher due to recency boost
        assert ranked[0].doc_id == "statute:new"

    def test_diversity_penalty(self, reranker):
        """Test same doc_id gets penalized."""
        results = [
            RetrievalResult(
                chunk_id="chunk:statute:212.05:0",
                doc_id="statute:212.05",
                doc_type="statute",
                level="parent",
                text="First chunk",
                score=0.90,
            ),
            RetrievalResult(
                chunk_id="chunk:statute:212.05:1",
                doc_id="statute:212.05",
                doc_type="statute",
                level="child",
                text="Second chunk from same doc",
                score=0.88,
            ),
            RetrievalResult(
                chunk_id="chunk:statute:212.08:0",
                doc_id="statute:212.08",
                doc_type="statute",
                level="parent",
                text="Different doc",
                score=0.85,
            ),
        ]

        ranked = reranker.rerank(
            results,
            prefer_primary=False,
            prefer_recent=False,
            diversity_penalty=0.1,
        )

        # Second chunk from 212.05 should be penalized
        # 212.08 might rank higher than the second 212.05 chunk
        assert ranked[0].doc_id == "statute:212.05"


class TestGraphExpander:
    """Tests for GraphExpander."""

    @pytest.fixture
    def mock_neo4j(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        client.run_query = MagicMock(return_value=[])
        client.health_check = MagicMock(return_value=True)
        return client

    @pytest.fixture
    def expander(self, mock_neo4j):
        """Create a GraphExpander with mock client."""
        return GraphExpander(mock_neo4j)

    def test_get_parent_chunk(self, expander, mock_neo4j):
        """Test parent chunk retrieval."""
        mock_neo4j.run_query.return_value = [
            {
                "id": "chunk:statute:212.05:0",
                "text": "Parent text",
                "ancestry": "Florida Statutes > Chapter 212",
                "citation": "Fla. Stat. § 212.05",
            }
        ]

        parent = expander.get_parent_chunk("chunk:statute:212.05:1")

        assert parent is not None
        assert parent["id"] == "chunk:statute:212.05:0"
        mock_neo4j.run_query.assert_called_once()

    def test_get_parent_chunk_not_found(self, expander, mock_neo4j):
        """Test parent chunk not found."""
        mock_neo4j.run_query.return_value = []

        parent = expander.get_parent_chunk("chunk:statute:212.05:1")

        assert parent is None

    def test_expand_statute(self, expander, mock_neo4j):
        """Test statute expansion finds related docs."""
        # Mock get_interpretation_chain
        with patch("src.retrieval.graph_expander.get_interpretation_chain") as mock_chain:
            mock_chain.return_value = MagicMock(
                implementing_rules=[MagicMock(id="rule:12A-1.001")],
                interpreting_cases=[MagicMock(id="case:123")],
                interpreting_taas=[],
            )

            related = expander.expand_statute("statute:212.05")

            assert "rule:12A-1.001" in related
            assert "case:123" in related

    def test_expand_results_adds_context(self, expander, mock_neo4j):
        """Test expand_results enriches results."""
        mock_neo4j.run_query.return_value = []

        results = [
            RetrievalResult(
                chunk_id="chunk:statute:212.05:1",
                doc_id="statute:212.05",
                doc_type="statute",
                level="child",
                text="Child chunk",
                score=0.85,
            )
        ]

        # Mock the parent chunk query
        mock_neo4j.run_query.return_value = [
            {"id": "chunk:statute:212.05:0", "text": "Parent", "ancestry": "", "citation": ""}
        ]

        with patch.object(expander, "get_citations_for_chunk", return_value=[]):
            with patch.object(expander, "expand_statute", return_value=[]):
                expanded = expander.expand_results(results)

        assert len(expanded) == 1
        assert expanded[0].parent_chunk_id == "chunk:statute:212.05:0"


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.fixture
    def mock_weaviate(self):
        """Create a mock Weaviate client."""
        client = MagicMock()
        client.health_check = MagicMock(return_value=True)
        return client

    @pytest.fixture
    def mock_neo4j(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        client.health_check = MagicMock(return_value=True)
        client.run_query = MagicMock(return_value=[])
        return client

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.1] * 1024)
        return embedder

    @pytest.fixture
    def retriever(self, mock_weaviate, mock_neo4j, mock_embedder):
        """Create a HybridRetriever with mocks."""
        return HybridRetriever(
            weaviate_client=mock_weaviate,
            neo4j_client=mock_neo4j,
            embedder=mock_embedder,
        )

    def test_retrieve_returns_results(self, retriever, mock_weaviate):
        """Test basic retrieval returns results."""
        # Mock search result
        mock_result = MagicMock()
        mock_result.chunk_id = "chunk:statute:212.05:0"
        mock_result.doc_id = "statute:212.05"
        mock_result.doc_type = "statute"
        mock_result.level = "parent"
        mock_result.text = "Test text"
        mock_result.text_with_ancestry = "Florida > Test text"
        mock_result.ancestry = "Florida"
        mock_result.citation = "Fla. Stat. § 212.05"
        mock_result.effective_date = None
        mock_result.token_count = 100
        mock_result.score = 0.85
        mock_result.distance = 0.15

        mock_weaviate.hybrid_search.return_value = [mock_result]

        results = retriever.retrieve(
            "test query",
            top_k=10,
            expand_graph=False,
            rerank=False,
        )

        assert len(results) == 1
        assert results[0].chunk_id == "chunk:statute:212.05:0"
        assert results[0].score == 0.85

    def test_doc_type_filtering(self, retriever, mock_weaviate):
        """Test filtering by doc_type."""
        mock_weaviate.hybrid_search.return_value = []

        retriever.retrieve(
            "test query",
            doc_types=["statute", "rule"],
            expand_graph=False,
            rerank=False,
        )

        # Check that filters were passed
        call_kwargs = mock_weaviate.hybrid_search.call_args[1]
        assert call_kwargs["filters"] == {"doc_type": ["statute", "rule"]}

    def test_single_doc_type_filter(self, retriever, mock_weaviate):
        """Test single doc_type filter format."""
        mock_weaviate.hybrid_search.return_value = []

        retriever.retrieve(
            "test query",
            doc_types=["statute"],
            expand_graph=False,
            rerank=False,
        )

        call_kwargs = mock_weaviate.hybrid_search.call_args[1]
        assert call_kwargs["filters"] == {"doc_type": "statute"}

    def test_vector_search_method(self, retriever, mock_weaviate):
        """Test vector_search method."""
        mock_result = MagicMock()
        mock_result.chunk_id = "chunk:statute:212.05:0"
        mock_result.doc_id = "statute:212.05"
        mock_result.doc_type = "statute"
        mock_result.level = "parent"
        mock_result.text = "Test text"
        mock_result.text_with_ancestry = None
        mock_result.ancestry = None
        mock_result.citation = None
        mock_result.effective_date = None
        mock_result.token_count = None
        mock_result.score = 0.85
        mock_result.distance = 0.15

        mock_weaviate.vector_search.return_value = [mock_result]

        results = retriever.vector_search("test query")

        assert len(results) == 1
        assert results[0].source == "vector"

    def test_keyword_search_method(self, retriever, mock_weaviate):
        """Test keyword_search method."""
        mock_result = MagicMock()
        mock_result.chunk_id = "chunk:statute:212.05:0"
        mock_result.doc_id = "statute:212.05"
        mock_result.doc_type = "statute"
        mock_result.level = "parent"
        mock_result.text = "Test text"
        mock_result.text_with_ancestry = None
        mock_result.ancestry = None
        mock_result.citation = None
        mock_result.effective_date = None
        mock_result.token_count = None
        mock_result.score = 0.85
        mock_result.distance = None

        mock_weaviate.keyword_search.return_value = [mock_result]

        results = retriever.keyword_search("test query")

        assert len(results) == 1
        assert results[0].source == "keyword"


class TestHybridRetrieverIntegration:
    """Integration tests (require services)."""

    @pytest.fixture
    def retriever(self):
        """Create a real retriever with actual clients."""
        try:
            from src.retrieval import create_retriever

            return create_retriever()
        except Exception as e:
            pytest.skip(f"Could not create retriever: {e}")

    @pytest.mark.integration
    def test_end_to_end_retrieval(self, retriever):
        """Test full retrieval pipeline."""
        results = retriever.retrieve(
            "Florida sales tax rate",
            top_k=5,
        )

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.integration
    def test_filtered_retrieval(self, retriever):
        """Test retrieval with doc_type filter."""
        results = retriever.retrieve(
            "tax exemption",
            top_k=5,
            doc_types=["statute"],
        )

        assert len(results) > 0
        assert all(r.doc_type == "statute" for r in results)
