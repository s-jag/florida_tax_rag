"""Hybrid retrieval combining vector search, keyword search, and graph traversal."""

from __future__ import annotations

import logging
from datetime import date

from src.graph.client import Neo4jClient
from src.vector.client import SearchResult, WeaviateClient
from src.vector.embeddings import VoyageEmbedder

from .graph_expander import GraphExpander
from .models import RetrievalResult
from .reranker import LegalReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval system combining vector search, keyword search, and graph traversal.

    This retriever:
    1. Embeds the query using Voyage AI
    2. Performs hybrid search in Weaviate (combining BM25 and vector similarity)
    3. Optionally expands results using Neo4j graph traversal
    4. Re-ranks results using legal-specific heuristics
    """

    def __init__(
        self,
        weaviate_client: WeaviateClient,
        neo4j_client: Neo4jClient,
        embedder: VoyageEmbedder,
        reranker: LegalReranker | None = None,
        graph_expander: GraphExpander | None = None,
    ):
        """Initialize the hybrid retriever.

        Args:
            weaviate_client: Client for Weaviate vector store
            neo4j_client: Client for Neo4j knowledge graph
            embedder: Voyage AI embedder for query embedding
            reranker: Optional custom reranker (defaults to LegalReranker)
            graph_expander: Optional custom graph expander
        """
        self.weaviate = weaviate_client
        self.neo4j = neo4j_client
        self.embedder = embedder
        self.reranker = reranker or LegalReranker()
        self.graph_expander = graph_expander or GraphExpander(neo4j_client)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.5,
        doc_types: list[str] | None = None,
        min_date: date | None = None,
        max_date: date | None = None,
        expand_graph: bool = True,
        rerank: bool = True,
        prefer_recent: bool = True,
        prefer_primary: bool = True,
    ) -> list[RetrievalResult]:
        """Perform hybrid retrieval.

        Args:
            query: User's search query
            top_k: Number of results to return
            alpha: Balance between vector (1.0) and keyword (0.0) search
            doc_types: Filter by document types (statute, rule, case, taa)
            min_date: Filter by minimum effective date
            max_date: Filter by maximum effective date
            expand_graph: Whether to expand results with graph traversal
            rerank: Whether to apply legal re-ranking
            prefer_recent: Whether to boost newer documents (in reranking)
            prefer_primary: Whether to boost primary authorities (in reranking)

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # 1. Embed the query
        query_vector = self.embedder.embed_query(query)

        # 2. Build filters
        filters = self._build_filters(doc_types)

        # 3. Perform hybrid search in Weaviate
        # Get more results than needed for reranking
        fetch_limit = top_k * 2 if rerank else top_k

        search_results = self.weaviate.hybrid_search(
            query=query,
            query_vector=query_vector,
            alpha=alpha,
            limit=fetch_limit,
            filters=filters,
        )

        logger.info(f"Weaviate returned {len(search_results)} results")

        # 4. Convert to RetrievalResult
        results = [self._to_retrieval_result(sr) for sr in search_results]

        # 5. Apply date filtering if specified (Weaviate doesn't support date ranges well)
        if min_date or max_date:
            results = self._filter_by_date(results, min_date, max_date)

        # 6. Graph expansion (optional)
        if expand_graph and self.neo4j.health_check():
            logger.info("Expanding results with graph traversal")
            results = self.graph_expander.expand_results(results)
        elif expand_graph:
            logger.warning("Neo4j not available, skipping graph expansion")

        # 7. Re-rank (optional)
        if rerank:
            logger.info("Reranking results")
            results = self.reranker.rerank(
                results,
                prefer_recent=prefer_recent,
                prefer_primary=prefer_primary,
            )

        # 8. Return top_k results
        return results[:top_k]

    def vector_search(
        self,
        query: str,
        top_k: int = 20,
        doc_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Perform vector-only search (no keyword matching).

        Args:
            query: User's search query
            top_k: Number of results to return
            doc_types: Filter by document types

        Returns:
            List of RetrievalResult objects
        """
        query_vector = self.embedder.embed_query(query)
        filters = self._build_filters(doc_types)

        search_results = self.weaviate.vector_search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters,
        )

        results = [self._to_retrieval_result(sr) for sr in search_results]
        for r in results:
            r.source = "vector"

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        doc_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Perform keyword-only search (BM25, no vector similarity).

        Args:
            query: User's search query
            top_k: Number of results to return
            doc_types: Filter by document types

        Returns:
            List of RetrievalResult objects
        """
        filters = self._build_filters(doc_types)

        search_results = self.weaviate.keyword_search(
            query=query,
            limit=top_k,
            filters=filters,
        )

        results = [self._to_retrieval_result(sr) for sr in search_results]
        for r in results:
            r.source = "keyword"

        return results

    def _build_filters(
        self,
        doc_types: list[str] | None,
    ) -> dict | None:
        """Build Weaviate filter dict.

        Args:
            doc_types: List of document types to filter by

        Returns:
            Filter dict or None if no filters
        """
        if not doc_types:
            return None

        # Single type vs multiple types
        if len(doc_types) == 1:
            return {"doc_type": doc_types[0]}
        else:
            return {"doc_type": doc_types}

    def _filter_by_date(
        self,
        results: list[RetrievalResult],
        min_date: date | None,
        max_date: date | None,
    ) -> list[RetrievalResult]:
        """Filter results by effective date range.

        Args:
            results: Results to filter
            min_date: Minimum effective date (inclusive)
            max_date: Maximum effective date (inclusive)

        Returns:
            Filtered results
        """
        filtered = []
        for r in results:
            if r.effective_date is None:
                # Include results without date
                filtered.append(r)
            else:
                if min_date and r.effective_date < min_date:
                    continue
                if max_date and r.effective_date > max_date:
                    continue
                filtered.append(r)
        return filtered

    def _to_retrieval_result(self, sr: SearchResult) -> RetrievalResult:
        """Convert Weaviate SearchResult to RetrievalResult.

        Args:
            sr: Weaviate search result

        Returns:
            RetrievalResult with mapped fields
        """
        # Determine score type based on distance presence
        vector_score = None
        keyword_score = None

        if sr.distance is not None:
            # Vector search includes distance
            vector_score = sr.score
        else:
            # Keyword search doesn't include distance
            keyword_score = sr.score

        # Convert datetime to date if present
        effective_date = None
        if sr.effective_date:
            effective_date = sr.effective_date.date()

        return RetrievalResult(
            chunk_id=sr.chunk_id,
            doc_id=sr.doc_id,
            doc_type=sr.doc_type,
            level=sr.level,
            text=sr.text,
            text_with_ancestry=sr.text_with_ancestry,
            ancestry=sr.ancestry,
            citation=sr.citation,
            effective_date=effective_date,
            token_count=sr.token_count,
            score=sr.score,
            vector_score=vector_score,
            keyword_score=keyword_score,
            source="hybrid",
        )


def create_retriever() -> HybridRetriever:
    """Factory function to create a HybridRetriever with default configuration.

    Returns:
        Configured HybridRetriever instance
    """
    weaviate = WeaviateClient()
    neo4j = Neo4jClient()
    embedder = VoyageEmbedder()

    return HybridRetriever(
        weaviate_client=weaviate,
        neo4j_client=neo4j,
        embedder=embedder,
    )
