"""Multi-query retriever that executes sub-queries and merges results."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from .hybrid import HybridRetriever
from .models import RetrievalResult
from .query_decomposer import DecompositionResult, QueryDecomposer, SubQuery

logger = logging.getLogger(__name__)


class SubQueryResult(BaseModel):
    """Results for a single sub-query."""

    sub_query: SubQuery = Field(..., description="The sub-query that was executed")
    results: list[RetrievalResult] = Field(
        default_factory=list, description="Retrieval results for this sub-query"
    )

    @property
    def result_count(self) -> int:
        """Number of results for this sub-query."""
        return len(self.results)


class MultiRetrievalResult(BaseModel):
    """Combined results from multi-query retrieval."""

    original_query: str = Field(..., description="The original user query")
    decomposition: DecompositionResult = Field(
        ..., description="Decomposition result (may be is_simple=True)"
    )
    sub_query_results: list[SubQueryResult] = Field(
        default_factory=list, description="Results for each sub-query"
    )
    merged_results: list[RetrievalResult] = Field(
        default_factory=list, description="Merged and deduplicated results"
    )

    @property
    def total_results(self) -> int:
        """Total results before deduplication."""
        return sum(sqr.result_count for sqr in self.sub_query_results)

    @property
    def unique_doc_ids(self) -> int:
        """Number of unique documents in merged results."""
        return len(set(r.doc_id for r in self.merged_results))

    @property
    def unique_chunk_ids(self) -> int:
        """Number of unique chunks in merged results."""
        return len(set(r.chunk_id for r in self.merged_results))


class MultiQueryRetriever:
    """Executes multiple sub-queries and merges results.

    This retriever:
    1. Decomposes complex queries into sub-queries
    2. Runs retrieval for each sub-query in parallel
    3. Merges and deduplicates results
    4. Boosts scores for chunks appearing in multiple sub-queries
    """

    # Score boost for chunks appearing in multiple sub-queries
    MULTI_MATCH_BOOST = 0.1

    def __init__(
        self,
        decomposer: QueryDecomposer,
        retriever: HybridRetriever,
        max_parallel: int = 3,
        multi_match_boost: float = 0.1,
    ):
        """Initialize the multi-query retriever.

        Args:
            decomposer: QueryDecomposer for breaking down queries
            retriever: HybridRetriever for executing searches
            max_parallel: Maximum parallel retrievals
            multi_match_boost: Score boost per additional sub-query match
        """
        self.decomposer = decomposer
        self.retriever = retriever
        self.max_parallel = max_parallel
        self.multi_match_boost = multi_match_boost

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        decompose: bool = True,
        per_query_top_k: int | None = None,
        **retrieval_kwargs: Any,
    ) -> MultiRetrievalResult:
        """Retrieve documents using query decomposition.

        Args:
            query: User's search query
            top_k: Number of final results to return
            decompose: Whether to attempt decomposition
            per_query_top_k: Results per sub-query (defaults to top_k)
            **retrieval_kwargs: Additional args for HybridRetriever

        Returns:
            MultiRetrievalResult with merged results and sub-query attribution
        """
        query = query.strip()
        per_query_top_k = per_query_top_k or top_k

        # Decompose if enabled
        if decompose:
            decomposition = await self.decomposer.decompose(query)
        else:
            decomposition = DecompositionResult(
                original_query=query,
                sub_queries=[],
                reasoning="Decomposition disabled",
                is_simple=True,
            )

        # Simple query path
        if decomposition.is_simple:
            logger.info(f"Simple query, running single retrieval: {query[:50]}...")
            results = await self._run_single_retrieval(query, top_k, **retrieval_kwargs)
            return MultiRetrievalResult(
                original_query=query,
                decomposition=decomposition,
                sub_query_results=[],
                merged_results=results,
            )

        # Complex query path: parallel sub-query retrieval
        logger.info(f"Running {decomposition.query_count} sub-queries for: {query[:50]}...")
        sub_query_results = await self._parallel_retrieve(
            decomposition.sub_queries,
            per_query_top_k,
            **retrieval_kwargs,
        )

        # Merge and deduplicate
        merged = self._merge_results(sub_query_results, top_k)

        return MultiRetrievalResult(
            original_query=query,
            decomposition=decomposition,
            sub_query_results=sub_query_results,
            merged_results=merged,
        )

    async def retrieve_simple(
        self,
        query: str,
        top_k: int = 20,
        **retrieval_kwargs: Any,
    ) -> list[RetrievalResult]:
        """Simple retrieval without decomposition.

        Convenience method for when decomposition is not needed.

        Args:
            query: User's search query
            top_k: Number of results to return
            **retrieval_kwargs: Additional args for HybridRetriever

        Returns:
            List of RetrievalResult
        """
        return await self._run_single_retrieval(query, top_k, **retrieval_kwargs)

    async def _run_single_retrieval(
        self,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Run a single retrieval in executor.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional retrieval args

        Returns:
            List of RetrievalResult
        """
        return await asyncio.to_thread(
            self.retriever.retrieve,
            query,
            top_k=top_k,
            **kwargs,
        )

    async def _parallel_retrieve(
        self,
        sub_queries: list[SubQuery],
        top_k: int,
        **kwargs: Any,
    ) -> list[SubQueryResult]:
        """Run retrievals for sub-queries in parallel.

        Args:
            sub_queries: List of sub-queries to execute
            top_k: Results per sub-query
            **kwargs: Additional retrieval args

        Returns:
            List of SubQueryResult
        """
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_one(sq: SubQuery) -> SubQueryResult:
            async with semaphore:
                logger.debug(f"Running sub-query: {sq.text[:50]}...")
                results = await asyncio.to_thread(
                    self.retriever.retrieve,
                    sq.text,
                    top_k=top_k,
                    **kwargs,
                )
                return SubQueryResult(sub_query=sq, results=results)

        # Run all sub-queries in parallel (limited by semaphore)
        results = await asyncio.gather(
            *[run_one(sq) for sq in sub_queries],
            return_exceptions=True,
        )

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Sub-query failed: {r}")
            else:
                valid_results.append(r)

        return valid_results

    def _merge_results(
        self,
        sub_query_results: list[SubQueryResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge results from multiple sub-queries.

        Deduplicates by chunk_id and boosts scores for chunks
        appearing in multiple sub-queries.

        Args:
            sub_query_results: Results from each sub-query
            top_k: Number of final results to return

        Returns:
            Merged and reranked results
        """
        # Track chunks by ID with their best result and match count
        chunk_map: dict[str, tuple[RetrievalResult, int, list[str]]] = {}

        for sqr in sub_query_results:
            sub_query_text = sqr.sub_query.text
            for result in sqr.results:
                chunk_id = result.chunk_id

                if chunk_id in chunk_map:
                    existing_result, match_count, matched_queries = chunk_map[chunk_id]
                    # Keep the higher-scoring result
                    if result.score > existing_result.score:
                        chunk_map[chunk_id] = (
                            result,
                            match_count + 1,
                            matched_queries + [sub_query_text],
                        )
                    else:
                        chunk_map[chunk_id] = (
                            existing_result,
                            match_count + 1,
                            matched_queries + [sub_query_text],
                        )
                else:
                    chunk_map[chunk_id] = (result, 1, [sub_query_text])

        # Apply multi-match boost and collect results
        merged: list[RetrievalResult] = []
        for chunk_id, (result, match_count, matched_queries) in chunk_map.items():
            # Apply boost for appearing in multiple sub-queries
            if match_count > 1:
                boost = self.multi_match_boost * (match_count - 1)
                result.score += boost
                logger.debug(
                    f"Boosted {chunk_id} by {boost:.3f} (matched {match_count} sub-queries)"
                )
            merged.append(result)

        # Sort by score descending
        merged.sort(key=lambda r: r.score, reverse=True)

        # Return top_k
        return merged[:top_k]


def create_multi_retriever(
    decomposer: QueryDecomposer | None = None,
    retriever: HybridRetriever | None = None,
) -> MultiQueryRetriever:
    """Factory function to create a MultiQueryRetriever with default configuration.

    Args:
        decomposer: Optional QueryDecomposer (created if None)
        retriever: Optional HybridRetriever (created if None)

    Returns:
        Configured MultiQueryRetriever instance
    """
    if decomposer is None:
        from .query_decomposer import create_decomposer

        decomposer = create_decomposer()

    if retriever is None:
        from .hybrid import create_retriever

        retriever = create_retriever()

    return MultiQueryRetriever(
        decomposer=decomposer,
        retriever=retriever,
    )
