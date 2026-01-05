"""Node functions for the Tax Agent graph."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Any

import structlog

from .state import TaxAgentState

logger = structlog.get_logger(__name__)


async def decompose_query(state: TaxAgentState) -> dict[str, Any]:
    """Break down the query into sub-queries.

    Uses QueryDecomposer to analyze the query and generate sub-queries.
    Simple queries (is_simple=True) skip to direct retrieval.

    Args:
        state: Current agent state with original_query

    Returns:
        State updates with sub_queries, is_simple_query, decomposition_reasoning
    """
    from src.retrieval import create_decomposer

    query = state["original_query"]
    logger.info("node_started", node="decompose_query", query=query[:50])

    try:
        decomposer = create_decomposer()
        result = await decomposer.decompose(query)

        # Convert SubQuery objects to dicts for state storage
        sub_queries = [sq.model_dump() for sq in result.sub_queries]

        # If simple query, create a single "sub-query" from original
        if result.is_simple:
            sub_queries = [{"text": query, "type": "general", "priority": 1}]

        logger.info(
            "node_completed",
            node="decompose_query",
            sub_query_count=len(sub_queries),
            is_simple=result.is_simple,
        )

        return {
            "sub_queries": sub_queries,
            "current_sub_query_idx": 0,
            "decomposition_reasoning": result.reasoning,
            "is_simple_query": result.is_simple,
            "reasoning_steps": [f"Decomposition: {result.reasoning}"],
            "current_iteration": state.get("current_iteration", 0) + 1,
            "max_iterations": state.get("max_iterations", 3),
        }

    except Exception as e:
        logger.error("node_failed", node="decompose_query", error=str(e))
        # Fallback: treat as simple query
        return {
            "sub_queries": [{"text": query, "type": "general", "priority": 1}],
            "current_sub_query_idx": 0,
            "decomposition_reasoning": f"Decomposition failed: {e}",
            "is_simple_query": True,
            "reasoning_steps": [f"Decomposition failed, using original query: {e}"],
            "errors": [f"Decomposition error: {e}"],
            "current_iteration": state.get("current_iteration", 0) + 1,
            "max_iterations": state.get("max_iterations", 3),
        }


async def retrieve_for_subquery(state: TaxAgentState) -> dict[str, Any]:
    """Run hybrid retrieval for current sub-query.

    Uses HybridRetriever with current sub-query or original query.
    Accumulates results in retrieved_chunks.

    Args:
        state: Current agent state with sub_queries and current_sub_query_idx

    Returns:
        State updates with current_retrieval_results, retrieved_chunks (appended)
    """
    from src.retrieval import create_retriever

    sub_queries = state.get("sub_queries", [])
    idx = state.get("current_sub_query_idx", 0)

    if idx >= len(sub_queries):
        # No more sub-queries, use original
        query_text = state["original_query"]
    else:
        query_text = sub_queries[idx].get("text", state["original_query"])

    logger.info(
        "node_started",
        node="retrieve_for_subquery",
        query=query_text[:50],
        sub_query_idx=idx,
    )

    try:
        retriever = create_retriever()

        # HybridRetriever.retrieve() is synchronous - wrap in to_thread
        results = await asyncio.to_thread(
            retriever.retrieve,
            query_text,
            top_k=20,
            expand_graph=False,  # We expand separately in expand_with_graph
            rerank=True,
        )

        # Convert RetrievalResult objects to dicts for state
        results_dicts = [r.model_dump() for r in results]

        logger.info(
            "node_completed",
            node="retrieve_for_subquery",
            result_count=len(results),
            sub_query_idx=idx,
        )

        return {
            "current_retrieval_results": results_dicts,
            "retrieved_chunks": results_dicts,  # Accumulates via Annotated[list, add]
            "current_sub_query_idx": idx + 1,
            "reasoning_steps": [
                f"Retrieved {len(results)} chunks for: {query_text[:50]}..."
            ],
        }

    except Exception as e:
        logger.error("node_failed", node="retrieve_for_subquery", error=str(e))
        return {
            "current_retrieval_results": [],
            "retrieved_chunks": [],
            "current_sub_query_idx": idx + 1,
            "reasoning_steps": [f"Retrieval failed for sub-query {idx}: {e}"],
            "errors": [f"Retrieval error: {e}"],
        }


async def expand_with_graph(state: TaxAgentState) -> dict[str, Any]:
    """Use Neo4j to expand context with related documents.

    For statutes: find implementing rules, interpreting cases/TAAs
    For rules: find parent statutes
    For cases: find cited authorities

    Args:
        state: Current agent state with current_retrieval_results

    Returns:
        State updates with graph_context, interpretation_chains
    """
    from src.graph import Neo4jClient
    from src.graph.queries import get_citing_documents, get_interpretation_chain

    results = state.get("current_retrieval_results", [])
    logger.info(
        "node_started", node="expand_with_graph", result_count=len(results)
    )

    graph_context: list[dict[str, Any]] = []
    interpretation_chains: dict[str, Any] = {}

    try:
        client = Neo4jClient()

        for result in results:
            doc_id = result.get("doc_id", "")
            doc_type = result.get("doc_type", "")

            try:
                if doc_type == "statute":
                    # Extract section number (e.g., "statute:212.05" -> "212.05")
                    section = doc_id.split(":")[-1] if ":" in doc_id else doc_id

                    chain = await asyncio.to_thread(
                        get_interpretation_chain, client, section
                    )
                    if chain:
                        interpretation_chains[doc_id] = chain.model_dump()
                        # Add related docs to context
                        for rule in chain.implementing_rules:
                            graph_context.append({
                                "target_doc_id": rule.id,
                                "target_citation": rule.full_citation,
                                "relation_type": "IMPLEMENTS",
                                "context_snippet": None,
                            })
                        for case in chain.interpreting_cases:
                            graph_context.append({
                                "target_doc_id": case.id,
                                "target_citation": case.full_citation,
                                "relation_type": "INTERPRETS",
                                "context_snippet": None,
                            })
                        for taa in chain.interpreting_taas:
                            graph_context.append({
                                "target_doc_id": taa.id,
                                "target_citation": taa.full_citation,
                                "relation_type": "INTERPRETS",
                                "context_snippet": None,
                            })

                elif doc_type in ("rule", "case", "taa"):
                    # Find citing/cited documents
                    citing = await asyncio.to_thread(
                        get_citing_documents, client, doc_id
                    )
                    for doc in citing[:5]:  # Limit to 5
                        graph_context.append({
                            "target_doc_id": doc.id,
                            "target_citation": doc.full_citation,
                            "relation_type": "CITES",
                            "context_snippet": None,
                        })

            except Exception as e:
                logger.warning(
                    "graph_expansion_failed",
                    doc_id=doc_id,
                    error=str(e),
                )

        logger.info(
            "node_completed",
            node="expand_with_graph",
            graph_context_count=len(graph_context),
            chains_found=len(interpretation_chains),
        )

        return {
            "graph_context": graph_context,
            "interpretation_chains": interpretation_chains,
            "reasoning_steps": [
                f"Graph expansion found {len(graph_context)} related documents"
            ],
        }

    except Exception as e:
        logger.error("node_failed", node="expand_with_graph", error=str(e))
        return {
            "graph_context": [],
            "interpretation_chains": {},
            "reasoning_steps": [f"Graph expansion failed: {e}"],
            "errors": [f"Graph expansion error: {e}"],
        }


async def score_relevance(state: TaxAgentState) -> dict[str, Any]:
    """Use LLM to score relevance of each retrieved chunk.

    Scores each chunk 0-1 for relevance to original query.
    Stores in relevance_scores dict.

    Args:
        state: Current agent state with current_retrieval_results

    Returns:
        State updates with relevance_scores
    """
    import anthropic

    from config.settings import get_settings
    from src.retrieval.prompts import RELEVANCE_PROMPT

    query = state["original_query"]
    results = state.get("current_retrieval_results", [])

    logger.info(
        "node_started", node="score_relevance", chunk_count=len(results)
    )

    settings = get_settings()
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value()
    )

    relevance_scores: dict[str, float] = {}

    async def score_one(chunk: dict) -> tuple[str, float, str]:
        """Score a single chunk."""
        chunk_id = chunk.get("chunk_id", "unknown")
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            doc_type=chunk.get("doc_type", "unknown"),
            citation=chunk.get("citation", "N/A"),
            text=chunk.get("text", "")[:2000],  # Limit text length
        )

        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            # Try to parse JSON from response
            # Handle case where response might have markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            score = float(data.get("score", 5)) / 10.0  # Normalize to 0-1
            reasoning = data.get("reasoning", "")
            return chunk_id, score, reasoning
        except Exception as e:
            logger.warning(
                "scoring_failed", chunk_id=chunk_id, error=str(e)
            )
            return chunk_id, 0.5, f"Scoring failed: {e}"

    # Score chunks in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(5)

    async def limited_score(chunk: dict) -> tuple[str, float, str]:
        async with semaphore:
            return await score_one(chunk)

    tasks = [limited_score(chunk) for chunk in results]
    scored = await asyncio.gather(*tasks)

    for chunk_id, score, _reasoning in scored:
        relevance_scores[chunk_id] = score

    # Calculate average score for logging
    avg_score = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0

    logger.info(
        "node_completed",
        node="score_relevance",
        chunks_scored=len(relevance_scores),
        avg_score=round(avg_score, 3),
    )

    return {
        "relevance_scores": relevance_scores,
        "reasoning_steps": [
            f"Scored {len(relevance_scores)} chunks for relevance (avg: {avg_score:.2f})"
        ],
    }


async def filter_irrelevant(state: TaxAgentState) -> dict[str, Any]:
    """Remove chunks below relevance threshold.

    Uses relevance_threshold (default 0.5).
    Produces filtered_chunks list.

    Args:
        state: Current agent state with relevance_scores

    Returns:
        State updates with filtered_chunks
    """
    results = state.get("current_retrieval_results", [])
    scores = state.get("relevance_scores", {})
    threshold = state.get("relevance_threshold", 0.5)

    logger.info(
        "node_started",
        node="filter_irrelevant",
        chunk_count=len(results),
        threshold=threshold,
    )

    # Score and sort chunks
    scored_chunks: list[tuple[dict, float]] = []
    for chunk in results:
        chunk_id = chunk.get("chunk_id")
        score = scores.get(chunk_id, 0.5)  # Default to 0.5 if not scored
        scored_chunks.append((chunk, score))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Filter: keep those above threshold OR top 10 (whichever is more)
    filtered: list[dict] = []
    for chunk, score in scored_chunks:
        if score >= threshold or len(filtered) < 10:
            filtered.append(chunk)

    # Log what was filtered
    removed = len(results) - len(filtered)

    logger.info(
        "node_completed",
        node="filter_irrelevant",
        filtered_count=len(filtered),
        removed_count=removed,
    )

    return {
        "filtered_chunks": filtered,
        "reasoning_steps": [
            f"Filtered {removed} chunks below threshold {threshold}. "
            f"Kept {len(filtered)} chunks."
        ],
    }


async def check_temporal_validity(state: TaxAgentState) -> dict[str, Any]:
    """Verify chunks apply to the relevant tax year.

    Infers tax year from query if not specified.
    Filters out superseded/amended documents.

    Args:
        state: Current agent state with filtered_chunks

    Returns:
        State updates with query_tax_year, temporally_valid_chunks
    """
    query = state["original_query"]
    filtered = state.get("filtered_chunks", [])

    logger.info(
        "node_started",
        node="check_temporal_validity",
        chunk_count=len(filtered),
    )

    # Extract year from query (e.g., "2023", "tax year 2024")
    year_match = re.search(r"\b(20\d{2})\b", query)
    query_tax_year = int(year_match.group(1)) if year_match else date.today().year

    # Check each chunk's effective date
    valid_chunks: list[dict] = []
    warnings: list[str] = []

    for chunk in filtered:
        effective_date = chunk.get("effective_date")

        if effective_date:
            # Parse date if string
            if isinstance(effective_date, str):
                try:
                    effective_date = date.fromisoformat(effective_date)
                except ValueError:
                    effective_date = None

            if effective_date and effective_date.year > query_tax_year:
                # Document is from after the query year - may not be applicable
                citation = chunk.get("citation", chunk.get("chunk_id"))
                warnings.append(
                    f"Document {citation} effective {effective_date} "
                    f"may not apply to {query_tax_year}"
                )
                continue

        valid_chunks.append(chunk)

    # Set needs_more_info if too few valid chunks
    needs_more_info = len(valid_chunks) < 3

    logger.info(
        "node_completed",
        node="check_temporal_validity",
        tax_year=query_tax_year,
        valid_count=len(valid_chunks),
        warning_count=len(warnings),
    )

    return {
        "query_tax_year": query_tax_year,
        "temporally_valid_chunks": valid_chunks,
        "needs_more_info": needs_more_info,
        "reasoning_steps": [
            f"Temporal check for tax year {query_tax_year}: "
            f"{len(valid_chunks)} valid, {len(warnings)} warnings"
        ]
        + warnings,
    }


async def synthesize_answer(state: TaxAgentState) -> dict[str, Any]:
    """Generate final answer with validated citations.

    Uses TaxLawGenerator to:
    1. Format chunks into structured context
    2. Generate answer with Claude using tax attorney system prompt
    3. Extract and validate citations against source chunks
    4. Calculate confidence based on source quality and verification

    Args:
        state: Current agent state with temporally_valid_chunks

    Returns:
        State updates with final_answer, citations, confidence
    """
    from src.generation import TaxLawGenerator, format_chunks_for_context

    valid_chunks = state.get("temporally_valid_chunks", [])
    reasoning_steps = state.get("reasoning_steps", [])
    query = state["original_query"]

    logger.info(
        "node_started",
        node="synthesize_answer",
        chunk_count=len(valid_chunks),
        query=query[:50],
    )

    # Initialize generator (it will create its own client)
    generator = TaxLawGenerator()

    try:
        # Generate response with citations
        response = await generator.generate(
            query=query,
            chunks=valid_chunks[:10],  # Top 10 chunks
            reasoning_steps=reasoning_steps,
        )

        # Convert ValidatedCitation to state Citation format
        citations: list[dict[str, Any]] = [
            {
                "doc_id": c.chunk_id or "",
                "citation": c.citation_text,
                "doc_type": c.doc_type,
                "text_snippet": c.raw_text[:200] if c.raw_text else "",
            }
            for c in response.citations
        ]

        # Count verified citations
        verified_count = sum(1 for c in response.citations if c.verified)
        total_count = len(response.citations)

        logger.info(
            "node_completed",
            node="synthesize_answer",
            citation_count=total_count,
            verified_count=verified_count,
            confidence=round(response.confidence, 2),
        )

        return {
            "final_answer": response.answer,
            "citations": citations,
            "confidence": response.confidence,
            "reasoning_steps": [
                f"Generated answer with {total_count} citations. "
                f"Verified: {verified_count}/{total_count}. "
                f"Confidence: {response.confidence:.2f}"
            ],
            "_synthesis_context": format_chunks_for_context(valid_chunks[:10]),
        }

    except Exception as e:
        logger.error("node_failed", node="synthesize_answer", error=str(e))

        # Fallback: return context without generation
        # Prepare basic citations from chunks
        fallback_citations: list[dict[str, Any]] = [
            {
                "doc_id": c.get("doc_id", ""),
                "citation": c.get("citation", f"Source {i+1}"),
                "doc_type": c.get("doc_type", "unknown"),
                "text_snippet": c.get("text", "")[:200],
            }
            for i, c in enumerate(valid_chunks[:10])
        ]

        return {
            "final_answer": None,
            "citations": fallback_citations,
            "confidence": 0.0,
            "errors": [f"Generation failed: {str(e)}"],
            "reasoning_steps": [f"Generation failed: {str(e)}"],
            "_synthesis_context": format_chunks_for_context(valid_chunks[:10]),
        }
