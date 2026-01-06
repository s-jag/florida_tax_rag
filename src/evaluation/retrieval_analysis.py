"""Retrieval analysis system for evaluating search quality."""

from __future__ import annotations

import logging
from typing import Optional

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult

from .metrics import citations_match, normalize_citation
from .models import EvalDataset, EvalQuestion
from .retrieval_metrics import (
    AlphaTuningResult,
    MethodComparisonResult,
    QueryRetrievalResult,
    RetrievalAnalysisReport,
    RetrievalMetrics,
    find_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class RetrievalAnalyzer:
    """Analyzes retrieval quality against golden dataset."""

    def __init__(
        self,
        retriever: HybridRetriever,
        dataset: EvalDataset,
    ):
        """Initialize the analyzer.

        Args:
            retriever: The HybridRetriever to analyze
            dataset: Evaluation dataset with expected citations
        """
        self.retriever = retriever
        self.dataset = dataset
        self._questions_map = {q.id: q for q in dataset.questions}

    def _extract_citation_from_doc_id(self, doc_id: str) -> str:
        """Extract citation from doc_id format.

        Handles formats like:
        - "statute:212.05" -> "212.05"
        - "rule:12a-1.001" -> "12a-1.001"

        Args:
            doc_id: Document ID from retrieval result

        Returns:
            Extracted citation string
        """
        if ":" in doc_id:
            return doc_id.split(":", 1)[1]
        return doc_id

    def _doc_matches_expected(self, doc_id: str, expected: str) -> bool:
        """Check if a retrieved doc_id matches an expected citation.

        Uses the existing citations_match logic for subsection handling.

        Args:
            doc_id: Retrieved document ID (e.g., "statute:212.05")
            expected: Expected citation (e.g., "212.05")

        Returns:
            True if they match
        """
        citation = self._extract_citation_from_doc_id(doc_id)
        return citations_match(citation, expected)

    def _create_matcher(self):
        """Create a matcher function for metric calculations."""
        return lambda doc_id, expected: self._doc_matches_expected(doc_id, expected)

    def analyze_query(
        self,
        question: EvalQuestion,
        top_k: int = 20,
        method: str = "hybrid",
        alpha: float = 0.5,
    ) -> QueryRetrievalResult:
        """Analyze retrieval quality for a single query.

        Args:
            question: Evaluation question with expected citations
            top_k: Number of results to retrieve
            method: Retrieval method ("vector", "keyword", "hybrid", "graph")
            alpha: Alpha parameter for hybrid search

        Returns:
            QueryRetrievalResult with detailed analysis
        """
        # Build expected citations list
        expected = question.expected_statutes + question.expected_rules

        # Perform retrieval based on method
        if method == "vector":
            results = self.retriever.vector_search(
                question.question,
                top_k=top_k,
            )
        elif method == "keyword":
            results = self.retriever.keyword_search(
                question.question,
                top_k=top_k,
            )
        elif method == "graph":
            results = self.retriever.retrieve(
                question.question,
                top_k=top_k,
                alpha=alpha,
                expand_graph=True,
                rerank=True,
            )
        else:  # hybrid (default)
            results = self.retriever.retrieve(
                question.question,
                top_k=top_k,
                alpha=alpha,
                expand_graph=False,
                rerank=True,
            )

        # Extract doc_ids and citations
        retrieved_doc_ids = [r.doc_id for r in results]
        retrieved_citations = [
            self._extract_citation_from_doc_id(r.doc_id) for r in results
        ]
        retrieved_scores = [r.score for r in results]

        # Find which expected docs were retrieved and at what rank
        found_expected = {}
        for exp in expected:
            found_rank = 0
            for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
                if self._doc_matches_expected(doc_id, exp):
                    found_rank = rank
                    break
            found_expected[exp] = found_rank

        # Create matcher for metrics
        matcher = self._create_matcher()

        # Calculate metrics
        rr = mean_reciprocal_rank(expected, retrieved_doc_ids, matcher)
        ndcg_5 = ndcg_at_k(expected, retrieved_doc_ids, 5, matcher)
        ndcg_10 = ndcg_at_k(expected, retrieved_doc_ids, 10, matcher)
        r_at_5 = recall_at_k(expected, retrieved_doc_ids, 5, matcher)
        r_at_10 = recall_at_k(expected, retrieved_doc_ids, 10, matcher)
        r_at_20 = recall_at_k(expected, retrieved_doc_ids, 20, matcher)

        # Find irrelevant docs in top-k
        irrelevant_5 = [
            doc_id
            for doc_id in retrieved_doc_ids[:5]
            if not any(self._doc_matches_expected(doc_id, e) for e in expected)
        ]
        irrelevant_10 = [
            doc_id
            for doc_id in retrieved_doc_ids[:10]
            if not any(self._doc_matches_expected(doc_id, e) for e in expected)
        ]

        return QueryRetrievalResult(
            question_id=question.id,
            question_text=question.question,
            expected_citations=expected,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_citations=retrieved_citations,
            retrieved_scores=retrieved_scores,
            found_expected=found_expected,
            reciprocal_rank=rr,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            recall_at_5=r_at_5,
            recall_at_10=r_at_10,
            recall_at_20=r_at_20,
            irrelevant_in_top_5=irrelevant_5,
            irrelevant_in_top_10=irrelevant_10,
        )

    def compare_retrieval_methods(
        self,
        question: EvalQuestion,
        top_k: int = 20,
        alpha: float = 0.5,
    ) -> MethodComparisonResult:
        """Compare all retrieval methods for a single query.

        Args:
            question: Evaluation question
            top_k: Number of results per method
            alpha: Alpha for hybrid search

        Returns:
            MethodComparisonResult with all method results
        """
        expected = question.expected_statutes + question.expected_rules

        # Run all methods
        vector_result = self.analyze_query(question, top_k, method="vector")
        keyword_result = self.analyze_query(question, top_k, method="keyword")
        hybrid_result = self.analyze_query(question, top_k, method="hybrid", alpha=alpha)
        graph_result = self.analyze_query(question, top_k, method="graph", alpha=alpha)

        # Determine best method
        methods = {
            "vector": vector_result.reciprocal_rank,
            "keyword": keyword_result.reciprocal_rank,
            "hybrid": hybrid_result.reciprocal_rank,
            "graph": graph_result.reciprocal_rank,
        }
        best_method = max(methods, key=methods.get)

        return MethodComparisonResult(
            question_id=question.id,
            question_text=question.question,
            expected_citations=expected,
            vector_result=vector_result,
            keyword_result=keyword_result,
            hybrid_result=hybrid_result,
            graph_result=graph_result,
            best_method=best_method,
            best_mrr=methods[best_method],
        )

    def tune_alpha(
        self,
        alphas: Optional[list[float]] = None,
        top_k: int = 20,
        progress_callback: Optional[callable] = None,
    ) -> list[AlphaTuningResult]:
        """Tune alpha parameter across all questions.

        Args:
            alphas: List of alpha values to test
            top_k: Number of results to retrieve
            progress_callback: Optional callback(alpha, i, n)

        Returns:
            List of AlphaTuningResult for each alpha
        """
        if alphas is None:
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        results = []
        matcher = self._create_matcher()

        for alpha in alphas:
            logger.info(f"Testing alpha={alpha}")

            per_query_mrr = {}
            all_recall_5 = []
            all_recall_10 = []
            all_ndcg_10 = []

            for i, question in enumerate(self.dataset.questions):
                query_result = self.analyze_query(
                    question,
                    top_k=top_k,
                    method="hybrid",
                    alpha=alpha,
                )
                per_query_mrr[question.id] = query_result.reciprocal_rank
                all_recall_5.append(query_result.recall_at_5)
                all_recall_10.append(query_result.recall_at_10)
                all_ndcg_10.append(query_result.ndcg_at_10)

                if progress_callback:
                    progress_callback(alpha, i + 1, len(self.dataset.questions))

            # Calculate aggregate metrics
            n = len(self.dataset.questions)
            mrr = sum(per_query_mrr.values()) / n if n else 0.0

            results.append(
                AlphaTuningResult(
                    alpha=alpha,
                    mrr=mrr,
                    recall_at_5=sum(all_recall_5) / n if n else 0.0,
                    recall_at_10=sum(all_recall_10) / n if n else 0.0,
                    ndcg_at_10=sum(all_ndcg_10) / n if n else 0.0,
                    per_query_mrr=per_query_mrr,
                )
            )

        return results

    def run_full_analysis(
        self,
        alphas: Optional[list[float]] = None,
        top_k: int = 20,
        progress_callback: Optional[callable] = None,
    ) -> RetrievalAnalysisReport:
        """Run full retrieval analysis.

        Args:
            alphas: Alpha values for tuning
            top_k: Number of results
            progress_callback: Optional callback(step, i, n)

        Returns:
            Complete RetrievalAnalysisReport
        """
        comparisons = []
        failed_queries = []

        # Aggregate metrics by method
        method_results: dict[str, list[QueryRetrievalResult]] = {
            "vector": [],
            "keyword": [],
            "hybrid": [],
            "graph": [],
        }

        logger.info(f"Analyzing {len(self.dataset.questions)} questions...")

        for i, question in enumerate(self.dataset.questions):
            logger.info(f"Analyzing {question.id} ({i+1}/{len(self.dataset.questions)})")

            if progress_callback:
                progress_callback("compare", i + 1, len(self.dataset.questions))

            try:
                comparison = self.compare_retrieval_methods(question, top_k=top_k)
                comparisons.append(comparison)

                # Collect per-method results
                method_results["vector"].append(comparison.vector_result)
                method_results["keyword"].append(comparison.keyword_result)
                method_results["hybrid"].append(comparison.hybrid_result)
                method_results["graph"].append(comparison.graph_result)

                # Track failed queries (no expected docs in top-20)
                if comparison.hybrid_result.recall_at_20 == 0:
                    failed_queries.append(comparison.hybrid_result)

            except Exception as e:
                logger.error(f"Error analyzing {question.id}: {e}")

        # Calculate aggregate metrics for each method
        def aggregate_metrics(results: list[QueryRetrievalResult]) -> RetrievalMetrics:
            n = len(results)
            if n == 0:
                return RetrievalMetrics()

            return RetrievalMetrics(
                mrr=sum(r.reciprocal_rank for r in results) / n,
                ndcg_at_5=sum(r.ndcg_at_5 for r in results) / n,
                ndcg_at_10=sum(r.ndcg_at_10 for r in results) / n,
                recall_at_5=sum(r.recall_at_5 for r in results) / n,
                recall_at_10=sum(r.recall_at_10 for r in results) / n,
                recall_at_20=sum(r.recall_at_20 for r in results) / n,
            )

        logger.info("Aggregating metrics by method...")

        method_metrics = {
            method: aggregate_metrics(results)
            for method, results in method_results.items()
        }

        # Alpha tuning
        logger.info("Running alpha tuning...")
        alpha_results = self.tune_alpha(alphas=alphas, top_k=top_k)
        optimal = max(alpha_results, key=lambda r: r.mrr)

        # Find best method
        best_method = max(method_metrics, key=lambda m: method_metrics[m].mrr)

        return RetrievalAnalysisReport(
            vector_metrics=method_metrics["vector"],
            keyword_metrics=method_metrics["keyword"],
            hybrid_metrics=method_metrics["hybrid"],
            graph_metrics=method_metrics["graph"],
            alpha_tuning=alpha_results,
            optimal_alpha=optimal.alpha,
            failed_queries=failed_queries,
            comparisons=comparisons,
            best_overall_method=best_method,
            best_method_mrr=method_metrics[best_method].mrr,
        )


def debug_retrieval(
    retriever: HybridRetriever,
    query: str,
    expected_citations: Optional[list[str]] = None,
    top_k: int = 20,
) -> dict:
    """Deep dive debugging for a single query.

    Shows detailed results for each retrieval method including
    scores, document types, and text previews.

    Args:
        retriever: HybridRetriever instance
        query: Query string to analyze
        expected_citations: Optional expected citations for analysis
        top_k: Number of results

    Returns:
        Dict with detailed debugging info for each method
    """
    results = {
        "query": query,
        "expected": expected_citations or [],
        "methods": {},
    }

    methods_config = [
        ("vector", lambda: retriever.vector_search(query, top_k=top_k)),
        ("keyword", lambda: retriever.keyword_search(query, top_k=top_k)),
        (
            "hybrid",
            lambda: retriever.retrieve(
                query, top_k=top_k, alpha=0.5, expand_graph=False
            ),
        ),
        (
            "graph",
            lambda: retriever.retrieve(
                query, top_k=top_k, alpha=0.5, expand_graph=True
            ),
        ),
    ]

    for method_name, retrieval_fn in methods_config:
        try:
            retrieved = retrieval_fn()

            results["methods"][method_name] = [
                {
                    "rank": i + 1,
                    "doc_id": r.doc_id,
                    "doc_type": r.doc_type,
                    "citation": r.citation,
                    "score": round(r.score, 4),
                    "vector_score": round(r.vector_score, 4) if r.vector_score else None,
                    "keyword_score": (
                        round(r.keyword_score, 4) if r.keyword_score else None
                    ),
                    "graph_boost": round(r.graph_boost, 4) if r.graph_boost else 0.0,
                    "text_preview": r.text[:200] if r.text else "",
                    "matches_expected": (
                        any(
                            citations_match(
                                r.doc_id.split(":", 1)[1] if ":" in r.doc_id else r.doc_id,
                                exp,
                            )
                            for exp in (expected_citations or [])
                        )
                        if expected_citations
                        else None
                    ),
                }
                for i, r in enumerate(retrieved)
            ]
        except Exception as e:
            results["methods"][method_name] = {"error": str(e)}

    return results


def generate_retrieval_markdown_report(report: RetrievalAnalysisReport) -> str:
    """Generate markdown report from retrieval analysis.

    Args:
        report: RetrievalAnalysisReport to format

    Returns:
        Formatted markdown string
    """
    lines = []

    lines.append("# Retrieval Analysis Report")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Best Overall Method:** {report.best_overall_method}")
    lines.append(f"**Best MRR:** {report.best_method_mrr:.4f}")
    lines.append(f"**Optimal Alpha:** {report.optimal_alpha}")
    lines.append(f"**Failed Queries:** {len(report.failed_queries)}")
    lines.append("")

    # Method comparison table
    lines.append("## Method Comparison")
    lines.append("")
    lines.append("| Method | MRR | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | Recall@20 |")
    lines.append("|--------|-----|--------|---------|----------|-----------|-----------|")

    for method, metrics in [
        ("Vector", report.vector_metrics),
        ("Keyword", report.keyword_metrics),
        ("Hybrid", report.hybrid_metrics),
        ("Graph", report.graph_metrics),
    ]:
        lines.append(
            f"| {method} | {metrics.mrr:.4f} | {metrics.ndcg_at_5:.4f} | "
            f"{metrics.ndcg_at_10:.4f} | {metrics.recall_at_5:.4f} | "
            f"{metrics.recall_at_10:.4f} | {metrics.recall_at_20:.4f} |"
        )
    lines.append("")

    # Alpha tuning
    lines.append("## Alpha Tuning Results")
    lines.append("")
    lines.append("| Alpha | MRR | Recall@5 | Recall@10 | NDCG@10 |")
    lines.append("|-------|-----|----------|-----------|---------|")

    for result in report.alpha_tuning:
        lines.append(
            f"| {result.alpha:.2f} | {result.mrr:.4f} | "
            f"{result.recall_at_5:.4f} | {result.recall_at_10:.4f} | "
            f"{result.ndcg_at_10:.4f} |"
        )
    lines.append("")

    # Failed queries
    if report.failed_queries:
        lines.append("## Failed Queries (No Expected Docs in Top-20)")
        lines.append("")
        for q in report.failed_queries:
            lines.append(f"### {q.question_id}")
            lines.append(f"**Query:** {q.question_text}")
            lines.append(f"**Expected:** {', '.join(q.expected_citations)}")
            lines.append(f"**Top-5 Retrieved:** {', '.join(q.retrieved_doc_ids[:5])}")
            lines.append("")

    # Per-query analysis
    lines.append("## Per-Query Analysis")
    lines.append("")
    lines.append("| Question | Best Method | MRR | Recall@10 | Expected Found |")
    lines.append("|----------|-------------|-----|-----------|----------------|")

    for comp in report.comparisons:
        found = sum(1 for r in comp.hybrid_result.found_expected.values() if r > 0)
        total = len(comp.hybrid_result.expected_citations)
        lines.append(
            f"| {comp.question_id} | {comp.best_method} | "
            f"{comp.best_mrr:.4f} | {comp.hybrid_result.recall_at_10:.4f} | "
            f"{found}/{total} |"
        )
    lines.append("")

    # Recommendations section
    lines.append("## Recommendations")
    lines.append("")

    # Determine recommendations based on analysis
    if report.optimal_alpha > 0.6:
        lines.append(
            f"1. **Increase semantic search weight:** Optimal alpha ({report.optimal_alpha}) "
            "suggests vector similarity is more effective than keyword matching."
        )
    elif report.optimal_alpha < 0.4:
        lines.append(
            f"1. **Increase keyword search weight:** Optimal alpha ({report.optimal_alpha}) "
            "suggests keyword/BM25 matching is more effective than vector similarity."
        )
    else:
        lines.append(
            f"1. **Balanced approach works best:** Optimal alpha ({report.optimal_alpha}) "
            "indicates both vector and keyword search contribute equally."
        )

    if report.graph_metrics.mrr > report.hybrid_metrics.mrr:
        improvement = (
            (report.graph_metrics.mrr - report.hybrid_metrics.mrr)
            / report.hybrid_metrics.mrr
            * 100
        )
        lines.append(
            f"2. **Enable graph expansion:** Graph-enhanced retrieval improves MRR by {improvement:.1f}%."
        )
    else:
        lines.append(
            "2. **Graph expansion has limited impact:** Consider disabling for faster retrieval."
        )

    if report.failed_queries:
        lines.append(
            f"3. **Address {len(report.failed_queries)} failing queries:** "
            "These queries return no relevant documents in top-20."
        )

    return "\n".join(lines)
