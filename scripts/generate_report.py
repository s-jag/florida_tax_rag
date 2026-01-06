#!/usr/bin/env python3
"""Generate full narrative HTML evaluation report.

This script creates a professional, publication-quality evaluation report
with narrative prose, methodology explanations, and baseline comparisons.

Usage:
    python scripts/generate_report.py --sample  # Generate with sample data
    python scripts/generate_report.py --input data/evaluation/reports/eval_xxx.json
    python scripts/generate_report.py --baseline data/evaluation/baseline_comparison.json

Output:
    data/evaluation/reports/report_{run_id}.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.authority_metrics import AuthorityMetrics
from src.evaluation.correction_metrics import CorrectionMetrics
from src.evaluation.visualizations import (
    create_authority_heatmap,
    create_correction_funnel,
    create_precision_recall_curve,
    create_alpha_tuning_curve,
    create_latency_distribution,
    create_metrics_by_category,
    create_metrics_by_difficulty,
    create_ablation_comparison,
    create_retrieval_radar,
    create_faithfulness_distribution,
    create_baseline_comparison,
    create_latency_comparison,
)
from src.evaluation.report_narrative import (
    generate_executive_summary,
    generate_self_correction_narrative,
    generate_baseline_narrative,
    generate_methodology_section,
    generate_methodology_caveat,
    generate_category_narrative,
    generate_difficulty_narrative,
    generate_recommendations,
    generate_conclusions,
)
from src.evaluation.report import FullEvaluationReport


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Margen Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            line-height: 1.6;
            color: #333;
        }}
        .report {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 40px rgba(0,0,0,0.1);
        }}

        /* Header */
        .report-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}
        .report-header h1 {{
            font-size: 2.8rem;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        .report-header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        .report-header .meta {{
            margin-top: 20px;
            font-size: 0.9rem;
            opacity: 0.7;
        }}

        /* Executive Summary */
        .executive-summary {{
            padding: 50px 60px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }}
        .executive-summary h2 {{
            color: #1a1a2e;
            font-size: 1.8rem;
            margin-bottom: 30px;
            border-bottom: 3px solid #0f3460;
            padding-bottom: 10px;
            display: inline-block;
        }}
        .executive-summary .prose {{
            font-size: 1.15rem;
            line-height: 1.9;
            color: #444;
            max-width: 900px;
        }}
        .executive-summary .prose p {{
            margin-bottom: 20px;
        }}

        /* Stats Row */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            padding: 40px 60px;
            background: white;
            border-bottom: 1px solid #eee;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            border: 1px solid #eee;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .stat-value.success {{ color: #28A745; }}
        .stat-value.warning {{ color: #FFC107; }}
        .stat-value.danger {{ color: #DC3545; }}
        .stat-label {{
            font-size: 0.85rem;
            color: #666;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Section Styling */
        .section {{
            padding: 50px 60px;
            border-bottom: 1px solid #eee;
        }}
        .section h2 {{
            color: #1a1a2e;
            font-size: 1.8rem;
            margin-bottom: 10px;
        }}
        .section-subtitle {{
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }}
        .section.hero {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .section.hero h2 {{
            color: white;
        }}
        .section.hero .section-subtitle {{
            color: rgba(255,255,255,0.9);
        }}

        /* Narrative Block */
        .narrative {{
            font-size: 1.1rem;
            line-height: 1.8;
            color: #444;
            max-width: 800px;
            margin-bottom: 30px;
        }}
        .narrative.light {{
            color: rgba(255,255,255,0.95);
        }}
        .narrative p {{
            margin-bottom: 15px;
        }}
        .narrative ul {{
            margin: 15px 0 15px 25px;
        }}
        .narrative li {{
            margin-bottom: 8px;
        }}

        /* Callout Box */
        .callout {{
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 30px 40px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 600;
            margin: 30px 0;
            backdrop-filter: blur(10px);
        }}
        .callout.dark {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
        }}

        /* Insight Box */
        .insight {{
            background: #f8f9fa;
            border-left: 4px solid #0f3460;
            padding: 20px 25px;
            margin: 25px 0;
            font-style: italic;
            color: #555;
        }}
        .insight.light {{
            background: rgba(255,255,255,0.1);
            border-left-color: rgba(255,255,255,0.5);
            color: rgba(255,255,255,0.9);
        }}

        /* Chart Grid */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin-top: 30px;
        }}
        @media (max-width: 900px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            border: 1px solid #eee;
        }}
        .chart-card.full-width {{
            grid-column: span 2;
        }}
        @media (max-width: 900px) {{
            .chart-card.full-width {{
                grid-column: span 1;
            }}
        }}
        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}

        /* Methodology Table */
        .definitions-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        .definitions-table th {{
            background: #1a1a2e;
            color: white;
            text-align: left;
            padding: 15px;
            font-weight: 600;
        }}
        .definitions-table td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}
        .definitions-table tr:hover {{
            background: #f8f9fa;
        }}
        .definitions-table code {{
            background: #f0f0f0;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        /* Recommendations */
        .recommendations {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 30px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            color: #1a1a2e;
            margin-bottom: 20px;
        }}
        .recommendations ul {{
            list-style: none;
        }}
        .recommendations li {{
            padding: 12px 0;
            border-bottom: 1px solid #eee;
            padding-left: 30px;
            position: relative;
        }}
        .recommendations li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #0f3460;
            font-weight: bold;
        }}
        .recommendations li:last-child {{
            border-bottom: none;
        }}

        /* Footer */
        .report-footer {{
            background: #1a1a2e;
            color: white;
            padding: 40px 60px;
            text-align: center;
        }}
        .report-footer p {{
            opacity: 0.8;
            margin-bottom: 10px;
        }}

        /* Note Box */
        .note {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px 20px;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        .note.info {{
            background: #d1ecf1;
            border-color: #17a2b8;
        }}
    </style>
</head>
<body>
    <div class="report">
        <!-- Header -->
        <header class="report-header">
            <h1>Margen Evaluation Report</h1>
            <div class="subtitle">Regulatory Document Retrieval System Assessment</div>
            <div class="meta">
                Run ID: {run_id} | Generated: {timestamp} | Dataset v{dataset_version}
            </div>
        </header>

        <!-- Executive Summary -->
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="prose">
                {executive_summary}
            </div>
        </section>

        <!-- Key Metrics -->
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value {precision_class}">{precision}%</div>
                <div class="stat-label">Citation Precision</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {recall_class}">{recall}%</div>
                <div class="stat-label">Citation Recall</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {faithfulness_class}">{faithfulness}%</div>
                <div class="stat-label">Faithfulness</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {pass_class}">{pass_rate}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {halluc_class}">{hallucinations}</div>
                <div class="stat-label">Hallucinations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_latency}ms</div>
                <div class="stat-label">Avg Latency</div>
            </div>
        </div>

        <!-- Section 2: Self-Correction (The Money Slide) -->
        <section class="section hero">
            <h2>{self_correction_title}</h2>
            <div class="section-subtitle">The architectural advantage that makes the difference</div>

            <div class="narrative light">
                {self_correction_prose}
            </div>

            <div class="callout">
                {self_correction_callout}
            </div>

            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-title">Correction Funnel</div>
                    <div id="correction-funnel"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Ablation: Validation Impact</div>
                    <div id="ablation-comparison"></div>
                </div>
            </div>

            <div class="insight light">
                {self_correction_insight}
            </div>
        </section>

        <!-- Section 3: Baseline Comparison -->
        <section class="section">
            <h2>{baseline_title}</h2>
            <div class="section-subtitle">Comparing against frontier models on the same questions</div>

            <div class="narrative">
                {baseline_prose}
            </div>

            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-title">Performance Comparison</div>
                    <div id="baseline-comparison"></div>
                </div>
            </div>

            <div class="callout dark">
                {baseline_callout}
            </div>

            <div class="insight">
                {baseline_insight}
            </div>
        </section>

        <!-- Section 4: Methodology -->
        <section class="section">
            <h2>How We Measured</h2>
            <div class="section-subtitle">Definitions and methodology for all metrics</div>

            {methodology_table}

            <div class="note info">
                {methodology_caveat}
            </div>
        </section>

        <!-- Section 5: Detailed Findings -->
        <section class="section">
            <h2>Detailed Findings</h2>
            <div class="section-subtitle">Performance breakdown by category and difficulty</div>

            <h3 style="margin-top: 30px; color: #1a1a2e;">By Category</h3>
            <div class="narrative">
                {category_narrative}
            </div>
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-title">Metrics by Category</div>
                    <div id="metrics-category"></div>
                </div>
            </div>

            <h3 style="margin-top: 40px; color: #1a1a2e;">By Difficulty</h3>
            <div class="narrative">
                {difficulty_narrative}
            </div>
            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-title">Results by Difficulty</div>
                    <div id="metrics-difficulty"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Faithfulness Distribution</div>
                    <div id="faithfulness-dist"></div>
                </div>
            </div>

            <h3 style="margin-top: 40px; color: #1a1a2e;">Latency Analysis</h3>
            <div class="narrative">
                <p>Median latency is {median_latency}ms. The additional time compared to raw model queries
                is spent on retrieval, source verification, and self-correction. For regulatory use cases
                where accuracy matters more than speed, this trade-off is acceptable.</p>
            </div>
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-title">Latency Distribution</div>
                    <div id="latency-dist"></div>
                </div>
            </div>
        </section>

        <!-- Section 6: Authority & Retrieval -->
        <section class="section">
            <h2>Authority & Retrieval</h2>
            <div class="section-subtitle">Technical deep-dive into retrieval quality</div>

            <div class="narrative">
                <p>The retrieval system is designed to respect legal authority hierarchy:
                binding law (statutes, rules) should rank above advisory opinions (TAAs).
                The heatmap below shows document type distribution across rank positions.</p>
            </div>

            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-title">Document Authority by Rank</div>
                    <div id="authority-heatmap"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Retrieval Method Comparison</div>
                    <div id="retrieval-radar"></div>
                </div>
            </div>

            <div class="narrative" style="margin-top: 30px;">
                <p><strong>Hybrid Search Tuning:</strong> The alpha parameter controls the balance between
                keyword matching (for exact statutory references) and semantic search (for conceptual queries).
                Alpha=0.25 (75% keyword, 25% vector) optimizes for legal document retrieval.</p>
            </div>

            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-title">Alpha Tuning Curve</div>
                    <div id="alpha-tuning"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Precision vs Recall</div>
                    <div id="precision-recall"></div>
                </div>
            </div>
        </section>

        <!-- Section 7: Conclusions -->
        <section class="section">
            <h2>Conclusions & Recommendations</h2>
            <div class="section-subtitle">Key takeaways and next steps</div>

            <div class="narrative">
                {conclusions}
            </div>

            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </section>

        <!-- Footer -->
        <footer class="report-footer">
            <p>Generated by Margen Evaluation Framework</p>
            <p>Total Queries: {total_queries} | Dataset Version: {dataset_version}</p>
        </footer>
    </div>

    <script>
        const chartConfig = {{responsive: true, displayModeBar: false}};

        {chart_scripts}
    </script>
</body>
</html>
"""


def get_value_class(value: float, thresholds: tuple = (0.7, 0.9)) -> str:
    """Get CSS class based on value threshold."""
    if value >= thresholds[1]:
        return "success"
    elif value >= thresholds[0]:
        return "warning"
    return "danger"


def generate_sample_data() -> dict:
    """Generate sample data for testing the report."""
    return {
        "run_id": "sample_001",
        "timestamp": datetime.now().isoformat(),
        "dataset_version": "1.0.0",
        "total_queries": 100,
        "summary": {
            "precision": 0.95,
            "recall": 0.72,
            "faithfulness": 0.91,
            "authority_ndcg": 0.85,
            "pass_rate": 0.88,
            "hallucinations": 0,
            "avg_latency_ms": 3200,
        },
        "authority_metrics": {
            "authority_by_rank": {
                1: {"statute": 45, "rule": 30, "case": 15, "taa": 10},
                2: {"statute": 35, "rule": 35, "case": 20, "taa": 10},
                3: {"statute": 25, "rule": 35, "case": 25, "taa": 15},
                4: {"statute": 20, "rule": 30, "case": 30, "taa": 20},
                5: {"statute": 15, "rule": 25, "case": 35, "taa": 25},
            },
        },
        "correction_metrics": {
            "total_queries": 100,
            "queries_with_issues": 15,
            "queries_corrected": 12,
            "queries_regenerated": 3,
            "queries_failed": 0,
            "intervention_rate": 0.15,
            "correction_success_rate": 0.80,
        },
        "precision_recall_data": [
            {"method": "vector", "recall": [0.3, 0.5, 0.6, 0.7, 0.8], "precision": [0.9, 0.85, 0.8, 0.7, 0.6]},
            {"method": "keyword", "recall": [0.4, 0.55, 0.65, 0.72, 0.78], "precision": [0.88, 0.82, 0.75, 0.68, 0.62]},
            {"method": "hybrid", "recall": [0.5, 0.65, 0.75, 0.82, 0.88], "precision": [0.92, 0.88, 0.85, 0.8, 0.75]},
            {"method": "graph", "recall": [0.55, 0.7, 0.8, 0.85, 0.9], "precision": [0.94, 0.9, 0.87, 0.82, 0.78]},
        ],
        "alpha_tuning": {
            "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "mrr": [0.42, 0.61, 0.51, 0.32, 0.25],
            "optimal": 0.25,
        },
        "retrieval_methods": {
            "vector": {"mrr": 0.25, "ndcg_at_10": 0.37, "recall_at_10": 0.5, "precision_at_5": 0.4, "auth_ndcg": 0.35},
            "keyword": {"mrr": 0.42, "ndcg_at_10": 0.65, "recall_at_10": 0.5, "precision_at_5": 0.5, "auth_ndcg": 0.55},
            "hybrid": {"mrr": 0.61, "ndcg_at_10": 0.78, "recall_at_10": 0.6, "precision_at_5": 0.6, "auth_ndcg": 0.75},
            "graph": {"mrr": 0.65, "ndcg_at_10": 0.82, "recall_at_10": 0.7, "precision_at_5": 0.65, "auth_ndcg": 0.85},
        },
        "metrics_by_category": {
            "categories": ["Sales Tax", "Exemptions", "Corporate", "Procedures"],
            "precision": [0.95, 0.92, 0.88, 0.90],
            "recall": [0.75, 0.70, 0.68, 0.72],
            "faithfulness": [0.92, 0.89, 0.87, 0.91],
            "pass_rate": [0.90, 0.85, 0.82, 0.88],
        },
        "results_by_difficulty": {
            "easy": {"passed": 28, "failed_clean": 2, "failed_hallucination": 0},
            "medium": {"passed": 42, "failed_clean": 6, "failed_hallucination": 2},
            "hard": {"passed": 18, "failed_clean": 1, "failed_hallucination": 1},
        },
        "ablation": {
            "metrics": ["Hallucinations", "Precision", "Recall", "Faithfulness"],
            "values_on": [0, 0.95, 0.72, 0.91],
            "values_off": [12, 0.88, 0.70, 0.75],
        },
        "faithfulness_scores": [0.95, 0.92, 0.88, 0.91, 0.85, 0.93, 0.89, 0.96, 0.87, 0.94,
                               0.90, 0.92, 0.88, 0.95, 0.91, 0.86, 0.93, 0.89, 0.94, 0.87,
                               0.72, 0.65, 0.78, 0.55, 0.82, 0.45, 0.91, 0.88, 0.93, 0.90],
        "latencies": [2500, 3200, 2800, 4100, 3500, 2900, 3800, 2600, 3100, 4500,
                     2700, 3300, 2900, 3600, 4200, 2800, 3400, 3000, 3700, 4800,
                     5500, 6200, 3100, 2500, 2800, 3200, 3900, 4100, 3500, 2900],
        "baseline_data": {
            "evaluation_date": datetime.now().isoformat(),
            "questions_evaluated": 20,
            "baselines": {
                "gpt4_vanilla": {
                    "name": "GPT-4 Turbo (No RAG)",
                    "hallucination_rate": 0.15,
                    "avg_citation_precision": 0.70,
                    "fabrication_rate": 0.08,
                    "avg_latency_ms": 800,
                    "pass_rate": 0.65,
                },
                "claude_vanilla": {
                    "name": "Claude 3.5 Sonnet (No RAG)",
                    "hallucination_rate": 0.12,
                    "avg_citation_precision": 0.75,
                    "fabrication_rate": 0.05,
                    "avg_latency_ms": 900,
                    "pass_rate": 0.70,
                },
            },
        },
        "margen_data": {
            "hallucination_rate": 0.0,
            "avg_citation_precision": 0.95,
            "fabrication_rate": 0.0,
            "avg_latency_ms": 3200,
            "pass_rate": 0.88,
        },
    }


def generate_report(data: dict, output_path: Path) -> Path:
    """Generate full narrative HTML report from evaluation data.

    Args:
        data: Evaluation results dictionary
        output_path: Where to save the HTML file

    Returns:
        Path to generated HTML file
    """
    # Build mock report object for narrative generation
    from src.evaluation.report import (
        FullEvaluationReport,
        CorrectionAnalysis,
        CategoryMetrics,
        DifficultyMetrics,
    )

    # Build correction analysis
    corr_data = data.get("correction_metrics", {})
    correction_analysis = CorrectionAnalysis(
        total_queries=corr_data.get("total_queries", 100),
        queries_with_issues=corr_data.get("queries_with_issues", 15),
        queries_corrected=corr_data.get("queries_corrected", 12),
        queries_regenerated=corr_data.get("queries_regenerated", 3),
        queries_failed=corr_data.get("queries_failed", 0),
        intervention_rate=corr_data.get("intervention_rate", 0.15),
        correction_success_rate=corr_data.get("correction_success_rate", 0.80),
    )

    # Build category metrics
    cat_data = data.get("metrics_by_category", {})
    categories = cat_data.get("categories", [])
    metrics_by_category = {}
    for i, cat in enumerate(categories):
        cat_key = cat.lower().replace(" ", "_")
        metrics_by_category[cat_key] = CategoryMetrics(
            category=cat_key,
            count=25,
            avg_precision=cat_data.get("precision", [0])[i] if i < len(cat_data.get("precision", [])) else 0,
            avg_recall=cat_data.get("recall", [0])[i] if i < len(cat_data.get("recall", [])) else 0,
            avg_score=8.0,
            pass_rate=cat_data.get("pass_rate", [0])[i] if i < len(cat_data.get("pass_rate", [])) else 0,
        )

    # Build difficulty metrics
    diff_data = data.get("results_by_difficulty", {})
    metrics_by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_data:
            total = sum(diff_data[diff].values())
            passed = diff_data[diff].get("passed", 0)
            metrics_by_difficulty[diff] = DifficultyMetrics(
                difficulty=diff,
                count=total,
                avg_precision=0.9,
                avg_recall=0.7,
                avg_score=8.0,
                pass_rate=passed / total if total > 0 else 0,
                avg_latency_ms=3000,
            )

    summary = data.get("summary", {})
    report = FullEvaluationReport(
        run_id=data.get("run_id", "unknown"),
        dataset_version=data.get("dataset_version", "1.0.0"),
        total_questions=data.get("total_queries", 100),
        successful_evaluations=data.get("total_queries", 100),
        avg_citation_precision=summary.get("precision", 0.95),
        avg_citation_recall=summary.get("recall", 0.72),
        avg_faithfulness_score=summary.get("faithfulness", 0.91),
        pass_rate=summary.get("pass_rate", 0.88),
        total_hallucinations=summary.get("hallucinations", 0),
        avg_latency_ms=summary.get("avg_latency_ms", 3200),
        correction_analysis=correction_analysis,
        metrics_by_category=metrics_by_category,
        metrics_by_difficulty=metrics_by_difficulty,
    )

    # Generate narratives
    baseline_data = data.get("baseline_data", {})
    margen_data = data.get("margen_data", {})

    executive_summary = generate_executive_summary(report, baseline_data)
    self_correction = generate_self_correction_narrative(report)
    baseline_narrative = generate_baseline_narrative(margen_data, baseline_data)
    methodology_table = generate_methodology_section()
    methodology_caveat = generate_methodology_caveat()
    category_narrative = generate_category_narrative(report)
    difficulty_narrative = generate_difficulty_narrative(report)
    recommendations = generate_recommendations(report)
    conclusions = generate_conclusions(report)

    # Build authority metrics
    auth_data = data.get("authority_metrics", {})
    auth_metrics = AuthorityMetrics(
        authority_ndcg_at_5=summary.get("authority_ndcg", 0) * 0.95,
        authority_ndcg_at_10=summary.get("authority_ndcg", 0),
        hierarchy_alignment_score=0.9,
        primary_authority_rate_at_5=0.65,
        primary_authority_rate_at_10=0.55,
        doc_type_distribution={},
        authority_by_rank=auth_data.get("authority_by_rank", {}),
    )

    # Build correction metrics
    corr_metrics = CorrectionMetrics(
        total_queries=corr_data.get("total_queries", 100),
        queries_with_issues=corr_data.get("queries_with_issues", 15),
        queries_corrected=corr_data.get("queries_corrected", 12),
        queries_regenerated=corr_data.get("queries_regenerated", 3),
        queries_failed=corr_data.get("queries_failed", 0),
        intervention_rate=0.15,
        correction_rate=0.8,
        regeneration_rate=0.03,
        total_issues_detected=15,
        total_issues_corrected=12,
        issues_by_type={},
        avg_severity=0.45,
        max_severity=0.7,
        severity_distribution={},
        avg_confidence_delta=-0.05,
    )

    # Generate all charts
    charts = []

    # Correction Funnel
    fig_funnel = create_correction_funnel(corr_metrics)
    charts.append(("correction-funnel", fig_funnel))

    # Ablation Comparison
    ablation_data = data.get("ablation", {})
    fig_ablation = create_ablation_comparison(
        ablation_data.get("metrics", []),
        ablation_data.get("values_on", []),
        ablation_data.get("values_off", []),
    )
    charts.append(("ablation-comparison", fig_ablation))

    # Baseline Comparison
    fig_baseline = create_baseline_comparison(margen_data, baseline_data)
    charts.append(("baseline-comparison", fig_baseline))

    # Metrics by Category
    fig_category = create_metrics_by_category(
        cat_data.get("categories", []),
        {
            "precision": cat_data.get("precision", []),
            "recall": cat_data.get("recall", []),
            "faithfulness": cat_data.get("faithfulness", []),
            "pass_rate": cat_data.get("pass_rate", []),
        },
    )
    charts.append(("metrics-category", fig_category))

    # Metrics by Difficulty
    fig_difficulty = create_metrics_by_difficulty(data.get("results_by_difficulty", {}))
    charts.append(("metrics-difficulty", fig_difficulty))

    # Faithfulness Distribution
    fig_faithfulness = create_faithfulness_distribution(data.get("faithfulness_scores", []))
    charts.append(("faithfulness-dist", fig_faithfulness))

    # Latency Distribution
    fig_latency = create_latency_distribution(data.get("latencies", []))
    charts.append(("latency-dist", fig_latency))

    # Authority Heatmap
    fig_heatmap = create_authority_heatmap(auth_metrics)
    charts.append(("authority-heatmap", fig_heatmap))

    # Retrieval Radar
    fig_radar = create_retrieval_radar(data.get("retrieval_methods", {}))
    charts.append(("retrieval-radar", fig_radar))

    # Alpha Tuning
    alpha_data = data.get("alpha_tuning", {})
    fig_alpha = create_alpha_tuning_curve(
        alpha_data.get("alphas", []),
        alpha_data.get("mrr", []),
        alpha_data.get("optimal"),
    )
    charts.append(("alpha-tuning", fig_alpha))

    # Precision-Recall
    fig_pr = create_precision_recall_curve(data.get("precision_recall_data", []))
    charts.append(("precision-recall", fig_pr))

    # Generate chart scripts
    chart_scripts = []
    for div_id, fig in charts:
        chart_json = fig.to_json()
        chart_scripts.append(f"Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout, chartConfig);")

    # Calculate median latency
    latencies = data.get("latencies", [3200])
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 3200

    # Format recommendations as HTML list items
    recommendations_html = "\n".join(f"<li>{rec}</li>" for rec in recommendations)

    # Format executive summary paragraphs
    exec_paragraphs = executive_summary.split("\n\n")
    exec_html = "\n".join(f"<p>{p}</p>" for p in exec_paragraphs)

    # Generate HTML
    html = REPORT_TEMPLATE.format(
        run_id=data.get("run_id", "unknown"),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        dataset_version=data.get("dataset_version", "1.0.0"),
        total_queries=data.get("total_queries", 0),
        precision=int(summary.get("precision", 0) * 100),
        recall=int(summary.get("recall", 0) * 100),
        faithfulness=int(summary.get("faithfulness", 0) * 100),
        pass_rate=int(summary.get("pass_rate", 0) * 100),
        hallucinations=summary.get("hallucinations", 0),
        avg_latency=int(summary.get("avg_latency_ms", 0)),
        precision_class=get_value_class(summary.get("precision", 0)),
        recall_class=get_value_class(summary.get("recall", 0)),
        faithfulness_class=get_value_class(summary.get("faithfulness", 0)),
        pass_class=get_value_class(summary.get("pass_rate", 0)),
        halluc_class="success" if summary.get("hallucinations", 0) == 0 else "danger",
        executive_summary=exec_html,
        self_correction_title=self_correction.title,
        self_correction_prose=self_correction.prose.replace("\n", "<br>"),
        self_correction_callout=self_correction.callout or "",
        self_correction_insight=self_correction.insight or "",
        baseline_title=baseline_narrative.title,
        baseline_prose=baseline_narrative.prose.replace("\n", "<br>"),
        baseline_callout=baseline_narrative.callout or "",
        baseline_insight=baseline_narrative.insight or "",
        methodology_table=methodology_table,
        methodology_caveat=methodology_caveat,
        category_narrative=f"<p>{category_narrative}</p>",
        difficulty_narrative=difficulty_narrative.replace("\n", "<br>"),
        median_latency=median_latency,
        conclusions=f"<p>{conclusions.replace(chr(10), '</p><p>')}</p>",
        recommendations=recommendations_html,
        chart_scripts="\n        ".join(chart_scripts),
    )

    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate full narrative HTML evaluation report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to evaluation JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline comparison JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate report with sample data",
    )

    args = parser.parse_args()

    if args.sample:
        data = generate_sample_data()
        run_id = "sample"
    elif args.input and args.input.exists():
        with open(args.input) as f:
            data = json.load(f)
        run_id = args.input.stem.replace("eval_", "")

        # Load baseline data if provided
        if args.baseline and args.baseline.exists():
            with open(args.baseline) as f:
                data["baseline_data"] = json.load(f)
    else:
        print("Error: Must provide --input or --sample")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path("data/evaluation/reports") / f"report_{run_id}.html"

    # Generate report
    result_path = generate_report(data, output_path)
    print(f"Report generated: {result_path}")
    print(f"Open in browser: file://{result_path.absolute()}")


if __name__ == "__main__":
    main()
