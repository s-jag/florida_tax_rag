#!/usr/bin/env python3
"""Generate interactive HTML dashboard from evaluation results.

Usage:
    python scripts/generate_dashboard.py --input data/evaluation/reports/eval_xxx.json
    python scripts/generate_dashboard.py --sample  # Generate with sample data

Output:
    data/evaluation/reports/dashboard_{run_id}.html
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
    create_ablation_comparison,
    create_alpha_tuning_curve,
    create_authority_heatmap,
    create_correction_funnel,
    create_faithfulness_distribution,
    create_latency_distribution,
    create_metrics_by_category,
    create_metrics_by_difficulty,
    create_precision_recall_curve,
    create_retrieval_radar,
)

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Margen Evaluation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            padding: 30px 0;
        }}
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2E86AB;
        }}
        .stat-value.success {{ color: #28A745; }}
        .stat-value.warning {{ color: #FFC107; }}
        .stat-value.danger {{ color: #DC3545; }}
        .stat-label {{
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        @media (max-width: 1000px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .chart-card.wide {{
            grid-column: span 2;
        }}
        @media (max-width: 1000px) {{
            .chart-card.wide {{
                grid-column: span 1;
            }}
        }}
        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .footer {{
            text-align: center;
            color: white;
            padding: 30px 0;
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        .section-header {{
            color: white;
            font-size: 1.5rem;
            margin: 40px 0 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Margen Evaluation Dashboard</h1>
            <div class="subtitle">Run ID: {run_id} | Generated: {timestamp}</div>
        </div>

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
                <div class="stat-value {authority_class}">{authority_ndcg}%</div>
                <div class="stat-label">Authority NDCG</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {pass_class}">{pass_rate}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {halluc_class}">{hallucinations}</div>
                <div class="stat-label">Hallucinations</div>
            </div>
        </div>

        <h2 class="section-header">Authority & Retrieval Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-title">Document Authority by Rank</div>
                <div id="authority-heatmap"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Retrieval Method Comparison</div>
                <div id="retrieval-radar"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Precision vs Recall</div>
                <div id="precision-recall"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Alpha Tuning</div>
                <div id="alpha-tuning"></div>
            </div>
        </div>

        <h2 class="section-header">Self-Correction Analysis</h2>
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

        <h2 class="section-header">Quality Breakdown</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-title">Metrics by Category</div>
                <div id="metrics-category"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Results by Difficulty</div>
                <div id="metrics-difficulty"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Faithfulness Distribution</div>
                <div id="faithfulness-dist"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Latency Distribution</div>
                <div id="latency-dist"></div>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Margen Evaluation Framework</p>
            <p>Total Queries: {total_queries} | Dataset Version: {dataset_version}</p>
        </div>
    </div>

    <script>
        // Chart configurations
        const chartConfig = {{responsive: true, displayModeBar: false}};

        // Render all charts
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
    """Generate sample data for testing the dashboard."""
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
        },
        "precision_recall_data": [
            {
                "method": "vector",
                "recall": [0.3, 0.5, 0.6, 0.7, 0.8],
                "precision": [0.9, 0.85, 0.8, 0.7, 0.6],
            },
            {
                "method": "keyword",
                "recall": [0.4, 0.55, 0.65, 0.72, 0.78],
                "precision": [0.88, 0.82, 0.75, 0.68, 0.62],
            },
            {
                "method": "hybrid",
                "recall": [0.5, 0.65, 0.75, 0.82, 0.88],
                "precision": [0.92, 0.88, 0.85, 0.8, 0.75],
            },
            {
                "method": "graph",
                "recall": [0.55, 0.7, 0.8, 0.85, 0.9],
                "precision": [0.94, 0.9, 0.87, 0.82, 0.78],
            },
        ],
        "alpha_tuning": {
            "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "mrr": [0.42, 0.61, 0.51, 0.32, 0.25],
            "optimal": 0.25,
        },
        "retrieval_methods": {
            "vector": {
                "mrr": 0.25,
                "ndcg_at_10": 0.37,
                "recall_at_10": 0.5,
                "precision_at_5": 0.4,
                "auth_ndcg": 0.35,
            },
            "keyword": {
                "mrr": 0.42,
                "ndcg_at_10": 0.65,
                "recall_at_10": 0.5,
                "precision_at_5": 0.5,
                "auth_ndcg": 0.55,
            },
            "hybrid": {
                "mrr": 0.61,
                "ndcg_at_10": 0.78,
                "recall_at_10": 0.6,
                "precision_at_5": 0.6,
                "auth_ndcg": 0.75,
            },
            "graph": {
                "mrr": 0.65,
                "ndcg_at_10": 0.82,
                "recall_at_10": 0.7,
                "precision_at_5": 0.65,
                "auth_ndcg": 0.85,
            },
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
        "faithfulness_scores": [
            0.95,
            0.92,
            0.88,
            0.91,
            0.85,
            0.93,
            0.89,
            0.96,
            0.87,
            0.94,
            0.90,
            0.92,
            0.88,
            0.95,
            0.91,
            0.86,
            0.93,
            0.89,
            0.94,
            0.87,
            0.72,
            0.65,
            0.78,
            0.55,
            0.82,
            0.45,
            0.91,
            0.88,
            0.93,
            0.90,
        ],
        "latencies": [
            2500,
            3200,
            2800,
            4100,
            3500,
            2900,
            3800,
            2600,
            3100,
            4500,
            2700,
            3300,
            2900,
            3600,
            4200,
            2800,
            3400,
            3000,
            3700,
            4800,
            5500,
            6200,
            3100,
            2500,
            2800,
            3200,
            3900,
            4100,
            3500,
            2900,
        ],
    }


def generate_dashboard(data: dict, output_path: Path) -> Path:
    """Generate HTML dashboard from evaluation data.

    Args:
        data: Evaluation results dictionary
        output_path: Where to save the HTML file

    Returns:
        Path to generated HTML file
    """
    # Build authority metrics
    auth_data = data.get("authority_metrics", {})
    auth_metrics = AuthorityMetrics(
        authority_ndcg_at_5=data.get("summary", {}).get("authority_ndcg", 0) * 0.95,
        authority_ndcg_at_10=data.get("summary", {}).get("authority_ndcg", 0),
        hierarchy_alignment_score=0.9,
        primary_authority_rate_at_5=0.65,
        primary_authority_rate_at_10=0.55,
        doc_type_distribution={},
        authority_by_rank=auth_data.get("authority_by_rank", {}),
    )

    # Build correction metrics
    corr_data = data.get("correction_metrics", {})
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

    # 1. Authority Heatmap
    fig1 = create_authority_heatmap(auth_metrics)
    charts.append(("authority-heatmap", fig1))

    # 2. Correction Funnel
    fig2 = create_correction_funnel(corr_metrics)
    charts.append(("correction-funnel", fig2))

    # 3. Precision-Recall Curve
    fig3 = create_precision_recall_curve(data.get("precision_recall_data", []))
    charts.append(("precision-recall", fig3))

    # 4. Alpha Tuning
    alpha_data = data.get("alpha_tuning", {})
    fig4 = create_alpha_tuning_curve(
        alpha_data.get("alphas", []),
        alpha_data.get("mrr", []),
        alpha_data.get("optimal"),
    )
    charts.append(("alpha-tuning", fig4))

    # 5. Latency Distribution
    fig5 = create_latency_distribution(data.get("latencies", []))
    charts.append(("latency-dist", fig5))

    # 6. Metrics by Category
    cat_data = data.get("metrics_by_category", {})
    fig6 = create_metrics_by_category(
        cat_data.get("categories", []),
        {
            "precision": cat_data.get("precision", []),
            "recall": cat_data.get("recall", []),
            "faithfulness": cat_data.get("faithfulness", []),
            "pass_rate": cat_data.get("pass_rate", []),
        },
    )
    charts.append(("metrics-category", fig6))

    # 7. Metrics by Difficulty
    fig7 = create_metrics_by_difficulty(data.get("results_by_difficulty", {}))
    charts.append(("metrics-difficulty", fig7))

    # 8. Ablation Comparison
    ablation_data = data.get("ablation", {})
    fig8 = create_ablation_comparison(
        ablation_data.get("metrics", []),
        ablation_data.get("values_on", []),
        ablation_data.get("values_off", []),
    )
    charts.append(("ablation-comparison", fig8))

    # 9. Retrieval Radar
    fig9 = create_retrieval_radar(data.get("retrieval_methods", {}))
    charts.append(("retrieval-radar", fig9))

    # 10. Faithfulness Distribution
    fig10 = create_faithfulness_distribution(data.get("faithfulness_scores", []))
    charts.append(("faithfulness-dist", fig10))

    # Generate chart scripts
    chart_scripts = []
    for div_id, fig in charts:
        chart_json = fig.to_json()
        chart_scripts.append(
            f"Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout, chartConfig);"
        )

    # Get summary values
    summary = data.get("summary", {})

    # Generate HTML
    html = DASHBOARD_TEMPLATE.format(
        run_id=data.get("run_id", "unknown"),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        dataset_version=data.get("dataset_version", "1.0.0"),
        total_queries=data.get("total_queries", 0),
        precision=int(summary.get("precision", 0) * 100),
        recall=int(summary.get("recall", 0) * 100),
        faithfulness=int(summary.get("faithfulness", 0) * 100),
        authority_ndcg=int(summary.get("authority_ndcg", 0) * 100),
        pass_rate=int(summary.get("pass_rate", 0) * 100),
        hallucinations=summary.get("hallucinations", 0),
        precision_class=get_value_class(summary.get("precision", 0)),
        recall_class=get_value_class(summary.get("recall", 0)),
        faithfulness_class=get_value_class(summary.get("faithfulness", 0)),
        authority_class=get_value_class(summary.get("authority_ndcg", 0)),
        pass_class=get_value_class(summary.get("pass_rate", 0)),
        halluc_class="success" if summary.get("hallucinations", 0) == 0 else "danger",
        chart_scripts="\n        ".join(chart_scripts),
    )

    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML dashboard from evaluation results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to evaluation JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate dashboard with sample data",
    )

    args = parser.parse_args()

    if args.sample:
        data = generate_sample_data()
        run_id = "sample"
    elif args.input and args.input.exists():
        with open(args.input) as f:
            data = json.load(f)
        run_id = args.input.stem.replace("eval_", "")
    else:
        print("Error: Must provide --input or --sample")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path("data/evaluation/reports") / f"dashboard_{run_id}.html"

    # Generate dashboard
    result_path = generate_dashboard(data, output_path)
    print(f"Dashboard generated: {result_path}")
    print(f"Open in browser: file://{result_path.absolute()}")


if __name__ == "__main__":
    main()
