"""Interactive visualizations for RAG evaluation using Plotly.

Generates standalone HTML graphs for embedding in dashboards and reports.
"""

from __future__ import annotations

from typing import Any, Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .authority_metrics import AuthorityMetrics, AUTHORITY_HIERARCHY
from .correction_metrics import CorrectionMetrics


# Color schemes
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "info": "#17A2B8",
    "statute": "#2E86AB",
    "rule": "#28A745",
    "case": "#FFC107",
    "taa": "#A23B72",
}

AUTHORITY_COLORS = {
    "statute": COLORS["statute"],
    "rule": COLORS["rule"],
    "case": COLORS["case"],
    "taa": COLORS["taa"],
}


def create_authority_heatmap(
    authority_metrics: AuthorityMetrics,
    title: str = "Document Authority by Rank Position",
) -> go.Figure:
    """Create heatmap showing document type distribution across rank positions.

    Graph 1: Authority Heatmap
    X-Axis: Document Rank (1st, 2nd, 3rd, 4th, 5th)
    Y-Axis: Document Type (Statute, Rule, Case, TAA)
    Color: Count of documents at that position

    Args:
        authority_metrics: AuthorityMetrics with authority_by_rank data
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Build matrix
    doc_types = AUTHORITY_HIERARCHY
    ranks = list(range(1, 6))

    z = []
    for dt in doc_types:
        row = []
        for rank in ranks:
            count = authority_metrics.authority_by_rank.get(rank, {}).get(dt, 0)
            row.append(count)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{r}{'st' if r == 1 else 'nd' if r == 2 else 'rd' if r == 3 else 'th'}" for r in ranks],
        y=[dt.title() for dt in doc_types],
        colorscale="RdYlGn",
        text=z,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Rank: %{x}<br>Type: %{y}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Rank Position",
        yaxis_title="Document Type",
        height=400,
        margin=dict(l=100, r=50, t=80, b=50),
    )

    return fig


def create_correction_funnel(
    metrics: CorrectionMetrics,
    title: str = "Self-Correction Funnel",
) -> go.Figure:
    """Create funnel diagram showing correction process.

    Graph 2: Correction Funnel (Sankey-style)
    Shows: Total Queries → Issues Detected → Corrected → Final Output

    Args:
        metrics: CorrectionMetrics instance
        title: Chart title

    Returns:
        Plotly Figure
    """
    stages = [
        "Total Queries",
        "Clean (No Issues)",
        "Issues Detected",
        "Auto-Corrected",
        "Regenerated",
        "Final Hallucinations",
    ]

    clean = metrics.total_queries - metrics.queries_with_issues
    final_issues = metrics.queries_failed

    values = [
        metrics.total_queries,
        clean,
        metrics.queries_with_issues,
        metrics.queries_corrected,
        metrics.queries_regenerated,
        final_issues,
    ]

    colors = [
        COLORS["primary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["info"],
        COLORS["secondary"],
        COLORS["danger"] if final_issues > 0 else COLORS["success"],
    ]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=colors),
        connector=dict(line=dict(color="royalblue", width=2)),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=500,
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig


def create_precision_recall_curve(
    data: list[dict],
    title: str = "Precision vs Recall at Different K Values",
) -> go.Figure:
    """Create precision-recall curve for different retrieval methods.

    Graph 3: Precision-Recall Curve
    X-Axis: Recall@k
    Y-Axis: Precision@k
    Lines: Different retrieval methods

    Args:
        data: List of dicts with 'method', 'k_values', 'precision', 'recall'
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    method_colors = {
        "vector": COLORS["primary"],
        "keyword": COLORS["secondary"],
        "hybrid": COLORS["success"],
        "graph": COLORS["info"],
    }

    for method_data in data:
        method = method_data["method"]
        color = method_colors.get(method, COLORS["primary"])

        fig.add_trace(go.Scatter(
            x=method_data["recall"],
            y=method_data["precision"],
            mode="lines+markers",
            name=method.title(),
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate=(
                f"{method.title()}<br>"
                "Recall: %{x:.2f}<br>"
                "Precision: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Recall@k",
        yaxis_title="Precision@k",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        height=450,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=50),
    )

    return fig


def create_alpha_tuning_curve(
    alpha_values: list[float],
    mrr_values: list[float],
    optimal_alpha: Optional[float] = None,
    title: str = "Hybrid Search Alpha Tuning",
) -> go.Figure:
    """Create curve showing MRR vs alpha parameter.

    Graph 4: Alpha Tuning Curve
    X-Axis: Alpha (0.0 → 1.0)
    Y-Axis: MRR
    Annotations: Pure Keyword at 0, Pure Vector at 1

    Args:
        alpha_values: List of alpha values tested
        mrr_values: Corresponding MRR scores
        optimal_alpha: Optimal alpha value to highlight
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=alpha_values,
        y=mrr_values,
        mode="lines+markers",
        name="MRR",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10),
        fill="tozeroy",
        fillcolor="rgba(46, 134, 171, 0.2)",
    ))

    # Add optimal point annotation
    if optimal_alpha is not None:
        optimal_idx = alpha_values.index(optimal_alpha) if optimal_alpha in alpha_values else None
        if optimal_idx is not None:
            fig.add_annotation(
                x=optimal_alpha,
                y=mrr_values[optimal_idx],
                text=f"Optimal: α={optimal_alpha}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLORS["success"],
                font=dict(color=COLORS["success"], size=12),
                ax=40,
                ay=-40,
            )

    # Add endpoint annotations
    fig.add_annotation(
        x=0, y=mrr_values[0] if mrr_values else 0,
        text="Pure Keyword",
        showarrow=False,
        yshift=20,
        font=dict(size=10, color="gray"),
    )
    fig.add_annotation(
        x=1, y=mrr_values[-1] if mrr_values else 0,
        text="Pure Vector",
        showarrow=False,
        yshift=20,
        font=dict(size=10, color="gray"),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Alpha (0=Keyword, 1=Vector)",
        yaxis_title="Mean Reciprocal Rank (MRR)",
        xaxis=dict(range=[-0.05, 1.05]),
        height=400,
        margin=dict(l=60, r=50, t=80, b=50),
    )

    return fig


def create_latency_distribution(
    latencies: list[float],
    title: str = "Query Latency Distribution",
    show_percentiles: bool = True,
) -> go.Figure:
    """Create histogram of query latencies with percentile markers.

    Graph 5: Latency Distribution
    X-Axis: Latency (ms) bins
    Y-Axis: Count of queries
    Annotations: p50, p95, p99 lines

    Args:
        latencies: List of latency values in milliseconds
        title: Chart title
        show_percentiles: Whether to show percentile lines

    Returns:
        Plotly Figure
    """
    import numpy as np

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=latencies,
        nbinsx=30,
        name="Latency",
        marker_color=COLORS["primary"],
        opacity=0.75,
    ))

    if show_percentiles and latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        for val, name, color in [
            (p50, "p50", COLORS["success"]),
            (p95, "p95", COLORS["warning"]),
            (p99, "p99", COLORS["danger"]),
        ]:
            fig.add_vline(
                x=val,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{name}: {val:.0f}ms",
                annotation_position="top",
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        height=400,
        margin=dict(l=60, r=50, t=80, b=50),
        bargap=0.1,
    )

    return fig


def create_metrics_by_category(
    categories: list[str],
    metrics_data: dict[str, list[float]],
    title: str = "Metrics by Category",
) -> go.Figure:
    """Create grouped bar chart of metrics by category.

    Graph 6: Metrics by Category
    X-Axis: Categories (sales_tax, exemptions, corporate, procedures)
    Y-Axis: Score (0-1)
    Bars: Different metrics

    Args:
        categories: List of category names
        metrics_data: Dict mapping metric name to list of values per category
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    metric_colors = {
        "precision": COLORS["primary"],
        "recall": COLORS["secondary"],
        "faithfulness": COLORS["success"],
        "pass_rate": COLORS["info"],
    }

    for metric_name, values in metrics_data.items():
        color = metric_colors.get(metric_name, COLORS["primary"])
        fig.add_trace(go.Bar(
            name=metric_name.replace("_", " ").title(),
            x=categories,
            y=values,
            marker_color=color,
            text=[f"{v:.0%}" for v in values],
            textposition="auto",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Category",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        barmode="group",
        height=450,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=80),
    )

    return fig


def create_metrics_by_difficulty(
    results_by_difficulty: dict[str, dict[str, int]],
    title: str = "Results by Difficulty",
) -> go.Figure:
    """Create stacked bar chart showing pass/fail by difficulty.

    Graph 7: Metrics by Difficulty
    X-Axis: Difficulty (Easy, Medium, Hard)
    Y-Axis: Count
    Segments: Passed, Failed (no hallucination), Failed (hallucination)

    Args:
        results_by_difficulty: Dict mapping difficulty to outcome counts
        title: Chart title

    Returns:
        Plotly Figure
    """
    difficulties = ["easy", "medium", "hard"]

    passed = [results_by_difficulty.get(d, {}).get("passed", 0) for d in difficulties]
    failed_clean = [results_by_difficulty.get(d, {}).get("failed_clean", 0) for d in difficulties]
    failed_halluc = [results_by_difficulty.get(d, {}).get("failed_hallucination", 0) for d in difficulties]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Passed",
        x=[d.title() for d in difficulties],
        y=passed,
        marker_color=COLORS["success"],
    ))
    fig.add_trace(go.Bar(
        name="Failed (No Hallucination)",
        x=[d.title() for d in difficulties],
        y=failed_clean,
        marker_color=COLORS["warning"],
    ))
    fig.add_trace(go.Bar(
        name="Failed (Hallucination)",
        x=[d.title() for d in difficulties],
        y=failed_halluc,
        marker_color=COLORS["danger"],
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Difficulty",
        yaxis_title="Count",
        barmode="stack",
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=50),
    )

    return fig


def create_ablation_comparison(
    metrics: list[str],
    values_on: list[float],
    values_off: list[float],
    title: str = "Ablation: Validation ON vs OFF",
) -> go.Figure:
    """Create side-by-side bar chart comparing validation on/off.

    Graph 8: Ablation Comparison
    X-Axis: Metrics
    Y-Axis: Score/Count
    Bars: Validation ON vs OFF with delta annotations

    Args:
        metrics: List of metric names
        values_on: Values with validation enabled
        values_off: Values with validation disabled
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Validation OFF",
        x=metrics,
        y=values_off,
        marker_color=COLORS["danger"],
        text=[f"{v:.2f}" for v in values_off],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Validation ON",
        x=metrics,
        y=values_on,
        marker_color=COLORS["success"],
        text=[f"{v:.2f}" for v in values_on],
        textposition="auto",
    ))

    # Add delta annotations
    for i, (on, off, metric) in enumerate(zip(values_on, values_off, metrics)):
        delta = on - off
        if delta != 0:
            color = COLORS["success"] if delta > 0 else COLORS["danger"]
            sign = "+" if delta > 0 else ""
            fig.add_annotation(
                x=metric,
                y=max(on, off) + 0.05,
                text=f"{sign}{delta:.2f}",
                showarrow=False,
                font=dict(color=color, size=12, weight="bold"),
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        height=450,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=80),
    )

    return fig


def create_retrieval_radar(
    methods_data: dict[str, dict[str, float]],
    title: str = "Retrieval Method Comparison",
) -> go.Figure:
    """Create radar chart comparing retrieval methods.

    Graph 9: Retrieval Method Radar
    Axes: MRR, NDCG@10, Recall@10, Precision@5, Authority-NDCG
    Lines: Vector, Keyword, Hybrid, Graph

    Args:
        methods_data: Dict mapping method name to metric scores
        title: Chart title

    Returns:
        Plotly Figure
    """
    metrics = ["MRR", "NDCG@10", "Recall@10", "Precision@5", "Auth-NDCG"]

    fig = go.Figure()

    method_colors = {
        "vector": COLORS["primary"],
        "keyword": COLORS["secondary"],
        "hybrid": COLORS["success"],
        "graph": COLORS["info"],
    }

    for method, scores in methods_data.items():
        values = [scores.get(m.lower().replace("-", "_").replace("@", "_at_"), 0) for m in metrics]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=method.title(),
            line=dict(color=method_colors.get(method, COLORS["primary"]), width=2),
            fill="toself",
            fillcolor=method_colors.get(method, COLORS["primary"]).replace(")", ", 0.1)").replace("rgb", "rgba"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        height=500,
        margin=dict(l=80, r=80, t=80, b=50),
    )

    return fig


def create_faithfulness_distribution(
    scores: list[float],
    title: str = "Citation Faithfulness Distribution",
) -> go.Figure:
    """Create histogram of faithfulness scores.

    Graph 10: Faithfulness Distribution
    X-Axis: Faithfulness Score (0-1)
    Y-Axis: Count of claims
    Goal: Distribution heavily right-skewed (most >0.9)

    Args:
        scores: List of faithfulness scores (0-1)
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Color bins by quality
    colors = []
    for score in scores:
        if score >= 0.8:
            colors.append(COLORS["success"])
        elif score >= 0.5:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["danger"])

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color=COLORS["success"],
        opacity=0.75,
    ))

    # Add threshold lines
    fig.add_vline(
        x=0.8,
        line_dash="dash",
        line_color=COLORS["success"],
        annotation_text="Good (0.8)",
        annotation_position="top",
    )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text="Acceptable (0.5)",
        annotation_position="top",
    )

    # Calculate and show mean
    if scores:
        mean_score = sum(scores) / len(scores)
        fig.add_annotation(
            x=0.5,
            y=1,
            xref="paper",
            yref="paper",
            text=f"Mean: {mean_score:.2f}",
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Faithfulness Score",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1.05]),
        height=400,
        margin=dict(l=60, r=50, t=80, b=50),
    )

    return fig


def create_summary_metrics_card(
    metrics: dict[str, Any],
    title: str = "Evaluation Summary",
) -> go.Figure:
    """Create a summary card with key metrics.

    Args:
        metrics: Dict with key metric values
        title: Card title

    Returns:
        Plotly Figure with indicator gauges
    """
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        ],
        subplot_titles=[
            "Citation Precision", "Citation Recall", "Faithfulness",
            "Authority NDCG", "Pass Rate", "Hallucinations"
        ],
    )

    # Row 1
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics.get("precision", 0) * 100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["primary"]}},
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics.get("recall", 0) * 100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["secondary"]}},
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics.get("faithfulness", 0) * 100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["success"]}},
    ), row=1, col=3)

    # Row 2
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics.get("authority_ndcg", 0) * 100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["info"]}},
    ), row=2, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics.get("pass_rate", 0) * 100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["success"]}},
    ), row=2, col=2)

    halluc_count = metrics.get("hallucinations", 0)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=halluc_count,
        delta={"reference": 0, "increasing": {"color": COLORS["danger"]}},
        number={"font": {"color": COLORS["success"] if halluc_count == 0 else COLORS["danger"]}},
    ), row=2, col=3)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=500,
        margin=dict(l=30, r=30, t=100, b=30),
    )

    return fig


def create_baseline_comparison(
    margen_data: dict,
    baseline_data: dict,
    title: str = "Margen vs Frontier Models",
) -> go.Figure:
    """Create side-by-side bar chart comparing Margen to frontier models.

    Graph 11: Baseline Comparison
    X-Axis: Metrics (Hallucination Rate, Citation Precision, etc.)
    Bars: Margen, GPT-4, Claude

    Args:
        margen_data: Dict with Margen's metrics
        baseline_data: Dict with baseline model results
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Metrics to compare (displayed in inverted order for some)
    metrics = [
        ("Hallucination Rate", "hallucination_rate", False),
        ("Citation Precision", "avg_citation_precision", True),
        ("Pass Rate", "pass_rate", True),
    ]

    metric_labels = [m[0] for m in metrics]

    # Margen values
    margen_values = []
    for label, key, higher_is_better in metrics:
        val = margen_data.get(key, 0)
        if isinstance(val, float) and val <= 1:
            val = val * 100  # Convert to percentage
        margen_values.append(val)

    fig.add_trace(go.Bar(
        name="Margen",
        x=metric_labels,
        y=margen_values,
        marker_color=COLORS["success"],
        text=[f"{v:.0f}%" for v in margen_values],
        textposition="auto",
    ))

    # Baseline model colors
    baseline_colors = {
        "gpt4_vanilla": COLORS["primary"],
        "claude_vanilla": COLORS["secondary"],
    }

    baselines = baseline_data.get("baselines", {})

    for model_key, model_data in baselines.items():
        model_name = model_data.get("name", model_key.replace("_", " ").title())
        model_values = []

        for label, key, higher_is_better in metrics:
            val = model_data.get(key, 0)
            if isinstance(val, float) and val <= 1:
                val = val * 100
            model_values.append(val)

        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_labels,
            y=model_values,
            marker_color=baseline_colors.get(model_key, COLORS["info"]),
            text=[f"{v:.0f}%" for v in model_values],
            textposition="auto",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Metric",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 110]),
        barmode="group",
        height=450,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=50, t=80, b=80),
    )

    # Add annotations for key insights
    fig.add_annotation(
        text="Lower is better",
        x=0,
        y=-0.15,
        xref="x",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig.add_annotation(
        text="Higher is better",
        x=1,
        y=-0.15,
        xref="x",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


def create_latency_comparison(
    margen_latency: float,
    baseline_latencies: dict[str, float],
    title: str = "Latency Comparison",
) -> go.Figure:
    """Create bar chart comparing latencies across systems.

    Args:
        margen_latency: Margen's average latency in ms
        baseline_latencies: Dict mapping model name to latency
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    systems = ["Margen"]
    latencies = [margen_latency]
    colors = [COLORS["success"]]

    for model, latency in baseline_latencies.items():
        systems.append(model.replace("_", " ").title())
        latencies.append(latency)
        colors.append(COLORS["primary"] if "gpt" in model.lower() else COLORS["secondary"])

    fig.add_trace(go.Bar(
        x=systems,
        y=latencies,
        marker_color=colors,
        text=[f"{l:.0f}ms" for l in latencies],
        textposition="auto",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="System",
        yaxis_title="Average Latency (ms)",
        height=400,
        margin=dict(l=60, r=50, t=80, b=50),
    )

    return fig
