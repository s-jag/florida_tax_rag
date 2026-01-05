#!/usr/bin/env python3
"""Visualize the Tax Agent graph structure.

Usage:
    python scripts/visualize_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import create_tax_agent_graph, get_graph_visualization


def main() -> int:
    """Main entry point."""
    print("=" * 70)
    print("TAX AGENT GRAPH STRUCTURE")
    print("=" * 70)

    # Create graph
    graph = create_tax_agent_graph()

    # Get visualization
    viz = get_graph_visualization()
    print("\nASCII Visualization:")
    print(viz)

    # Print node info
    graph_def = graph.get_graph()
    print("\nNodes:")
    for node_id in graph_def.nodes:
        if not node_id.startswith("__"):
            print(f"  - {node_id}")

    print("\nEdges:")
    for edge in graph_def.edges:
        source = edge.source if not edge.source.startswith("__") else "START"
        target = edge.target if not edge.target.startswith("__") else "END"
        print(f"  {source} -> {target}")

    print("\n" + "=" * 70)
    print("Graph created successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
