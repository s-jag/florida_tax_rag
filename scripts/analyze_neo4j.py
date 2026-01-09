#!/usr/bin/env python3
"""Analyze and optimize Neo4j query performance.

Usage:
    python scripts/analyze_neo4j.py [--explain] [--update-indexes]

Options:
    --explain          Run EXPLAIN on key queries
    --update-indexes   Add missing indexes from schema
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.client import Neo4jClient
from src.graph.schema import get_schema_queries

# Key queries to analyze
KEY_QUERIES = [
    {
        "name": "Get statute by section",
        "query": "MATCH (s:Statute {section: '212.05'}) RETURN s",
    },
    {
        "name": "Get interpretation chain",
        "query": """
            MATCH (s:Statute {section: '212.05'})
            OPTIONAL MATCH (r:Rule)-[:IMPLEMENTS|AUTHORITY]->(s)
            OPTIONAL MATCH (c:Case)-[:INTERPRETS|CITES]->(s)
            OPTIONAL MATCH (t:TAA)-[:INTERPRETS|CITES]->(s)
            RETURN s, collect(DISTINCT r) AS rules,
                   collect(DISTINCT c) AS cases,
                   collect(DISTINCT t) AS taas
        """,
    },
    {
        "name": "Get citing documents",
        "query": """
            MATCH (source:Document)-[r]->(target:Document {id: 'statute:212.05'})
            WHERE type(r) IN ['CITES', 'IMPLEMENTS', 'AUTHORITY', 'INTERPRETS']
            RETURN DISTINCT source
        """,
    },
    {
        "name": "Get document with chunks",
        "query": """
            MATCH (d:Document {id: 'statute:212.05'})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d, collect(c) AS chunks
        """,
    },
    {
        "name": "Get chunks by doc_id",
        "query": """
            MATCH (c:Chunk {doc_id: 'statute:212.05'})
            RETURN c
        """,
    },
]


def get_existing_indexes(client: Neo4jClient) -> set[str]:
    """Get names of existing indexes."""
    query = "SHOW INDEXES YIELD name RETURN name"
    results = client.run_query(query)
    return {r["name"] for r in results}


def run_explain(client: Neo4jClient, query: str, name: str) -> dict:
    """Run EXPLAIN on a query and parse the output."""
    explain_query = f"EXPLAIN {query}"

    try:
        results = client.run_query(explain_query)

        # The EXPLAIN output contains plan information
        return {
            "name": name,
            "status": "ok",
            "plan": str(results) if results else "No plan available",
        }
    except Exception as e:
        return {
            "name": name,
            "status": "error",
            "error": str(e),
        }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze and optimize Neo4j query performance")
    parser.add_argument("--explain", action="store_true", help="Run EXPLAIN on key queries")
    parser.add_argument("--update-indexes", action="store_true", help="Add missing indexes")
    args = parser.parse_args()

    print("=" * 60)
    print("NEO4J QUERY PERFORMANCE ANALYSIS")
    print("=" * 60)
    print()

    try:
        with Neo4jClient() as client:
            # Check connection
            if not client.health_check():
                print("ERROR: Cannot connect to Neo4j")
                return 1

            print("Connected to Neo4j")
            print()

            # Show existing indexes
            print("-" * 40)
            print("EXISTING INDEXES")
            print("-" * 40)

            existing = get_existing_indexes(client)
            for name in sorted(existing):
                print(f"  {name}")

            print()

            # Update indexes if requested
            if args.update_indexes:
                print("-" * 40)
                print("UPDATING INDEXES")
                print("-" * 40)

                for query in get_schema_queries():
                    if query.startswith("CREATE INDEX"):
                        try:
                            client.run_write(query)
                            # Extract index name
                            idx_name = query.split()[2]
                            print(f"  Created/verified: {idx_name}")
                        except Exception as e:
                            print(f"  Error: {e}")

                print()

            # Run EXPLAIN if requested
            if args.explain:
                print("-" * 40)
                print("QUERY EXPLAIN ANALYSIS")
                print("-" * 40)

                for q in KEY_QUERIES:
                    result = run_explain(client, q["query"], q["name"])
                    print(f"\n{result['name']}:")
                    if result["status"] == "ok":
                        print("  Status: OK")
                        # Print condensed plan info
                        plan = result["plan"]
                        if "NodeByLabelScan" in plan:
                            print("  WARNING: Uses NodeByLabelScan (table scan)")
                        if "NodeIndexSeek" in plan:
                            print("  Using: NodeIndexSeek (good)")
                        if "NodeUniqueIndexSeek" in plan:
                            print("  Using: NodeUniqueIndexSeek (optimal)")
                    else:
                        print(f"  ERROR: {result['error']}")

                print()

            # Show statistics
            print("-" * 40)
            print("DATABASE STATISTICS")
            print("-" * 40)

            node_counts = client.get_node_counts()
            print("Node counts:")
            for label, count in sorted(node_counts.items()):
                print(f"  {label}: {count}")

            edge_counts = client.get_edge_counts()
            print("\nRelationship counts:")
            for rel_type, count in sorted(edge_counts.items()):
                print(f"  {rel_type}: {count}")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print()
    print("=" * 60)
    print("Analysis complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
