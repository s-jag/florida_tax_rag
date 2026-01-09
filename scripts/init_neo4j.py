#!/usr/bin/env python3
"""Initialize Neo4j with Florida Tax RAG data.

Usage:
    python scripts/init_neo4j.py [--clear] [--verify]

Options:
    --clear     Clear existing data before loading
    --verify    Run verification queries after loading
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.client import Neo4jClient
from src.graph.loader import init_schema, load_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_counts(client: Neo4jClient) -> bool:
    """Verify expected counts in the database.

    Args:
        client: Neo4j client

    Returns:
        True if all counts match expected values
    """
    print()
    print("-" * 40)
    print("VERIFICATION")
    print("-" * 40)

    expected_nodes = {
        "Document": 1152,
        "Chunk": 3022,
    }

    actual_nodes = client.get_node_counts()

    all_ok = True
    print("Node counts:")
    for label, expected_count in expected_nodes.items():
        actual_count = actual_nodes.get(label, 0)
        status = "OK" if actual_count == expected_count else "MISMATCH"
        print(f"  {label}: {actual_count} (expected {expected_count}) [{status}]")
        if actual_count != expected_count:
            all_ok = False

    # Check additional labels
    for label in ["Statute", "Rule", "Case", "TAA"]:
        count = actual_nodes.get(label, 0)
        print(f"  {label}: {count}")

    # Check relationships
    edge_counts = client.get_edge_counts()
    print()
    print("Relationship counts:")
    for rel_type in sorted(edge_counts.keys()):
        count = edge_counts[rel_type]
        print(f"  {rel_type}: {count}")

    # Verify specific counts
    expected_edges = {
        "HAS_CHUNK": 3022,
        "CHILD_OF": 1870,
    }

    for rel_type, expected_count in expected_edges.items():
        actual_count = edge_counts.get(rel_type, 0)
        if actual_count != expected_count:
            print(f"  WARNING: {rel_type} mismatch: {actual_count} vs {expected_count}")
            all_ok = False

    return all_ok


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Initialize Neo4j with Florida Tax RAG data")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before loading")
    parser.add_argument("--verify", action="store_true", help="Verify counts after loading")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "processed"
    corpus_path = data_dir / "corpus.json"
    chunks_path = data_dir / "chunks.json"
    citations_path = data_dir / "citation_graph.json"

    # Verify files exist
    for path in [corpus_path, chunks_path, citations_path]:
        if not path.exists():
            logger.error(f"Missing required file: {path}")
            return 1

    print("=" * 60)
    print("FLORIDA TAX RAG - NEO4J INITIALIZATION")
    print("=" * 60)
    print()
    print(f"Data directory: {data_dir}")
    print(f"  corpus.json:        {corpus_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  chunks.json:        {chunks_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  citation_graph.json: {citations_path.stat().st_size / 1024:.2f} KB")
    print()

    try:
        with Neo4jClient() as client:
            # Health check
            print("Connecting to Neo4j...")
            if not client.health_check():
                logger.error("Cannot connect to Neo4j. Is it running?")
                print()
                print("To start Neo4j:")
                print("  make docker-up")
                print("  make docker-wait")
                return 1

            print("Connected to Neo4j")
            print()

            # Clear if requested
            if args.clear:
                print("-" * 40)
                print("CLEARING DATABASE")
                print("-" * 40)
                client.clear_database()
                print("Database cleared")
                print()

            # Initialize schema
            print("-" * 40)
            print("INITIALIZING SCHEMA")
            print("-" * 40)
            start_time = time.time()
            init_schema(client)
            schema_time = time.time() - start_time
            print(f"Schema initialized in {schema_time:.2f}s")
            print()

            # Load all data
            print("-" * 40)
            print("LOADING DATA")
            print("-" * 40)
            load_start = time.time()
            stats = load_all(client, corpus_path, chunks_path, citations_path)
            time.time() - load_start
            print()

            # Print summary
            print("-" * 40)
            print("SUMMARY")
            print("-" * 40)
            total_time = time.time() - start_time
            print(f"Total time: {total_time:.2f}s")
            print()
            print("Nodes created:")
            print(f"  Documents: {stats['documents']['nodes_created']}")
            print(f"  Chunks:    {stats['chunks']['nodes_created']}")
            print()
            print("Relationships created:")
            print(f"  HAS_CHUNK: {stats['chunks']['relationships_created']}")
            print(f"  CHILD_OF:  {stats['hierarchy']['relationships_created']}")
            print(f"  Citations: {stats['citations']['relationships_created']}")

            # Verify if requested
            if args.verify:
                if verify_counts(client):
                    print()
                    print("Verification: PASSED")
                else:
                    print()
                    print("Verification: FAILED")
                    return 1

    except Exception as e:
        logger.exception(f"Error during initialization: {e}")
        return 1

    print()
    print("=" * 60)
    print("Done! View the graph at http://localhost:7474")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
