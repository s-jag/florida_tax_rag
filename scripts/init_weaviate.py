#!/usr/bin/env python3
"""Initialize Weaviate schema for Florida Tax RAG.

Usage:
    python scripts/init_weaviate.py [--delete] [--verify]

Options:
    --delete    Delete existing collection before creating
    --verify    Verify collection was created successfully

Note: This script only initializes the schema. Data loading requires
embeddings from Voyage AI and will be done separately.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector.client import WeaviateClient
from src.vector.schema import VOYAGE_LAW_2_DIMENSION, CollectionName

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_collection(client: WeaviateClient) -> bool:
    """Verify the LegalChunk collection was created correctly.

    Args:
        client: Weaviate client

    Returns:
        True if collection exists and has correct schema
    """
    print()
    print("-" * 40)
    print("VERIFICATION")
    print("-" * 40)

    info = client.get_collection_info()

    if info is None:
        print("ERROR: Collection does not exist")
        return False

    print(f"Collection: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Object count: {info['object_count']}")
    print()
    print("Properties:")

    expected_properties = {
        "chunk_id",
        "doc_id",
        "doc_type",
        "level",
        "ancestry",
        "citation",
        "text",
        "text_with_ancestry",
        "effective_date",
        "token_count",
    }

    actual_properties = {p["name"] for p in info["properties"]}

    for prop in info["properties"]:
        status = "OK" if prop["name"] in expected_properties else "UNEXPECTED"
        print(f"  {prop['name']}: {prop['data_type']} [{status}]")

    missing = expected_properties - actual_properties
    if missing:
        print()
        print(f"MISSING PROPERTIES: {missing}")
        return False

    print()
    print(f"Vector dimension: {VOYAGE_LAW_2_DIMENSION} (Voyage AI voyage-law-2)")

    return True


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Initialize Weaviate schema for Florida Tax RAG")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing collection before creating",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify collection was created successfully",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG - WEAVIATE INITIALIZATION")
    print("=" * 60)
    print()
    print(f"Collection: {CollectionName.LEGAL_CHUNK.value}")
    print(f"Vector dimension: {VOYAGE_LAW_2_DIMENSION}")
    print()

    try:
        with WeaviateClient() as client:
            # Health check
            print("Connecting to Weaviate...")
            if not client.health_check():
                logger.error("Cannot connect to Weaviate. Is it running?")
                print()
                print("To start Weaviate:")
                print("  make docker-up")
                print("  make docker-wait")
                return 1

            print("Connected to Weaviate")
            print()

            # Delete if requested
            if args.delete:
                print("-" * 40)
                print("DELETING COLLECTION")
                print("-" * 40)
                if client.delete_collection():
                    print(f"Deleted collection '{CollectionName.LEGAL_CHUNK.value}'")
                else:
                    print("Collection did not exist")
                print()

            # Initialize schema
            print("-" * 40)
            print("INITIALIZING SCHEMA")
            print("-" * 40)
            if client.init_schema():
                print(f"Created collection '{CollectionName.LEGAL_CHUNK.value}'")
            else:
                print(f"Collection '{CollectionName.LEGAL_CHUNK.value}' already exists")
            print()

            # Verify if requested
            if args.verify:
                if verify_collection(client):
                    print("Verification: PASSED")
                else:
                    print("Verification: FAILED")
                    return 1

    except Exception as e:
        logger.exception(f"Error during initialization: {e}")
        return 1

    print()
    print("=" * 60)
    print("Schema initialized!")
    print()
    print("Next steps:")
    print("  1. Configure Voyage AI API key in .env")
    print("  2. Run embedding script to load data with vectors")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
