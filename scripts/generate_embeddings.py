#!/usr/bin/env python3
"""Generate embeddings for all chunks using Voyage AI.

Usage:
    python scripts/generate_embeddings.py [--sample N] [--resume] [--verify] [--no-cache]

Options:
    --sample N   Only embed N chunks (for testing, saves to sample_embeddings.npz)
    --resume     Resume from last checkpoint
    --verify     Verify embeddings after generation
    --no-cache   Disable Redis caching
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunking import LegalChunk
from src.vector.embeddings import (
    VoyageEmbedder,
    create_embedder_with_cache,
    verify_embeddings,
)
from src.vector.schema import VOYAGE_LAW_2_DIMENSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
SAMPLE_EMBEDDINGS_PATH = DATA_DIR / "sample_embeddings.npz"
CHECKPOINT_PATH = DATA_DIR / ".embeddings_checkpoint.json"
STATS_PATH = DATA_DIR / "embedding_stats.json"


def load_chunks(path: Path = CHUNKS_PATH) -> list[LegalChunk]:
    """Load chunks from JSON file.

    Args:
        path: Path to chunks.json

    Returns:
        List of LegalChunk objects
    """
    logger.info(f"Loading chunks from {path}")
    with open(path) as f:
        data = json.load(f)

    chunks = [LegalChunk(**chunk) for chunk in data["chunks"]]
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def save_embeddings(
    chunk_ids: list[str],
    embeddings: np.ndarray,
    output_path: Path,
) -> None:
    """Save embeddings to numpy .npz format.

    Args:
        chunk_ids: List of chunk IDs
        embeddings: 2D numpy array of embeddings
        output_path: Output file path
    """
    logger.info(f"Saving {len(chunk_ids)} embeddings to {output_path}")
    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids, dtype=object),
        embeddings=embeddings.astype(np.float32),  # Use float32 for efficiency
    )
    file_size = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved embeddings: {file_size:.2f} MB")


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load embeddings from .npz file.

    Args:
        path: Path to embeddings.npz

    Returns:
        Tuple of (chunk_ids, embeddings)
    """
    data = np.load(path, allow_pickle=True)
    return data["chunk_ids"].tolist(), data["embeddings"]


def save_checkpoint(completed_ids: set[str], path: Path = CHECKPOINT_PATH) -> None:
    """Save progress checkpoint.

    Args:
        completed_ids: Set of completed chunk IDs
        path: Checkpoint file path
    """
    checkpoint = {
        "completed_ids": list(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "count": len(completed_ids),
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)
    logger.debug(f"Saved checkpoint: {len(completed_ids)} completed")


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> set[str]:
    """Load completed chunk IDs from checkpoint.

    Args:
        path: Checkpoint file path

    Returns:
        Set of completed chunk IDs
    """
    if not path.exists():
        return set()

    with open(path) as f:
        data = json.load(f)

    completed = set(data.get("completed_ids", []))
    logger.info(f"Loaded checkpoint: {len(completed)} previously completed")
    return completed


def save_stats(stats: dict, path: Path = STATS_PATH) -> None:
    """Save embedding statistics.

    Args:
        stats: Statistics dictionary
        path: Output path
    """
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks using Voyage AI")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only embed N chunks (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify embeddings after generation",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG - EMBEDDING GENERATION")
    print("=" * 60)
    print()
    print("Model: voyage-law-2")
    print(f"Dimension: {VOYAGE_LAW_2_DIMENSION}")
    print("Batch size: 72 (conservative for token limit)")
    print()

    # Load chunks
    if not CHUNKS_PATH.exists():
        logger.error(f"Chunks file not found: {CHUNKS_PATH}")
        return 1

    chunks = load_chunks()

    # Sample mode
    if args.sample:
        chunks = chunks[: args.sample]
        output_path = SAMPLE_EMBEDDINGS_PATH
        print(f"Sample mode: embedding {len(chunks)} chunks")
    else:
        output_path = EMBEDDINGS_PATH
        print(f"Full mode: embedding {len(chunks)} chunks")
    print()

    # Resume from checkpoint
    completed_ids: set[str] = set()
    if args.resume and not args.sample:
        completed_ids = load_checkpoint()
        if completed_ids:
            chunks = [c for c in chunks if c.id not in completed_ids]
            print(f"Resuming: {len(completed_ids)} already done, {len(chunks)} remaining")
            print()

    if not chunks:
        print("No chunks to embed!")
        return 0

    # Create embedder
    print("-" * 40)
    print("INITIALIZING EMBEDDER")
    print("-" * 40)

    try:
        if args.no_cache:
            embedder = VoyageEmbedder()
            print("Redis caching: DISABLED")
        else:
            embedder = create_embedder_with_cache(use_cache=True)
            print("Redis caching: ENABLED")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        logger.info("Proceeding without caching")
        embedder = VoyageEmbedder()
        print("Redis caching: DISABLED (connection failed)")

    print()

    # Embed chunks
    print("-" * 40)
    print("GENERATING EMBEDDINGS")
    print("-" * 40)

    start_time = time.time()

    try:
        results = embedder.embed_chunks(chunks, show_progress=True)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return 1

    elapsed = time.time() - start_time
    print()

    # Extract results
    chunk_ids = [chunk_id for chunk_id, _ in results]
    embeddings = np.array([emb for _, emb in results])

    # Add previously completed if resuming
    if completed_ids and not args.sample:
        # Load previous embeddings
        if EMBEDDINGS_PATH.exists():
            prev_ids, prev_embeddings = load_embeddings(EMBEDDINGS_PATH)
            # Combine
            all_ids = prev_ids + chunk_ids
            all_embeddings = np.vstack([prev_embeddings, embeddings])
            chunk_ids = all_ids
            embeddings = all_embeddings

    # Save embeddings
    print("-" * 40)
    print("SAVING EMBEDDINGS")
    print("-" * 40)

    save_embeddings(chunk_ids, embeddings, output_path)

    # Clear checkpoint after successful save
    if CHECKPOINT_PATH.exists() and not args.sample:
        CHECKPOINT_PATH.unlink()
        logger.info("Cleared checkpoint file")

    # Verify if requested
    if args.verify:
        print()
        print("-" * 40)
        print("VERIFICATION")
        print("-" * 40)

        verification = verify_embeddings(embeddings.tolist())

        print(f"Count: {verification['count']}")
        print(
            f"Dimension: {verification['dimension']} (expected {verification['expected_dimension']})"
        )
        print(f"Dimension OK: {verification['dimension_ok']}")
        print(f"Avg L2 norm: {verification['avg_norm']:.4f}")
        print(f"Std L2 norm: {verification['std_norm']:.4f}")
        print(f"Normalized: {verification['normalized']}")
        print()

        if verification["valid"]:
            print("Verification: PASSED")
        else:
            print("Verification: FAILED")
            return 1

    # Statistics
    stats = embedder.get_stats()
    stats.update(
        {
            "total_chunks": len(chunk_ids),
            "elapsed_seconds": elapsed,
            "output_file": str(output_path),
            "timestamp": datetime.now().isoformat(),
        }
    )

    if not args.sample:
        save_stats(stats, STATS_PATH)

    # Summary
    print()
    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Chunks embedded: {len(chunk_ids)}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Output: {output_path}")

    # Cost estimate
    avg_tokens = 362  # From chunking stats
    total_tokens = stats["texts_embedded"] * avg_tokens
    cost_per_million = 0.12  # voyage-law-2 pricing
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    print(f"Estimated cost: ${estimated_cost:.4f}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
