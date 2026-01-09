#!/usr/bin/env python3
"""Load chunks and embeddings into Weaviate.

Usage:
    python scripts/load_weaviate.py [--resume] [--batch-size N] [--reset]

Options:
    --resume       Resume from last checkpoint
    --batch-size N Number of chunks per batch (default 100)
    --reset        Delete and recreate the collection before loading
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector.client import WeaviateClient

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
CHECKPOINT_PATH = DATA_DIR / ".weaviate_checkpoint.json"
LOAD_LOG_PATH = DATA_DIR / "weaviate_load.log"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load chunks from JSON file.

    Args:
        path: Path to chunks.json

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Loading chunks from {path}")
    with open(path) as f:
        data = json.load(f)

    chunks = data["chunks"]
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> tuple[list[str], np.ndarray]:
    """Load embeddings from .npz file.

    Args:
        path: Path to embeddings.npz

    Returns:
        Tuple of (chunk_ids, embeddings array)
    """
    logger.info(f"Loading embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    chunk_ids = data["chunk_ids"].tolist()
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(chunk_ids)} embeddings with shape {embeddings.shape}")
    return chunk_ids, embeddings


def chunk_to_weaviate_properties(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert a LegalChunk dict to Weaviate properties.

    Args:
        chunk: Chunk dictionary from chunks.json

    Returns:
        Dictionary of Weaviate properties
    """
    # Convert effective_date string to datetime
    effective_date = None
    if chunk.get("effective_date"):
        try:
            effective_date = datetime.fromisoformat(chunk["effective_date"])
        except (ValueError, TypeError):
            pass

    return {
        "chunk_id": chunk["id"],
        "doc_id": chunk["doc_id"],
        "doc_type": chunk["doc_type"],
        "level": chunk["level"],  # Already a string from JSON
        "ancestry": chunk.get("ancestry", ""),
        "citation": chunk.get("citation", ""),
        "text": chunk["text"],
        "text_with_ancestry": chunk.get("text_with_ancestry", chunk["text"]),
        "effective_date": effective_date,
        "token_count": chunk.get("token_count", 0),
    }


def save_checkpoint(loaded_ids: set[str], path: Path = CHECKPOINT_PATH) -> None:
    """Save progress checkpoint.

    Args:
        loaded_ids: Set of loaded chunk IDs
        path: Checkpoint file path
    """
    checkpoint = {
        "loaded_ids": list(loaded_ids),
        "timestamp": datetime.now().isoformat(),
        "count": len(loaded_ids),
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)
    logger.debug(f"Saved checkpoint: {len(loaded_ids)} loaded")


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

    loaded = set(data.get("loaded_ids", []))
    logger.info(f"Loaded checkpoint: {len(loaded)} previously loaded")
    return loaded


def save_load_log(stats: dict[str, Any], path: Path = LOAD_LOG_PATH) -> None:
    """Save load statistics to log file.

    Args:
        stats: Statistics dictionary
        path: Log file path
    """
    with open(path, "w") as f:
        f.write("WEAVIATE LOAD LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {stats['timestamp']}\n")
        f.write(f"Total chunks: {stats['total_chunks']}\n")
        f.write(f"Loaded: {stats['loaded']}\n")
        f.write(f"Skipped (checkpoint): {stats['skipped']}\n")
        f.write(f"Errors: {stats['errors']}\n")
        f.write(f"Elapsed time: {stats['elapsed_seconds']:.2f}s\n")
        f.write(f"Rate: {stats['rate']:.2f} chunks/s\n\n")

        if stats.get("failed_chunks"):
            f.write("Failed chunks:\n")
            for chunk_id in stats["failed_chunks"]:
                f.write(f"  - {chunk_id}\n")
        else:
            f.write("No failed chunks.\n")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Load chunks and embeddings into Weaviate")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per batch (default 100)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before loading",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG - WEAVIATE LOADER")
    print("=" * 60)
    print()

    # Check files exist
    if not CHUNKS_PATH.exists():
        logger.error(f"Chunks file not found: {CHUNKS_PATH}")
        return 1
    if not EMBEDDINGS_PATH.exists():
        logger.error(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        return 1

    # Load data
    print("-" * 40)
    print("LOADING DATA")
    print("-" * 40)

    chunks = load_chunks()
    embedding_ids, embeddings = load_embeddings()

    # Create embedding lookup
    embedding_map = {chunk_id: embeddings[i] for i, chunk_id in enumerate(embedding_ids)}
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(embedding_map)}")
    print()

    # Match chunks to embeddings
    matched_chunks = []
    matched_vectors = []
    missing_embeddings = []

    for chunk in chunks:
        chunk_id = chunk["id"]
        if chunk_id in embedding_map:
            matched_chunks.append(chunk)
            matched_vectors.append(embedding_map[chunk_id].tolist())
        else:
            missing_embeddings.append(chunk_id)

    if missing_embeddings:
        logger.warning(f"{len(missing_embeddings)} chunks missing embeddings")
        for chunk_id in missing_embeddings[:5]:
            logger.warning(f"  Missing: {chunk_id}")
        if len(missing_embeddings) > 5:
            logger.warning(f"  ... and {len(missing_embeddings) - 5} more")

    print(f"Matched: {len(matched_chunks)} chunks with embeddings")
    print()

    # Resume from checkpoint
    loaded_ids: set[str] = set()
    if args.resume:
        loaded_ids = load_checkpoint()
        if loaded_ids:
            # Filter out already loaded chunks
            len(matched_chunks)
            filtered = [
                (c, v) for c, v in zip(matched_chunks, matched_vectors) if c["id"] not in loaded_ids
            ]
            matched_chunks = [c for c, _ in filtered]
            matched_vectors = [v for _, v in filtered]
            print(f"Resuming: {len(loaded_ids)} already loaded, {len(matched_chunks)} remaining")
            print()

    if not matched_chunks:
        print("No chunks to load!")
        return 0

    # Connect to Weaviate
    print("-" * 40)
    print("CONNECTING TO WEAVIATE")
    print("-" * 40)

    client = WeaviateClient()

    if not client.health_check():
        logger.error("Weaviate is not reachable")
        return 1

    print("Weaviate connection: OK")

    # Reset collection if requested
    if args.reset:
        print("Resetting collection...")
        client.delete_collection()
        client.init_schema()
        loaded_ids = set()  # Clear checkpoint
        print("Collection reset complete")
    else:
        # Ensure collection exists
        client.init_schema()

    print()

    # Load data in batches
    print("-" * 40)
    print("LOADING DATA INTO WEAVIATE")
    print("-" * 40)
    print(f"Batch size: {args.batch_size}")
    print()

    start_time = time.time()
    total_loaded = 0
    total_errors = 0
    failed_chunks: list[str] = []

    # Process in batches with progress bar
    num_batches = (len(matched_chunks) + args.batch_size - 1) // args.batch_size
    batch_iterator = tqdm(range(num_batches), desc="Loading batches")

    for batch_idx in batch_iterator:
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(matched_chunks))

        batch_chunks = matched_chunks[start_idx:end_idx]
        batch_vectors = matched_vectors[start_idx:end_idx]

        # Convert to Weaviate format
        batch_properties = [chunk_to_weaviate_properties(c) for c in batch_chunks]

        try:
            result = client.batch_insert(
                batch_properties, batch_vectors, batch_size=len(batch_properties)
            )
            total_loaded += result["inserted"]
            total_errors += result["errors"]

            if result["errors"] > 0:
                for chunk in batch_chunks:
                    failed_chunks.append(chunk["id"])

            # Update checkpoint every 10 batches
            if (batch_idx + 1) % 10 == 0:
                for chunk in batch_chunks:
                    loaded_ids.add(chunk["id"])
                save_checkpoint(loaded_ids)

        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {e}")
            total_errors += len(batch_chunks)
            for chunk in batch_chunks:
                failed_chunks.append(chunk["id"])

        # Update progress bar
        batch_iterator.set_postfix(loaded=total_loaded, errors=total_errors)

    elapsed = time.time() - start_time

    # Final checkpoint
    for chunk in matched_chunks:
        loaded_ids.add(chunk["id"])
    save_checkpoint(loaded_ids)

    # Clear checkpoint on success
    if total_errors == 0 and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Cleared checkpoint file (all chunks loaded successfully)")

    print()

    # Verify
    print("-" * 40)
    print("VERIFICATION")
    print("-" * 40)

    collection_info = client.get_collection_info()
    if collection_info:
        weaviate_count = collection_info["object_count"]
        print(f"Weaviate object count: {weaviate_count}")
        print(f"Expected count: {len(chunks)}")

        if weaviate_count == len(chunks):
            print("Count verification: PASSED")
        else:
            print(f"Count verification: WARNING (diff: {weaviate_count - len(chunks)})")
    else:
        print("Could not get collection info")

    print()

    # Summary
    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)

    rate = total_loaded / elapsed if elapsed > 0 else 0
    print(f"Total chunks: {len(chunks)}")
    print(f"Loaded: {total_loaded}")
    print(f"Skipped (checkpoint): {len(loaded_ids) - total_loaded}")
    print(f"Errors: {total_errors}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Rate: {rate:.2f} chunks/s")

    # Save log
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "loaded": total_loaded,
        "skipped": len(loaded_ids) - total_loaded,
        "errors": total_errors,
        "elapsed_seconds": elapsed,
        "rate": rate,
        "failed_chunks": failed_chunks,
    }
    save_load_log(stats)
    print(f"\nLog saved: {LOAD_LOG_PATH}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    # Close client
    client.close()

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
