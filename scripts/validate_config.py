#!/usr/bin/env python3
"""Validate configuration and test service connections.

Usage:
    python scripts/validate_config.py              # Full validation
    python scripts/validate_config.py --quick      # Settings only (no service tests)
    python scripts/validate_config.py --service neo4j   # Test specific service
    python scripts/validate_config.py --verbose    # Show all settings values
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_settings() -> tuple[bool, list[str]]:
    """Validate that settings can be loaded and pass validation.

    Returns:
        Tuple of (success, list of error messages)
    """
    errors = []

    try:
        from config.settings import get_settings

        # Clear cache to force reload
        get_settings.cache_clear()
        settings = get_settings()

        # Check required fields have values
        if not settings.voyage_api_key.get_secret_value():
            errors.append("VOYAGE_API_KEY is empty")
        if not settings.anthropic_api_key.get_secret_value():
            errors.append("ANTHROPIC_API_KEY is empty")
        if not settings.neo4j_password.get_secret_value():
            errors.append("NEO4J_PASSWORD is empty")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"Failed to load settings: {e}"]


def test_neo4j_connection() -> tuple[bool, str]:
    """Test Neo4j connection.

    Returns:
        Tuple of (success, message)
    """
    try:
        from src.graph.client import Neo4jClient

        client = Neo4jClient()

        if client.health_check():
            client.close()
            return True, "Neo4j connected successfully"
        else:
            client.close()
            return False, "Neo4j health check failed"

    except Exception as e:
        return False, f"Neo4j connection failed: {e}"


def test_weaviate_connection() -> tuple[bool, str]:
    """Test Weaviate connection.

    Returns:
        Tuple of (success, message)
    """
    try:
        from src.vector.client import WeaviateClient

        client = WeaviateClient()

        if client.health_check():
            client.close()
            return True, "Weaviate connected successfully"
        else:
            client.close()
            return False, "Weaviate health check failed"

    except Exception as e:
        return False, f"Weaviate connection failed: {e}"


def test_redis_connection() -> tuple[bool, str]:
    """Test Redis connection.

    Returns:
        Tuple of (success, message)
    """
    try:
        import redis

        from config.settings import get_settings

        settings = get_settings()
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
        r.ping()
        r.close()
        return True, "Redis connected successfully"

    except Exception as e:
        return False, f"Redis connection failed: {e}"


def test_voyage_api_key() -> tuple[bool, str]:
    """Test Voyage AI API key with a minimal request.

    Returns:
        Tuple of (success, message)
    """
    try:
        import voyageai

        from config.settings import get_settings

        settings = get_settings()
        client = voyageai.Client(api_key=settings.voyage_api_key.get_secret_value())

        # Make a minimal embedding request
        result = client.embed(["test"], model=settings.embedding_model)
        if result.embeddings and len(result.embeddings[0]) > 0:
            return True, f"Voyage AI API key valid (model: {settings.embedding_model})"
        else:
            return False, "Voyage AI returned empty embeddings"

    except Exception as e:
        return False, f"Voyage AI API test failed: {e}"


def test_anthropic_api_key() -> tuple[bool, str]:
    """Test Anthropic API key with a minimal request.

    Returns:
        Tuple of (success, message)
    """
    try:
        import anthropic

        from config.settings import get_settings

        settings = get_settings()
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

        # Make a minimal completion request
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        if response.content:
            return True, f"Anthropic API key valid (model: {settings.llm_model})"
        else:
            return False, "Anthropic returned empty response"

    except Exception as e:
        return False, f"Anthropic API test failed: {e}"


def print_settings_summary(verbose: bool = False) -> None:
    """Print current settings summary."""
    from config.settings import get_settings

    settings = get_settings()

    print("\n" + "=" * 60)
    print("SETTINGS SUMMARY")
    print("=" * 60)

    print(f"\nEnvironment: {settings.env}")
    print(f"Debug: {settings.debug}")
    print(f"Log Level: {settings.log_level}")

    print(f"\nNeo4j URI: {settings.neo4j_uri}")
    print(f"Neo4j User: {settings.neo4j_user}")

    print(f"\nWeaviate URL: {settings.weaviate_url}")

    print(f"\nRedis: {settings.redis_host}:{settings.redis_port}/{settings.redis_db}")

    print(f"\nRetrieval Top-K: {settings.retrieval_top_k}")
    print(f"Hybrid Alpha: {settings.hybrid_alpha}")
    print(f"Graph Expansion: {settings.expand_graph}")

    print(f"\nLLM Model: {settings.llm_model}")
    print(f"LLM Temperature: {settings.llm_temperature}")
    print(f"Max Tokens: {settings.max_tokens}")

    print(f"\nEmbedding Model: {settings.embedding_model}")

    print(f"\nRate Limit: {settings.rate_limit_per_minute}/min")

    if verbose:
        print("\n" + "-" * 60)
        print("API KEYS (masked)")
        print("-" * 60)
        voyage_key = settings.voyage_api_key.get_secret_value()
        anthropic_key = settings.anthropic_api_key.get_secret_value()
        print(f"Voyage: {voyage_key[:8]}...{voyage_key[-4:] if len(voyage_key) > 12 else '****'}")
        print(
            f"Anthropic: {anthropic_key[:8]}...{anthropic_key[-4:] if len(anthropic_key) > 12 else '****'}"
        )

        if settings.openai_api_key:
            openai_key = settings.openai_api_key.get_secret_value()
            print(
                f"OpenAI: {openai_key[:8]}...{openai_key[-4:] if len(openai_key) > 12 else '****'}"
            )
        else:
            print("OpenAI: (not configured)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate configuration and test service connections"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only validate settings (no service connection tests)",
    )
    parser.add_argument(
        "--service",
        choices=["neo4j", "weaviate", "redis", "voyage", "anthropic"],
        help="Test specific service only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all settings values including masked API keys",
    )
    parser.add_argument(
        "--skip-api-tests",
        action="store_true",
        help="Skip API key validation tests (Voyage, Anthropic)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG - CONFIGURATION VALIDATION")
    print("=" * 60)

    all_passed = True

    # Step 1: Validate settings
    print("\n[1/2] Validating settings...")
    success, errors = validate_settings()
    if success:
        print("  Settings: OK")
    else:
        print("  Settings: FAILED")
        for error in errors:
            print(f"    - {error}")
        all_passed = False

    # Show settings summary
    if success:
        print_settings_summary(verbose=args.verbose)

    # Step 2: Test service connections
    if not args.quick:
        print("\n[2/2] Testing service connections...")

        service_tests = []

        if args.service:
            # Test specific service only
            if args.service == "neo4j":
                service_tests.append(("Neo4j", test_neo4j_connection))
            elif args.service == "weaviate":
                service_tests.append(("Weaviate", test_weaviate_connection))
            elif args.service == "redis":
                service_tests.append(("Redis", test_redis_connection))
            elif args.service == "voyage":
                service_tests.append(("Voyage AI", test_voyage_api_key))
            elif args.service == "anthropic":
                service_tests.append(("Anthropic", test_anthropic_api_key))
        else:
            # Test all services
            service_tests = [
                ("Neo4j", test_neo4j_connection),
                ("Weaviate", test_weaviate_connection),
                ("Redis", test_redis_connection),
            ]

            if not args.skip_api_tests:
                service_tests.extend(
                    [
                        ("Voyage AI", test_voyage_api_key),
                        ("Anthropic", test_anthropic_api_key),
                    ]
                )

        for name, test_func in service_tests:
            print(f"\n  Testing {name}...", end=" ")
            success, message = test_func()
            if success:
                print("OK")
                print(f"    {message}")
            else:
                print("FAILED")
                print(f"    {message}")
                # Redis and API tests are optional - don't fail validation
                if name in ("Neo4j", "Weaviate"):
                    all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED")
        print("=" * 60)
        return 0
    else:
        print("VALIDATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
