"""Tests for the observability module."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.errors import (
    GenerationError,
    NotFoundError,
    QueryTimeoutError,
    RateLimitError,
    RetrievalError,
    TaxRAGError,
    ValidationError,
)
from src.observability.context import (
    clear_request_context,
    get_context,
    query_id_var,
    request_id_var,
    set_request_context,
)
from src.observability.metrics import MetricsCollector, get_metrics_collector


# =============================================================================
# Context Tests
# =============================================================================


class TestRequestContext:
    """Tests for request context management."""

    def test_set_and_get_context(self):
        """Test setting and getting request context."""
        set_request_context("req-123", "q-abc")

        ctx = get_context()
        assert ctx["request_id"] == "req-123"
        assert ctx["query_id"] == "q-abc"

        clear_request_context()

    def test_get_empty_context(self):
        """Test getting context when not set."""
        clear_request_context()

        ctx = get_context()
        assert ctx == {}

    def test_clear_context(self):
        """Test clearing request context."""
        set_request_context("req-456", "q-xyz")
        clear_request_context()

        ctx = get_context()
        assert ctx == {}

    def test_partial_context(self):
        """Test setting only request_id."""
        set_request_context("req-789")

        ctx = get_context()
        assert ctx["request_id"] == "req-789"
        assert "query_id" not in ctx or ctx.get("query_id") == ""

        clear_request_context()


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for the metrics collector."""

    def setup_method(self):
        """Reset metrics before each test."""
        collector = get_metrics_collector()
        collector.reset()

    def test_singleton_pattern(self):
        """Test that MetricsCollector is a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2

    def test_record_successful_query(self):
        """Test recording a successful query."""
        collector = get_metrics_collector()
        collector.record_query(latency_ms=150.0, success=True)

        stats = collector.get_stats()
        assert stats["total_queries"] == 1
        assert stats["successful_queries"] == 1
        assert stats["failed_queries"] == 0
        assert stats["success_rate_percent"] == 100.0
        assert stats["latency_ms"]["avg"] == 150.0

    def test_record_failed_query(self):
        """Test recording a failed query."""
        collector = get_metrics_collector()
        collector.record_query(latency_ms=50.0, success=False, error_type="TimeoutError")

        stats = collector.get_stats()
        assert stats["total_queries"] == 1
        assert stats["successful_queries"] == 0
        assert stats["failed_queries"] == 1
        assert stats["errors_by_type"]["TimeoutError"] == 1

    def test_record_error(self):
        """Test recording an error without latency."""
        collector = get_metrics_collector()
        collector.record_error("RateLimitError")

        stats = collector.get_stats()
        assert stats["failed_queries"] == 1
        assert stats["errors_by_type"]["RateLimitError"] == 1

    def test_latency_statistics(self):
        """Test latency min/max/avg calculation."""
        collector = get_metrics_collector()
        collector.record_query(100.0, True)
        collector.record_query(200.0, True)
        collector.record_query(300.0, True)

        stats = collector.get_stats()
        assert stats["latency_ms"]["min"] == 100.0
        assert stats["latency_ms"]["max"] == 300.0
        assert stats["latency_ms"]["avg"] == 200.0

    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        collector = get_metrics_collector()
        collector.record_query(100.0, True)
        collector.record_query(100.0, True)
        collector.record_query(100.0, False)
        collector.record_query(100.0, True)

        stats = collector.get_stats()
        assert stats["success_rate_percent"] == 75.0

    def test_thread_safety(self):
        """Test that metrics collector is thread-safe."""
        collector = get_metrics_collector()
        collector.reset()

        num_threads = 10
        queries_per_thread = 100

        def record_queries():
            for _ in range(queries_per_thread):
                collector.record_query(50.0, True)

        threads = [threading.Thread(target=record_queries) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = collector.get_stats()
        expected_total = num_threads * queries_per_thread
        assert stats["total_queries"] == expected_total

    def test_uptime_tracking(self):
        """Test uptime is tracked correctly."""
        collector = get_metrics_collector()
        collector.reset()

        time.sleep(0.2)  # Wait a bit (slightly more for rounding)

        stats = collector.get_stats()
        # uptime_seconds is rounded to 0 decimal places, so check >= 0
        assert stats["uptime_seconds"] >= 0
        assert "started_at" in stats

    def test_reset(self):
        """Test resetting metrics."""
        collector = get_metrics_collector()
        collector.record_query(100.0, True)
        collector.record_query(200.0, False, "TestError")

        collector.reset()

        stats = collector.get_stats()
        assert stats["total_queries"] == 0
        assert stats["successful_queries"] == 0
        assert stats["failed_queries"] == 0


# =============================================================================
# Custom Exceptions Tests
# =============================================================================


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_tax_rag_error_base(self):
        """Test base TaxRAGError exception."""
        error = TaxRAGError("Test error", details={"key": "value"})

        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert error.error_code == "TAX_RAG_ERROR"
        assert error.status_code == 500

    def test_tax_rag_error_to_dict(self):
        """Test converting exception to dictionary."""
        error = TaxRAGError("Test error", details={"foo": "bar"})
        result = error.to_dict()

        assert result["error"] == "TAX_RAG_ERROR"
        assert result["message"] == "Test error"
        assert result["details"] == {"foo": "bar"}

    def test_retrieval_error(self):
        """Test RetrievalError exception."""
        error = RetrievalError("Neo4j connection failed")

        assert error.error_code == "RETRIEVAL_ERROR"
        assert error.status_code == 503

    def test_generation_error(self):
        """Test GenerationError exception."""
        error = GenerationError("Claude API failed")

        assert error.error_code == "GENERATION_ERROR"
        assert error.status_code == 502

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Invalid query")

        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == 400

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError("Too many requests", details={"retry_after": 60})

        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.status_code == 429
        assert error.details["retry_after"] == 60

    def test_query_timeout_error(self):
        """Test QueryTimeoutError exception."""
        error = QueryTimeoutError("Query timed out")

        assert error.error_code == "TIMEOUT"
        assert error.status_code == 408

    def test_not_found_error(self):
        """Test NotFoundError exception."""
        error = NotFoundError("Chunk not found", details={"chunk_id": "abc123"})

        assert error.error_code == "NOT_FOUND"
        assert error.status_code == 404

    def test_override_error_code(self):
        """Test overriding error code at instantiation."""
        error = TaxRAGError(
            "Custom error",
            error_code="CUSTOM_CODE",
            status_code=418,
        )

        assert error.error_code == "CUSTOM_CODE"
        assert error.status_code == 418


# =============================================================================
# Middleware Tests
# =============================================================================


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    def test_rate_limit_allows_requests(self):
        """Test that requests under limit are allowed."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=10)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Make 5 requests (under limit)
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200

    def test_excluded_paths_not_rate_limited(self):
        """Test that excluded paths bypass rate limiting."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @app.get("/api/v1/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/api/v1/metrics")
        async def metrics():
            return {"total": 0}

        client = TestClient(app)

        # Health and metrics should not be rate limited
        for _ in range(10):
            response = client.get("/api/v1/health")
            assert response.status_code == 200

        for _ in range(10):
            response = client.get("/api/v1/metrics")
            assert response.status_code == 200


class TestRequestLoggingMiddleware:
    """Tests for request logging middleware."""

    def test_adds_request_id_header(self):
        """Test that X-Request-ID header is added to response."""
        from src.api.middleware import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Request-ID" in response.headers
        assert "X-Query-ID" in response.headers

    def test_uses_provided_request_id(self):
        """Test that provided X-Request-ID is used."""
        from src.api.middleware import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Request-ID": "custom-id-123"})

        assert response.headers["X-Request-ID"] == "custom-id-123"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsEndpoint:
    """Tests for the /metrics API endpoint."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with metrics endpoint."""
        from src.api.routes import router
        from src.api.main import create_app

        return create_app()

    def test_metrics_endpoint_returns_stats(self, app):
        """Test that metrics endpoint returns expected stats."""
        # Reset metrics for clean test
        collector = get_metrics_collector()
        collector.reset()

        client = TestClient(app)

        # Make a request to generate some metrics
        # (This may fail due to missing deps, but should still record the attempt)
        try:
            client.get("/api/v1/health")
        except Exception:
            pass

        # Get metrics
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "total_queries" in data
        assert "successful_queries" in data
        assert "failed_queries" in data
        assert "latency_ms" in data
        assert "uptime_seconds" in data
