"""Tests for src/api/middleware.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from src.api.errors import RateLimitError
from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_request() -> MagicMock:
    """Create a mock Starlette request."""
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/api/v1/query"
    request.query_params = {}
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def mock_response() -> Response:
    """Create a mock Starlette response."""
    return Response(content="OK", status_code=200)


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock FastAPI app."""
    return MagicMock()


# =============================================================================
# RequestLoggingMiddleware Tests
# =============================================================================


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware class."""

    async def test_generates_request_id(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should generate request ID if not present in headers."""
        mock_request.headers = {}

        async def call_next(request):
            return mock_response

        middleware = RequestLoggingMiddleware(mock_app)

        with patch("src.api.middleware.set_request_context"):
            with patch("src.api.middleware.clear_request_context"):
                with patch("src.api.middleware.get_metrics_collector") as mock_metrics:
                    mock_collector = MagicMock()
                    mock_metrics.return_value = mock_collector

                    response = await middleware.dispatch(mock_request, call_next)

        assert "X-Request-ID" in response.headers
        assert "X-Query-ID" in response.headers

    async def test_extracts_request_id_from_header(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should use X-Request-ID from headers if present."""
        mock_request.headers = {"X-Request-ID": "test-request-id-123"}

        async def call_next(request):
            return mock_response

        middleware = RequestLoggingMiddleware(mock_app)

        with patch("src.api.middleware.set_request_context"):
            with patch("src.api.middleware.clear_request_context"):
                with patch("src.api.middleware.get_metrics_collector") as mock_metrics:
                    mock_collector = MagicMock()
                    mock_metrics.return_value = mock_collector

                    response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["X-Request-ID"] == "test-request-id-123"

    async def test_records_success_metrics(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should record success metrics for successful requests."""

        async def call_next(request):
            return mock_response

        middleware = RequestLoggingMiddleware(mock_app)

        with patch("src.api.middleware.set_request_context"):
            with patch("src.api.middleware.clear_request_context"):
                with patch("src.api.middleware.get_metrics_collector") as mock_metrics:
                    mock_collector = MagicMock()
                    mock_metrics.return_value = mock_collector

                    await middleware.dispatch(mock_request, call_next)

        mock_collector.record_query.assert_called_once()
        # record_query is called with positional args (latency_ms, success)
        call_args = mock_collector.record_query.call_args
        # The second positional arg is success=True
        assert call_args[0][1] is True or call_args[1].get("success") is True

    async def test_records_error_metrics(
        self, mock_request: MagicMock, mock_app: MagicMock
    ) -> None:
        """Should record error metrics for failed requests."""

        async def call_next(request):
            raise ValueError("Test error")

        middleware = RequestLoggingMiddleware(mock_app)

        with patch("src.api.middleware.set_request_context"):
            with patch("src.api.middleware.clear_request_context"):
                with patch("src.api.middleware.get_metrics_collector") as mock_metrics:
                    mock_collector = MagicMock()
                    mock_metrics.return_value = mock_collector

                    with pytest.raises(ValueError):
                        await middleware.dispatch(mock_request, call_next)

        mock_collector.record_query.assert_called_once()
        call_args = mock_collector.record_query.call_args
        assert call_args[1]["success"] is False
        assert call_args[1]["error_type"] == "ValueError"

    async def test_measures_latency(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should measure request latency."""

        async def call_next(request):
            return mock_response

        middleware = RequestLoggingMiddleware(mock_app)

        with patch("src.api.middleware.set_request_context"):
            with patch("src.api.middleware.clear_request_context"):
                with patch("src.api.middleware.get_metrics_collector") as mock_metrics:
                    mock_collector = MagicMock()
                    mock_metrics.return_value = mock_collector

                    await middleware.dispatch(mock_request, call_next)

        call_args = mock_collector.record_query.call_args
        latency = call_args[0][0]
        assert latency >= 0  # Latency should be non-negative


# =============================================================================
# RateLimitMiddleware Tests
# =============================================================================


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware class."""

    async def test_allows_under_limit(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should allow requests under the rate limit."""

        async def call_next(request):
            return mock_response

        middleware = RateLimitMiddleware(mock_app, requests_per_minute=10)
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200

    async def test_blocks_over_limit(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should block requests over the rate limit."""

        async def call_next(request):
            return mock_response

        middleware = RateLimitMiddleware(mock_app, requests_per_minute=2)

        # First two requests should succeed
        await middleware.dispatch(mock_request, call_next)
        await middleware.dispatch(mock_request, call_next)

        # Third request should be rate limited
        with pytest.raises(RateLimitError):
            await middleware.dispatch(mock_request, call_next)

    async def test_excludes_health_endpoint(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should exclude health endpoint from rate limiting."""
        mock_request.url.path = "/api/v1/health"

        async def call_next(request):
            return mock_response

        middleware = RateLimitMiddleware(mock_app, requests_per_minute=1)

        # Health endpoint should not be rate limited
        for _ in range(10):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200

    async def test_excludes_metrics_endpoint(
        self, mock_request: MagicMock, mock_response: Response, mock_app: MagicMock
    ) -> None:
        """Should exclude metrics endpoint from rate limiting."""
        mock_request.url.path = "/api/v1/metrics"

        async def call_next(request):
            return mock_response

        middleware = RateLimitMiddleware(mock_app, requests_per_minute=1)

        # Metrics endpoint should not be rate limited
        for _ in range(5):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200

    async def test_per_ip_tracking(self, mock_response: Response, mock_app: MagicMock) -> None:
        """Should track rate limit per IP address."""

        async def call_next(request):
            return mock_response

        middleware = RateLimitMiddleware(mock_app, requests_per_minute=2)

        # Create two mock requests from different IPs
        request1 = MagicMock(spec=Request)
        request1.url.path = "/api/v1/query"
        request1.client = MagicMock()
        request1.client.host = "192.168.1.1"

        request2 = MagicMock(spec=Request)
        request2.url.path = "/api/v1/query"
        request2.client = MagicMock()
        request2.client.host = "192.168.1.2"

        # Each IP should have its own limit
        await middleware.dispatch(request1, call_next)
        await middleware.dispatch(request1, call_next)
        await middleware.dispatch(request2, call_next)
        await middleware.dispatch(request2, call_next)

        # IP1 should now be rate limited
        with pytest.raises(RateLimitError):
            await middleware.dispatch(request1, call_next)

        # IP2 should also be rate limited
        with pytest.raises(RateLimitError):
            await middleware.dispatch(request2, call_next)

    def test_cleanup_old_entries(self, mock_app: MagicMock) -> None:
        """Should remove entries older than 1 minute."""
        middleware = RateLimitMiddleware(mock_app, requests_per_minute=100)

        current_time = time.time()

        # Add old entries
        middleware.requests["old_ip"] = [current_time - 120, current_time - 90]
        # Add recent entries
        middleware.requests["recent_ip"] = [current_time - 30, current_time - 10]

        middleware._cleanup_old_entries(current_time)

        # Old IP should be removed
        assert "old_ip" not in middleware.requests
        # Recent IP should remain with only entries < 60s old
        assert "recent_ip" in middleware.requests
        assert len(middleware.requests["recent_ip"]) == 2
