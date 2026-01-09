"""FastAPI middleware for logging, metrics, and rate limiting."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.context import clear_request_context, set_request_context
from src.observability.metrics import get_metrics_collector

from .errors import RateLimitError

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and tracing.

    Features:
    - Generates/extracts request ID from X-Request-ID header
    - Generates short query ID for tracing through agent nodes
    - Logs request start and completion with timing
    - Binds context vars for downstream logging
    - Records metrics for each request
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with logging and tracing."""
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        query_id = str(uuid.uuid4())[:8]  # Short ID for easier tracing

        # Set context for logging throughout the request
        set_request_context(request_id, query_id)

        # Bind context vars for structlog
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            query_id=query_id,
        )

        # Log request start
        logger.info(
            "request_started",
            method=request.method,
            path=str(request.url.path),
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
        )

        start_time = time.perf_counter()
        metrics = get_metrics_collector()

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            )

            # Record metrics
            success = response.status_code < 400
            metrics.record_query(latency_ms, success)

            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Query-ID"] = query_id

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log request failure
            logger.error(
                "request_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=round(latency_ms, 2),
            )

            # Record error metrics
            metrics.record_query(latency_ms, success=False, error_type=type(e).__name__)

            raise

        finally:
            # Clean up context
            structlog.contextvars.unbind_contextvars("request_id", "query_id")
            clear_request_context()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware.

    Features:
    - Per-IP rate limiting with sliding window
    - Configurable requests per minute
    - Automatic cleanup of old entries
    - Excludes health and metrics endpoints

    Note: This is suitable for single-instance deployments.
    For distributed deployments, use Redis-based rate limiting.
    """

    # Endpoints to exclude from rate limiting
    EXCLUDED_PATHS = {"/api/v1/health", "/api/v1/metrics", "/", "/docs", "/redoc"}

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        cleanup_interval: int = 100,
    ) -> None:
        """Initialize rate limiter.

        Args:
            app: The FastAPI/Starlette app.
            requests_per_minute: Max requests per IP per minute.
            cleanup_interval: Clean old entries every N requests.
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.request_count = 0
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        with self.lock:
            # Periodic cleanup of old entries
            self.request_count += 1
            if self.request_count % self.cleanup_interval == 0:
                self._cleanup_old_entries(current_time)

            # Get requests in the last minute for this IP
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if current_time - t < 60
            ]

            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                logger.warning(
                    "rate_limit_exceeded",
                    client_ip=client_ip,
                    requests_in_window=len(self.requests[client_ip]),
                    limit=self.requests_per_minute,
                )
                raise RateLimitError(
                    message=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                    details={
                        "limit": self.requests_per_minute,
                        "window_seconds": 60,
                        "retry_after_seconds": 60,
                    },
                )

            # Record this request
            self.requests[client_ip].append(current_time)

        return await call_next(request)

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff = current_time - 60
        empty_ips = []

        for ip, timestamps in self.requests.items():
            self.requests[ip] = [t for t in timestamps if t > cutoff]
            if not self.requests[ip]:
                empty_ips.append(ip)

        # Remove IPs with no recent requests
        for ip in empty_ips:
            del self.requests[ip]
