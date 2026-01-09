"""Simple in-memory metrics collection for the API."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryMetrics:
    """Container for query-related metrics."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    errors_by_type: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Thread-safe singleton metrics collector.

    Collects basic API metrics including:
    - Query counts (total, success, failure)
    - Latency statistics (avg, min, max)
    - Error breakdown by type

    Example:
        metrics = get_metrics_collector()
        metrics.record_query(latency_ms=150.5, success=True)
        stats = metrics.get_stats()
    """

    _instance: MetricsCollector | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> MetricsCollector:
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._metrics = QueryMetrics()
                    instance._metrics_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def record_query(
        self,
        latency_ms: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record metrics for a completed query.

        Args:
            latency_ms: Request latency in milliseconds.
            success: Whether the query succeeded.
            error_type: Type of error if failed (e.g., "TimeoutError").
        """
        with self._metrics_lock:
            m = self._metrics
            m.total_queries += 1
            m.total_latency_ms += latency_ms

            # Update min/max latency
            if m.min_latency_ms is None or latency_ms < m.min_latency_ms:
                m.min_latency_ms = latency_ms
            if m.max_latency_ms is None or latency_ms > m.max_latency_ms:
                m.max_latency_ms = latency_ms

            if success:
                m.successful_queries += 1
            else:
                m.failed_queries += 1
                if error_type:
                    m.errors_by_type[error_type] = m.errors_by_type.get(error_type, 0) + 1

    def record_error(self, error_type: str) -> None:
        """Record an error without a query (e.g., rate limit before processing).

        Args:
            error_type: Type of error.
        """
        with self._metrics_lock:
            m = self._metrics
            m.total_queries += 1
            m.failed_queries += 1
            m.errors_by_type[error_type] = m.errors_by_type.get(error_type, 0) + 1

    def get_stats(self) -> dict:
        """Get current metrics as a dictionary.

        Returns:
            Dictionary with all metrics suitable for JSON serialization.
        """
        with self._metrics_lock:
            m = self._metrics
            avg_latency = m.total_latency_ms / m.total_queries if m.total_queries > 0 else 0.0
            success_rate = (
                (m.successful_queries / m.total_queries * 100) if m.total_queries > 0 else 0.0
            )
            uptime_seconds = (datetime.utcnow() - m.started_at).total_seconds()

            return {
                "total_queries": m.total_queries,
                "successful_queries": m.successful_queries,
                "failed_queries": m.failed_queries,
                "success_rate_percent": round(success_rate, 2),
                "latency_ms": {
                    "avg": round(avg_latency, 2),
                    "min": round(m.min_latency_ms, 2) if m.min_latency_ms else None,
                    "max": round(m.max_latency_ms, 2) if m.max_latency_ms else None,
                },
                "errors_by_type": dict(m.errors_by_type),
                "uptime_seconds": round(uptime_seconds, 0),
                "started_at": m.started_at.isoformat(),
            }

    def reset(self) -> None:
        """Reset all metrics. Useful for testing."""
        with self._metrics_lock:
            self._metrics = QueryMetrics()


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector instance."""
    return MetricsCollector()
