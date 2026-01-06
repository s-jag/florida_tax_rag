"""Observability module for logging, metrics, and tracing."""

from .context import (
    clear_request_context,
    get_context,
    query_id_var,
    request_id_var,
    set_request_context,
)
from .logging import configure_logging, get_logger
from .metrics import MetricsCollector, get_metrics_collector
from .profiler import (
    PipelineProfiler,
    ProfileSummary,
    clear_profiler,
    get_profiler,
    profile_request,
    set_profiler,
)

__all__ = [
    # Context
    "request_id_var",
    "query_id_var",
    "get_context",
    "set_request_context",
    "clear_request_context",
    # Logging
    "configure_logging",
    "get_logger",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    # Profiler
    "PipelineProfiler",
    "ProfileSummary",
    "get_profiler",
    "set_profiler",
    "clear_profiler",
    "profile_request",
]
