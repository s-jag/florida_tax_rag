"""Pipeline profiler for per-stage timing in the query pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from .logging import get_logger

logger = get_logger(__name__)

# Context variable to store profiler instance per request
_profiler_var: ContextVar[Optional["PipelineProfiler"]] = ContextVar(
    "pipeline_profiler", default=None
)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, end_time: Optional[float] = None) -> None:
        """Mark stage as complete and calculate duration."""
        self.end_time = end_time or time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class ProfileSummary:
    """Summary of all profiled stages for a request."""

    request_id: str
    total_ms: float
    stages: dict[str, float]
    stage_order: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "total_ms": round(self.total_ms, 2),
            "stages": {k: round(v, 2) for k, v in self.stages.items()},
            "stage_order": self.stage_order,
            "metadata": self.metadata,
        }


class PipelineProfiler:
    """Profile pipeline stages for performance analysis.

    Uses context variables to maintain state across async operations.
    Thread-safe through contextvars isolation.

    Example:
        # Set up profiler for request
        profiler = PipelineProfiler("request-123")
        set_profiler(profiler)

        # Time a stage
        with profiler.stage("decompose"):
            result = await decompose_query(state)

        # Get summary at end
        summary = profiler.get_summary()
    """

    def __init__(self, request_id: str) -> None:
        """Initialize profiler for a request.

        Args:
            request_id: Unique request identifier for correlation.
        """
        self.request_id = request_id
        self.start_time = time.perf_counter()
        self._stages: list[StageMetrics] = []
        self._current_stage: Optional[StageMetrics] = None
        self._metadata: dict[str, Any] = {}

    @contextmanager
    def stage(
        self, name: str, **metadata: Any
    ) -> Generator[StageMetrics, None, None]:
        """Context manager to time a pipeline stage.

        Args:
            name: Name of the stage (e.g., "decompose", "retrieve", "synthesize")
            **metadata: Additional metadata to attach to this stage

        Yields:
            StageMetrics object for the stage

        Example:
            with profiler.stage("retrieve", sub_query_count=3):
                results = await retriever.retrieve(query)
        """
        stage_metrics = StageMetrics(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        self._current_stage = stage_metrics
        self._stages.append(stage_metrics)

        try:
            yield stage_metrics
        finally:
            stage_metrics.complete()
            self._current_stage = None

            logger.debug(
                "stage_completed",
                stage=name,
                duration_ms=round(stage_metrics.duration_ms or 0, 2),
                request_id=self.request_id,
                **metadata,
            )

    def record_stage(self, name: str, duration_ms: float, **metadata: Any) -> None:
        """Record a stage that was already timed externally.

        Args:
            name: Name of the stage
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        now = time.perf_counter()
        stage = StageMetrics(
            name=name,
            start_time=now - (duration_ms / 1000),
            end_time=now,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self._stages.append(stage)

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to the overall profile.

        Args:
            **kwargs: Key-value pairs to add
        """
        self._metadata.update(kwargs)

    def get_summary(self) -> ProfileSummary:
        """Get summary of all profiled stages.

        Returns:
            ProfileSummary with timing breakdown
        """
        stages_dict = {}
        stage_order = []

        for stage in self._stages:
            if stage.duration_ms is not None:
                # If same stage appears multiple times, sum durations
                if stage.name in stages_dict:
                    stages_dict[stage.name] += stage.duration_ms
                else:
                    stages_dict[stage.name] = stage.duration_ms
                    stage_order.append(stage.name)

        total_ms = (time.perf_counter() - self.start_time) * 1000

        return ProfileSummary(
            request_id=self.request_id,
            total_ms=total_ms,
            stages=stages_dict,
            stage_order=stage_order,
            metadata=self._metadata,
        )

    def get_stage_timings(self) -> dict[str, float]:
        """Get just the stage name to duration mapping.

        Returns:
            Dictionary mapping stage names to duration in ms
        """
        return self.get_summary().stages


def set_profiler(profiler: PipelineProfiler) -> None:
    """Set the profiler for the current async context.

    Args:
        profiler: PipelineProfiler instance
    """
    _profiler_var.set(profiler)


def get_profiler() -> Optional[PipelineProfiler]:
    """Get the profiler for the current async context.

    Returns:
        PipelineProfiler instance or None if not set
    """
    return _profiler_var.get()


def clear_profiler() -> None:
    """Clear the profiler from the current async context."""
    _profiler_var.set(None)


@contextmanager
def profile_request(request_id: str) -> Generator[PipelineProfiler, None, None]:
    """Context manager to set up profiling for a request.

    Args:
        request_id: Unique request identifier

    Yields:
        PipelineProfiler instance

    Example:
        with profile_request("req-123") as profiler:
            with profiler.stage("decompose"):
                await decompose()
            with profiler.stage("retrieve"):
                await retrieve()
            summary = profiler.get_summary()
    """
    profiler = PipelineProfiler(request_id)
    set_profiler(profiler)
    try:
        yield profiler
    finally:
        # Log summary
        summary = profiler.get_summary()
        logger.info(
            "request_profile",
            request_id=request_id,
            total_ms=round(summary.total_ms, 2),
            stages=summary.stages,
        )
        clear_profiler()
