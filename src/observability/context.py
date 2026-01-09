"""Request context propagation using contextvars."""

from __future__ import annotations

from contextvars import ContextVar

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
query_id_var: ContextVar[str] = ContextVar("query_id", default="")


def get_context() -> dict[str, str]:
    """Get current request context for logging.

    Returns a dict with request_id and query_id that can be spread
    into structured log calls.

    Example:
        ctx = get_context()
        logger.info("processing", **ctx, step="decompose")
    """
    ctx = {}
    request_id = request_id_var.get("")
    query_id = query_id_var.get("")

    if request_id:
        ctx["request_id"] = request_id
    if query_id:
        ctx["query_id"] = query_id

    return ctx


def set_request_context(request_id: str, query_id: str | None = None) -> None:
    """Set request context for the current async context.

    Args:
        request_id: Unique request identifier (from header or generated)
        query_id: Short query identifier for tracking through agent nodes
    """
    request_id_var.set(request_id)
    if query_id:
        query_id_var.set(query_id)


def clear_request_context() -> None:
    """Clear request context after request completes."""
    request_id_var.set("")
    query_id_var.set("")
