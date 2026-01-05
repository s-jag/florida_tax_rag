"""Centralized structlog configuration for the application."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog


def configure_logging(
    json_output: bool = True,
    log_level: str = "INFO",
    include_timestamp: bool = True,
) -> None:
    """Configure structlog for the application.

    This should be called once at application startup, typically in
    the FastAPI lifespan context manager.

    Args:
        json_output: If True, output JSON logs (production). If False,
                    use colored console output (development).
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        include_timestamp: Include ISO timestamp in logs.
    """
    # Set up standard logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Build processor chain
    processors: list = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Merge context vars (request_id, query_id)
        structlog.contextvars.merge_contextvars,
        # Handle positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exception info
        structlog.processors.format_exc_info,
        # Add call site info (file, line, function)
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    # Add timestamp if requested
    if include_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    # Add final renderer based on output mode
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        A bound structlog logger that will include context vars.

    Example:
        logger = get_logger(__name__)
        logger.info("processing_started", query="test", chunks=10)
    """
    return structlog.get_logger(name)
