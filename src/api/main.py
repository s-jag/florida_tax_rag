"""FastAPI application for the Florida Tax RAG system."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.observability.context import request_id_var
from src.observability.logging import configure_logging, get_logger
from src.observability.metrics import get_metrics_collector

from .dependencies import cleanup_clients, get_neo4j_client, get_weaviate_client
from .errors import RateLimitError, TaxRAGError
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .routes import router

logger = get_logger(__name__)


# =============================================================================
# Lifespan Context Manager
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Startup:
    - Configure structured logging
    - Initialize database connections
    - Warm up connection pools
    - Compile agent graph (via dependency)

    Shutdown:
    - Close database connections
    - Cleanup resources
    """
    # Configure structured logging
    # Use JSON output in production, console output in development
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    log_level = os.getenv("LOG_LEVEL", "INFO")
    configure_logging(json_output=is_production, log_level=log_level)

    logger.info("api_startup_started", environment="production" if is_production else "development")

    # Warm up connections
    try:
        neo4j = get_neo4j_client()
        weaviate = get_weaviate_client()

        # Verify connectivity
        if not neo4j.health_check():
            logger.warning("neo4j_not_available", status="unavailable")
        else:
            logger.info("neo4j_connected", status="connected")

        if not weaviate.health_check():
            logger.warning("weaviate_not_available", status="unavailable")
        else:
            logger.info("weaviate_connected", status="connected")

    except Exception as e:
        logger.error("connection_initialization_failed", error=str(e), error_type=type(e).__name__)
        # Continue anyway - health endpoint will report status

    logger.info("api_startup_complete", status="ready")

    yield

    # Shutdown
    logger.info("api_shutdown_started")
    await cleanup_clients()
    logger.info("api_shutdown_complete")


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Florida Tax RAG API",
        description="Agentic RAG system for Florida tax law questions",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ==========================================================================
    # Middleware Stack (order matters - last added runs first)
    # ==========================================================================

    # CORS must be added first (runs last in request processing)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Alternative frontend
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

    # Request logging middleware (runs first, logs all requests)
    app.add_middleware(RequestLoggingMiddleware)

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(TaxRAGError)
    async def handle_tax_rag_error(
        request: Request, exc: TaxRAGError
    ) -> JSONResponse:
        """Handle custom TaxRAG exceptions."""
        request_id = request_id_var.get("unknown")

        logger.error(
            "tax_rag_error",
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            status_code=exc.status_code,
        )

        # Record error in metrics
        metrics = get_metrics_collector()
        metrics.record_error(exc.error_code)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(RateLimitError)
    async def handle_rate_limit_error(
        request: Request, exc: RateLimitError
    ) -> JSONResponse:
        """Handle rate limit exceeded errors with Retry-After header."""
        request_id = request_id_var.get("unknown")

        logger.warning(
            "rate_limit_error",
            message=exc.message,
            client_ip=request.client.host if request.client else "unknown",
        )

        # Record error in metrics
        metrics = get_metrics_collector()
        metrics.record_error("RATE_LIMIT_EXCEEDED")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat(),
            },
            headers={"Retry-After": str(exc.details.get("retry_after_seconds", 60))},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        request_id = request_id_var.get("unknown")

        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "code": "VALIDATION_ERROR",
                    "message": error["msg"],
                    "field": ".".join(str(loc) for loc in error["loc"]),
                }
            )

        logger.warning(
            "validation_error",
            errors=errors,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request_id,
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = request_id_var.get("unknown")

        logger.exception(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

        # Record error in metrics
        metrics = get_metrics_collector()
        metrics.record_error(type(exc).__name__)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "request_id": request_id,
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error_type": type(exc).__name__},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # ==========================================================================
    # Include Routers
    # ==========================================================================

    app.include_router(router, prefix="/api/v1", tags=["RAG"])

    # ==========================================================================
    # Root Endpoint
    # ==========================================================================

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Florida Tax RAG API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create app instance for uvicorn
app = create_app()
