"""FastAPI application for the Florida Tax RAG system."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import cleanup_clients, get_neo4j_client, get_weaviate_client
from .routes import router

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Context Manager
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Startup:
    - Initialize database connections
    - Warm up connection pools
    - Compile agent graph (via dependency)

    Shutdown:
    - Close database connections
    - Cleanup resources
    """
    logger.info("Starting Florida Tax RAG API...")

    # Warm up connections
    try:
        neo4j = get_neo4j_client()
        weaviate = get_weaviate_client()

        # Verify connectivity
        if not neo4j.health_check():
            logger.warning("Neo4j not available at startup")
        else:
            logger.info("Neo4j connection established")

        if not weaviate.health_check():
            logger.warning("Weaviate not available at startup")
        else:
            logger.info("Weaviate connection established")

    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        # Continue anyway - health endpoint will report status

    logger.info("Florida Tax RAG API ready")

    yield

    # Shutdown
    logger.info("Shutting down Florida Tax RAG API...")
    await cleanup_clients()
    logger.info("Cleanup complete")


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
    # CORS Middleware
    # ==========================================================================

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

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "code": "VALIDATION_ERROR",
                    "message": error["msg"],
                    "field": ".".join(str(loc) for loc in error["loc"]),
                }
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request.headers.get("X-Request-ID", "unknown"),
                "error": "Validation failed",
                "details": errors,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "request_id": request.headers.get("X-Request-ID", "unknown"),
                "error": "Internal server error",
                "details": [{"code": "INTERNAL_ERROR", "message": str(exc)}],
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
