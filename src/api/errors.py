"""Custom exception hierarchy for the Florida Tax RAG API."""

from __future__ import annotations

from typing import Any


class TaxRAGError(Exception):
    """Base exception for all Tax RAG application errors.

    Attributes:
        error_code: Machine-readable error code for clients.
        status_code: HTTP status code to return.
        message: Human-readable error message.
        details: Additional error context.
    """

    error_code: str = "TAX_RAG_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Additional context (e.g., query, chunk_id).
            error_code: Override the class error code.
            status_code: Override the class status code.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

        if error_code:
            self.error_code = error_code
        if status_code:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class RetrievalError(TaxRAGError):
    """Error during document retrieval from Neo4j or Weaviate.

    Raised when:
    - Neo4j connection fails
    - Weaviate search fails
    - Graph traversal errors
    """

    error_code = "RETRIEVAL_ERROR"
    status_code = 503  # Service Unavailable


class GenerationError(TaxRAGError):
    """Error during LLM generation.

    Raised when:
    - Claude API call fails
    - Response parsing fails
    - Context too long
    """

    error_code = "GENERATION_ERROR"
    status_code = 502  # Bad Gateway (upstream service error)


class ValidationError(TaxRAGError):
    """Error during input or response validation.

    Raised when:
    - Query too short/long
    - Invalid options
    - Response validation fails
    """

    error_code = "VALIDATION_ERROR"
    status_code = 400  # Bad Request


class RateLimitError(TaxRAGError):
    """API rate limit exceeded.

    Raised when:
    - Too many requests from same IP
    - Upstream API rate limited
    """

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429  # Too Many Requests


class QueryTimeoutError(TaxRAGError):
    """Query execution timed out.

    Raised when:
    - Agent graph execution exceeds timeout
    - Database query times out
    """

    error_code = "TIMEOUT"
    status_code = 408  # Request Timeout


class ServiceUnavailableError(TaxRAGError):
    """Required service is unavailable.

    Raised when:
    - Neo4j is down
    - Weaviate is down
    - Redis cache unavailable
    """

    error_code = "SERVICE_UNAVAILABLE"
    status_code = 503  # Service Unavailable


class NotFoundError(TaxRAGError):
    """Requested resource not found.

    Raised when:
    - Chunk ID doesn't exist
    - Statute section not found
    - Document ID not found
    """

    error_code = "NOT_FOUND"
    status_code = 404  # Not Found
