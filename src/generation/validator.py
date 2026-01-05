"""Response validation for hallucination detection."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.generation.formatter import format_chunks_for_context
from src.generation.models import (
    DetectedHallucination,
    HallucinationType,
    ValidationResult,
)
from src.generation.prompts import HALLUCINATION_DETECTION_PROMPT

logger = structlog.get_logger(__name__)

# Thresholds for validation decisions
REGENERATION_THRESHOLD = 0.7  # Average severity above this triggers regeneration
CORRECTION_THRESHOLD = 0.3  # Average severity above this triggers correction
ACCURACY_PASS_THRESHOLD = 0.8  # Overall accuracy below this fails validation


class ResponseValidator:
    """Validates generated responses for hallucinations.

    Uses LLM-based semantic verification to detect:
    - Unsupported claims not found in sources
    - Misquoted or paraphrased text that changes meaning
    - Fabricated citations
    - Outdated information
    - Misattributed claims
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-haiku-3-5-20241022",
        max_tokens: int = 4096,
    ):
        """Initialize the validator.

        Args:
            client: Anthropic client instance. If None, one will be created.
            model: Model ID to use. Haiku recommended for cost efficiency.
            max_tokens: Maximum tokens in the validation response.
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _ensure_client(self) -> Any:
        """Ensure Anthropic client is available."""
        if self.client is None:
            import anthropic

            from config.settings import get_settings

            settings = get_settings()
            self.client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self.client

    async def validate_response(
        self,
        response_text: str,
        query: str,
        chunks: list[dict],
    ) -> ValidationResult:
        """Validate a generated response for hallucinations.

        Args:
            response_text: The generated answer to validate.
            query: The original user query.
            chunks: Source chunks used for generation.

        Returns:
            ValidationResult with detected hallucinations and accuracy score.
        """
        logger.info(
            "validation_started",
            response_length=len(response_text),
            chunk_count=len(chunks),
        )

        # Handle empty response
        if not response_text or not response_text.strip():
            return ValidationResult(
                hallucinations=[],
                verified_claims=[],
                overall_accuracy=0.0,
                needs_regeneration=True,
                needs_correction=False,
                validation_metadata={"error": "Empty response"},
            )

        # Format context for the prompt
        formatted_context = format_chunks_for_context(chunks)

        prompt = HALLUCINATION_DETECTION_PROMPT.format(
            context=formatted_context,
            response=response_text,
            query=query,
        )

        try:
            client = self._ensure_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            result_data = self._parse_validation_response(result_text)

            # Convert to ValidationResult
            hallucinations = []
            for h in result_data.get("hallucinations", []):
                try:
                    hallucination_type = HallucinationType(
                        h.get("hallucination_type", "unsupported_claim")
                    )
                except ValueError:
                    hallucination_type = HallucinationType.UNSUPPORTED_CLAIM

                hallucinations.append(
                    DetectedHallucination(
                        claim_text=h.get("claim_text", ""),
                        hallucination_type=hallucination_type,
                        cited_source=h.get("cited_source"),
                        actual_source_text=h.get("actual_source_text"),
                        severity=float(h.get("severity", 0.5)),
                        reasoning=h.get("reasoning", ""),
                        suggested_correction=h.get("suggested_correction"),
                    )
                )

            overall_accuracy = float(result_data.get("overall_accuracy", 1.0))
            verified_claims = result_data.get("verified_claims", [])

            # Determine if regeneration or correction is needed
            avg_severity = (
                sum(h.severity for h in hallucinations) / len(hallucinations)
                if hallucinations
                else 0.0
            )
            max_severity = max((h.severity for h in hallucinations), default=0.0)

            needs_regeneration = (
                max_severity >= 0.9  # Any critical error
                or avg_severity >= REGENERATION_THRESHOLD
                or overall_accuracy < 0.5
            )
            needs_correction = not needs_regeneration and (
                avg_severity >= CORRECTION_THRESHOLD
                or overall_accuracy < ACCURACY_PASS_THRESHOLD
            )

            validation_result = ValidationResult(
                hallucinations=hallucinations,
                verified_claims=verified_claims,
                overall_accuracy=overall_accuracy,
                needs_regeneration=needs_regeneration,
                needs_correction=needs_correction,
                validation_metadata={
                    "model": self.model,
                    "avg_severity": avg_severity,
                    "max_severity": max_severity,
                    "hallucination_count": len(hallucinations),
                    "verified_claim_count": len(verified_claims),
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

            logger.info(
                "validation_completed",
                accuracy=overall_accuracy,
                hallucination_count=len(hallucinations),
                needs_regeneration=needs_regeneration,
                needs_correction=needs_correction,
            )

            return validation_result

        except Exception as e:
            logger.error("validation_failed", error=str(e))
            # On validation failure, return a passing result with warning (fail-open)
            return ValidationResult(
                hallucinations=[],
                verified_claims=[],
                overall_accuracy=0.5,  # Unknown accuracy
                needs_regeneration=False,
                needs_correction=False,
                validation_metadata={"error": str(e), "model": self.model},
            )

    def _parse_validation_response(self, response_text: str) -> dict:
        """Parse JSON from validation LLM response.

        Args:
            response_text: Raw LLM response text.

        Returns:
            Parsed JSON dict.
        """
        # Handle markdown code blocks
        text = response_text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    text = stripped[4:].strip()
                    break
                elif stripped.startswith("{"):
                    text = stripped
                    break

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("validation_json_parse_failed", text=response_text[:200])
            return {
                "hallucinations": [],
                "verified_claims": [],
                "overall_accuracy": 0.5,
            }

    async def detect_hallucinations(
        self,
        response_text: str,
        query: str,
        chunks: list[dict],
    ) -> list[DetectedHallucination]:
        """Convenience method to just get hallucinations list.

        Args:
            response_text: The generated answer to validate.
            query: The original user query.
            chunks: Source chunks used for generation.

        Returns:
            List of detected hallucinations.
        """
        result = await self.validate_response(response_text, query, chunks)
        return result.hallucinations
