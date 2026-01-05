"""Response correction for hallucination removal."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import structlog

from src.generation.formatter import format_chunks_for_context
from src.generation.models import (
    CorrectionResult,
    DetectedHallucination,
    ValidationResult,
)
from src.generation.prompts import CORRECTION_PROMPT

logger = structlog.get_logger(__name__)


@dataclass
class SimpleCorrectionsResult:
    """Result of simple correction attempt."""

    all_corrected: bool
    result: CorrectionResult | None = None
    remaining_hallucinations: list[DetectedHallucination] | None = None


class ResponseCorrector:
    """Corrects responses by removing or fixing hallucinated content.

    Strategies:
    1. For correctable claims: Replace with suggested correction
    2. For uncorrectable claims: Remove and add disclaimer
    3. For severe cases: Flag for regeneration instead
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
    ):
        """Initialize the corrector.

        Args:
            client: Anthropic client instance.
            model: Model ID for correction generation.
            max_tokens: Maximum tokens in corrected response.
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

    async def correct(
        self,
        response_text: str,
        validation_result: ValidationResult,
        chunks: list[dict],
    ) -> CorrectionResult:
        """Correct a response based on validation results.

        Args:
            response_text: The original response to correct.
            validation_result: Results from ResponseValidator.
            chunks: Source chunks for reference.

        Returns:
            CorrectionResult with corrected answer.
        """
        logger.info(
            "correction_started",
            hallucination_count=len(validation_result.hallucinations),
            original_accuracy=validation_result.overall_accuracy,
        )

        # If no hallucinations, return original
        if not validation_result.hallucinations:
            return CorrectionResult(
                corrected_answer=response_text,
                corrections_made=[],
                disclaimers_added=[],
                confidence_adjustment=0.0,
            )

        # Try simple corrections first for low-severity issues
        simple_corrections = self._apply_simple_corrections(
            response_text,
            validation_result.hallucinations,
        )

        if simple_corrections.all_corrected:
            return simple_corrections.result  # type: ignore

        # Use LLM for complex corrections
        return await self._llm_correct(
            response_text,
            validation_result,
            chunks,
        )

    def _apply_simple_corrections(
        self,
        response_text: str,
        hallucinations: list[DetectedHallucination],
    ) -> SimpleCorrectionsResult:
        """Apply simple string-based corrections.

        For hallucinations with suggested_correction and low severity,
        we can do direct text replacement.
        """
        corrected = response_text
        corrections_made = []
        remaining = []

        for h in hallucinations:
            if h.suggested_correction and h.severity < 0.5:
                # Direct replacement
                if h.claim_text in corrected:
                    corrected = corrected.replace(
                        h.claim_text,
                        h.suggested_correction,
                        1,  # Only first occurrence
                    )
                    claim_preview = (
                        h.claim_text[:50] + "..."
                        if len(h.claim_text) > 50
                        else h.claim_text
                    )
                    correction_preview = (
                        h.suggested_correction[:50] + "..."
                        if len(h.suggested_correction) > 50
                        else h.suggested_correction
                    )
                    corrections_made.append(
                        f"Corrected: '{claim_preview}' -> '{correction_preview}'"
                    )
                else:
                    remaining.append(h)
            else:
                remaining.append(h)

        all_corrected = len(remaining) == 0

        if all_corrected:
            # Calculate confidence adjustment based on corrections made
            confidence_adjustment = -0.1 * len(corrections_made)

            return SimpleCorrectionsResult(
                all_corrected=True,
                result=CorrectionResult(
                    corrected_answer=corrected,
                    corrections_made=corrections_made,
                    disclaimers_added=[],
                    confidence_adjustment=max(confidence_adjustment, -0.3),
                ),
            )

        return SimpleCorrectionsResult(
            all_corrected=False,
            result=None,
            remaining_hallucinations=remaining,
        )

    async def _llm_correct(
        self,
        response_text: str,
        validation_result: ValidationResult,
        chunks: list[dict],
    ) -> CorrectionResult:
        """Use LLM to correct complex hallucinations."""

        # Format hallucinations for prompt
        hallucination_desc = "\n".join(
            [
                f"- Claim: \"{h.claim_text}\"\n"
                f"  Type: {h.hallucination_type.value}\n"
                f"  Severity: {h.severity}\n"
                f"  Issue: {h.reasoning}\n"
                f"  Suggestion: {h.suggested_correction or 'Remove claim'}"
                for h in validation_result.hallucinations
            ]
        )

        formatted_context = format_chunks_for_context(chunks)

        prompt = CORRECTION_PROMPT.format(
            response=response_text,
            hallucinations=hallucination_desc,
            context=formatted_context,
        )

        try:
            client = self._ensure_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            corrected_answer = response.content[0].text

            # Extract any disclaimers that were added
            disclaimers = self._extract_disclaimers(corrected_answer)

            # Calculate confidence adjustment
            avg_severity = sum(
                h.severity for h in validation_result.hallucinations
            ) / len(validation_result.hallucinations)
            confidence_adjustment = -0.2 - (avg_severity * 0.3)

            logger.info(
                "correction_completed",
                corrections_made=len(validation_result.hallucinations),
                confidence_adjustment=confidence_adjustment,
            )

            return CorrectionResult(
                corrected_answer=corrected_answer,
                corrections_made=[
                    f"Addressed: {h.claim_text[:50]}..."
                    for h in validation_result.hallucinations
                ],
                disclaimers_added=disclaimers,
                confidence_adjustment=max(confidence_adjustment, -0.5),
            )

        except Exception as e:
            logger.error("correction_failed", error=str(e))
            # Fallback: add disclaimer to original
            disclaimer = (
                "\n\n**Note:** This response could not be fully verified against "
                "available sources. Please consult official Florida tax resources."
            )
            return CorrectionResult(
                corrected_answer=response_text + disclaimer,
                corrections_made=[],
                disclaimers_added=[disclaimer],
                confidence_adjustment=-0.3,
            )

    def _extract_disclaimers(self, text: str) -> list[str]:
        """Extract disclaimer sections from corrected text."""
        disclaimers = []

        # Look for Caveats section
        caveat_match = re.search(
            r"(?:Caveats?|Note|Disclaimer):\s*(.+?)(?=\n\n|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if caveat_match:
            disclaimers.append(caveat_match.group(1).strip())

        return disclaimers
