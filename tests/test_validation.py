"""Tests for hallucination detection and correction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.corrector import ResponseCorrector
from src.generation.models import (
    CorrectionResult,
    DetectedHallucination,
    HallucinationType,
    ValidationResult,
)
from src.generation.validator import ResponseValidator


class TestResponseValidator:
    """Tests for ResponseValidator."""

    @pytest.mark.asyncio
    async def test_validate_clean_response(self):
        """Test that a fully-supported response passes validation."""
        validator = ResponseValidator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        {
            "hallucinations": [],
            "verified_claims": [
                "Florida imposes a 6% state sales tax per 212.05",
                "Certain food items are exempt under 212.08"
            ],
            "overall_accuracy": 1.0
        }
        """
            )
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 100

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="Florida has a 6% sales tax [Source: 212.05]",
                query="What is the sales tax rate?",
                chunks=[{"text": "Florida imposes 6% sales tax...", "citation": "212.05"}],
            )

        assert isinstance(result, ValidationResult)
        assert result.overall_accuracy == 1.0
        assert len(result.hallucinations) == 0
        assert not result.needs_regeneration
        assert not result.needs_correction

    @pytest.mark.asyncio
    async def test_detect_hallucinated_citation(self):
        """Test detection of fabricated citation."""
        validator = ResponseValidator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        {
            "hallucinations": [
                {
                    "claim_text": "per Section 999.99",
                    "hallucination_type": "fabricated_citation",
                    "cited_source": "999.99",
                    "actual_source_text": null,
                    "severity": 0.9,
                    "reasoning": "Section 999.99 does not exist in Florida Statutes",
                    "suggested_correction": null
                }
            ],
            "verified_claims": [],
            "overall_accuracy": 0.3
        }
        """
            )
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 150

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="The tax rate is 10% per Section 999.99",
                query="What is the tax rate?",
                chunks=[{"text": "6% sales tax...", "citation": "212.05"}],
            )

        assert len(result.hallucinations) == 1
        assert (
            result.hallucinations[0].hallucination_type
            == HallucinationType.FABRICATED_CITATION
        )
        assert result.hallucinations[0].severity == 0.9
        assert result.needs_regeneration  # High severity triggers regeneration

    @pytest.mark.asyncio
    async def test_detect_misquoted_text(self):
        """Test detection of misquoted source text."""
        validator = ResponseValidator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        {
            "hallucinations": [
                {
                    "claim_text": "The exemption applies to all food items",
                    "hallucination_type": "misquoted_text",
                    "cited_source": "212.08(1)",
                    "actual_source_text": "The exemption applies only to food for human consumption sold for off-premises consumption",
                    "severity": 0.6,
                    "reasoning": "The response overstates the scope of the exemption",
                    "suggested_correction": "The exemption applies to food for human consumption sold for off-premises consumption"
                }
            ],
            "verified_claims": ["Sales tax rate is 6%"],
            "overall_accuracy": 0.7
        }
        """
            )
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="The exemption applies to all food items [Source: 212.08(1)]",
                query="What food is exempt?",
                chunks=[
                    {
                        "text": "food for human consumption sold for off-premises...",
                        "citation": "212.08(1)",
                    }
                ],
            )

        assert len(result.hallucinations) == 1
        assert (
            result.hallucinations[0].hallucination_type
            == HallucinationType.MISQUOTED_TEXT
        )
        assert result.hallucinations[0].suggested_correction is not None
        assert not result.needs_regeneration  # Medium severity
        assert result.needs_correction

    @pytest.mark.asyncio
    async def test_empty_response_needs_regeneration(self):
        """Test that empty response triggers regeneration."""
        validator = ResponseValidator()

        result = await validator.validate_response(
            response_text="",
            query="What is the tax rate?",
            chunks=[],
        )

        assert result.needs_regeneration
        assert result.overall_accuracy == 0.0

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test graceful handling of validation errors."""
        validator = ResponseValidator()

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(side_effect=Exception("API Error"))
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="Test response",
                query="Test query",
                chunks=[],
            )

        # Should return passing result on error (fail-open)
        assert not result.needs_regeneration
        assert not result.needs_correction
        assert "error" in result.validation_metadata


class TestResponseCorrector:
    """Tests for ResponseCorrector."""

    @pytest.mark.asyncio
    async def test_no_corrections_needed(self):
        """Test that response without hallucinations returns unchanged."""
        corrector = ResponseCorrector()

        validation_result = ValidationResult(
            hallucinations=[],
            verified_claims=["All claims verified"],
            overall_accuracy=1.0,
            needs_correction=False,
            needs_regeneration=False,
        )

        result = await corrector.correct(
            response_text="Original response text",
            validation_result=validation_result,
            chunks=[],
        )

        assert result.corrected_answer == "Original response text"
        assert len(result.corrections_made) == 0
        assert result.confidence_adjustment == 0.0

    @pytest.mark.asyncio
    async def test_simple_correction(self):
        """Test simple text replacement correction."""
        corrector = ResponseCorrector()

        validation_result = ValidationResult(
            hallucinations=[
                DetectedHallucination(
                    claim_text="7% sales tax",
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=0.4,
                    reasoning="Source says 6%",
                    suggested_correction="6% sales tax",
                )
            ],
            verified_claims=["Florida imposes sales tax"],
            overall_accuracy=0.8,
            needs_correction=True,
            needs_regeneration=False,
        )

        result = await corrector.correct(
            response_text="Florida has a 7% sales tax on most goods.",
            validation_result=validation_result,
            chunks=[],
        )

        assert "6% sales tax" in result.corrected_answer
        assert "7%" not in result.corrected_answer
        assert len(result.corrections_made) > 0

    @pytest.mark.asyncio
    async def test_llm_correction_for_complex_case(self):
        """Test LLM-based correction for complex hallucinations."""
        corrector = ResponseCorrector()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        Florida imposes a 6% state sales tax on most tangible personal property
        [Source: 212.05]. Food for human consumption sold for off-premises
        consumption is generally exempt [Source: 212.08(1)].

        Caveats: The scope of food exemptions has specific conditions that should
        be verified for each situation.
        """
            )
        ]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        corrector.client = mock_client

        validation_result = ValidationResult(
            hallucinations=[
                DetectedHallucination(
                    claim_text="All food is exempt from sales tax",
                    hallucination_type=HallucinationType.OVERGENERALIZATION,
                    severity=0.7,
                    reasoning="Only specific food categories are exempt",
                    suggested_correction=None,  # Complex, needs LLM
                )
            ],
            verified_claims=[],
            overall_accuracy=0.5,
            needs_correction=True,
            needs_regeneration=False,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.corrector.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await corrector.correct(
                response_text="All food is exempt from sales tax in Florida.",
                validation_result=validation_result,
                chunks=[
                    {"text": "food for human consumption...", "citation": "212.08(1)"}
                ],
            )

        assert "Caveats" in result.corrected_answer or len(result.disclaimers_added) > 0
        assert result.confidence_adjustment < 0

    @pytest.mark.asyncio
    async def test_correction_error_handling(self):
        """Test graceful handling of correction errors."""
        corrector = ResponseCorrector()

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(
            side_effect=Exception("Correction Error")
        )
        corrector.client = mock_client

        validation_result = ValidationResult(
            hallucinations=[
                DetectedHallucination(
                    claim_text="Some claim",
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=0.8,  # High severity, won't use simple correction
                    reasoning="Not in sources",
                    suggested_correction=None,
                )
            ],
            verified_claims=[],
            overall_accuracy=0.5,
            needs_correction=True,
            needs_regeneration=False,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.corrector.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await corrector.correct(
                response_text="Original response",
                validation_result=validation_result,
                chunks=[],
            )

        # Should return original with disclaimer on error
        assert "Original response" in result.corrected_answer
        assert len(result.disclaimers_added) > 0
        assert result.confidence_adjustment < 0


class TestValidationEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_nonexistent_law_query(self):
        """Test handling of queries about laws that don't exist."""
        validator = ResponseValidator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        {
            "hallucinations": [
                {
                    "claim_text": "The cryptocurrency tax rate in Florida is 2%",
                    "hallucination_type": "unsupported_claim",
                    "severity": 1.0,
                    "reasoning": "Florida does not have a specific cryptocurrency tax",
                    "suggested_correction": null
                }
            ],
            "verified_claims": [],
            "overall_accuracy": 0.0
        }
        """
            )
        ]
        mock_response.usage.input_tokens = 300
        mock_response.usage.output_tokens = 100

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="The cryptocurrency tax rate in Florida is 2%",
                query="What is Florida's crypto tax?",
                chunks=[],  # No relevant chunks
            )

        assert result.overall_accuracy == 0.0
        assert result.needs_regeneration

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_claims(self):
        """Test response with both valid and invalid claims."""
        validator = ResponseValidator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
        {
            "hallucinations": [
                {
                    "claim_text": "plus a mandatory 2% local tax",
                    "hallucination_type": "unsupported_claim",
                    "severity": 0.5,
                    "reasoning": "Local surtaxes vary by county and are not mandatory statewide",
                    "suggested_correction": "plus discretionary local surtaxes that vary by county"
                }
            ],
            "verified_claims": [
                "Florida has a 6% state sales tax",
                "Sales tax applies to tangible personal property"
            ],
            "overall_accuracy": 0.75
        }
        """
            )
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 150

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        validator.client = mock_client

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "src.generation.validator.asyncio.to_thread", side_effect=mock_to_thread
        ):
            result = await validator.validate_response(
                response_text="Florida has 6% state sales tax plus a mandatory 2% local tax",
                query="What is the total sales tax?",
                chunks=[{"text": "6% state rate...", "citation": "212.05"}],
            )

        assert len(result.hallucinations) == 1
        assert len(result.verified_claims) == 2
        assert 0.7 <= result.overall_accuracy <= 0.8
        assert result.needs_correction
        assert not result.needs_regeneration

    def test_json_parsing_with_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        validator = ResponseValidator()

        response_text = """```json
{
    "hallucinations": [],
    "verified_claims": ["Test claim"],
    "overall_accuracy": 0.9
}
```"""

        result = validator._parse_validation_response(response_text)

        assert result["overall_accuracy"] == 0.9
        assert len(result["verified_claims"]) == 1

    def test_json_parsing_with_invalid_json(self):
        """Test handling of invalid JSON."""
        validator = ResponseValidator()

        response_text = "This is not valid JSON"

        result = validator._parse_validation_response(response_text)

        # Should return default values
        assert result["overall_accuracy"] == 0.5
        assert result["hallucinations"] == []


class TestRouteAfterValidation:
    """Tests for route_after_validation edge function."""

    def test_accept_when_validation_passed(self):
        """Test routing to accept when validation passed."""
        from src.agent.edges import route_after_validation

        state = {
            "original_query": "test",
            "validation_passed": True,
        }

        result = route_after_validation(state)
        assert result == "accept"

    def test_regenerate_when_needed_and_under_limit(self):
        """Test routing to regenerate when needed and under limit."""
        from src.agent.edges import route_after_validation

        state = {
            "original_query": "test",
            "validation_passed": False,
            "validation_result": {
                "needs_regeneration": True,
                "needs_correction": False,
            },
            "regeneration_count": 0,
            "max_regenerations": 2,
        }

        result = route_after_validation(state)
        assert result == "regenerate"

    def test_correct_when_regeneration_at_limit(self):
        """Test routing to correct when regeneration limit reached."""
        from src.agent.edges import route_after_validation

        state = {
            "original_query": "test",
            "validation_passed": False,
            "validation_result": {
                "needs_regeneration": True,
                "needs_correction": False,
            },
            "regeneration_count": 2,
            "max_regenerations": 2,
        }

        # Should route to correct instead of regenerate
        result = route_after_validation(state)
        assert result == "correct"

    def test_correct_when_needs_correction(self):
        """Test routing to correct when correction needed."""
        from src.agent.edges import route_after_validation

        state = {
            "original_query": "test",
            "validation_passed": False,
            "validation_result": {
                "needs_regeneration": False,
                "needs_correction": True,
            },
        }

        result = route_after_validation(state)
        assert result == "correct"


class TestValidationNodeIntegration:
    """Integration tests for validation in the agent nodes."""

    @pytest.mark.asyncio
    async def test_validate_response_node_execution(self):
        """Test validate_response node runs correctly."""
        from src.agent.nodes import validate_response

        state = {
            "original_query": "What is the sales tax rate?",
            "final_answer": "Florida has a 6% sales tax [Source: 212.05]",
            "temporally_valid_chunks": [
                {"text": "6% state sales tax", "citation": "212.05", "doc_type": "statute"}
            ],
        }

        # Mock the validator - patch at the source module level
        with patch("src.generation.validator.ResponseValidator") as MockValidator:
            mock_validator = MagicMock()
            mock_result = ValidationResult(
                hallucinations=[],
                verified_claims=["6% sales tax"],
                overall_accuracy=1.0,
                needs_regeneration=False,
                needs_correction=False,
            )
            mock_validator.validate_response = AsyncMock(return_value=mock_result)
            MockValidator.return_value = mock_validator

            result = await validate_response(state)

        assert result["validation_passed"] is True
        assert result["validation_result"]["overall_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_validate_response_skips_empty_answer(self):
        """Test validate_response skips validation for empty answer."""
        from src.agent.nodes import validate_response

        state = {
            "original_query": "What is the sales tax rate?",
            "final_answer": None,
            "temporally_valid_chunks": [],
        }

        result = await validate_response(state)

        assert result["validation_passed"] is False
        assert result["validation_result"] is None

    @pytest.mark.asyncio
    async def test_correct_response_node_execution(self):
        """Test correct_response node runs correctly."""
        from src.agent.nodes import correct_response

        state = {
            "original_query": "What is the tax rate?",
            "final_answer": "The tax rate is 7%",
            "validation_result": {
                "hallucinations": [
                    {
                        "claim_text": "7%",
                        "hallucination_type": "unsupported_claim",
                        "severity": 0.4,
                        "reasoning": "Should be 6%",
                        "suggested_correction": "6%",
                        "cited_source": None,
                        "actual_source_text": None,
                    }
                ],
                "verified_claims": [],
                "overall_accuracy": 0.7,
                "needs_regeneration": False,
                "needs_correction": True,
                "validation_metadata": {},
            },
            "temporally_valid_chunks": [],
            "confidence": 0.8,
        }

        # Mock the corrector - patch at the source module level
        with patch("src.generation.corrector.ResponseCorrector") as MockCorrector:
            mock_corrector = MagicMock()
            mock_result = CorrectionResult(
                corrected_answer="The tax rate is 6%",
                corrections_made=["Changed 7% to 6%"],
                disclaimers_added=[],
                confidence_adjustment=-0.1,
            )
            mock_corrector.correct = AsyncMock(return_value=mock_result)
            MockCorrector.return_value = mock_corrector

            result = await correct_response(state)

        assert "6%" in result["final_answer"]
        assert result["confidence"] < 0.8  # Adjusted down
        assert result["validation_passed"] is True
