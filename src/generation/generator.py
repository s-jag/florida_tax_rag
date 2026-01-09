"""Tax law response generator with citation validation."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import structlog

from config.prompts import CONTEXT_TEMPLATE
from config.prompts import GENERATION_SYSTEM_PROMPT as SYSTEM_PROMPT
from src.generation.formatter import format_chunks_for_context
from src.generation.models import (
    ExtractedCitation,
    GeneratedResponse,
    ValidatedCitation,
)

logger = structlog.get_logger(__name__)

# Regex patterns for citation extraction
CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")

# Patterns to identify citation types and extract section numbers
STATUTE_PATTERN = re.compile(
    r"(?:Fla\.?\s*Stat\.?\s*)?§\s*([\d.]+(?:\([^)]+\))*)",
    re.IGNORECASE,
)
RULE_PATTERN = re.compile(
    r"(?:Rule\s+|F\.?A\.?C\.?\s*R?\.?\s*)([\dA-Za-z\-.]+)",
    re.IGNORECASE,
)
CASE_PATTERN = re.compile(r"v\.\s+\w+|vs\.\s+\w+", re.IGNORECASE)


class TaxLawGenerator:
    """Generate legally-accurate tax law responses with citations.

    This generator:
    1. Formats retrieval chunks into structured context
    2. Calls Claude with a specialized tax attorney system prompt
    3. Extracts citations from the response
    4. Validates citations against provided chunks
    5. Calculates confidence based on source quality and verification
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
    ):
        """Initialize the generator.

        Args:
            client: Anthropic client instance. If None, one will be created.
            model: Model ID to use for generation.
            max_tokens: Maximum tokens in the response.
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
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        return self.client

    async def generate(
        self,
        query: str,
        chunks: list[dict],
        reasoning_steps: list[str] | None = None,
    ) -> GeneratedResponse:
        """Generate a response with validated citations.

        Args:
            query: The user's tax law question.
            chunks: Retrieved chunks to use as context.
            reasoning_steps: Optional reasoning steps from earlier nodes.

        Returns:
            GeneratedResponse with answer, citations, and confidence.
        """
        logger.info(
            "generation_started",
            query_length=len(query),
            chunk_count=len(chunks),
        )
        start_time = time.time()

        # Track which chunks were provided
        chunks_used = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]

        # Handle empty context
        if not chunks:
            logger.warning("generation_no_chunks", query=query[:50])
            return GeneratedResponse(
                answer="I don't have sufficient legal context to answer this question. "
                "Please ensure relevant Florida tax law documents have been retrieved.",
                citations=[],
                chunks_used=[],
                confidence=0.0,
                warnings=["No legal documents provided in context"],
                generation_metadata={
                    "model": self.model,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
            )

        # Format context
        formatted_context = format_chunks_for_context(chunks)
        prompt = CONTEXT_TEMPLATE.format(
            formatted_chunks=formatted_context,
            query=query,
        )

        # Call LLM
        try:
            client = self._ensure_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        except Exception as e:
            logger.error("generation_llm_error", error=str(e))
            return GeneratedResponse(
                answer=f"Error generating response: {str(e)}",
                citations=[],
                chunks_used=chunks_used,
                confidence=0.0,
                warnings=[f"LLM call failed: {str(e)}"],
                generation_metadata={
                    "model": self.model,
                    "error": str(e),
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
            )

        # Extract and validate citations
        extracted_citations = self.extract_citations(answer_text)
        validated_citations = self.validate_citations(extracted_citations, chunks)

        # Calculate confidence
        confidence = self._calculate_confidence(chunks, validated_citations)

        # Generate warnings
        warnings = []
        unverified_count = sum(1 for c in validated_citations if not c.verified)
        if unverified_count > 0:
            warnings.append(
                f"{unverified_count} citation(s) could not be verified against provided sources"
            )

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "generation_completed",
            duration_ms=duration_ms,
            citation_count=len(validated_citations),
            verified_count=len(validated_citations) - unverified_count,
            confidence=confidence,
        )

        return GeneratedResponse(
            answer=answer_text,
            citations=validated_citations,
            chunks_used=chunks_used,
            confidence=confidence,
            warnings=warnings,
            generation_metadata={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": duration_ms,
            },
        )

    def extract_citations(self, response_text: str) -> list[ExtractedCitation]:
        """Extract citations from generated response text.

        Looks for [Source: ...] patterns and identifies citation types.

        Args:
            response_text: The LLM-generated answer text.

        Returns:
            List of ExtractedCitation objects.
        """
        citations = []

        for match in CITATION_PATTERN.finditer(response_text):
            citation_text = match.group(1).strip()
            position = match.start()

            # Determine citation type
            citation_type = self._classify_citation(citation_text)

            citations.append(
                ExtractedCitation(
                    citation_text=citation_text,
                    position=position,
                    citation_type=citation_type,
                )
            )

        logger.debug(
            "citations_extracted",
            count=len(citations),
            types=[c.citation_type for c in citations],
        )

        return citations

    def _classify_citation(self, citation_text: str) -> str:
        """Classify a citation by its type.

        Args:
            citation_text: The extracted citation text.

        Returns:
            Citation type: 'statute', 'rule', 'case', 'taa', or 'unknown'.
        """
        text_lower = citation_text.lower()

        # Check for case patterns FIRST (before statute, since "State" contains "stat")
        if CASE_PATTERN.search(citation_text):
            return "case"

        # Check for TAA patterns
        if "taa" in text_lower or "technical assistance" in text_lower:
            return "taa"

        # Check for statute patterns (§ symbol or explicit "statute" word)
        if "§" in citation_text or "statute" in text_lower:
            return "statute"

        # Check for rule patterns
        if RULE_PATTERN.search(citation_text) or "f.a.c" in text_lower:
            return "rule"

        return "unknown"

    def validate_citations(
        self,
        citations: list[ExtractedCitation],
        chunks: list[dict],
    ) -> list[ValidatedCitation]:
        """Validate extracted citations against source chunks.

        Matches citation section numbers against chunk citations to verify
        the LLM only cited documents that were actually provided.

        Args:
            citations: List of extracted citations.
            chunks: List of chunk dictionaries with citation info.

        Returns:
            List of ValidatedCitation objects with verification status.
        """
        validated = []

        for citation in citations:
            # Try to find a matching chunk
            match_result = self._find_matching_chunk(citation, chunks)

            if match_result:
                chunk_id, chunk_text, doc_type = match_result
                validated.append(
                    ValidatedCitation(
                        citation_text=citation.citation_text,
                        chunk_id=chunk_id,
                        verified=True,
                        raw_text=chunk_text[:500],  # First 500 chars
                        doc_type=doc_type,
                    )
                )
            else:
                # Citation not found - potentially hallucinated
                validated.append(
                    ValidatedCitation(
                        citation_text=citation.citation_text,
                        chunk_id=None,
                        verified=False,
                        raw_text="",
                        doc_type=citation.citation_type,
                    )
                )

        return validated

    def _find_matching_chunk(
        self,
        citation: ExtractedCitation,
        chunks: list[dict],
    ) -> tuple[str, str, str] | None:
        """Find a chunk that matches the extracted citation.

        Uses section number matching to handle variations in citation format.

        Args:
            citation: The extracted citation to match.
            chunks: List of chunks to search.

        Returns:
            Tuple of (chunk_id, text, doc_type) if found, None otherwise.
        """
        citation_text = citation.citation_text

        # Extract section number from citation
        section_number = self._extract_section_number(citation_text)

        # Also extract base section (e.g., "212.05" from "212.05(1)(a)")
        base_section = None
        if section_number:
            # Get the base section number without subsections
            base_match = re.match(r"([\d.]+)", section_number)
            if base_match:
                base_section = base_match.group(1)

        for chunk in chunks:
            chunk_citation = chunk.get("citation", "")
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "")
            doc_type = chunk.get("doc_type", "unknown")

            # Direct match
            if citation_text.lower() in chunk_citation.lower():
                return (chunk_id, chunk_text, doc_type)

            # Section number match
            if section_number:
                chunk_section = self._extract_section_number(chunk_citation)

                # Exact section match
                if chunk_section and section_number in chunk_section:
                    return (chunk_id, chunk_text, doc_type)

                # Base section match (e.g., "212.05(1)" matches "212.05")
                if base_section and chunk_section:
                    chunk_base = re.match(r"([\d.]+)", chunk_section)
                    if chunk_base and base_section == chunk_base.group(1):
                        return (chunk_id, chunk_text, doc_type)

                # Also check chunk_id for section number
                if section_number in chunk_id:
                    return (chunk_id, chunk_text, doc_type)

                # Check chunk_id for base section
                if base_section and base_section in chunk_id:
                    return (chunk_id, chunk_text, doc_type)

        return None

    def _extract_section_number(self, citation_text: str) -> str | None:
        """Extract the section number from a citation.

        Args:
            citation_text: Citation string to parse.

        Returns:
            Section number string or None if not found.
        """
        # Try statute pattern first
        statute_match = STATUTE_PATTERN.search(citation_text)
        if statute_match:
            return statute_match.group(1)

        # Try rule pattern
        rule_match = RULE_PATTERN.search(citation_text)
        if rule_match:
            return rule_match.group(1)

        # Try to find any number pattern like "212.05"
        number_match = re.search(r"(\d+\.\d+(?:\([^)]+\))*)", citation_text)
        if number_match:
            return number_match.group(1)

        return None

    def _calculate_confidence(
        self,
        chunks: list[dict],
        validated_citations: list[ValidatedCitation],
    ) -> float:
        """Calculate confidence score for the response.

        Confidence is based on:
        - Source type weights (statute=1.0, rule=0.9, case=0.7, taa=0.6)
        - Citation verification rate
        - Number of sources used

        Args:
            chunks: Source chunks used in context.
            validated_citations: Validated citations from the response.

        Returns:
            Confidence score between 0 and 1.
        """
        if not chunks:
            return 0.0

        # Source type weights
        doc_type_weights = {
            "statute": 1.0,
            "rule": 0.9,
            "case": 0.7,
            "taa": 0.6,
        }

        # Calculate source quality score (top 5 chunks)
        source_weights = [doc_type_weights.get(c.get("doc_type", ""), 0.5) for c in chunks[:5]]
        source_score = sum(source_weights) / max(len(source_weights), 1)

        # Calculate verification score
        if validated_citations:
            verified_count = sum(1 for c in validated_citations if c.verified)
            verification_score = verified_count / len(validated_citations)
        else:
            # No citations extracted - neutral
            verification_score = 0.5

        # Combine scores (60% source quality, 40% verification)
        confidence = (source_score * 0.6) + (verification_score * 0.4)

        # Ensure bounds
        return max(0.0, min(1.0, confidence))
