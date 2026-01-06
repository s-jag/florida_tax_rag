"""Faithfulness evaluation using LLM-as-a-judge.

Checks whether generated claims are actually supported by source documents.
Uses GPT-4 to determine if the source text entails the claim.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

FAITHFULNESS_PROMPT = """You are an expert legal analyst evaluating whether a claim is supported by source text.

Given a CLAIM from an AI-generated answer and the SOURCE TEXT it cites, determine if the source actually supports the claim.

## Evaluation Criteria

1. **SUPPORTED**: The source text directly states or strongly implies the claim. Minor paraphrasing is acceptable.
2. **PARTIALLY_SUPPORTED**: The source text relates to the claim but doesn't fully support it, or the claim overgeneralizes.
3. **NOT_SUPPORTED**: The claim cannot be derived from the source text, or the source says something different.
4. **CONTRADICTED**: The source text directly contradicts the claim.

## Instructions

Analyze the claim against the source text carefully. Consider:
- Does the source text contain the specific facts in the claim?
- Are numbers, rates, dates accurate?
- Is the scope of the claim appropriate (not overgeneralized)?
- Are any qualifiers or conditions preserved?

Respond with a JSON object:
```json
{
  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_SUPPORTED" | "CONTRADICTED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your judgment",
  "specific_issues": ["List any specific inaccuracies or issues found"]
}
```

## CLAIM
{claim}

## SOURCE TEXT
{source_text}

## Your Analysis (JSON only):"""


@dataclass
class FaithfulnessResult:
    """Result of faithfulness check for a single claim."""

    claim: str
    source_citation: str
    source_text: str
    verdict: str  # SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, CONTRADICTED
    confidence: float
    reasoning: str
    specific_issues: list[str] = field(default_factory=list)
    is_faithful: bool = False  # True if SUPPORTED or PARTIALLY_SUPPORTED

    def __post_init__(self):
        self.is_faithful = self.verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED")


@dataclass
class FaithfulnessReport:
    """Aggregated faithfulness results for an answer."""

    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    faithfulness_score: float  # 0-1, weighted score
    results: list[FaithfulnessResult]

    @property
    def has_issues(self) -> bool:
        """Check if any claims have faithfulness issues."""
        return self.unsupported_claims > 0 or self.contradicted_claims > 0


class FaithfulnessChecker:
    """LLM-as-a-judge for checking claim faithfulness."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_concurrent: int = 5,
    ):
        """Initialize the faithfulness checker.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Model to use for evaluation
            max_concurrent: Maximum concurrent API calls
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def check_claim(
        self,
        claim: str,
        source_text: str,
        source_citation: str = "",
    ) -> FaithfulnessResult:
        """Check if a single claim is supported by the source text.

        Args:
            claim: The claim to verify
            source_text: The source text that allegedly supports the claim
            source_citation: Citation reference for the source

        Returns:
            FaithfulnessResult with verdict and reasoning
        """
        prompt = FAITHFULNESS_PROMPT.format(
            claim=claim,
            source_text=source_text,
        )

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )

                content = response.choices[0].message.content or ""

                # Parse JSON response
                result = self._parse_response(content)

                return FaithfulnessResult(
                    claim=claim,
                    source_citation=source_citation,
                    source_text=source_text[:500],  # Truncate for storage
                    verdict=result.get("verdict", "NOT_SUPPORTED"),
                    confidence=result.get("confidence", 0.5),
                    reasoning=result.get("reasoning", ""),
                    specific_issues=result.get("specific_issues", []),
                )

            except Exception as e:
                # Return uncertain result on error
                return FaithfulnessResult(
                    claim=claim,
                    source_citation=source_citation,
                    source_text=source_text[:500],
                    verdict="NOT_SUPPORTED",
                    confidence=0.0,
                    reasoning=f"Error during evaluation: {str(e)}",
                    specific_issues=["Evaluation failed"],
                )

    def _parse_response(self, content: str) -> dict:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try parsing entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Default response if parsing fails
            return {
                "verdict": "NOT_SUPPORTED",
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM response",
                "specific_issues": ["Response parsing error"],
            }

    async def check_claims(
        self,
        claims: list[dict],
    ) -> FaithfulnessReport:
        """Check faithfulness of multiple claims.

        Args:
            claims: List of dicts with 'claim', 'source_text', 'source_citation'

        Returns:
            FaithfulnessReport with aggregated results
        """
        if not claims:
            return FaithfulnessReport(
                total_claims=0,
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                faithfulness_score=1.0,
                results=[],
            )

        # Check all claims concurrently
        tasks = [
            self.check_claim(
                claim=c["claim"],
                source_text=c["source_text"],
                source_citation=c.get("source_citation", ""),
            )
            for c in claims
        ]

        results = await asyncio.gather(*tasks)

        # Aggregate results
        supported = sum(1 for r in results if r.verdict == "SUPPORTED")
        partial = sum(1 for r in results if r.verdict == "PARTIALLY_SUPPORTED")
        unsupported = sum(1 for r in results if r.verdict == "NOT_SUPPORTED")
        contradicted = sum(1 for r in results if r.verdict == "CONTRADICTED")

        # Calculate weighted score
        # SUPPORTED = 1.0, PARTIALLY = 0.5, NOT_SUPPORTED = 0.0, CONTRADICTED = -0.5
        total = len(results)
        score = (
            supported * 1.0 +
            partial * 0.5 +
            unsupported * 0.0 +
            contradicted * -0.5
        ) / total if total > 0 else 1.0

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return FaithfulnessReport(
            total_claims=total,
            supported_claims=supported,
            partially_supported_claims=partial,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            faithfulness_score=score,
            results=list(results),
        )


def extract_claims_from_answer(
    answer: str,
    citations: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """Extract claim-source pairs from an answer for faithfulness checking.

    Args:
        answer: The generated answer text
        citations: List of citation dicts with 'citation', 'doc_id', etc.
        chunks: List of source chunks with 'chunk_id', 'text', etc.

    Returns:
        List of dicts with 'claim', 'source_text', 'source_citation'
    """
    claims = []

    # Create chunk lookup by various IDs
    chunk_lookup = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        doc_id = chunk.get("doc_id", "")
        citation = chunk.get("citation", "")

        chunk_lookup[chunk_id] = chunk
        chunk_lookup[doc_id] = chunk
        if citation:
            chunk_lookup[citation] = chunk

    # Extract sentences with citations
    # Pattern: sentence ending with [Source: X]
    citation_pattern = r"([^.!?]+[.!?])\s*\[Source:\s*([^\]]+)\]"

    for match in re.finditer(citation_pattern, answer):
        sentence = match.group(1).strip()
        source_ref = match.group(2).strip()

        # Try to find the source text
        source_text = ""
        source_citation = source_ref

        # Look up in chunks
        for key in [source_ref, f"statute:{source_ref}", f"rule:{source_ref}"]:
            if key in chunk_lookup:
                source_text = chunk_lookup[key].get("text", "")
                break

        # Also check citations list
        if not source_text:
            for cit in citations:
                if source_ref in cit.get("citation", "") or source_ref in cit.get("doc_id", ""):
                    doc_id = cit.get("doc_id", "")
                    if doc_id in chunk_lookup:
                        source_text = chunk_lookup[doc_id].get("text", "")
                        break

        if source_text:
            claims.append({
                "claim": sentence,
                "source_text": source_text,
                "source_citation": source_citation,
            })

    return claims


async def check_answer_faithfulness(
    answer: str,
    citations: list[dict],
    chunks: list[dict],
    checker: Optional[FaithfulnessChecker] = None,
) -> FaithfulnessReport:
    """Check faithfulness of an entire answer.

    Args:
        answer: The generated answer text
        citations: List of citations from the answer
        chunks: Source chunks used for generation
        checker: FaithfulnessChecker instance (creates one if not provided)

    Returns:
        FaithfulnessReport with all claim results
    """
    if checker is None:
        checker = FaithfulnessChecker()

    # Extract claims
    claims = extract_claims_from_answer(answer, citations, chunks)

    # Check faithfulness
    return await checker.check_claims(claims)
