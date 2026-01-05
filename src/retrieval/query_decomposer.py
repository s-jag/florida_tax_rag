"""Query decomposition using Claude LLM."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, field_validator

from .prompts import CLASSIFICATION_PROMPT, DECOMPOSITION_PROMPT, SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of tax queries."""

    DEFINITION = "definition"  # What is...
    EXEMPTION = "exemption"  # Is X exempt...
    RATE = "rate"  # What is the rate...
    PROCEDURE = "procedure"  # How do I...
    PENALTY = "penalty"  # What is the penalty...
    LOCAL = "local"  # County-specific
    TEMPORAL = "temporal"  # Time-based
    STATUTE = "statute"  # Direct statute reference
    RULE = "rule"  # Direct rule reference
    GENERAL = "general"  # General/other


class SubQuery(BaseModel):
    """A decomposed sub-query."""

    text: str = Field(..., description="The sub-query text for retrieval")
    type: QueryType = Field(..., description="Type of sub-query")
    priority: int = Field(
        default=3, ge=1, le=5, description="Priority 1 (highest) to 5 (lowest)"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Why this sub-query was generated"
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        """Normalize type string to enum value."""
        if isinstance(v, str):
            v = v.lower().strip()
            # Map common variations
            type_map = {
                "def": "definition",
                "exempt": "exemption",
                "rates": "rate",
                "proc": "procedure",
                "penalties": "penalty",
                "county": "local",
                "time": "temporal",
                "stat": "statute",
            }
            return type_map.get(v, v)
        return v


class DecompositionResult(BaseModel):
    """Result of query decomposition."""

    original_query: str = Field(..., description="The original user query")
    sub_queries: list[SubQuery] = Field(
        default_factory=list, description="Decomposed sub-queries"
    )
    reasoning: str = Field(..., description="Explanation of decomposition decision")
    is_simple: bool = Field(
        default=False, description="True if query didn't need decomposition"
    )

    @property
    def query_count(self) -> int:
        """Number of sub-queries generated."""
        return len(self.sub_queries)


class QueryDecomposer:
    """Uses Claude to break complex tax questions into sub-queries.

    This decomposer:
    1. Analyzes query complexity using heuristics
    2. Uses Claude LLM to decompose complex queries
    3. Returns structured sub-queries for parallel retrieval
    """

    # Heuristic thresholds
    MIN_WORDS_FOR_DECOMPOSITION = 8
    COMPLEXITY_KEYWORDS = {
        "and",
        "or",
        "both",
        "also",
        "as well as",
        "in addition",
        "multiple",
        "various",
        "different",
        "compare",
        "versus",
        "vs",
    }

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        """Initialize the query decomposer.

        Args:
            client: Anthropic client (created from settings if None)
            model: Claude model to use
            max_tokens: Maximum tokens for LLM response
        """
        if client is None:
            from config.settings import get_settings

            settings = get_settings()
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose a complex query into sub-queries.

        Args:
            query: User's tax question

        Returns:
            DecompositionResult with sub-queries or is_simple=True
        """
        query = query.strip()

        # Quick heuristic check
        if not self._should_decompose(query):
            logger.info(f"Query too simple for decomposition: {query[:50]}...")
            return DecompositionResult(
                original_query=query,
                sub_queries=[],
                reasoning="Query is simple and direct; no decomposition needed",
                is_simple=True,
            )

        # Use LLM for decomposition
        try:
            result = await self._decompose_with_llm(query)
            logger.info(
                f"Decomposed query into {result.query_count} sub-queries "
                f"(is_simple={result.is_simple})"
            )
            return result
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback: treat as simple query
            return DecompositionResult(
                original_query=query,
                sub_queries=[],
                reasoning=f"Decomposition failed ({e}); treating as simple query",
                is_simple=True,
            )

    async def classify_query(self, query: str) -> QueryType:
        """Classify query type for retrieval optimization.

        Args:
            query: User's tax question

        Returns:
            QueryType classification
        """
        try:
            result = await self._classify_with_llm(query)
            return result
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return QueryType.GENERAL

    def _should_decompose(self, query: str) -> bool:
        """Quick heuristic check if decomposition is needed.

        Args:
            query: User's query

        Returns:
            True if query likely needs decomposition
        """
        # Very short queries don't need decomposition
        words = query.split()
        if len(words) < self.MIN_WORDS_FOR_DECOMPOSITION:
            return False

        # Check for complexity keywords
        query_lower = query.lower()
        has_complexity_keyword = any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS)

        # Check for multiple question marks
        has_multiple_questions = query.count("?") > 1

        # Check for geographic specificity (county names)
        florida_counties = {
            "miami",
            "miami-dade",
            "broward",
            "palm beach",
            "hillsborough",
            "orange",
            "pinellas",
            "duval",
            "lee",
            "polk",
        }
        has_county_reference = any(county in query_lower for county in florida_counties)

        # Decompose if any complexity indicator is present
        return has_complexity_keyword or has_multiple_questions or has_county_reference

    async def _decompose_with_llm(self, query: str) -> DecompositionResult:
        """Use Claude to decompose the query.

        Args:
            query: User's query

        Returns:
            DecompositionResult from LLM
        """
        prompt = DECOMPOSITION_PROMPT.format(query=query)

        # Run sync Anthropic call in executor
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        content = response.content[0].text
        return self._parse_decomposition_response(query, content)

    async def _classify_with_llm(self, query: str) -> QueryType:
        """Use Claude to classify the query type.

        Args:
            query: User's query

        Returns:
            QueryType classification
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip().lower()

        # Map response to QueryType
        try:
            return QueryType(content)
        except ValueError:
            logger.warning(f"Unknown query type '{content}', defaulting to GENERAL")
            return QueryType.GENERAL

    def _parse_decomposition_response(
        self, original_query: str, response: str
    ) -> DecompositionResult:
        """Parse LLM response into DecompositionResult.

        Args:
            original_query: The original user query
            response: Raw LLM response text

        Returns:
            Parsed DecompositionResult
        """
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON
            json_str = response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return as simple query on parse failure
            return DecompositionResult(
                original_query=original_query,
                sub_queries=[],
                reasoning=f"Failed to parse LLM response: {e}",
                is_simple=True,
            )

        # Build sub-queries
        sub_queries = []
        for sq_data in data.get("sub_queries", []):
            try:
                sq = SubQuery(
                    text=sq_data.get("text", ""),
                    type=sq_data.get("type", "general"),
                    priority=sq_data.get("priority", 3),
                )
                if sq.text:  # Only add non-empty sub-queries
                    sub_queries.append(sq)
            except Exception as e:
                logger.warning(f"Failed to parse sub-query {sq_data}: {e}")
                continue

        return DecompositionResult(
            original_query=original_query,
            sub_queries=sub_queries,
            reasoning=data.get("reasoning", ""),
            is_simple=data.get("is_simple", len(sub_queries) == 0),
        )


def create_decomposer(
    model: str = "claude-sonnet-4-20250514",
) -> QueryDecomposer:
    """Factory function to create a QueryDecomposer with default configuration.

    Args:
        model: Claude model to use

    Returns:
        Configured QueryDecomposer instance
    """
    return QueryDecomposer(model=model)
