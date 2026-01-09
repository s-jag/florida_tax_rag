"""Unit tests for Tax Law generation layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.formatter import format_chunk_for_citation, format_chunks_for_context
from src.generation.generator import TaxLawGenerator
from src.generation.models import ExtractedCitation, GeneratedResponse, ValidatedCitation


class TestCitationExtraction:
    """Test citation extraction regex patterns."""

    def test_extract_statute_citation(self):
        """Test extracting statute citation with section symbol."""
        generator = TaxLawGenerator()
        text = "Sales tax is imposed [Source: § 212.05(1)] on tangible goods."
        citations = generator.extract_citations(text)

        assert len(citations) == 1
        assert "212.05" in citations[0].citation_text
        assert citations[0].citation_type == "statute"

    def test_extract_rule_citation(self):
        """Test extracting FAC rule citation."""
        generator = TaxLawGenerator()
        text = "Per [Source: Rule 12A-1.005], this applies."
        citations = generator.extract_citations(text)

        assert len(citations) == 1
        assert "12A-1.005" in citations[0].citation_text
        assert citations[0].citation_type == "rule"

    def test_extract_multiple_citations(self):
        """Test extracting multiple different citation types."""
        generator = TaxLawGenerator()
        text = """Based on [Source: § 212.05] and [Source: Rule 12A-1.001],
                  with case law [Source: FL DOR v. Smith]."""
        citations = generator.extract_citations(text)

        assert len(citations) == 3
        types = [c.citation_type for c in citations]
        assert "statute" in types
        assert "rule" in types

    def test_no_citations(self):
        """Test handling text with no citations."""
        generator = TaxLawGenerator()
        text = "This is general text without citations."
        citations = generator.extract_citations(text)

        assert len(citations) == 0

    def test_extract_fla_stat_format(self):
        """Test extracting full 'Fla. Stat.' format."""
        generator = TaxLawGenerator()
        text = "As stated in [Source: Fla. Stat. § 212.08(7)]."
        citations = generator.extract_citations(text)

        assert len(citations) == 1
        assert "212.08" in citations[0].citation_text
        assert citations[0].citation_type == "statute"

    def test_extract_fac_format(self):
        """Test extracting F.A.C. format."""
        generator = TaxLawGenerator()
        text = "According to [Source: F.A.C. 12A-1.073]."
        citations = generator.extract_citations(text)

        assert len(citations) == 1
        assert "12A-1.073" in citations[0].citation_text
        assert citations[0].citation_type == "rule"

    def test_citation_position_tracked(self):
        """Test that citation position is correctly tracked."""
        generator = TaxLawGenerator()
        text = "Start [Source: § 212.05] middle [Source: Rule 12A-1.001] end."
        citations = generator.extract_citations(text)

        assert len(citations) == 2
        assert citations[0].position < citations[1].position
        assert citations[0].position == text.index("[Source: § 212.05]")


class TestCitationValidation:
    """Test citation validation against source chunks."""

    def test_validate_matching_citation(self):
        """Test that matching citations are verified."""
        generator = TaxLawGenerator()
        citations = [
            ExtractedCitation(
                citation_text="§ 212.05(1)",
                position=0,
                citation_type="statute",
            )
        ]
        chunks = [
            {
                "chunk_id": "chunk:statute:212.05:0",
                "citation": "Fla. Stat. § 212.05",
                "doc_type": "statute",
                "text": "Imposition of sales tax...",
            }
        ]

        validated = generator.validate_citations(citations, chunks)

        assert len(validated) == 1
        assert validated[0].verified is True
        assert validated[0].chunk_id == "chunk:statute:212.05:0"
        assert validated[0].doc_type == "statute"

    def test_flag_hallucinated_citation(self):
        """Test that non-matching citations are flagged as unverified."""
        generator = TaxLawGenerator()
        citations = [
            ExtractedCitation(
                citation_text="§ 999.99",  # Does not exist
                position=0,
                citation_type="statute",
            )
        ]
        chunks = [
            {
                "chunk_id": "chunk:statute:212.05:0",
                "citation": "Fla. Stat. § 212.05",
                "doc_type": "statute",
                "text": "Real text...",
            }
        ]

        validated = generator.validate_citations(citations, chunks)

        assert len(validated) == 1
        assert validated[0].verified is False
        assert validated[0].chunk_id is None

    def test_validate_partial_section_match(self):
        """Test matching when citation has subsection but chunk doesn't."""
        generator = TaxLawGenerator()
        citations = [
            ExtractedCitation(
                citation_text="§ 212.05(1)(a)",
                position=0,
                citation_type="statute",
            )
        ]
        chunks = [
            {
                "chunk_id": "chunk:statute:212.05:0",
                "citation": "Fla. Stat. § 212.05",
                "doc_type": "statute",
                "text": "Sales tax provisions...",
            }
        ]

        validated = generator.validate_citations(citations, chunks)

        assert len(validated) == 1
        # Should match on 212.05 base section
        assert validated[0].verified is True

    def test_validate_multiple_citations(self):
        """Test validating mix of valid and invalid citations."""
        generator = TaxLawGenerator()
        citations = [
            ExtractedCitation(
                citation_text="§ 212.05",
                position=0,
                citation_type="statute",
            ),
            ExtractedCitation(
                citation_text="§ 999.99",
                position=50,
                citation_type="statute",
            ),
            ExtractedCitation(
                citation_text="Rule 12A-1.001",
                position=100,
                citation_type="rule",
            ),
        ]
        chunks = [
            {
                "chunk_id": "chunk:statute:212.05:0",
                "citation": "Fla. Stat. § 212.05",
                "doc_type": "statute",
                "text": "...",
            },
            {
                "chunk_id": "chunk:rule:12A-1.001:0",
                "citation": "Fla. Admin. Code R. 12A-1.001",
                "doc_type": "rule",
                "text": "...",
            },
        ]

        validated = generator.validate_citations(citations, chunks)

        assert len(validated) == 3
        verified_count = sum(1 for c in validated if c.verified)
        assert verified_count == 2  # Two valid, one hallucinated


class TestFormatter:
    """Test chunk formatting utilities."""

    def test_format_single_chunk(self):
        """Test formatting a single chunk."""
        chunks = [
            {
                "doc_type": "statute",
                "citation": "Fla. Stat. § 212.05",
                "effective_date": "2024-01-01",
                "ancestry": "Chapter 212 > § 212.05",
                "text": "Sales tax imposed...",
            }
        ]
        formatted = format_chunks_for_context(chunks)

        assert "STATUTE" in formatted
        assert "212.05" in formatted
        assert "2024-01-01" in formatted
        assert "Chapter 212" in formatted
        assert "Sales tax imposed" in formatted

    def test_format_multiple_chunks(self):
        """Test formatting multiple chunks with different types."""
        chunks = [
            {"doc_type": "statute", "citation": "§ 212.05", "text": "Statute text..."},
            {"doc_type": "rule", "citation": "Rule 12A-1.001", "text": "Rule text..."},
            {"doc_type": "taa", "citation": "TAA 24A-001", "text": "TAA text..."},
        ]
        formatted = format_chunks_for_context(chunks)

        assert "Document 1" in formatted
        assert "Document 2" in formatted
        assert "Document 3" in formatted
        assert "STATUTE" in formatted
        assert "RULE" in formatted
        assert "TAA" in formatted

    def test_format_empty_chunks(self):
        """Test handling empty chunk list."""
        formatted = format_chunks_for_context([])

        assert "No legal documents provided" in formatted

    def test_format_chunk_with_date_object(self):
        """Test handling date objects in effective_date."""
        from datetime import date

        chunks = [
            {
                "doc_type": "statute",
                "citation": "§ 212.05",
                "effective_date": date(2024, 1, 1),
                "text": "Text...",
            }
        ]
        formatted = format_chunks_for_context(chunks)

        assert "2024-01-01" in formatted

    def test_format_chunk_for_citation(self):
        """Test single chunk citation formatting."""
        chunk = {
            "citation": "Fla. Stat. § 212.05",
            "doc_type": "statute",
        }
        formatted = format_chunk_for_citation(chunk, 1)

        assert "[1]" in formatted
        assert "212.05" in formatted
        assert "statute" in formatted


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_high_confidence_all_statutes_verified(self):
        """Test high confidence with all statute sources and verified citations."""
        generator = TaxLawGenerator()
        chunks = [{"doc_type": "statute"}] * 5
        citations = [
            ValidatedCitation(
                citation_text="§ 212.05",
                chunk_id="c1",
                verified=True,
                raw_text="...",
                doc_type="statute",
            )
        ] * 3

        confidence = generator._calculate_confidence(chunks, citations)

        # Source score: 5 statutes * 1.0 / 5 = 1.0
        # Verification: 3/3 = 1.0
        # Combined: 1.0 * 0.6 + 1.0 * 0.4 = 1.0
        assert confidence >= 0.9

    def test_low_confidence_taas_unverified(self):
        """Test low confidence with TAA sources and unverified citations."""
        generator = TaxLawGenerator()
        chunks = [{"doc_type": "taa"}] * 2
        citations = [
            ValidatedCitation(
                citation_text="§ 999.99",
                chunk_id=None,
                verified=False,
                raw_text="",
                doc_type="unknown",
            )
        ]

        confidence = generator._calculate_confidence(chunks, citations)

        # Source score: 2 TAAs * 0.6 / 2 = 0.6
        # Verification: 0/1 = 0.0
        # Combined: 0.6 * 0.6 + 0.0 * 0.4 = 0.36
        assert confidence < 0.5

    def test_medium_confidence_mixed_sources(self):
        """Test medium confidence with mixed source types."""
        generator = TaxLawGenerator()
        chunks = [
            {"doc_type": "statute"},
            {"doc_type": "rule"},
            {"doc_type": "case"},
            {"doc_type": "taa"},
        ]
        citations = [
            ValidatedCitation(
                citation_text="§ 212.05",
                chunk_id="c1",
                verified=True,
                raw_text="...",
                doc_type="statute",
            ),
            ValidatedCitation(
                citation_text="Rule 12A",
                chunk_id=None,
                verified=False,
                raw_text="",
                doc_type="rule",
            ),
        ]

        confidence = generator._calculate_confidence(chunks, citations)

        # Should be between 0.4 and 0.8
        assert 0.4 <= confidence <= 0.8

    def test_confidence_empty_chunks(self):
        """Test confidence is 0 with no chunks."""
        generator = TaxLawGenerator()
        confidence = generator._calculate_confidence([], [])
        assert confidence == 0.0

    def test_confidence_no_citations(self):
        """Test confidence with chunks but no extracted citations."""
        generator = TaxLawGenerator()
        chunks = [{"doc_type": "statute"}] * 3
        confidence = generator._calculate_confidence(chunks, [])

        # Source score counts, verification is neutral (0.5)
        # Should still have reasonable confidence from sources
        assert confidence > 0.3


class TestGeneratorGenerate:
    """Test the generate method with mocked LLM."""

    @pytest.mark.asyncio
    async def test_generate_with_chunks(self):
        """Test generation with valid chunks."""
        generator = TaxLawGenerator()

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="The sales tax rate is 6% [Source: § 212.05]. "
                "Local surtaxes may apply [Source: Rule 12A-1.001]."
            )
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 50

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)

        generator.client = mock_client

        chunks = [
            {
                "chunk_id": "chunk:statute:212.05:0",
                "citation": "Fla. Stat. § 212.05",
                "doc_type": "statute",
                "text": "Sales tax imposed at 6%...",
                "effective_date": "2024-01-01",
                "ancestry": "Chapter 212",
            },
            {
                "chunk_id": "chunk:rule:12A-1.001:0",
                "citation": "Fla. Admin. Code R. 12A-1.001",
                "doc_type": "rule",
                "text": "Definitions for Chapter 12A...",
            },
        ]

        # Patch asyncio.to_thread to directly call the function
        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("src.generation.generator.asyncio.to_thread", side_effect=mock_to_thread):
            result = await generator.generate(
                query="What is the sales tax rate?",
                chunks=chunks,
            )

        assert isinstance(result, GeneratedResponse)
        assert "6%" in result.answer
        assert len(result.citations) == 2
        assert result.chunks_used == [
            "chunk:statute:212.05:0",
            "chunk:rule:12A-1.001:0",
        ]
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_generate_empty_chunks(self):
        """Test generation with no chunks returns appropriate message."""
        generator = TaxLawGenerator()

        result = await generator.generate(
            query="What is the sales tax rate?",
            chunks=[],
        )

        assert isinstance(result, GeneratedResponse)
        assert "sufficient legal context" in result.answer.lower()
        assert result.confidence == 0.0
        assert "No legal documents provided" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self):
        """Test that LLM errors are handled gracefully."""
        generator = TaxLawGenerator()

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(side_effect=Exception("API Error"))
        generator.client = mock_client

        chunks = [
            {
                "chunk_id": "chunk:1",
                "citation": "§ 212.05",
                "doc_type": "statute",
                "text": "...",
            }
        ]

        with patch("asyncio.to_thread", new_callable=lambda: AsyncMock) as mock_thread:
            mock_thread.side_effect = Exception("API Error")

            result = await generator.generate(
                query="Test query",
                chunks=chunks,
            )

        assert isinstance(result, GeneratedResponse)
        assert "Error generating response" in result.answer
        assert result.confidence == 0.0
        assert "LLM call failed" in result.warnings[0]


class TestExtractSectionNumber:
    """Test section number extraction from citations."""

    def test_extract_simple_section(self):
        """Test extracting simple section number."""
        generator = TaxLawGenerator()
        section = generator._extract_section_number("§ 212.05")
        assert section == "212.05"

    def test_extract_section_with_subsection(self):
        """Test extracting section with subsection."""
        generator = TaxLawGenerator()
        section = generator._extract_section_number("§ 212.05(1)(a)")
        assert "212.05" in section

    def test_extract_rule_number(self):
        """Test extracting rule number."""
        generator = TaxLawGenerator()
        section = generator._extract_section_number("Rule 12A-1.073")
        assert "12A-1.073" in section

    def test_extract_fac_number(self):
        """Test extracting F.A.C. rule number."""
        generator = TaxLawGenerator()
        section = generator._extract_section_number("F.A.C. 12A-1.001")
        assert "12A-1.001" in section


class TestClassifyCitation:
    """Test citation type classification."""

    def test_classify_statute(self):
        """Test statute classification."""
        generator = TaxLawGenerator()

        assert generator._classify_citation("§ 212.05") == "statute"
        assert generator._classify_citation("Fla. Stat. § 212.05") == "statute"
        assert generator._classify_citation("Florida Statutes 212.05") == "statute"

    def test_classify_rule(self):
        """Test rule classification."""
        generator = TaxLawGenerator()

        assert generator._classify_citation("Rule 12A-1.001") == "rule"
        assert generator._classify_citation("F.A.C. 12A-1.001") == "rule"

    def test_classify_case(self):
        """Test case classification."""
        generator = TaxLawGenerator()

        assert generator._classify_citation("DOR v. Smith") == "case"
        assert generator._classify_citation("State vs. Jones") == "case"

    def test_classify_taa(self):
        """Test TAA classification."""
        generator = TaxLawGenerator()

        assert generator._classify_citation("TAA 24A-001") == "taa"
        assert generator._classify_citation("Technical Assistance Advisement") == "taa"

    def test_classify_unknown(self):
        """Test unknown citation classification."""
        generator = TaxLawGenerator()

        assert generator._classify_citation("Some random text") == "unknown"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_generation():
    """Test full agent flow with generation.

    This test requires all services (Weaviate, Neo4j, Anthropic) to be running.
    """
    from src.agent import create_tax_agent_graph

    graph = create_tax_agent_graph()

    result = await graph.ainvoke({"original_query": "What is the sales tax rate in Miami?"})

    # Verify answer was generated
    assert result.get("final_answer") is not None
    assert len(result["final_answer"]) > 50

    # Verify citations exist
    citations = result.get("citations", [])
    print(f"Generated {len(citations)} citations:")
    for c in citations:
        print(f"  {c.get('citation')} ({c.get('doc_type')})")

    # Verify confidence is in valid range
    confidence = result.get("confidence", -1)
    assert 0 <= confidence <= 1, f"Confidence {confidence} out of range [0, 1]"
