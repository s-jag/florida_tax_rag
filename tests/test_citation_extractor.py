"""Unit tests for citation extraction."""

import pytest

from src.ingestion.citation_extractor import (
    Citation,
    CitationType,
    RelationType,
    detect_relation_type,
    extract_all_citations,
    extract_case_citations,
    extract_chapter_citations,
    extract_rule_citations,
    extract_statute_citations,
    get_context,
)
from src.ingestion.build_citation_graph import (
    CitationIndex,
    ResolvedEdge,
    build_citation_index,
    deduplicate_edges,
    normalize_case_citation_for_index,
    resolve_citation,
)


class TestExtractStatuteCitations:
    """Tests for statute citation extraction."""

    def test_basic_section_symbol(self):
        """Test extraction with § symbol."""
        text = "See § 212.05 for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"
        assert citations[0].section == "212.05"
        assert citations[0].chapter == "212"
        assert citations[0].citation_type == CitationType.STATUTE

    def test_section_symbol_no_space(self):
        """Test extraction with § symbol and no space."""
        text = "See §212.05 for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"

    def test_section_word(self):
        """Test extraction with 'section' word."""
        text = "See Section 212.05 for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"

    def test_section_abbreviation(self):
        """Test extraction with 's.' abbreviation."""
        text = "See s. 212.05 for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"

    def test_fs_suffix(self):
        """Test extraction with 'F.S.' suffix."""
        text = "See 212.05, F.S. for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"

    def test_with_subsection(self):
        """Test extraction with subsection."""
        text = "See § 212.05(1)(a) for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05(1)(a)"
        assert citations[0].subsection == "(1)(a)"

    def test_with_complex_subsection(self):
        """Test extraction with complex subsection."""
        text = "See § 212.08(7)(a)2. for details."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert "(7)(a)" in citations[0].normalized

    def test_multiple_citations(self):
        """Test extraction of multiple citations."""
        text = "See § 212.05 and § 212.08(1). Also see 196.012, F.S."
        citations = extract_statute_citations(text)
        assert len(citations) == 3
        normalized = {c.normalized for c in citations}
        assert "212.05" in normalized
        assert "212.08(1)" in normalized
        assert "196.012" in normalized

    def test_deduplication(self):
        """Test that duplicate citations are deduplicated."""
        text = "See § 212.05 and also § 212.05 again."
        citations = extract_statute_citations(text)
        assert len(citations) == 1

    def test_html_artifact_newline(self):
        """Test handling of 's.\n775.082' artifact from HTML parsing."""
        text = "punishable as provided in s.\n775.082"
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].section == "775.082"

    def test_fla_stat_prefix(self):
        """Test extraction with 'Fla. Stat.' prefix."""
        text = "Pursuant to Fla. Stat. § 212.05."
        citations = extract_statute_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "212.05"


class TestExtractRuleCitations:
    """Tests for rule citation extraction."""

    def test_basic_rule(self):
        """Test basic rule citation extraction."""
        text = "See Rule 12A-1.005 for details."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "12A-1.005"
        assert citations[0].chapter == "12A-1"
        assert citations[0].citation_type == CitationType.RULE

    def test_rule_abbreviation(self):
        """Test extraction with 'R.' abbreviation."""
        text = "See R. 12A-1.005 for details."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "12A-1.005"

    def test_fac_suffix(self):
        """Test extraction with 'F.A.C.' suffix."""
        text = "See 12A-1.005, F.A.C. for details."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "12A-1.005"

    def test_fla_admin_code_prefix(self):
        """Test extraction with 'Fla. Admin. Code' prefix."""
        text = "Pursuant to Fla. Admin. Code R. 12A-1.005."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "12A-1.005"

    def test_with_subsection(self):
        """Test extraction with subsection."""
        text = "See Rule 12A-1.005(1)(a) for details."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert "(1)(a)" in citations[0].normalized

    def test_multiple_rules(self):
        """Test extraction of multiple rule citations."""
        text = "See Rule 12A-1.005 and Rule 12A-1.006(2)."
        citations = extract_rule_citations(text)
        assert len(citations) == 2
        normalized = {c.normalized for c in citations}
        assert "12A-1.005" in normalized
        assert "12A-1.006(2)" in normalized

    def test_lowercase_letter_normalized(self):
        """Test that lowercase letter is normalized to uppercase."""
        text = "See Rule 12a-1.005 for details."
        citations = extract_rule_citations(text)
        assert len(citations) == 1
        assert citations[0].normalized == "12A-1.005"


class TestExtractCaseCitations:
    """Tests for case citation extraction."""

    def test_southern_reporter_2d(self):
        """Test Southern Reporter 2d citation."""
        text = "See Harbor Ventures, Inc. v. Hutches, 366 So.2d 1173, 1174 (Fla. 1979)"
        citations = extract_case_citations(text)
        assert len(citations) >= 1
        assert any("366" in c.normalized for c in citations)
        assert any(c.volume == 366 for c in citations)

    def test_southern_reporter_3d(self):
        """Test Southern Reporter 3d citation."""
        text = "as held in 857 So. 3d 904 (Fla. 3d D.C.A. 2003)"
        citations = extract_case_citations(text)
        assert len(citations) >= 1
        assert any("857" in c.normalized for c in citations)

    def test_lexis_citation(self):
        """Test LEXIS citation."""
        text = "2002 Fla. LEXIS 1162"
        citations = extract_case_citations(text)
        assert len(citations) >= 1
        assert any("2002" in c.normalized and "LEXIS" in c.normalized for c in citations)

    def test_case_with_court_and_year(self):
        """Test extraction captures court and year."""
        text = "612 So. 2d 720 (Fla. 1992)"
        citations = extract_case_citations(text)
        assert len(citations) >= 1
        matching = [c for c in citations if c.volume == 612]
        assert len(matching) >= 1
        assert matching[0].year == 1992


class TestExtractChapterCitations:
    """Tests for chapter reference extraction."""

    def test_single_chapter(self):
        """Test single chapter reference."""
        text = "See Chapter 212 for sales tax provisions."
        citations = extract_chapter_citations(text)
        assert len(citations) == 1
        assert citations[0].chapter == "212"
        assert citations[0].citation_type == CitationType.CHAPTER

    def test_chapter_lowercase(self):
        """Test lowercase 'chapter'."""
        text = "as provided in chapter 196"
        citations = extract_chapter_citations(text)
        assert len(citations) == 1
        assert citations[0].chapter == "196"

    def test_chapter_range(self):
        """Test chapter range reference."""
        text = "as provided in chapters 192-197"
        citations = extract_chapter_citations(text)
        assert len(citations) == 1
        assert "192" in citations[0].normalized
        assert "197" in citations[0].normalized


class TestDetectRelationType:
    """Tests for relation type detection."""

    def test_implements_detection(self):
        """Test detection of 'implements' relation."""
        context = "Law Implemented: 212.05, 212.08, F.S."
        citation = Citation(
            raw_text="212.05",
            normalized="212.05",
            citation_type=CitationType.STATUTE,
        )
        relation = detect_relation_type(citation, context, "rule")
        assert relation == RelationType.IMPLEMENTS

    def test_authority_detection(self):
        """Test detection of 'authority' relation."""
        context = "Rulemaking Authority: 212.08(6), F.S."
        citation = Citation(
            raw_text="212.08(6)",
            normalized="212.08(6)",
            citation_type=CitationType.STATUTE,
        )
        relation = detect_relation_type(citation, context, "rule")
        assert relation == RelationType.AUTHORITY

    def test_specific_authority_detection(self):
        """Test detection of 'specific authority' keyword."""
        context = "Specific Authority: 213.06(1), F.S."
        citation = Citation(
            raw_text="213.06(1)",
            normalized="213.06(1)",
            citation_type=CitationType.STATUTE,
        )
        relation = detect_relation_type(citation, context, "rule")
        assert relation == RelationType.AUTHORITY

    def test_amends_detection(self):
        """Test detection of 'amends' relation."""
        context = "as amended by chapter 2023-45"
        citation = Citation(
            raw_text="chapter 2023-45",
            normalized="chapter:2023-45",
            citation_type=CitationType.CHAPTER,
        )
        relation = detect_relation_type(citation, context, "statute")
        assert relation == RelationType.AMENDS

    def test_supersedes_detection(self):
        """Test detection of 'supersedes' relation."""
        context = "this section supersedes and replaces former s. 212.05"
        citation = Citation(
            raw_text="s. 212.05",
            normalized="212.05",
            citation_type=CitationType.STATUTE,
        )
        relation = detect_relation_type(citation, context, "statute")
        assert relation == RelationType.SUPERSEDES

    def test_default_cites(self):
        """Test default relation is 'cites'."""
        context = "See also § 212.05 for related provisions."
        citation = Citation(
            raw_text="212.05",
            normalized="212.05",
            citation_type=CitationType.STATUTE,
        )
        relation = detect_relation_type(citation, context, "statute")
        assert relation == RelationType.CITES


class TestGetContext:
    """Tests for context extraction."""

    def test_context_window(self):
        """Test context window extraction."""
        text = "A" * 100 + "CITATION" + "B" * 100
        context = get_context(text, 100, 108, window=50)
        assert "CITATION" in context
        assert len(context) <= 108  # 50 + 8 + 50

    def test_context_at_start(self):
        """Test context at start of text."""
        text = "CITATION" + "B" * 100
        context = get_context(text, 0, 8, window=50)
        assert "CITATION" in context

    def test_context_at_end(self):
        """Test context at end of text."""
        text = "A" * 100 + "CITATION"
        context = get_context(text, 100, 108, window=50)
        assert "CITATION" in context


class TestExtractAllCitations:
    """Integration tests for combined extraction."""

    def test_mixed_citations(self):
        """Test extraction of mixed citation types."""
        text = """
        Pursuant to § 212.05, F.S., and Rule 12A-1.073, F.A.C.,
        as interpreted in 366 So.2d 1173 (Fla. 1979),
        the provisions of Chapter 212 apply.
        """
        citations = extract_all_citations(text)

        types = {c.citation_type for c in citations}
        assert CitationType.STATUTE in types
        assert CitationType.RULE in types
        assert CitationType.CASE in types
        assert CitationType.CHAPTER in types


class TestBuildCitationIndex:
    """Tests for citation index building."""

    def test_index_statutes(self):
        """Test indexing statute chunks."""
        chunks = [
            {
                "id": "chunk:statute:212.05:0",
                "doc_id": "statute:212.05",
                "doc_type": "statute",
                "citation": "Fla. Stat. § 212.05",
            },
            {
                "id": "chunk:statute:212.05:1",
                "doc_id": "statute:212.05",
                "doc_type": "statute",
                "citation": "Fla. Stat. § 212.05",
            },
        ]
        index = build_citation_index(chunks)

        assert "statute:212.05" in index.statute_to_chunks
        assert len(index.statute_to_chunks["statute:212.05"]) == 2
        assert "212.05" in index.section_to_doc

    def test_index_rules(self):
        """Test indexing rule chunks."""
        chunks = [
            {
                "id": "chunk:rule:12A-1.005:0",
                "doc_id": "rule:12A-1.005",
                "doc_type": "rule",
                "citation": "Fla. Admin. Code R. 12A-1.005",
            },
        ]
        index = build_citation_index(chunks)

        assert "rule:12A-1.005" in index.rule_to_chunks
        assert "12A-1.005" in index.rule_to_doc

    def test_index_cases(self):
        """Test indexing case chunks with citation lookup."""
        chunks = [
            {
                "id": "chunk:case:1234567:0",
                "doc_id": "case:1234567",
                "doc_type": "case",
                "citation": "State v. Taxpayer, 366 So. 2d 1173",
            },
        ]
        index = build_citation_index(chunks)

        assert "case:1234567" in index.case_to_chunks
        # Normalized citation should map to doc_id
        assert "366 so 2d 1173" in index.case_citation_to_id


class TestResolveCitation:
    """Tests for citation resolution."""

    def test_resolve_statute_exact(self):
        """Test exact statute resolution."""
        index = CitationIndex()
        index.statute_to_chunks["statute:212.05"] = ["chunk:statute:212.05:0"]
        index.section_to_doc["212.05"] = "statute:212.05"

        citation = Citation(
            raw_text="§ 212.05",
            normalized="212.05",
            citation_type=CitationType.STATUTE,
            section="212.05",
        )

        doc_id, chunk_id, confidence = resolve_citation(citation, index)

        assert doc_id == "statute:212.05"
        assert chunk_id == "chunk:statute:212.05:0"
        assert confidence == 1.0

    def test_resolve_statute_partial(self):
        """Test partial statute resolution (with subsection)."""
        index = CitationIndex()
        index.statute_to_chunks["statute:212.08"] = ["chunk:statute:212.08:0"]
        index.section_to_doc["212.08"] = "statute:212.08"

        citation = Citation(
            raw_text="§ 212.08(7)(a)",
            normalized="212.08(7)(a)",
            citation_type=CitationType.STATUTE,
            section="212.08(7)(a)",
        )

        doc_id, chunk_id, confidence = resolve_citation(citation, index)

        assert doc_id == "statute:212.08"
        assert confidence == 0.8

    def test_resolve_rule_exact(self):
        """Test exact rule resolution."""
        index = CitationIndex()
        index.rule_to_chunks["rule:12A-1.005"] = ["chunk:rule:12A-1.005:0"]
        index.rule_to_doc["12A-1.005"] = "rule:12A-1.005"

        citation = Citation(
            raw_text="Rule 12A-1.005",
            normalized="12A-1.005",
            citation_type=CitationType.RULE,
            section="12A-1.005",
        )

        doc_id, chunk_id, confidence = resolve_citation(citation, index)

        assert doc_id == "rule:12A-1.005"
        assert confidence == 1.0

    def test_resolve_unresolvable(self):
        """Test citation that cannot be resolved."""
        index = CitationIndex()

        citation = Citation(
            raw_text="§ 999.999",
            normalized="999.999",
            citation_type=CitationType.STATUTE,
            section="999.999",
        )

        doc_id, chunk_id, confidence = resolve_citation(citation, index)

        assert doc_id is None
        assert chunk_id is None
        assert confidence == 0.0


class TestDeduplicateEdges:
    """Tests for edge deduplication."""

    def test_deduplicate_keeps_highest_confidence(self):
        """Test that deduplication keeps highest confidence edge."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:statute:192.001:0",
                source_doc_id="statute:192.001",
                target_doc_id="statute:212.05",
                target_chunk_id="chunk:statute:212.05:0",
                relation_type=RelationType.CITES,
                citation_text="§ 212.05",
                confidence=0.8,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:statute:192.001:1",
                source_doc_id="statute:192.001",
                target_doc_id="statute:212.05",
                target_chunk_id="chunk:statute:212.05:0",
                relation_type=RelationType.CITES,
                citation_text="s. 212.05",
                confidence=1.0,
            ),
        ]

        unique = deduplicate_edges(edges)

        assert len(unique) == 1
        assert unique[0].confidence == 1.0

    def test_deduplicate_different_relations(self):
        """Test that different relation types are kept separate."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:rule:12A-1.005:0",
                source_doc_id="rule:12A-1.005",
                target_doc_id="statute:212.05",
                target_chunk_id="chunk:statute:212.05:0",
                relation_type=RelationType.AUTHORITY,
                citation_text="212.05, F.S.",
                confidence=1.0,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:rule:12A-1.005:0",
                source_doc_id="rule:12A-1.005",
                target_doc_id="statute:212.05",
                target_chunk_id="chunk:statute:212.05:0",
                relation_type=RelationType.IMPLEMENTS,
                citation_text="212.05, F.S.",
                confidence=1.0,
            ),
        ]

        unique = deduplicate_edges(edges)

        assert len(unique) == 2


class TestNormalizeCaseCitation:
    """Tests for case citation normalization."""

    def test_normalize_removes_periods(self):
        """Test that periods are removed."""
        result = normalize_case_citation_for_index("366 So. 2d 1173")
        assert "." not in result

    def test_normalize_lowercase(self):
        """Test that result is lowercase."""
        result = normalize_case_citation_for_index("366 So. 2d 1173")
        assert result == result.lower()

    def test_normalize_collapses_whitespace(self):
        """Test that multiple spaces are collapsed."""
        result = normalize_case_citation_for_index("366  So.  2d  1173")
        assert "  " not in result
