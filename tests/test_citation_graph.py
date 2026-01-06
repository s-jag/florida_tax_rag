"""Tests for src/ingestion/build_citation_graph.py."""

from __future__ import annotations

import pytest

from src.ingestion.build_citation_graph import (
    CitationIndex,
    ResolvedEdge,
    build_citation_graph,
    build_citation_index,
    deduplicate_edges,
    extract_chunk_citations,
    get_parent_chunk,
    normalize_case_citation_for_index,
    resolve_citation,
)
from src.ingestion.citation_extractor import Citation, CitationType, RelationType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Sample chunks for building citation index."""
    return [
        {
            "id": "chunk:statute:212.05:0",
            "doc_id": "statute:212.05",
            "doc_type": "statute",
            "citation": "Fla. Stat. § 212.05",
            "text": "Sales tax rate is 6 percent.",
        },
        {
            "id": "chunk:statute:212.05:1",
            "doc_id": "statute:212.05",
            "doc_type": "statute",
            "citation": "Fla. Stat. § 212.05",
            "text": "Subsection about exemptions.",
        },
        {
            "id": "chunk:statute:212.08:0",
            "doc_id": "statute:212.08",
            "doc_type": "statute",
            "citation": "Fla. Stat. § 212.08",
            "text": "Exemptions from sales tax.",
        },
        {
            "id": "chunk:rule:12A-1.001:0",
            "doc_id": "rule:12A-1.001",
            "doc_type": "rule",
            "citation": "Fla. Admin. Code R. 12A-1.001",
            "text": "Sales tax rules.",
        },
        {
            "id": "chunk:case:1234567:0",
            "doc_id": "case:1234567",
            "doc_type": "case",
            "citation": "DOR v. Taxpayer, 366 So. 2d 1173",
            "text": "Court case about sales tax.",
        },
        {
            "id": "chunk:taa:24A-001:0",
            "doc_id": "taa:24A-001",
            "doc_type": "taa",
            "citation": "TAA 24A-001",
            "text": "Technical assistance advisement.",
        },
    ]


@pytest.fixture
def citation_index(sample_chunks: list[dict]) -> CitationIndex:
    """Pre-built citation index."""
    return build_citation_index(sample_chunks)


# =============================================================================
# Normalize Case Citation Tests
# =============================================================================


class TestNormalizeCaseCitationForIndex:
    """Test normalize_case_citation_for_index function."""

    def test_removes_extra_whitespace(self) -> None:
        """Should collapse multiple spaces."""
        result = normalize_case_citation_for_index("366  So.  2d   1173")
        assert "  " not in result

    def test_removes_periods(self) -> None:
        """Should remove periods from abbreviations."""
        result = normalize_case_citation_for_index("366 So. 2d 1173")
        assert "." not in result

    def test_lowercases(self) -> None:
        """Should convert to lowercase."""
        result = normalize_case_citation_for_index("366 So. 2d 1173")
        assert result == result.lower()

    def test_handles_various_formats(self) -> None:
        """Should handle different citation formats."""
        citations = [
            "366 So.2d 1173",
            "366 So. 2d 1173",
            "366  So.  2d  1173",
        ]
        results = [normalize_case_citation_for_index(c) for c in citations]
        # All should normalize to similar form
        assert all("so" in r for r in results)
        assert all("1173" in r for r in results)


# =============================================================================
# Build Citation Index Tests
# =============================================================================


class TestBuildCitationIndex:
    """Test build_citation_index function."""

    def test_indexes_statutes(
        self, sample_chunks: list[dict], citation_index: CitationIndex
    ) -> None:
        """Should index statute chunks."""
        assert "statute:212.05" in citation_index.statute_to_chunks
        assert len(citation_index.statute_to_chunks["statute:212.05"]) == 2

    def test_indexes_rules(
        self, sample_chunks: list[dict], citation_index: CitationIndex
    ) -> None:
        """Should index rule chunks."""
        assert "rule:12A-1.001" in citation_index.rule_to_chunks

    def test_indexes_cases(
        self, sample_chunks: list[dict], citation_index: CitationIndex
    ) -> None:
        """Should index case chunks."""
        assert "case:1234567" in citation_index.case_to_chunks

    def test_indexes_taas(
        self, sample_chunks: list[dict], citation_index: CitationIndex
    ) -> None:
        """Should index TAA chunks."""
        assert "taa:24A-001" in citation_index.taa_to_chunks

    def test_builds_section_to_doc_mapping(
        self, citation_index: CitationIndex
    ) -> None:
        """Should map section numbers to doc IDs."""
        assert "212.05" in citation_index.section_to_doc
        assert citation_index.section_to_doc["212.05"] == "statute:212.05"

    def test_builds_rule_to_doc_mapping(self, citation_index: CitationIndex) -> None:
        """Should map rule numbers to doc IDs."""
        assert "12A-1.001" in citation_index.rule_to_doc
        assert citation_index.rule_to_doc["12A-1.001"] == "rule:12A-1.001"

    def test_builds_case_citation_mapping(
        self, citation_index: CitationIndex
    ) -> None:
        """Should map normalized case citations to doc IDs."""
        # The normalized form of "366 So. 2d 1173"
        assert any("1173" in k for k in citation_index.case_citation_to_id.keys())


# =============================================================================
# Get Parent Chunk Tests
# =============================================================================


class TestGetParentChunk:
    """Test get_parent_chunk function."""

    def test_returns_parent_chunk(self) -> None:
        """Should return chunk ending in :0."""
        chunks = [
            "chunk:statute:212.05:1",
            "chunk:statute:212.05:0",
            "chunk:statute:212.05:2",
        ]
        result = get_parent_chunk(chunks)
        assert result == "chunk:statute:212.05:0"

    def test_returns_first_if_no_parent(self) -> None:
        """Should return first chunk if no :0 found."""
        chunks = [
            "chunk:statute:212.05:1",
            "chunk:statute:212.05:2",
        ]
        result = get_parent_chunk(chunks)
        assert result == "chunk:statute:212.05:1"

    def test_returns_none_for_empty_list(self) -> None:
        """Should return None for empty list."""
        result = get_parent_chunk([])
        assert result is None


# =============================================================================
# Resolve Citation Tests
# =============================================================================


class TestResolveCitation:
    """Test resolve_citation function."""

    def test_resolves_statute_exact_match(self, citation_index: CitationIndex) -> None:
        """Should resolve exact statute match with confidence 1.0."""
        citation = Citation(
            raw_text="Section 212.05",
            normalized="212.05",
            citation_type=CitationType.STATUTE,
            section="212.05",
            start_pos=0,
            end_pos=14,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id == "statute:212.05"
        assert chunk_id is not None
        assert confidence == 1.0

    def test_resolves_statute_with_subsection(
        self, citation_index: CitationIndex
    ) -> None:
        """Should resolve statute with subsection to base section."""
        citation = Citation(
            raw_text="Section 212.05(1)(a)",
            normalized="212.05(1)(a)",
            citation_type=CitationType.STATUTE,
            section="212.05(1)(a)",
            start_pos=0,
            end_pos=20,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id == "statute:212.05"
        assert confidence == 0.8  # Partial match

    def test_resolves_rule_exact_match(self, citation_index: CitationIndex) -> None:
        """Should resolve exact rule match."""
        citation = Citation(
            raw_text="Rule 12A-1.001",
            normalized="12A-1.001",
            citation_type=CitationType.RULE,
            section="12A-1.001",
            start_pos=0,
            end_pos=14,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id == "rule:12A-1.001"
        assert confidence == 1.0

    def test_resolves_case_citation(self, citation_index: CitationIndex) -> None:
        """Should resolve case citation."""
        citation = Citation(
            raw_text="366 So. 2d 1173",
            normalized="366 So. 2d 1173",
            citation_type=CitationType.CASE,
            start_pos=0,
            end_pos=15,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id == "case:1234567"
        assert confidence == 0.9

    def test_returns_none_for_unresolvable(
        self, citation_index: CitationIndex
    ) -> None:
        """Should return None for unresolvable citation."""
        citation = Citation(
            raw_text="Section 999.99",
            normalized="999.99",
            citation_type=CitationType.STATUTE,
            section="999.99",
            start_pos=0,
            end_pos=14,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id is None
        assert chunk_id is None
        assert confidence == 0.0

    def test_chapter_returns_none(self, citation_index: CitationIndex) -> None:
        """Chapter citations should not resolve."""
        citation = Citation(
            raw_text="Chapter 212",
            normalized="212",
            citation_type=CitationType.CHAPTER,
            section="212",
            start_pos=0,
            end_pos=11,
        )
        doc_id, chunk_id, confidence = resolve_citation(citation, citation_index)

        assert doc_id is None
        assert confidence == 0.0


# =============================================================================
# Extract Chunk Citations Tests
# =============================================================================


class TestExtractChunkCitations:
    """Test extract_chunk_citations function."""

    def test_extracts_statute_citations(self, citation_index: CitationIndex) -> None:
        """Should extract statute citations from chunk."""
        chunk = {
            "id": "chunk:rule:12A-1.001:0",
            "doc_id": "rule:12A-1.001",
            "doc_type": "rule",
            "text": "This rule implements Section 212.05, F.S.",
        }
        relations = extract_chunk_citations(chunk, citation_index)

        assert len(relations) >= 1
        # Should find the 212.05 citation
        statute_relations = [
            r
            for r in relations
            if r.target_citation.citation_type == CitationType.STATUTE
        ]
        assert len(statute_relations) >= 1

    def test_filters_self_references(self, citation_index: CitationIndex) -> None:
        """Should filter out self-references."""
        chunk = {
            "id": "chunk:statute:212.05:0",
            "doc_id": "statute:212.05",
            "doc_type": "statute",
            "text": "Section 212.05 provides for sales tax at 6 percent.",
        }
        relations = extract_chunk_citations(chunk, citation_index)

        # Should not include self-reference to 212.05
        self_refs = [
            r
            for r in relations
            if r.target_citation.section == "212.05"
        ]
        assert len(self_refs) == 0


# =============================================================================
# Build Citation Graph Tests
# =============================================================================


class TestBuildCitationGraph:
    """Test build_citation_graph function."""

    def test_builds_graph_from_chunks(self) -> None:
        """Should build citation graph from chunks."""
        chunks = [
            {
                "id": "chunk:rule:12A-1.001:0",
                "doc_id": "rule:12A-1.001",
                "doc_type": "rule",
                "citation": "Fla. Admin. Code R. 12A-1.001",
                "text": "This rule implements Section 212.05, F.S.",
            },
            {
                "id": "chunk:statute:212.05:0",
                "doc_id": "statute:212.05",
                "doc_type": "statute",
                "citation": "Fla. Stat. § 212.05",
                "text": "Sales tax rate.",
            },
        ]

        resolved, unresolved = build_citation_graph(chunks)

        # Should have at least one resolved edge
        assert len(resolved) >= 1
        # The rule should cite the statute
        rule_to_statute = [
            e for e in resolved if e.source_doc_id == "rule:12A-1.001"
        ]
        assert len(rule_to_statute) >= 1

    def test_respects_min_confidence(self) -> None:
        """Should filter edges below min_confidence."""
        chunks = [
            {
                "id": "chunk:statute:212.05:0",
                "doc_id": "statute:212.05",
                "doc_type": "statute",
                "citation": "Fla. Stat. § 212.05",
                "text": "Text",
            },
        ]

        # With high min_confidence, should get fewer edges
        resolved_high, _ = build_citation_graph(chunks, min_confidence=0.99)
        resolved_low, _ = build_citation_graph(chunks, min_confidence=0.1)

        assert len(resolved_high) <= len(resolved_low)


# =============================================================================
# Deduplicate Edges Tests
# =============================================================================


class TestDeduplicateEdges:
    """Test deduplicate_edges function."""

    def test_removes_duplicates(self) -> None:
        """Should remove duplicate edges."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.8,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:a:1",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.9,
            ),
        ]

        result = deduplicate_edges(edges)

        assert len(result) == 1

    def test_keeps_highest_confidence(self) -> None:
        """Should keep edge with highest confidence."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.7,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:a:1",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.95,
            ),
        ]

        result = deduplicate_edges(edges)

        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_keeps_different_relations(self) -> None:
        """Should keep edges with different relation types."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.9,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.IMPLEMENTS,
                citation_text="Section 212.05",
                confidence=0.9,
            ),
        ]

        result = deduplicate_edges(edges)

        assert len(result) == 2

    def test_keeps_different_targets(self) -> None:
        """Should keep edges with different target docs."""
        edges = [
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:b",
                relation_type=RelationType.CITES,
                citation_text="Section 212.05",
                confidence=0.9,
            ),
            ResolvedEdge(
                source_chunk_id="chunk:a:0",
                source_doc_id="doc:a",
                target_doc_id="doc:c",
                relation_type=RelationType.CITES,
                citation_text="Section 212.08",
                confidence=0.9,
            ),
        ]

        result = deduplicate_edges(edges)

        assert len(result) == 2
