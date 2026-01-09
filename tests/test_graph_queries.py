"""Tests for graph queries."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.graph.queries import (
    ChunkNode,
    CitationEdge,
    DocumentNode,
    InterpretationChainResult,
    find_path_between,
    get_all_citations_for_chunk,
    get_cited_documents,
    get_citing_documents,
    get_document_with_chunks,
    get_interpretation_chain,
    get_statute_with_implementing_rules,
    get_statutes_by_chapter,
)


class TestDocumentNode:
    """Tests for DocumentNode model."""

    def test_required_fields(self):
        """Test required fields."""
        node = DocumentNode(
            id="statute:212.05",
            doc_type="statute",
            title="Tax on Sales",
            full_citation="Fla. Stat. Section 212.05",
        )
        assert node.id == "statute:212.05"
        assert node.section is None

    def test_optional_fields(self):
        """Test optional fields."""
        node = DocumentNode(
            id="statute:212.05",
            doc_type="statute",
            title="Tax on Sales",
            full_citation="Fla. Stat. Section 212.05",
            section="212.05",
            chapter="212",
        )
        assert node.section == "212.05"
        assert node.chapter == "212"


class TestChunkNode:
    """Tests for ChunkNode model."""

    def test_all_fields(self):
        """Test ChunkNode with all fields."""
        node = ChunkNode(
            id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            level="parent",
            ancestry="Florida Statutes > Chapter 212 > § 212.05",
            subsection_path="",
            citation="Fla. Stat. § 212.05",
            token_count=500,
        )
        assert node.id == "chunk:statute:212.05:0"
        assert node.level == "parent"


class TestCitationEdge:
    """Tests for CitationEdge model."""

    def test_all_fields(self):
        """Test CitationEdge with all fields."""
        edge = CitationEdge(
            source_id="statute:212.05",
            target_id="statute:212.02",
            relation_type="CITES",
            citation_text="s. 212.02",
            confidence=1.0,
        )
        assert edge.relation_type == "CITES"
        assert edge.confidence == 1.0


class TestInterpretationChainResult:
    """Tests for InterpretationChainResult model."""

    def test_empty_lists(self):
        """Test with empty lists."""
        statute = DocumentNode(
            id="statute:212.05",
            doc_type="statute",
            title="Tax on Sales",
            full_citation="Fla. Stat. § 212.05",
        )
        result = InterpretationChainResult(statute=statute)
        assert len(result.implementing_rules) == 0
        assert len(result.interpreting_cases) == 0
        assert len(result.interpreting_taas) == 0


class TestGetStatuteWithImplementingRules:
    """Tests for get_statute_with_implementing_rules."""

    def test_statute_not_found(self):
        """Test when statute is not found."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [{"s": None, "rules": []}]

        result = get_statute_with_implementing_rules(mock_client, "999.999")
        assert result is None

    def test_statute_with_no_rules(self):
        """Test statute with no implementing rules."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "s": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                    "title": "Tax on Sales",
                    "full_citation": "Fla. Stat. Section 212.05",
                },
                "rules": [],
            }
        ]

        result = get_statute_with_implementing_rules(mock_client, "212.05")

        assert result is not None
        assert result.statute.id == "statute:212.05"
        assert len(result.implementing_rules) == 0

    def test_statute_with_rules(self):
        """Test statute with implementing rules."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "s": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                    "title": "Tax on Sales",
                    "full_citation": "Fla. Stat. Section 212.05",
                },
                "rules": [
                    {
                        "id": "rule:12A-1.001",
                        "doc_type": "rule",
                        "title": "Sales Tax Rule",
                        "full_citation": "Fla. Admin. Code R. 12A-1.001",
                    },
                ],
            }
        ]

        result = get_statute_with_implementing_rules(mock_client, "212.05")

        assert result is not None
        assert len(result.implementing_rules) == 1
        assert result.implementing_rules[0].id == "rule:12A-1.001"


class TestGetAllCitationsForChunk:
    """Tests for get_all_citations_for_chunk."""

    def test_no_citations(self):
        """Test chunk with no citations."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = []

        result = get_all_citations_for_chunk(mock_client, "chunk:statute:212.05:0")
        assert len(result) == 0

    def test_with_citations(self):
        """Test chunk with citations."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "source_id": "statute:212.05",
                "target_id": "statute:212.02",
                "relation_type": "CITES",
                "citation_text": "s. 212.02",
                "confidence": 1.0,
            },
        ]

        result = get_all_citations_for_chunk(mock_client, "chunk:statute:212.05:0")

        assert len(result) == 1
        assert isinstance(result[0], CitationEdge)
        assert result[0].relation_type == "CITES"

    def test_multiple_citations(self):
        """Test chunk with multiple citations."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "source_id": "statute:212.05",
                "target_id": "statute:212.02",
                "relation_type": "CITES",
                "citation_text": "s. 212.02",
                "confidence": 1.0,
            },
            {
                "source_id": "statute:212.05",
                "target_id": "statute:212.03",
                "relation_type": "CITES",
                "citation_text": "s. 212.03",
                "confidence": 0.9,
            },
        ]

        result = get_all_citations_for_chunk(mock_client, "chunk:statute:212.05:0")

        assert len(result) == 2


class TestGetInterpretationChain:
    """Tests for get_interpretation_chain."""

    def test_statute_not_found(self):
        """Test when statute is not found."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [{"s": None, "rules": [], "cases": [], "taas": []}]

        result = get_interpretation_chain(mock_client, "999.999")
        assert result is None

    def test_full_chain(self):
        """Test complete interpretation chain."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "s": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                    "title": "Tax on Sales",
                    "full_citation": "Fla. Stat. Section 212.05",
                },
                "rules": [
                    {
                        "id": "rule:12A-1.001",
                        "doc_type": "rule",
                        "title": "Sales Tax Rule",
                        "full_citation": "Fla. Admin. Code R. 12A-1.001",
                    },
                ],
                "cases": [
                    {
                        "id": "case:12345",
                        "doc_type": "case",
                        "title": "State v. Taxpayer",
                        "full_citation": "State v. Taxpayer, 123 So. 3d 456",
                    },
                ],
                "taas": [],
            }
        ]

        result = get_interpretation_chain(mock_client, "212.05")

        assert result is not None
        assert result.statute.id == "statute:212.05"
        assert len(result.implementing_rules) == 1
        assert len(result.interpreting_cases) == 1
        assert len(result.interpreting_taas) == 0


class TestFindPathBetween:
    """Tests for find_path_between."""

    def test_direct_connection(self):
        """Test finding direct connection."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "from_id": "statute:212.05",
                "from_title": "Tax on Sales",
                "relation": "CITES",
                "to_id": "statute:212.02",
                "to_title": "Definitions",
            },
        ]

        result = find_path_between(mock_client, "statute:212.05", "statute:212.02")

        assert len(result) == 1
        assert result[0]["relation"] == "CITES"

    def test_no_path(self):
        """Test when no path exists."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = []

        result = find_path_between(mock_client, "statute:212.05", "case:99999")
        assert len(result) == 0


class TestGetDocumentWithChunks:
    """Tests for get_document_with_chunks."""

    def test_document_not_found(self):
        """Test when document is not found."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [{"d": None, "chunks": []}]

        result = get_document_with_chunks(mock_client, "statute:999.999")
        assert result is None

    def test_document_with_chunks(self):
        """Test document with chunks."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "d": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                    "title": "Tax on Sales",
                },
                "chunks": [
                    {
                        "chunk": {
                            "id": "chunk:statute:212.05:0",
                            "doc_id": "statute:212.05",
                            "level": "parent",
                        },
                        "parent_id": None,
                    },
                    {
                        "chunk": {
                            "id": "chunk:statute:212.05:1",
                            "doc_id": "statute:212.05",
                            "level": "child",
                        },
                        "parent_id": "chunk:statute:212.05:0",
                    },
                ],
            }
        ]

        result = get_document_with_chunks(mock_client, "statute:212.05")

        assert result is not None
        assert result["document"]["id"] == "statute:212.05"
        assert len(result["chunks"]) == 2


class TestGetCitingDocuments:
    """Tests for get_citing_documents."""

    def test_no_citing_documents(self):
        """Test document with no citing documents."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = []

        result = get_citing_documents(mock_client, "statute:212.05")
        assert len(result) == 0

    def test_with_citing_documents(self):
        """Test document with citing documents."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "source": {
                    "id": "rule:12A-1.001",
                    "doc_type": "rule",
                    "title": "Sales Tax Rule",
                    "full_citation": "Fla. Admin. Code R. 12A-1.001",
                },
            },
        ]

        result = get_citing_documents(mock_client, "statute:212.05")

        assert len(result) == 1
        assert isinstance(result[0], DocumentNode)
        assert result[0].id == "rule:12A-1.001"


class TestGetCitedDocuments:
    """Tests for get_cited_documents."""

    def test_no_cited_documents(self):
        """Test document that cites nothing."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = []

        result = get_cited_documents(mock_client, "case:12345")
        assert len(result) == 0

    def test_with_cited_documents(self):
        """Test document that cites other documents."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "target": {
                    "id": "statute:212.05",
                    "doc_type": "statute",
                    "title": "Tax on Sales",
                    "full_citation": "Fla. Stat. § 212.05",
                },
            },
        ]

        result = get_cited_documents(mock_client, "rule:12A-1.001")

        assert len(result) == 1
        assert result[0].id == "statute:212.05"


class TestGetStatutesByChapter:
    """Tests for get_statutes_by_chapter."""

    def test_empty_chapter(self):
        """Test chapter with no statutes."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = []

        result = get_statutes_by_chapter(mock_client, "999")
        assert len(result) == 0

    def test_chapter_with_statutes(self):
        """Test chapter with statutes."""
        mock_client = MagicMock()
        mock_client.run_query.return_value = [
            {
                "s": {
                    "id": "statute:212.01",
                    "doc_type": "statute",
                    "title": "Short Title",
                    "full_citation": "Fla. Stat. § 212.01",
                },
            },
            {
                "s": {
                    "id": "statute:212.02",
                    "doc_type": "statute",
                    "title": "Definitions",
                    "full_citation": "Fla. Stat. § 212.02",
                },
            },
        ]

        result = get_statutes_by_chapter(mock_client, "212")

        assert len(result) == 2
        assert result[0].id == "statute:212.01"
        assert result[1].id == "statute:212.02"
