"""Unit tests for chunking module."""

from datetime import date, datetime

import pytest

from src.ingestion.chunking import (
    ChunkLevel,
    LegalChunk,
    build_ancestry,
    chunk_case,
    chunk_corpus,
    chunk_document,
    chunk_rule,
    chunk_statute,
    chunk_taa,
    create_chunk,
    parse_top_level_subsections,
)
from src.ingestion.models import Corpus, CorpusMetadata, DocumentType, LegalDocument
from src.ingestion.tokenizer import count_tokens, get_encoder, truncate_to_tokens


class TestTokenizer:
    """Tests for tokenizer utilities."""

    def test_count_tokens_empty(self):
        """Test token count for empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_basic(self):
        """Test token count for basic text."""
        # "Hello world" should be ~2 tokens
        result = count_tokens("Hello world")
        assert result > 0
        assert result < 10

    def test_count_tokens_legal_text(self):
        """Test token count for legal-style text."""
        text = "Section 212.05, Florida Statutes, imposes a tax on sales."
        result = count_tokens(text)
        assert result > 0
        assert result < 50

    def test_get_encoder_returns_same_instance(self):
        """Test that encoder is cached."""
        enc1 = get_encoder()
        enc2 = get_encoder()
        assert enc1 is enc2

    def test_truncate_to_tokens_empty(self):
        """Test truncation of empty string."""
        assert truncate_to_tokens("", 100) == ""

    def test_truncate_to_tokens_short_text(self):
        """Test truncation when text is shorter than limit."""
        text = "Hello world"
        result = truncate_to_tokens(text, 100)
        assert result == text

    def test_truncate_to_tokens_long_text(self):
        """Test truncation of long text."""
        text = "This is a longer sentence that should be truncated. " * 10
        result = truncate_to_tokens(text, 10)
        assert count_tokens(result) <= 10


class TestParseSubsections:
    """Tests for subsection parsing."""

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        result = parse_top_level_subsections("")
        assert result == []

    def test_parse_no_subsections(self):
        """Test parsing text without subsections."""
        text = "This is just regular text without any subsection markers."
        result = parse_top_level_subsections(text)
        assert result == []

    def test_parse_single_subsection(self):
        """Test parsing single subsection marker."""
        text = """
(1)
This is subsection one content.
"""
        result = parse_top_level_subsections(text)
        assert len(result) == 1
        assert result[0][0] == "(1)"
        assert "subsection one content" in result[0][1]

    def test_parse_multiple_subsections(self):
        """Test parsing multiple subsection markers."""
        text = """Introduction text.

(1)
First subsection content here.

(2)
Second subsection content here.

(3)
Third subsection content here.
"""
        result = parse_top_level_subsections(text)
        assert len(result) == 3
        assert result[0][0] == "(1)"
        assert result[1][0] == "(2)"
        assert result[2][0] == "(3)"

    def test_parse_preserves_nested_markers(self):
        """Test that nested markers (a), (b) are preserved in content."""
        text = """
(1)
Main subsection with nested markers:
(a) First nested item
(b) Second nested item
"""
        result = parse_top_level_subsections(text)
        assert len(result) == 1
        # Nested markers should be in the content, not split out
        assert "(a)" in result[0][1]
        assert "(b)" in result[0][1]


class TestBuildAncestry:
    """Tests for ancestry string construction."""

    def test_ancestry_statute(self):
        """Test ancestry for statute document."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="Test text",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={
                "title_number": "XIV",
                "title_name": "TAXATION AND FINANCE",
                "chapter": "212",
            },
        )
        ancestry = build_ancestry(doc)
        assert "Florida Statutes" in ancestry
        assert "Title XIV" in ancestry
        assert "Chapter 212" in ancestry
        assert "212.05" in ancestry

    def test_ancestry_rule(self):
        """Test ancestry for rule document."""
        doc = LegalDocument(
            id="rule:12A-1.005",
            doc_type=DocumentType.RULE,
            title="Rule 12A-1.005",
            full_citation="Fla. Admin. Code R. 12A-1.005",
            text="Test text",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"chapter": "12A-1"},
        )
        ancestry = build_ancestry(doc)
        assert "Florida Administrative Code" in ancestry
        assert "Chapter 12A-1" in ancestry
        assert "Rule 12A-1.005" in ancestry

    def test_ancestry_taa(self):
        """Test ancestry for TAA document."""
        doc = LegalDocument(
            id="taa:25A-009",
            doc_type=DocumentType.TAA,
            title="TAA 25A-009",
            full_citation="Fla. DOR TAA 25A-009",
            text="Test text",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"tax_type": "Sales Tax"},
        )
        ancestry = build_ancestry(doc)
        assert "Technical Assistance Advisements" in ancestry
        assert "Sales Tax" in ancestry
        assert "TAA 25A-009" in ancestry

    def test_ancestry_case(self):
        """Test ancestry for case document."""
        doc = LegalDocument(
            id="case:12345",
            doc_type=DocumentType.CASE,
            title="State v. Taxpayer",
            full_citation="State v. Taxpayer, 123 So. 3d 456",
            text="Test text",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"court": "Florida Supreme Court"},
        )
        ancestry = build_ancestry(doc)
        assert "Florida Case Law" in ancestry
        assert "Florida Supreme Court" in ancestry
        assert "State v. Taxpayer" in ancestry


class TestCreateChunk:
    """Tests for chunk creation."""

    def test_create_parent_chunk(self):
        """Test creating a parent chunk."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="Test text for the statute.",
            effective_date=date(2024, 1, 1),
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )
        ancestry = "Florida Statutes > Chapter 212 > § 212.05"

        chunk = create_chunk(
            doc=doc,
            chunk_index=0,
            level=ChunkLevel.PARENT,
            text="Test text for the statute.",
            ancestry=ancestry,
        )

        assert chunk.id == "chunk:statute:212.05:0"
        assert chunk.doc_id == "statute:212.05"
        assert chunk.level == ChunkLevel.PARENT
        assert chunk.ancestry == ancestry
        assert chunk.subsection_path == ""
        assert chunk.text == "Test text for the statute."
        assert ancestry in chunk.text_with_ancestry
        assert chunk.citation == "Fla. Stat. § 212.05"
        assert chunk.effective_date == date(2024, 1, 1)
        assert chunk.doc_type == "statute"
        assert chunk.token_count > 0

    def test_create_child_chunk(self):
        """Test creating a child chunk."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="Full text",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )
        ancestry = "Florida Statutes > Chapter 212 > § 212.05"

        chunk = create_chunk(
            doc=doc,
            chunk_index=1,
            level=ChunkLevel.CHILD,
            text="Subsection content",
            ancestry=ancestry,
            subsection_path="(1)",
            parent_chunk_id="chunk:statute:212.05:0",
        )

        assert chunk.id == "chunk:statute:212.05:1"
        assert chunk.level == ChunkLevel.CHILD
        assert chunk.subsection_path == "(1)"
        assert chunk.parent_chunk_id == "chunk:statute:212.05:0"


class TestChunkStatute:
    """Tests for statute chunking."""

    def test_chunk_statute_without_subsections(self):
        """Test chunking statute without subsections."""
        doc = LegalDocument(
            id="statute:212.01",
            doc_type=DocumentType.STATUTE,
            title="§ 212.01",
            full_citation="Fla. Stat. § 212.01",
            text="This is a short statute without any subsection markers.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"chapter": "212"},
        )

        chunks = chunk_statute(doc)

        # Should have only parent chunk
        assert len(chunks) == 1
        assert chunks[0].level == ChunkLevel.PARENT
        assert chunks[0].child_chunk_ids == []

    def test_chunk_statute_with_subsections(self):
        """Test chunking statute with subsections."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="""General provisions for sales tax.

(1)
First subsection about rates.

(2)
Second subsection about exemptions.
""",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"chapter": "212"},
        )

        chunks = chunk_statute(doc)

        # Should have parent + 2 children
        assert len(chunks) == 3

        # First chunk is parent
        parent = chunks[0]
        assert parent.level == ChunkLevel.PARENT
        assert len(parent.child_chunk_ids) == 2

        # Children reference parent
        for child in chunks[1:]:
            assert child.level == ChunkLevel.CHILD
            assert child.parent_chunk_id == parent.id

    def test_chunk_statute_parent_child_linking(self):
        """Test that parent-child linking is bidirectional."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="""Intro.

(1)
First.

(2)
Second.
""",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )

        chunks = chunk_statute(doc)
        parent = chunks[0]
        children = chunks[1:]

        # Parent references all children
        for child in children:
            assert child.id in parent.child_chunk_ids

        # Children reference parent
        for child in children:
            assert child.parent_chunk_id == parent.id


class TestChunkRule:
    """Tests for rule chunking."""

    def test_chunk_rule_basic(self):
        """Test basic rule chunking."""
        doc = LegalDocument(
            id="rule:12A-1.005",
            doc_type=DocumentType.RULE,
            title="Rule 12A-1.005",
            full_citation="Fla. Admin. Code R. 12A-1.005",
            text="This is a simple rule without subsections.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"chapter": "12A-1"},
        )

        chunks = chunk_rule(doc)

        assert len(chunks) >= 1
        assert chunks[0].level == ChunkLevel.PARENT
        assert chunks[0].doc_type == "rule"


class TestChunkTaa:
    """Tests for TAA chunking."""

    def test_chunk_taa_short(self):
        """Test chunking short TAA (single chunk)."""
        doc = LegalDocument(
            id="taa:25A-001",
            doc_type=DocumentType.TAA,
            title="TAA 25A-001",
            full_citation="Fla. DOR TAA 25A-001",
            text="Short TAA text that should be a single chunk.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )

        chunks = chunk_taa(doc)

        # Short TAA should be single chunk
        assert len(chunks) == 1
        assert chunks[0].level == ChunkLevel.PARENT

    def test_chunk_taa_with_sections(self):
        """Test chunking TAA with logical sections."""
        # Create a long TAA with sections (>2000 tokens to trigger splitting)
        long_text = """TAA Introduction and summary.

REQUESTED ADVISEMENT

The taxpayer requests advisement on applicability of sales tax.

FACTS

The taxpayer operates a business in Florida. """ + "Additional facts about the business operations and circumstances. " * 500 + """

TAXPAYER'S POSITION

The taxpayer believes the transaction is exempt.

LAW AND DISCUSSION

Section 212.05, F.S., provides the applicable law. """ + "Discussion of the legal analysis and application continues here. " * 500 + """

CONCLUSION

Based on the above analysis, the transaction is taxable.
"""

        doc = LegalDocument(
            id="taa:25A-002",
            doc_type=DocumentType.TAA,
            title="TAA 25A-002",
            full_citation="Fla. DOR TAA 25A-002",
            text=long_text,
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )

        chunks = chunk_taa(doc)

        # Should have parent + section chunks
        assert len(chunks) > 1
        assert chunks[0].level == ChunkLevel.PARENT


class TestChunkCase:
    """Tests for case chunking."""

    def test_chunk_case_short(self):
        """Test chunking short case (single chunk)."""
        doc = LegalDocument(
            id="case:12345",
            doc_type=DocumentType.CASE,
            title="State v. Taxpayer",
            full_citation="State v. Taxpayer, 123 So. 3d 456",
            text="Short case opinion text that fits in one chunk.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={"court": "Florida Supreme Court"},
        )

        chunks = chunk_case(doc)

        assert len(chunks) == 1
        assert chunks[0].level == ChunkLevel.PARENT
        assert chunks[0].doc_type == "case"


class TestChunkDocument:
    """Tests for document routing."""

    def test_chunk_document_routes_statute(self):
        """Test that chunk_document routes statutes correctly."""
        doc = LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="§ 212.05",
            full_citation="Fla. Stat. § 212.05",
            text="Statute text.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )

        chunks = chunk_document(doc)
        assert all(c.doc_type == "statute" for c in chunks)

    def test_chunk_document_routes_rule(self):
        """Test that chunk_document routes rules correctly."""
        doc = LegalDocument(
            id="rule:12A-1.005",
            doc_type=DocumentType.RULE,
            title="Rule 12A-1.005",
            full_citation="Fla. Admin. Code R. 12A-1.005",
            text="Rule text.",
            source_url="http://example.com",
            scraped_at=datetime.now(),
            metadata={},
        )

        chunks = chunk_document(doc)
        assert all(c.doc_type == "rule" for c in chunks)


class TestChunkCorpus:
    """Tests for corpus-level chunking."""

    def test_chunk_corpus_processes_all_documents(self):
        """Test that chunk_corpus processes all documents."""
        docs = [
            LegalDocument(
                id="statute:212.05",
                doc_type=DocumentType.STATUTE,
                title="§ 212.05",
                full_citation="Fla. Stat. § 212.05",
                text="Statute text.",
                source_url="http://example.com",
                scraped_at=datetime.now(),
                metadata={},
            ),
            LegalDocument(
                id="rule:12A-1.005",
                doc_type=DocumentType.RULE,
                title="Rule 12A-1.005",
                full_citation="Fla. Admin. Code R. 12A-1.005",
                text="Rule text.",
                source_url="http://example.com",
                scraped_at=datetime.now(),
                metadata={},
            ),
        ]

        corpus = Corpus(
            metadata=CorpusMetadata(
                created_at=datetime.now(),
                total_documents=2,
                by_type={"statute": 1, "rule": 1},
            ),
            documents=docs,
        )

        chunks = chunk_corpus(corpus)

        # Should have at least one chunk per document
        assert len(chunks) >= 2

        # Should have both doc types
        doc_types = {c.doc_type for c in chunks}
        assert "statute" in doc_types
        assert "rule" in doc_types


class TestLegalChunkModel:
    """Tests for LegalChunk model."""

    def test_chunk_level_enum_values(self):
        """Test ChunkLevel enum values."""
        assert ChunkLevel.PARENT.value == "parent"
        assert ChunkLevel.CHILD.value == "child"

    def test_legal_chunk_serialization(self):
        """Test that LegalChunk can be serialized to JSON."""
        chunk = LegalChunk(
            id="chunk:statute:212.05:0",
            doc_id="statute:212.05",
            level=ChunkLevel.PARENT,
            ancestry="Florida Statutes > Chapter 212",
            subsection_path="",
            text="Test text",
            text_with_ancestry="Florida Statutes > Chapter 212\n\nTest text",
            parent_chunk_id=None,
            child_chunk_ids=["chunk:statute:212.05:1"],
            citation="Fla. Stat. § 212.05",
            effective_date=date(2024, 1, 1),
            doc_type="statute",
            token_count=10,
        )

        # Should be able to serialize
        data = chunk.model_dump(mode="json")
        assert data["id"] == "chunk:statute:212.05:0"
        assert data["level"] == "parent"
        assert data["effective_date"] == "2024-01-01"
