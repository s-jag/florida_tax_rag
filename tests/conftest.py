"""Shared pytest fixtures for Florida Tax RAG tests."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.chunking import ChunkLevel, LegalChunk
from src.ingestion.models import Corpus, CorpusMetadata, DocumentType, LegalDocument
from src.scrapers.models import (
    CaseMetadata,
    RawCase,
    RawRule,
    RawStatute,
    RawTAA,
    RuleMetadata,
    StatuteMetadata,
    TAAMetadata,
)


# =============================================================================
# Sample Raw Documents (Scraper Output)
# =============================================================================


@pytest.fixture
def sample_statute_metadata() -> StatuteMetadata:
    """Sample statute metadata."""
    return StatuteMetadata(
        title="TAXATION AND FINANCE",
        title_number=14,
        chapter=212,
        chapter_name="Tax on Sales, Use, and Other Transactions",
        section="212.05",
        section_name="Sales, storage, use tax",
        effective_date=date(2024, 1, 1),
        history=["s. 2, ch. 26319, 1949; s. 1, ch. 65-371"],
    )


@pytest.fixture
def sample_raw_statute(sample_statute_metadata: StatuteMetadata) -> RawStatute:
    """Sample raw statute from scraper."""
    return RawStatute(
        metadata=sample_statute_metadata,
        text="""212.05 Sales, storage, use tax.—
(1) It is hereby declared to be the legislative intent that every person is exercising a taxable privilege who engages in the business of selling tangible personal property at retail in this state.
(a) For the exercise of such privilege, a tax is levied on each taxable transaction or incident at the rate of 6 percent of the sales price of each item or article of tangible personal property when sold at retail in this state.
(b) Each occasional or isolated sale of an aircraft, boat, mobile home, or motor vehicle of a class or type which is required to be registered, licensed, titled, or documented in this state or by the United States Government shall be subject to tax.""",
        html="<html><body>...</body></html>",
        source_url="https://www.leg.state.fl.us/statutes/index.cfm?App_mode=Display_Statute&Search_String=&URL=0200-0299/0212/Sections/0212.05.html",
        scraped_at=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_rule_metadata() -> RuleMetadata:
    """Sample rule metadata."""
    return RuleMetadata(
        chapter="12A-1",
        rule_number="12A-1.001",
        title="Specific Exemptions",
        effective_date=date(2023, 7, 1),
        references_statutes=["212.02", "212.05", "212.08"],
        rulemaking_authority=["212.17(6)", "212.18(2)", "213.06(1)"],
        law_implemented=["212.02", "212.05", "212.055", "212.08"],
    )


@pytest.fixture
def sample_raw_rule(sample_rule_metadata: RuleMetadata) -> RawRule:
    """Sample raw rule from scraper."""
    return RawRule(
        metadata=sample_rule_metadata,
        text="""12A-1.001 Specific Exemptions.
(1) The following are exempt from the tax imposed by Chapter 212, F.S.:
(a) Occasional or isolated sales by a person who does not hold himself out as engaged in business.
(b) Sales of services or tangible personal property to the United States Government.""",
        html="<html><body>...</body></html>",
        source_url="https://www.flrules.org/gateway/RuleNo.asp?ID=12A-1.001",
        scraped_at=datetime(2024, 1, 15, 11, 0, 0),
    )


@pytest.fixture
def sample_taa_metadata() -> TAAMetadata:
    """Sample TAA metadata."""
    return TAAMetadata(
        taa_number="TAA 24A01-001",
        title="Sales Tax on Software as a Service (SaaS)",
        issue_date=date(2024, 1, 15),
        tax_type="Sales and Use Tax",
        tax_type_code="A",
        topics=["software", "SaaS", "digital products"],
        question="Is a subscription to cloud-based software subject to Florida sales tax?",
        answer="Cloud-based software (SaaS) that is accessed remotely and not downloaded is generally not subject to Florida sales tax as it is considered an information service rather than tangible personal property.",
        statutes_cited=["212.05", "212.08(7)(v)"],
        rules_cited=["12A-1.032"],
    )


@pytest.fixture
def sample_raw_taa(sample_taa_metadata: TAAMetadata) -> RawTAA:
    """Sample raw TAA from scraper."""
    return RawTAA(
        metadata=sample_taa_metadata,
        text="""TAA 24A01-001
Re: Sales Tax on Software as a Service (SaaS)

ISSUE: Is a subscription to cloud-based software subject to Florida sales tax?

RESPONSE: Cloud-based software (SaaS) that is accessed remotely and not downloaded is generally not subject to Florida sales tax as it is considered an information service rather than tangible personal property.

Pursuant to Section 212.05, F.S., and Rule 12A-1.032, F.A.C., the determination depends on whether the software is delivered electronically or accessed remotely.""",
        pdf_path="/tmp/taa/TAA_24A01-001.pdf",
        source_url="https://floridarevenue.com/taxes/taxesfees/Pages/taa.aspx",
        scraped_at=datetime(2024, 1, 15, 12, 0, 0),
    )


@pytest.fixture
def sample_case_metadata() -> CaseMetadata:
    """Sample case metadata."""
    return CaseMetadata(
        case_name="Dept. of Revenue v. XYZ Corp.",
        case_name_full="Florida Department of Revenue v. XYZ Corporation",
        citations=["350 So. 3d 123 (Fla. 2024)"],
        court="Supreme Court of Florida",
        court_id="fla",
        date_filed=date(2024, 3, 15),
        docket_number="SC2023-1234",
        judges="Chief Justice Muniz",
        statutes_cited=["212.05", "212.08(7)(b)"],
        cases_cited=[1093614, 2045789],
        cluster_id=9876543,
    )


@pytest.fixture
def sample_raw_case(sample_case_metadata: CaseMetadata) -> RawCase:
    """Sample raw case from scraper."""
    return RawCase(
        metadata=sample_case_metadata,
        opinion_text="""MUNIZ, C.J.

This case presents the question of whether digital advertising services are subject to Florida sales tax under section 212.05, Florida Statutes.

We hold that digital advertising services do not constitute a sale of tangible personal property and are therefore not subject to sales tax.

The Department of Revenue argues that section 212.08(7)(b) applies, but we disagree.""",
        source_url="https://www.courtlistener.com/opinion/9876543/dept-of-revenue-v-xyz-corp/",
        pdf_url="https://www.courtlistener.com/pdf/9876543/dept-of-revenue-v-xyz-corp.pdf",
        scraped_at=datetime(2024, 3, 20, 14, 0, 0),
    )


# =============================================================================
# Sample Legal Documents (Unified Model)
# =============================================================================


@pytest.fixture
def sample_legal_document() -> LegalDocument:
    """Sample unified legal document."""
    return LegalDocument(
        id="statute:212.05",
        doc_type=DocumentType.STATUTE,
        title="Sales, storage, use tax",
        full_citation="Fla. Stat. § 212.05",
        text="""212.05 Sales, storage, use tax.—
(1) It is hereby declared to be the legislative intent that every person is exercising a taxable privilege who engages in the business of selling tangible personal property at retail in this state.
(a) For the exercise of such privilege, a tax is levied on each taxable transaction or incident at the rate of 6 percent of the sales price.""",
        effective_date=date(2024, 1, 1),
        source_url="https://www.leg.state.fl.us/statutes/",
        parent_id="chapter:212",
        cites_statutes=["212.02", "212.08"],
        cites_rules=["12A-1.001"],
        scraped_at=datetime(2024, 1, 15, 10, 30, 0),
        metadata={"chapter": 212, "title_number": 14},
    )


@pytest.fixture
def sample_legal_documents() -> list[LegalDocument]:
    """List of sample legal documents."""
    return [
        LegalDocument(
            id="statute:212.05",
            doc_type=DocumentType.STATUTE,
            title="Sales, storage, use tax",
            full_citation="Fla. Stat. § 212.05",
            text="The sales tax rate is 6 percent.",
            effective_date=date(2024, 1, 1),
            source_url="https://www.leg.state.fl.us/statutes/",
            scraped_at=datetime(2024, 1, 15),
        ),
        LegalDocument(
            id="statute:212.08",
            doc_type=DocumentType.STATUTE,
            title="Exemptions",
            full_citation="Fla. Stat. § 212.08",
            text="The following are exempt from sales tax...",
            effective_date=date(2024, 1, 1),
            source_url="https://www.leg.state.fl.us/statutes/",
            scraped_at=datetime(2024, 1, 15),
        ),
        LegalDocument(
            id="rule:12A-1.001",
            doc_type=DocumentType.RULE,
            title="Specific Exemptions",
            full_citation="Fla. Admin. Code R. 12A-1.001",
            text="The following are exempt from tax...",
            effective_date=date(2023, 7, 1),
            source_url="https://www.flrules.org/",
            cites_statutes=["212.05", "212.08"],
            scraped_at=datetime(2024, 1, 15),
        ),
    ]


@pytest.fixture
def sample_corpus(sample_legal_documents: list[LegalDocument]) -> Corpus:
    """Sample corpus with metadata."""
    return Corpus(
        metadata=CorpusMetadata(
            created_at=datetime(2024, 1, 15),
            total_documents=len(sample_legal_documents),
            by_type={"statute": 2, "rule": 1},
            version="1.0",
        ),
        documents=sample_legal_documents,
    )


# =============================================================================
# Sample Chunks
# =============================================================================


@pytest.fixture
def sample_parent_chunk() -> LegalChunk:
    """Sample parent chunk."""
    return LegalChunk(
        id="chunk:statute:212.05:0",
        doc_id="statute:212.05",
        level=ChunkLevel.PARENT,
        ancestry="Florida Statutes > Title XIV > Chapter 212 > § 212.05",
        subsection_path="",
        text="212.05 Sales, storage, use tax.— (1) It is hereby declared...",
        text_with_ancestry="Florida Statutes > Title XIV > Chapter 212 > § 212.05\n\n212.05 Sales, storage, use tax.— (1) It is hereby declared...",
        citation="Fla. Stat. § 212.05",
        effective_date=date(2024, 1, 1),
        doc_type="statute",
        token_count=150,
    )


@pytest.fixture
def sample_child_chunk(sample_parent_chunk: LegalChunk) -> LegalChunk:
    """Sample child chunk."""
    return LegalChunk(
        id="chunk:statute:212.05:1",
        doc_id="statute:212.05",
        level=ChunkLevel.CHILD,
        ancestry="Florida Statutes > Title XIV > Chapter 212 > § 212.05(1)",
        subsection_path="(1)",
        text="(1) It is hereby declared to be the legislative intent...",
        text_with_ancestry="Florida Statutes > Title XIV > Chapter 212 > § 212.05(1)\n\n(1) It is hereby declared to be the legislative intent...",
        parent_chunk_id=sample_parent_chunk.id,
        citation="Fla. Stat. § 212.05(1)",
        effective_date=date(2024, 1, 1),
        doc_type="statute",
        token_count=75,
    )


@pytest.fixture
def sample_chunks(sample_parent_chunk: LegalChunk, sample_child_chunk: LegalChunk) -> list[LegalChunk]:
    """List of sample chunks."""
    parent = sample_parent_chunk.model_copy()
    parent.child_chunk_ids = [sample_child_chunk.id]
    return [parent, sample_child_chunk]


# =============================================================================
# Mock Clients
# =============================================================================


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    """Mock Neo4j client for testing."""
    client = MagicMock()
    client.health_check.return_value = True
    client.run_query.return_value = []
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_weaviate_client() -> MagicMock:
    """Mock Weaviate client for testing."""
    client = MagicMock()
    client.health_check.return_value = True
    client.collection_exists.return_value = True
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock async HTTP client for scraper testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock Anthropic client for LLM testing."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Sample LLM response")]
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def mock_voyage_client() -> MagicMock:
    """Mock Voyage AI client for embedding testing."""
    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
    return client


# =============================================================================
# Sample HTML Pages (for scraper parsing tests)
# =============================================================================


@pytest.fixture
def sample_html_statute_page() -> str:
    """Sample HTML from Florida Statutes website."""
    return """<!DOCTYPE html>
<html>
<head><title>2024 Florida Statutes</title></head>
<body>
<div class="Statute">
<div class="SectionHead">
<a name="0212.05"></a>212.05 Sales, storage, use tax.—
</div>
<div class="Section">
<div class="Paragraph">
(1) It is hereby declared to be the legislative intent that every person is exercising a taxable privilege.
<br><br>
<div class="SubParagraph">
(a) For the exercise of such privilege, a tax is levied at the rate of 6 percent.
</div>
</div>
</div>
<div class="History">
History.—s. 2, ch. 26319, 1949; s. 1, ch. 65-371.
</div>
</div>
</body>
</html>"""


@pytest.fixture
def sample_html_rule_page() -> str:
    """Sample HTML from Florida Administrative Code website."""
    return """<!DOCTYPE html>
<html>
<head>
<meta name="citation_title" content="Specific Exemptions">
<title>12A-1.001 - Florida Administrative Code</title>
</head>
<body>
<h1>12A-1.001 Specific Exemptions</h1>
<table class="RuleContent">
<tr><td>
(1) The following are exempt from the tax imposed by Chapter 212, F.S.:
(a) Occasional or isolated sales.
(b) Sales to the United States Government.
</td></tr>
</table>
<div class="Authority">
Rulemaking Authority: 212.17(6), 212.18(2), 213.06(1) FS.
</div>
<div class="History">
Law Implemented: 212.02, 212.05, 212.08 FS.
</div>
<div class="EffectiveDate">
Effective Date: 7-1-2023
</div>
</body>
</html>"""


@pytest.fixture
def sample_html_taa_index() -> str:
    """Sample HTML from TAA SharePoint index page."""
    return """<!DOCTYPE html>
<html>
<body>
<table class="ms-listviewtable">
<tr>
<td><a href="/taa/TAA%2024A01-001.pdf">TAA 24A01-001</a></td>
<td>Sales Tax on SaaS</td>
<td>1/15/2024</td>
</tr>
<tr>
<td><a href="/taa/TAA%2024A01-002.pdf">TAA 24A01-002</a></td>
<td>Property Tax Exemptions</td>
<td>1/20/2024</td>
</tr>
</table>
</body>
</html>"""


@pytest.fixture
def sample_courtlistener_response() -> dict[str, Any]:
    """Sample response from CourtListener API."""
    return {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {
                "cluster": "https://www.courtlistener.com/api/rest/v3/clusters/9876543/",
                "cluster_id": 9876543,
                "case_name": "Dept. of Revenue v. XYZ Corp.",
                "case_name_full": "Florida Department of Revenue v. XYZ Corporation",
                "citation": ["350 So. 3d 123 (Fla. 2024)"],
                "court_id": "fla",
                "date_filed": "2024-03-15",
                "docket_number": "SC2023-1234",
                "snippet": "This case presents the question of whether digital advertising services are subject to Florida sales tax...",
            },
            {
                "cluster": "https://www.courtlistener.com/api/rest/v3/clusters/8765432/",
                "cluster_id": 8765432,
                "case_name": "Smith v. Dept. of Revenue",
                "citation": ["348 So. 3d 456 (Fla. 1st DCA 2024)"],
                "court_id": "flaapp1",
                "date_filed": "2024-02-10",
            },
        ],
    }


# =============================================================================
# Sample Evaluation Data
# =============================================================================


@pytest.fixture
def sample_eval_question() -> dict[str, Any]:
    """Sample evaluation question."""
    return {
        "id": "eval_001",
        "question": "What is the Florida state sales tax rate?",
        "category": "sales_tax",
        "difficulty": "easy",
        "expected_statutes": ["212.05"],
        "expected_rules": ["12A-1.001"],
        "expected_answer_contains": ["6%", "six percent"],
        "expected_answer_type": "numeric",
        "notes": "Standard rate question - should cite 212.05",
    }


@pytest.fixture
def sample_eval_dataset() -> dict[str, Any]:
    """Sample evaluation dataset."""
    return {
        "version": "1.0.0",
        "questions": [
            {
                "id": "eval_001",
                "question": "What is the Florida state sales tax rate?",
                "category": "sales_tax",
                "difficulty": "easy",
                "expected_statutes": ["212.05"],
                "expected_rules": [],
                "expected_answer_contains": ["6%"],
                "expected_answer_type": "numeric",
            },
            {
                "id": "eval_002",
                "question": "Are groceries exempt from sales tax in Florida?",
                "category": "exemptions",
                "difficulty": "easy",
                "expected_statutes": ["212.08"],
                "expected_rules": ["12A-1.001"],
                "expected_answer_contains": ["exempt", "food"],
                "expected_answer_type": "yes/no",
            },
        ],
    }


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config: Any) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Docker services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary directory for cache testing."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for data testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
