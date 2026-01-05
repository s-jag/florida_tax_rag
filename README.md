# Florida Tax Law RAG System

A Hybrid Agentic GraphRAG system for Florida Tax Law, combining vector search, knowledge graphs, and multi-agent orchestration for comprehensive legal research.

## Project Overview

This system provides intelligent retrieval and reasoning over Florida tax law documents including:
- **Florida Statutes** (Chapters 198-220: Taxation and Finance)
- **Florida Administrative Code** (Chapter 12A: Department of Revenue Rules)
- **Technical Assistance Advisements (TAAs)** from Florida DOR
- **Florida Case Law** from CourtListener API

## Architecture

```
florida_tax_rag/
├── src/
│   ├── scrapers/           # Data collection from legal sources
│   │   ├── base.py         # Base scraper with retry, caching, rate limiting
│   │   ├── statutes.py     # Florida Statutes scraper
│   │   ├── admin_code.py   # Florida Administrative Code scraper
│   │   ├── taa.py          # Technical Assistance Advisements scraper
│   │   ├── case_law.py     # CourtListener API case law scraper
│   │   ├── models.py       # Pydantic models for scraped data
│   │   └── utils.py        # Citation parsing utilities
│   └── ingestion/          # Document processing pipeline
│       ├── models.py       # Unified LegalDocument model
│       ├── consolidate.py  # Consolidation functions
│       ├── chunking.py     # Hierarchical chunking (parent/child)
│       ├── tokenizer.py    # Token counting (tiktoken)
│       ├── citation_extractor.py  # Citation extraction
│       └── build_citation_graph.py # Citation graph construction
├── scripts/
│   ├── scrape_statutes.py
│   ├── scrape_admin_code.py
│   ├── scrape_taa.py
│   ├── scrape_case_law.py
│   ├── audit_raw_data.py   # Data quality audit
│   ├── consolidate_corpus.py # Corpus consolidation
│   ├── chunk_corpus.py     # Hierarchical chunking
│   └── extract_citations.py # Citation graph extraction
├── data/
│   ├── raw/                # Raw scraped data
│   │   ├── statutes/       # 742 statute sections
│   │   ├── admin_code/     # 101 administrative rules
│   │   ├── taa/            # Technical Assistance Advisements + PDFs
│   │   └── case_law/       # 308 Florida Supreme Court cases
│   └── processed/          # Processed data
│       ├── corpus.json     # Unified document corpus (4.16 MB)
│       ├── chunks.json     # Hierarchical chunks (11.18 MB)
│       ├── citation_graph.json # Citation relationships (670 KB)
│       └── statistics.json # Consolidation metrics
└── tests/
```

## Data Sources & Coverage

| Source | Documents | Status | Notes |
|--------|-----------|--------|-------|
| Florida Statutes | 742 sections | Complete | Chapters 192-220 (Tax & Finance) |
| Admin Code (12A) | 101 rules | Complete | DOR tax rules |
| TAAs | 1 | Partial | Recent TAAs; historical requires URL enumeration |
| Case Law | 308 cases | Complete | FL Supreme Court via CourtListener |

### Consolidated Corpus Statistics

| Metric | Value |
|--------|-------|
| **Total Documents** | 1,152 |
| **Corpus Size** | 4.16 MB |
| **Statute Citations** | 786 |
| **Rule Citations** | 3 |
| **Case-to-Case Citations** | 2,469 |

### Chunking Statistics

| Metric | Value |
|--------|-------|
| **Total Chunks** | 3,022 |
| **Parent Chunks** | 1,152 |
| **Child Chunks** | 1,870 |
| **Avg Tokens/Chunk** | 362 |
| **Chunks <500 tokens** | 84% |

### Citation Graph Statistics

| Metric | Value |
|--------|-------|
| **Total Edges** | 1,126 |
| **Relation: cites** | 796 |
| **Relation: amends** | 148 |
| **Relation: authority** | 105 |
| **Relation: implements** | 41 |
| **Relation: supersedes** | 35 |

### Average Text Lengths

| Document Type | Avg Length | Min | Max |
|---------------|------------|-----|-----|
| Statutes | 3,482 chars | 59 | 207,504 |
| Rules | 3,589 chars | 1,215 | 44,211 |
| TAAs | 14,800 chars | - | - |
| Cases | 356 chars | 19 | 500 |

> **Note**: Case opinion text is truncated due to CourtListener API limitations. Full opinions available via PDF URLs.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/florida_tax_rag.git
cd florida_tax_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Usage

### Scraping Data

```bash
# Scrape Florida Statutes (tax chapters)
python scripts/scrape_statutes.py --chapters 212 213 214

# Scrape Administrative Code (Chapter 12A)
python scripts/scrape_admin_code.py --division 12A --delay 0.5

# Scrape Technical Assistance Advisements
python scripts/scrape_taa.py --dry-run  # Preview
python scripts/scrape_taa.py --max 20   # Scrape up to 20

# Scrape Case Law from CourtListener
python scripts/scrape_case_law.py --dry-run           # Count available
python scripts/scrape_case_law.py --court fla --max 100  # FL Supreme Court
python scripts/scrape_case_law.py --list-courts       # Show court IDs
```

### Data Models

All scraped data uses Pydantic models with computed citation fields:

```python
from src.scrapers import RawStatute, RawRule, RawTAA, RawCase

# Each model includes:
# - metadata: Type-specific structured fields
# - text: Plain text content
# - source_url: Original source
# - scraped_at: Timestamp

# Citation helpers
statute.metadata.full_citation  # "Fla. Stat. § 212.05"
rule.metadata.full_citation     # "Fla. Admin. Code R. 12A-1.001"
taa.metadata.full_citation      # "Fla. DOR TAA 25A-009"
case.metadata.full_citation     # "Case Name, 215 So. 3d 46"
```

## Key Features

### Scraper Infrastructure
- **Rate limiting**: Configurable delays between requests
- **Retry logic**: Exponential backoff on failures
- **Caching**: Avoid redundant fetches
- **Structured logging**: Via structlog

### Citation Extraction
Automatic extraction of cross-references:
- Statute citations: `§ 212.05`, `Section 212.02(10)(i), F.S.`
- Rule citations: `Rule 12A-1.073, F.A.C.`
- Case citations: CourtListener cluster IDs

### PDF Processing
TAAs are distributed as PDFs:
- Primary: pdfplumber (better text extraction)
- Fallback: pypdf
- Extracted text + metadata saved as JSON

## Dependencies

### Core
- `langchain` / `langgraph` - Agent orchestration
- `weaviate-client` - Vector database
- `neo4j` - Graph database
- `voyageai` - Legal embeddings
- `anthropic` - Claude LLM

### Scraping
- `httpx` - Async HTTP client
- `beautifulsoup4` / `lxml` - HTML parsing
- `pdfplumber` / `pypdf` - PDF text extraction
- `tenacity` - Retry logic

### Data Validation
- `pydantic` - Data models and validation

## Configuration

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_key
VOYAGE_API_KEY=your_key
WEAVIATE_URL=http://localhost:8080
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/

# Format
ruff format src/
```

## Roadmap

- [x] **Phase 1: Data Collection**
  - [x] Florida Statutes scraper
  - [x] Administrative Code scraper
  - [x] TAA scraper (PDF extraction)
  - [x] Case Law scraper (CourtListener API)

- [x] **Phase 2: Data Processing**
  - [x] Data quality audit (`scripts/audit_raw_data.py`)
  - [x] Unified document model (`src/ingestion/models.py`)
  - [x] Corpus consolidation (`data/processed/corpus.json`)
  - [x] Hierarchical chunking (`src/ingestion/chunking.py`)
  - [x] Citation extraction (`src/ingestion/citation_extractor.py`)
  - [x] Citation graph construction (`data/processed/citation_graph.json`)

- [ ] **Phase 3: Knowledge Base**
  - [ ] Vector embeddings (Voyage AI)
  - [ ] Neo4j knowledge graph
  - [ ] Weaviate vector store

- [ ] **Phase 4: Retrieval & Agents**
  - [ ] Hybrid retrieval (vector + graph)
  - [ ] Multi-agent orchestration
  - [ ] Query decomposition
  - [ ] Citation verification

## Documentation

- [SCRAPING_NOTES.md](./SCRAPING_NOTES.md) - Detailed scraping documentation
- [data/processed/README.md](./data/processed/README.md) - Unified corpus schema documentation

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [CourtListener](https://www.courtlistener.com/) - Free legal research API
- [Florida Legislature](https://www.leg.state.fl.us/) - Official statutes
- [Florida Administrative Code](https://www.flrules.org/) - Administrative rules
- [Florida DOR](https://floridarevenue.com/) - Tax guidance documents
