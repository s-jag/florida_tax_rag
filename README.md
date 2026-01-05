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
│   ├── ingestion/          # Document processing pipeline
│   │   ├── models.py       # Unified LegalDocument model
│   │   ├── consolidate.py  # Consolidation functions
│   │   ├── chunking.py     # Hierarchical chunking (parent/child)
│   │   ├── tokenizer.py    # Token counting (tiktoken)
│   │   ├── citation_extractor.py  # Citation extraction
│   │   └── build_citation_graph.py # Citation graph construction
│   ├── graph/              # Neo4j knowledge graph
│   │   ├── schema.py       # Node labels, edge types, constraints
│   │   ├── client.py       # Neo4jClient with connection pooling
│   │   ├── loader.py       # Data loading functions
│   │   └── queries.py      # Graph query functions
│   ├── vector/             # Weaviate vector store
│   │   ├── schema.py       # LegalChunk collection schema
│   │   ├── client.py       # WeaviateClient with hybrid search
│   │   └── embeddings.py   # VoyageEmbedder + Redis cache
│   ├── retrieval/          # Hybrid retrieval system
│   │   ├── models.py       # RetrievalResult, CitationContext
│   │   ├── hybrid.py       # HybridRetriever (vector + keyword + graph)
│   │   ├── graph_expander.py # Neo4j graph expansion
│   │   ├── reranker.py     # Legal-specific reranking
│   │   ├── query_decomposer.py # Claude-powered query decomposition
│   │   ├── multi_retriever.py  # Multi-query parallel retrieval
│   │   └── prompts.py      # LLM prompts for decomposition/scoring
│   ├── agent/              # LangGraph agentic workflow
│   │   ├── state.py        # TaxAgentState TypedDict
│   │   ├── nodes.py        # 7 node functions (decompose, retrieve, etc.)
│   │   ├── edges.py        # Conditional routing logic
│   │   └── graph.py        # StateGraph definition
│   └── generation/         # LLM response generation with citations
│       ├── prompts.py      # Tax attorney system prompt
│       ├── formatter.py    # Chunk formatting for context
│       ├── generator.py    # TaxLawGenerator class
│       └── models.py       # GeneratedResponse, ValidatedCitation
├── config/
│   └── settings.py         # Pydantic settings from environment
├── scripts/
│   ├── scrape_statutes.py
│   ├── scrape_admin_code.py
│   ├── scrape_taa.py
│   ├── scrape_case_law.py
│   ├── audit_raw_data.py   # Data quality audit
│   ├── consolidate_corpus.py # Corpus consolidation
│   ├── chunk_corpus.py     # Hierarchical chunking
│   ├── extract_citations.py # Citation graph extraction
│   ├── init_neo4j.py       # Initialize Neo4j schema + load data
│   ├── init_weaviate.py    # Initialize Weaviate schema
│   ├── generate_embeddings.py # Generate Voyage AI embeddings
│   ├── load_weaviate.py    # Load chunks + embeddings into Weaviate
│   ├── verify_vector_store.py # Verify Weaviate data
│   ├── test_retrieval.py   # Test hybrid retrieval
│   ├── test_decomposition.py # Test query decomposition
│   └── visualize_agent.py  # Visualize LangGraph agent workflow
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
│       ├── embeddings.npz  # Voyage AI embeddings (11 MB)
│       └── statistics.json # Consolidation metrics
├── docker-compose.yml      # Neo4j, Weaviate, Redis services
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

### Embedding Statistics

| Metric | Value |
|--------|-------|
| **Model** | voyage-law-2 |
| **Dimension** | 1,024 |
| **Total Embeddings** | 3,022 |
| **File Size** | 11 MB |
| **Normalized** | Yes (L2 norm = 1.0) |

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

## Docker Services

The system uses three containerized services:

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| Neo4j | neo4j:5.15-community | 7474, 7687 | Knowledge graph (citation network) |
| Weaviate | weaviate:1.28.2 | 8080, 50051 | Vector store (hybrid search) |
| Redis | redis:7-alpine | 6379 | Embedding cache |

### Starting Services

```bash
# Start all services
docker-compose up -d

# Check health status
docker-compose ps

# View logs
docker-compose logs -f weaviate
```

### Neo4j Browser

Access the Neo4j browser at http://localhost:7474
- Username: `neo4j`
- Password: `florida_tax_rag_dev`

## Database Setup

### Initialize Neo4j Knowledge Graph

```bash
# Load schema and data (1,152 documents, 3,022 chunks, 1,126 citations)
python scripts/init_neo4j.py --verify

# To clear and reload
python scripts/init_neo4j.py --clear --verify
```

**Neo4j Statistics:**
| Metric | Count |
|--------|-------|
| Documents | 1,152 |
| Chunks | 3,022 |
| HAS_CHUNK edges | 3,022 |
| CHILD_OF edges | 1,870 |
| Citation edges | 1,126 |

### Initialize Weaviate Vector Store

```bash
# Create LegalChunk collection schema
python scripts/init_weaviate.py --verify

# To delete and recreate
python scripts/init_weaviate.py --delete --verify
```

### Generate Embeddings

```bash
# Generate embeddings for all chunks using Voyage AI
python scripts/generate_embeddings.py --verify

# Test with sample first
python scripts/generate_embeddings.py --sample 10 --verify

# Resume interrupted run
python scripts/generate_embeddings.py --resume
```

### Load Data into Weaviate

```bash
# Load chunks and embeddings into Weaviate
python scripts/load_weaviate.py

# Reset and reload from scratch
python scripts/load_weaviate.py --reset

# Verify the vector store
python scripts/verify_vector_store.py
```

**Weaviate Statistics:**
| Metric | Value |
|--------|-------|
| Collection | LegalChunk |
| Objects | 3,022 |
| Vector Dimension | 1,024 |
| BM25 Config | b=0.75, k1=1.2 |

**Properties:** chunk_id, doc_id, doc_type, level, ancestry, citation, text, text_with_ancestry, effective_date, token_count

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

## Retrieval System

The retrieval system combines vector search, keyword search, and graph traversal for comprehensive legal document retrieval.

### Hybrid Retrieval

```python
from src.retrieval import create_retriever

retriever = create_retriever()

# Basic retrieval
results = retriever.retrieve(
    "What is the Florida sales tax rate?",
    top_k=10,
    alpha=0.5,  # Balance: 0=keyword, 1=vector
)

for r in results:
    print(f"{r.citation} (score: {r.score:.3f})")
    print(f"  Type: {r.doc_type}, Text: {r.text[:100]}...")
```

### Query Decomposition

Complex queries are automatically decomposed into sub-queries for better coverage:

```python
from src.retrieval import create_decomposer
import asyncio

decomposer = create_decomposer()

result = asyncio.run(decomposer.decompose(
    "Do I owe sales tax on software consulting services in Miami?"
))

print(f"Is Simple: {result.is_simple}")
print(f"Sub-queries: {result.query_count}")
for sq in result.sub_queries:
    print(f"  [{sq.priority}] {sq.type}: {sq.text}")
```

**Example Decomposition:**
```
Query: "Do I owe sales tax on software consulting services in Miami?"

Sub-queries (6):
  [1] definition: Florida sales tax definition of software consulting services
  [1] statute: Florida Statutes Chapter 212 taxability of professional services
  [1] statute: Florida sales tax on computer software services and consulting
  [2] rule: Florida Administrative Code 12A-1 rules on software services taxation
  [2] local: Miami-Dade County sales tax surtax rates and application
  [3] exemption: Florida sales tax exemptions for professional services
```

### Multi-Query Retrieval

Run decomposed queries in parallel and merge results:

```python
from src.retrieval import create_multi_retriever
import asyncio

multi_retriever = create_multi_retriever()

result = asyncio.run(multi_retriever.retrieve(
    "Is there sales tax on SaaS products delivered to Florida customers?",
    top_k=10,
))

print(f"Unique documents: {result.unique_doc_ids}")
print(f"Merged results: {len(result.merged_results)}")
```

### Testing Retrieval

```bash
# Test hybrid retrieval
python scripts/test_retrieval.py --query "sales tax exemptions"

# Test query decomposition
python scripts/test_decomposition.py --query "Is SaaS taxable in Florida?"

# Run all decomposition tests
python scripts/test_decomposition.py --all

# Test multi-query retrieval (requires services)
python scripts/test_decomposition.py --query "..." --multi
```

### Retrieval Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│        QueryDecomposer              │
│  • Heuristic complexity check       │
│  • Claude LLM decomposition         │
│  • Returns list[SubQuery]           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        MultiQueryRetriever          │
│  • Parallel sub-query execution     │
│  • Uses HybridRetriever per query   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        HybridRetriever              │
│  1. Embed query (Voyage AI)         │
│  2. Hybrid search (Weaviate)        │
│  3. Graph expansion (Neo4j)         │
│  4. Legal reranking                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        Result Merging               │
│  • Deduplicate by chunk_id          │
│  • Boost multi-match chunks         │
│  • Return top_k results             │
└─────────────────────────────────────┘
```

### Reranking Weights

| Document Type | Weight | Description |
|--------------|--------|-------------|
| Statute | 1.0 | Primary authority (highest) |
| Rule | 0.9 | Implementing authority |
| Case | 0.8 | Interpretive authority |
| TAA | 0.7 | Advisory only |

## LangGraph Agent Workflow

The system uses a LangGraph StateGraph to orchestrate multi-step reasoning over tax law queries. The agent processes queries through 7 specialized nodes that progressively refine results.

### Agent Workflow Diagram

```
┌─────────────────┐
│  decompose_query │ ◄── Break query into sub-queries
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ retrieve_for_subquery│ ◄── Hybrid search (vector + keyword)
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ expand_ │ │ score_       │ ◄── Parallel: graph + relevance
│ graph   │ │ relevance    │
└────┬────┘ └──────┬───────┘
     └──────┬──────┘
            ▼
   ┌────────────────┐
   │filter_irrelevant│ ◄── Remove low-relevance chunks
   └───────┬────────┘
           │
           ▼ (loop if more sub-queries)
   ┌──────────────────────┐
   │check_temporal_validity│ ◄── Verify effective dates
   └───────┬──────────────┘
           │
           ▼
   ┌──────────────────┐
   │ synthesize_answer │ ◄── Prepare citations & context
   └──────────────────┘
```

### Node Descriptions

| Node | Purpose | Key Operations |
|------|---------|----------------|
| `decompose_query` | Break complex queries into sub-queries | Uses Claude LLM for decomposition |
| `retrieve_for_subquery` | Run hybrid retrieval | Vector + keyword search via Weaviate |
| `expand_with_graph` | Find related documents | Neo4j: implementing rules, interpreting cases |
| `score_relevance` | LLM-based relevance scoring | Parallel scoring with semaphore (5 concurrent) |
| `filter_irrelevant` | Remove low-quality chunks | Threshold: 0.5, minimum: 10 chunks |
| `check_temporal_validity` | Verify temporal applicability | Extract tax year, filter future docs |
| `synthesize_answer` | Prepare final response | Calculate confidence, format citations |

### Using the Agent

```python
from src.agent import create_tax_agent_graph
import asyncio

async def query_agent():
    graph = create_tax_agent_graph()

    result = await graph.ainvoke({
        "original_query": "Is software consulting taxable in Miami?"
    })

    print(f"Sub-queries: {len(result['sub_queries'])}")
    print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    print(f"Filtered chunks: {len(result['filtered_chunks'])}")
    print(f"Confidence: {result['confidence']:.2f}")

    for citation in result['citations'][:5]:
        print(f"  - {citation['citation']} ({citation['doc_type']})")

asyncio.run(query_agent())
```

### Confidence Scoring

Confidence is calculated based on the quality of retrieved sources:

| Source Type | Weight |
|-------------|--------|
| Statute | 1.0 |
| Rule | 0.9 |
| Case | 0.7 |
| TAA | 0.6 |

Formula: `confidence = sum(weights for top 5 chunks) / 5`

### Agent State

The `TaxAgentState` TypedDict tracks workflow progress:

```python
# Key state fields
original_query: str          # User's question
sub_queries: list            # Decomposed sub-queries
retrieved_chunks: list       # Accumulates across sub-queries (Annotated[list, add])
filtered_chunks: list        # After relevance filtering
temporally_valid_chunks: list # After temporal validation
citations: list[Citation]    # Prepared citations
confidence: float            # 0-1 confidence score
reasoning_steps: list        # Accumulated reasoning (Annotated[list, add])
errors: list                 # Accumulated errors (Annotated[list, add])
```

### Visualizing the Agent

```bash
# Generate workflow diagram
python scripts/visualize_agent.py
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
REDIS_HOST=localhost
REDIS_PORT=6379
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

## Makefile Commands

```bash
# Docker services
make docker-up          # Start Neo4j, Weaviate, Redis
make docker-down        # Stop services
make docker-logs        # Tail service logs
make docker-reset       # Wipe data and restart fresh
make docker-wait        # Wait for all services to be ready

# Vector store
make generate-embeddings  # Generate Voyage AI embeddings
make init-weaviate        # Initialize Weaviate schema
make load-weaviate        # Load chunks into Weaviate
make verify-weaviate      # Verify vector store

# Development
make install            # Install dependencies
make test               # Run tests
make lint               # Run linting
make format             # Format code
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

- [x] **Phase 3: Knowledge Base**
  - [x] Neo4j knowledge graph schema & data loading
  - [x] Weaviate vector store schema (hybrid search ready)
  - [x] Vector embeddings (Voyage AI voyage-law-2)
  - [x] Load embeddings into Weaviate (3,022 chunks)
  - [x] Hybrid search verification (keyword + vector + filters)

- [x] **Phase 4: Retrieval System**
  - [x] Hybrid retrieval (vector + keyword + graph)
  - [x] Legal-specific reranking (primary authority boost, recency)
  - [x] Query decomposition (Claude-powered)
  - [x] Multi-query parallel retrieval

- [x] **Phase 5: Agentic Workflow**
  - [x] LangGraph StateGraph definition
  - [x] Query decomposition node
  - [x] Hybrid retrieval node
  - [x] Graph expansion node (Neo4j integration)
  - [x] LLM-based relevance scoring
  - [x] Relevance filtering with threshold
  - [x] Temporal validity checking
  - [x] Citation preparation & confidence scoring
  - [x] Unit tests (16 tests passing)
  - [x] Integration tests

- [ ] **Phase 6: Answer Generation & API**
  - [ ] Full answer synthesis with Claude
  - [ ] Citation verification
  - [ ] FastAPI REST endpoint
  - [ ] Streaming responses

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
