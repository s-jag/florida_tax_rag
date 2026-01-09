# Replicating the Tax RAG System: Federal Tax Law Edition

A comprehensive guide to building a production-grade Retrieval-Augmented Generation (RAG) system for federal tax law, based on the Florida Tax RAG architecture.

**Target Repository**: `federal_tax_rag`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Federal Tax Data Sources](#3-federal-tax-data-sources)
4. [Project Structure](#4-project-structure)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Sequential Implementation Prompts](#6-sequential-implementation-prompts)

---

## 1. Architecture Overview

### 1.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FEDERAL TAX RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────────┐  │
│  │   Streamlit  │────▶│                  FastAPI Backend                    │  │
│  │   Frontend   │◀────│  /api/v1/query  /api/v1/stream  /api/v1/health     │  │
│  └──────────────┘     └───────────────────────┬─────────────────────────────┘  │
│                                               │                                 │
│                                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        LangGraph Agent Pipeline                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ Decompose│─▶│ Retrieve │─▶│  Expand  │─▶│  Score   │─▶│  Filter  │   │  │
│  │  │  Query   │  │ Parallel │  │  Graph   │  │Relevance │  │Irrelevant│   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  │       │                                                        │         │  │
│  │       │        ┌──────────┐  ┌──────────┐  ┌──────────┐       │         │  │
│  │       └───────▶│Synthesize│─▶│ Validate │─▶│ Correct  │◀──────┘         │  │
│  │                │  Answer  │  │ Response │  │  (if needed)               │  │
│  │                └──────────┘  └──────────┘  └──────────┘                 │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                       │                              │                          │
│          ┌────────────┴────────────┐    ┌───────────┴───────────┐              │
│          ▼                         ▼    ▼                       ▼              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   ┌──────────┐   │
│  │   Weaviate    │    │    Neo4j      │    │  Voyage AI    │   │  Claude  │   │
│  │ Vector Store  │    │ Knowledge     │    │  Embeddings   │   │   LLM    │   │
│  │  (Hybrid)     │    │    Graph      │    │ (voyage-law-2)│   │ (Sonnet) │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   └──────────┘   │
│          │                    │                                               │
│          └────────────────────┴──────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Redis Cache Layer                              │ │
│  │              (Embedding Cache + Query Result Cache)                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ▲
                                    │
┌───────────────────────────────────┴───────────────────────────────────────────┐
│                           DATA INGESTION PIPELINE                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    IRC      │  │  Treasury   │  │  Tax Court  │  │    IRS      │          │
│  │  Scraper    │  │ Regulations │  │   Cases     │  │  Guidance   │          │
│  │(uscode.gov) │  │  (eCFR)     │  │ (ustaxcourt)│  │  (PLRs/RRs) │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                │                  │
│         └────────────────┴────────────────┴────────────────┘                  │
│                                   │                                           │
│                                   ▼                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Consolidate │─▶│   Chunk     │─▶│  Generate   │─▶│    Load     │          │
│  │   Corpus    │  │ Documents   │  │ Embeddings  │  │  Databases  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘          │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

#### LangGraph Agent Pipeline

The heart of the system is a stateful, multi-node agent built with LangGraph. Each node performs a specific function:

| Node | Purpose | Key Operations |
|------|---------|----------------|
| **decompose_query** | Break complex queries into sub-queries | LLM analysis, complexity detection |
| **retrieve_for_subquery** | Parallel hybrid retrieval | Vector + BM25 search, deduplication |
| **expand_with_graph** | Citation network expansion | Neo4j traversal for related documents |
| **score_relevance** | LLM-based relevance scoring | Parallel scoring with semaphore |
| **filter_irrelevant** | Remove low-quality results | Threshold filtering (0.5), minimum 10 |
| **check_temporal_validity** | Tax year applicability | Effective date checking |
| **synthesize_answer** | Generate response | Claude with citations |
| **validate_response** | Hallucination detection | Claim verification against sources |
| **correct_response** | Self-correction | Fix minor issues, adjust confidence |

#### Agent State

```python
class TaxAgentState(TypedDict, total=False):
    # Input
    original_query: str

    # Decomposition
    sub_queries: list[SubQuery]
    current_sub_query_idx: int
    is_simple_query: bool
    decomposition_reasoning: str

    # Retrieval (accumulates across sub-queries)
    retrieved_chunks: Annotated[list, add]
    current_retrieval_results: list

    # Graph Expansion
    graph_context: list[CitationContext]
    interpretation_chains: dict

    # Filtering
    relevance_scores: dict[str, float]
    filtered_chunks: list
    temporally_valid_chunks: list

    # Output
    final_answer: str | None
    citations: list[Citation]
    confidence: float

    # Validation/Correction
    validation_result: dict | None
    correction_result: dict | None
    validation_passed: bool

    # Metadata
    reasoning_steps: Annotated[list, add]
    errors: Annotated[list, add]
```

### 1.3 Hybrid Retrieval Strategy

The retrieval system combines multiple search strategies:

```
Query → Embedding (Voyage AI voyage-law-2)
         │
    ┌────┴────┐
    ▼         ▼
Vector Search  Keyword Search
(Cosine 1024d)   (BM25)
    │         │
    └────┬────┘
         ▼
  Hybrid Merge (α=0.25)
  hybrid_score = 0.25*vector + 0.75*keyword
         │
         ▼
  Graph Expansion (Neo4j)
  IRC § → Regulations → Cases → PLRs
         │
         ▼
  Legal Reranker
  - Type weight: statute > regulation > case > PLR
  - Recency boost
  - Authority hierarchy
         │
         ▼
    Top-K Results
```

**Why α=0.25 (keyword-heavy)?**
Legal documents have high keyword specificity. Exact terms like "IRC § 61" or "26 CFR 1.61-1" must match precisely. Semantic similarity alone would miss these critical legal references.

### 1.4 Knowledge Graph Schema

```
Node Labels:
  :Document (base for all)
  :Statute (IRC sections)
  :Regulation (Treasury Regs)
  :Case (Tax Court, Circuit Courts)
  :Guidance (PLRs, Rev. Rulings, TAMs)
  :Chunk (document segments)

Relationships:
  (Regulation)-[:IMPLEMENTS]->(Statute)
  (Case)-[:INTERPRETS]->(Statute)
  (Case)-[:INTERPRETS]->(Regulation)
  (Guidance)-[:INTERPRETS]->(Statute)
  (Document)-[:CITES]->(Document)
  (Document)-[:AMENDS]->(Document)
  (Document)-[:SUPERSEDES]->(Document)
  (Document)-[:HAS_CHUNK]->(Chunk)
  (Chunk)-[:CHILD_OF]->(Chunk)
```

### 1.5 Authority Hierarchy

Federal tax law has a clear authority hierarchy that the system respects:

```
1. Internal Revenue Code (IRC)     [Weight: 3.0] - Primary statutory authority
2. Treasury Regulations            [Weight: 2.5] - Binding interpretive rules
3. Tax Court / Circuit Court Cases [Weight: 2.0] - Judicial interpretations
4. Revenue Rulings                 [Weight: 1.5] - IRS official guidance
5. Private Letter Rulings (PLRs)   [Weight: 1.0] - Taxpayer-specific (advisory only)
6. Technical Advice Memoranda      [Weight: 1.0] - Advisory
```

---

## 2. Technology Stack

### 2.1 Core Dependencies

```toml
[project]
name = "federal_tax_rag"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # LangChain Ecosystem
    "langchain>=0.3.0",
    "langgraph>=0.2.0",

    # Vector & Graph Databases
    "weaviate-client>=4.0.0",
    "neo4j>=5.0.0",

    # LLM Providers
    "anthropic>=0.40.0",
    "openai>=1.0.0",        # For evaluation
    "voyageai>=0.3.0",      # Legal embeddings

    # Web Framework
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",

    # Web Scraping
    "beautifulsoup4>=4.12.0",
    "httpx>=0.27.0",
    "lxml>=5.0.0",

    # PDF Processing
    "pypdf>=4.0.0",
    "pdfplumber>=0.11.0",

    # Configuration & Validation
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",

    # Utilities
    "tenacity>=9.0.0",      # Retry logic
    "structlog>=24.0.0",    # Structured logging
    "redis>=5.0.0",         # Caching
    "tiktoken>=0.5.0",      # Token counting
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]
```

### 2.2 External Services

| Service | Purpose | Model/Version | API Key Required |
|---------|---------|---------------|------------------|
| **Voyage AI** | Legal embeddings | `voyage-law-2` (1024d) | Yes |
| **Anthropic** | LLM generation | `claude-sonnet-4-20250514` | Yes |
| **OpenAI** | Evaluation judge | `gpt-4` | Optional |
| **Neo4j** | Knowledge graph | 5.0+ | Yes (password) |
| **Weaviate** | Vector store | 1.28+ | Optional (cloud) |
| **Redis** | Caching | 7+ | No (local) |

### 2.3 Docker Services

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/federal_tax_rag_dev
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - ./docker-data/neo4j:/data

  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.28.2
    ports:
      - "8080:8080"   # REST
      - "50051:50051" # gRPC
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: true
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
    volumes:
      - ./docker-data/weaviate:/var/lib/weaviate

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - ./docker-data/redis:/data
```

---

## 3. Federal Tax Data Sources

### 3.1 Internal Revenue Code (IRC)

**Source**: U.S. Code Title 26
**URL**: https://uscode.house.gov/browse/prelim@title26
**Alternative**: https://www.law.cornell.edu/uscode/text/26

**Structure**:
```
Title 26 - Internal Revenue Code
├── Subtitle A - Income Taxes (§§ 1-1564)
│   ├── Chapter 1 - Normal Taxes and Surtaxes
│   │   ├── Subchapter A - Determination of Tax Liability
│   │   ├── Subchapter B - Computation of Taxable Income
│   │   └── ...
│   └── Chapter 2-6 - Other taxes
├── Subtitle B - Estate and Gift Taxes (§§ 2001-2801)
├── Subtitle C - Employment Taxes (§§ 3101-3510)
├── Subtitle D - Miscellaneous Excise Taxes (§§ 4001-5000)
├── Subtitle E - Alcohol, Tobacco, and Firearms (§§ 5001-5891)
├── Subtitle F - Procedure and Administration (§§ 6001-7874)
└── Subtitles G-K - Other provisions
```

**Scraping Strategy**:
1. Fetch title index page
2. Parse subtitle/chapter/subchapter hierarchy
3. Extract individual section text with subsections
4. Parse effective dates from notes
5. Extract cross-references to other IRC sections

**Citation Format**: `IRC § 61(a)(1)`, `26 U.S.C. § 61`

### 3.2 Treasury Regulations

**Source**: Electronic Code of Federal Regulations (eCFR)
**URL**: https://www.ecfr.gov/current/title-26

**Structure**:
```
Title 26 - Internal Revenue
├── Chapter I - Internal Revenue Service
│   ├── Subchapter A - Income Tax
│   │   ├── Part 1 - Income Taxes (§§ 1.1-1.1563-1)
│   │   │   ├── § 1.61-1 Gross income
│   │   │   ├── § 1.61-2 Compensation for services
│   │   │   └── ...
│   │   └── Parts 2-17
│   ├── Subchapter B - Estate and Gift Taxes
│   ├── Subchapter C - Employment Taxes
│   └── Subchapters D-H
└── Chapters II-IV
```

**eCFR API**:
```
Base: https://www.ecfr.gov/api/versioner/v1
GET /titles - List titles
GET /full/{date}/title-{number} - Full title content
GET /structure/{date}/title-{number} - Structure only
```

**Scraping Strategy**:
1. Fetch title structure via API
2. Extract part/section hierarchy
3. Download section content in XML format
4. Parse implementing statutes (authority citations)
5. Extract effective dates

**Citation Format**: `Treas. Reg. § 1.61-1`, `26 CFR 1.61-1`

### 3.3 Tax Court Cases

**Source**: U.S. Tax Court
**URL**: https://www.ustaxcourt.gov/UstcInOp/OpinionSearch.aspx

**Case Types**:
- Regular opinions (precedential)
- Memorandum opinions (fact-specific)
- Summary opinions (small tax cases, not precedential)
- Bench opinions (oral decisions)

**Alternative Sources**:
- CourtListener API: https://www.courtlistener.com/api/rest/v3/
- PACER (requires account)

**Scraping Strategy**:
1. Search Tax Court website by date range
2. Download opinion PDFs
3. Extract text with pdfplumber
4. Parse case name, docket number, judge
5. Extract IRC citations from opinion text
6. Identify holding/reasoning sections

**Citation Format**: `Smith v. Commissioner, 123 T.C. 456 (2024)`

### 3.4 IRS Guidance Documents

**Source**: IRS Website
**URL**: https://www.irs.gov/

#### Revenue Rulings
**Path**: /pub/irs-irbs/ (Internal Revenue Bulletins)

#### Private Letter Rulings (PLRs)
**Path**: /irs/private-letter-rulings-technical-advice-memoranda-and-field-service-advice/
**Format**: PLR 202401001 (Year + Week + Sequence)

#### Revenue Procedures
**Path**: /pub/irs-irbs/

#### Technical Advice Memoranda (TAMs)
**Path**: Same as PLRs

**Scraping Strategy**:
1. Index Internal Revenue Bulletin archives
2. Download PDFs/HTML content
3. Parse document type and number
4. Extract issue addressed
5. Parse IRC/regulation citations
6. Identify holding section

**Citation Formats**:
- Revenue Ruling: `Rev. Rul. 2024-1`
- PLR: `PLR 202401001`
- Revenue Procedure: `Rev. Proc. 2024-1`
- TAM: `TAM 202401001`

### 3.5 Federal Circuit Court Cases

**Source**: CourtListener API
**URL**: https://www.courtlistener.com/api/rest/v3/

**Relevant Courts**:
- Supreme Court of the United States (scotus)
- U.S. Court of Appeals for Federal Circuit (cafc)
- All Circuit Courts (ca1-ca11, cadc)
- U.S. Court of Federal Claims (uscfc)

**API Endpoints**:
```
GET /search/?q=internal+revenue&court=cafc&type=o
GET /opinions/{id}/
GET /clusters/{id}/
```

**Scraping Strategy**:
1. Search by tax-related keywords
2. Filter by court and date
3. Download opinion text
4. Parse IRC citations
5. Extract holding

**Citation Format**: `Smith v. United States, 123 F.3d 456 (Fed. Cir. 2024)`

### 3.6 Data Source Summary

| Source | Document Type | Estimated Volume | Update Frequency |
|--------|--------------|------------------|------------------|
| IRC (uscode.house.gov) | Statutes | ~10,000 sections | Annual (tax acts) |
| Treasury Regs (eCFR) | Regulations | ~50,000 sections | Ongoing |
| Tax Court | Cases | ~500/year | Weekly |
| IRS Guidance | PLRs, RRs, etc. | ~2,000/year | Weekly |
| Circuit Courts | Appeals | ~200/year | Ongoing |

---

## 4. Project Structure

```
federal_tax_rag/
├── config/
│   ├── __init__.py
│   ├── settings.py              # Pydantic settings
│   └── prompts/
│       ├── __init__.py
│       ├── retrieval.py         # Query decomposition prompts
│       ├── generation.py        # Answer synthesis prompts
│       └── evaluation.py        # LLM judge prompts
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent/                   # LangGraph agent
│   │   ├── __init__.py
│   │   ├── state.py            # TaxAgentState definition
│   │   ├── nodes.py            # Node functions
│   │   ├── edges.py            # Conditional routing
│   │   └── graph.py            # Graph compilation
│   │
│   ├── api/                     # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py             # Application factory
│   │   ├── routes.py           # Endpoint definitions
│   │   ├── models.py           # Request/response schemas
│   │   ├── dependencies.py     # Dependency injection
│   │   ├── middleware.py       # CORS, rate limit, logging
│   │   ├── errors.py           # Custom exceptions
│   │   └── cache.py            # Query result caching
│   │
│   ├── retrieval/               # Retrieval system
│   │   ├── __init__.py
│   │   ├── hybrid.py           # Hybrid retriever
│   │   ├── query_decomposer.py # Query analysis
│   │   ├── graph_expander.py   # Neo4j expansion
│   │   ├── reranker.py         # Legal reranking
│   │   ├── models.py           # Retrieval models
│   │   └── multi_retriever.py  # Multi-strategy retrieval
│   │
│   ├── generation/              # Answer generation
│   │   ├── __init__.py
│   │   ├── generator.py        # TaxLawGenerator
│   │   ├── validator.py        # Hallucination detection
│   │   ├── corrector.py        # Self-correction
│   │   ├── formatter.py        # Context formatting
│   │   └── models.py           # Generation models
│   │
│   ├── vector/                  # Vector store
│   │   ├── __init__.py
│   │   ├── client.py           # Weaviate client
│   │   └── embeddings.py       # Voyage AI embeddings
│   │
│   ├── graph/                   # Knowledge graph
│   │   ├── __init__.py
│   │   ├── client.py           # Neo4j client
│   │   ├── schema.py           # Graph schema
│   │   ├── loader.py           # Data loading
│   │   └── queries.py          # Cypher queries
│   │
│   ├── scrapers/                # Data collection
│   │   ├── __init__.py
│   │   ├── base.py             # BaseScraper class
│   │   ├── models.py           # Scraper models
│   │   ├── irc.py              # IRC scraper
│   │   ├── treasury_regs.py    # eCFR scraper
│   │   ├── tax_court.py        # Tax Court scraper
│   │   ├── irs_guidance.py     # PLRs, Rev. Rulings
│   │   └── circuit_courts.py   # CourtListener scraper
│   │
│   ├── ingestion/               # Data processing
│   │   ├── __init__.py
│   │   ├── models.py           # LegalDocument, LegalChunk
│   │   ├── consolidate.py      # Corpus consolidation
│   │   ├── chunking.py         # Hierarchical chunking
│   │   ├── citation_extractor.py
│   │   └── build_citation_graph.py
│   │
│   ├── evaluation/              # Quality evaluation
│   │   ├── __init__.py
│   │   ├── runner.py           # Evaluation orchestrator
│   │   ├── models.py           # Eval models
│   │   ├── metrics.py          # Citation metrics
│   │   ├── authority_metrics.py # Legal hierarchy metrics
│   │   ├── faithfulness.py     # Faithfulness analysis
│   │   ├── llm_judge.py        # LLM evaluation
│   │   ├── correction_metrics.py
│   │   ├── report.py           # Report generation
│   │   └── visualizations.py
│   │
│   └── observability/           # Logging & metrics
│       ├── __init__.py
│       ├── logging.py          # Structured logging
│       ├── metrics.py          # Metrics collection
│       ├── profiler.py         # Pipeline profiling
│       └── context.py          # Request context
│
├── scripts/                     # Utility scripts
│   ├── scrape_irc.py
│   ├── scrape_treasury_regs.py
│   ├── scrape_tax_court.py
│   ├── scrape_irs_guidance.py
│   ├── scrape_circuit_courts.py
│   ├── consolidate_corpus.py
│   ├── chunk_corpus.py
│   ├── generate_embeddings.py
│   ├── extract_citations.py
│   ├── init_neo4j.py
│   ├── load_weaviate.py
│   ├── load_neo4j.py
│   ├── run_evaluation.py
│   └── wait_for_services.py
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Fixtures
│   ├── test_api.py
│   ├── test_agent_graph.py
│   ├── test_agent_nodes.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_vector_client.py
│   ├── test_graph_client.py
│   ├── test_chunking.py
│   ├── test_scrapers/
│   │   ├── test_irc_scraper.py
│   │   ├── test_treasury_regs_scraper.py
│   │   └── ...
│   └── integration/
│       └── test_full_pipeline.py
│
├── streamlit_app/
│   └── app.py                   # Streamlit frontend
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Scraped data
│   │   ├── irc/
│   │   ├── treasury_regs/
│   │   ├── tax_court/
│   │   └── irs_guidance/
│   └── processed/               # Processed data
│       ├── corpus.json
│       ├── chunks.json
│       └── embeddings.npz
│
├── docker-data/                 # Docker volumes (gitignored)
├── .env.example                 # Environment template
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
└── README.md
```

---

## 5. Implementation Walkthrough

### Phase 1: Project Foundation (Prompts 1-5)
- Repository setup
- Configuration management
- Docker environment
- Basic project structure

### Phase 2: Data Collection (Prompts 6-11)
- Base scraper infrastructure
- IRC scraper
- Treasury Regulations scraper
- Tax Court scraper
- IRS Guidance scraper
- Circuit Courts scraper

### Phase 3: Data Processing (Prompts 12-16)
- Data models
- Corpus consolidation
- Hierarchical chunking
- Citation extraction
- Embedding generation

### Phase 4: Storage Layer (Prompts 17-20)
- Weaviate client and schema
- Neo4j client and schema
- Data loading scripts
- Verification scripts

### Phase 5: Retrieval System (Prompts 21-24)
- Query decomposition
- Hybrid retrieval
- Graph expansion
- Legal reranking

### Phase 6: Generation Pipeline (Prompts 25-27)
- Answer generator
- Hallucination validator
- Self-correction

### Phase 7: Agent Orchestration (Prompts 28-29)
- LangGraph state and nodes
- Agent graph compilation

### Phase 8: API & Deployment (Prompts 30-32)
- FastAPI application
- Streamlit frontend
- Production deployment

---

## 6. Sequential Implementation Prompts

Below are 32 detailed, self-contained prompts for implementing the Federal Tax RAG system. Each prompt should be given to Claude Code sequentially. The prompts instruct the agent to enter plan mode, implement the feature, test it, and commit the changes.

---

### Prompt 1: Repository Initialization

```
I'm building a Federal Tax RAG system called "federal_tax_rag". This is the first prompt in a series of ~30 prompts that will build out the complete system.

**Your Task**: Initialize the repository with the foundational project structure.

**Enter plan mode first** to design the approach.

**Requirements**:
1. Create a new git repository with proper .gitignore
2. Create pyproject.toml with the following dependencies:
   - langchain>=0.3.0, langgraph>=0.2.0
   - weaviate-client>=4.0.0, neo4j>=5.0.0
   - anthropic>=0.40.0, openai>=1.0.0, voyageai>=0.3.0
   - fastapi>=0.115.0, uvicorn[standard]>=0.32.0
   - beautifulsoup4>=4.12.0, httpx>=0.27.0, lxml>=5.0.0
   - pypdf>=4.0.0, pdfplumber>=0.11.0
   - pydantic>=2.0.0, pydantic-settings>=2.0.0, python-dotenv>=1.0.0
   - tenacity>=9.0.0, structlog>=24.0.0, redis>=5.0.0, tiktoken>=0.5.0
   - Dev dependencies: pytest>=8.0.0, pytest-asyncio>=0.24.0, pytest-cov>=6.0.0, ruff>=0.8.0, mypy>=1.13.0
3. Configure ruff (line-length=100, target py311)
4. Configure pytest (asyncio_mode="auto", testpaths=["tests"])
5. Create empty directory structure:
   - config/, src/, scripts/, tests/, streamlit_app/, data/raw/, data/processed/
   - src/agent/, src/api/, src/retrieval/, src/generation/, src/vector/, src/graph/, src/scrapers/, src/ingestion/, src/evaluation/, src/observability/
6. Create .env.example with placeholder API keys

**Testing**:
- Verify pyproject.toml is valid: `pip install -e .` should work
- Verify ruff config: `ruff check .` should run without config errors

**Commit**: Create commit with message: "chore: initialize federal_tax_rag repository with project structure"

**Verification before next prompt**:
- [ ] Git repository initialized
- [ ] pyproject.toml valid and installable
- [ ] Directory structure created
- [ ] .gitignore includes data/, docker-data/, .env, __pycache__, etc.
```

---

### Prompt 2: Configuration Management

```
Continuing the Federal Tax RAG build. Previous step created the repository structure.

**Your Task**: Implement the configuration management system using Pydantic Settings.

**Enter plan mode first** to design the settings structure.

**Requirements**:
1. Create config/__init__.py with get_settings() function
2. Create config/settings.py with Settings class:
   ```python
   class Settings(BaseSettings):
       # Environment
       env: str = "development"  # development | staging | production
       debug: bool = True
       log_level: str = "INFO"

       # API Keys (Required)
       voyage_api_key: SecretStr
       anthropic_api_key: SecretStr
       neo4j_password: SecretStr

       # Optional APIs
       openai_api_key: SecretStr | None = None
       weaviate_api_key: SecretStr | None = None

       # Neo4j
       neo4j_uri: str = "bolt://localhost:7687"
       neo4j_user: str = "neo4j"
       neo4j_connection_timeout: int = 30
       neo4j_max_connection_pool_size: int = 50

       # Weaviate
       weaviate_url: str = "http://localhost:8080"
       weaviate_grpc_port: int = 50051
       weaviate_timeout: int = 30

       # Redis
       redis_url: str = "redis://localhost:6379/0"

       # Retrieval
       retrieval_top_k: int = 20
       hybrid_alpha: float = 0.25
       expand_graph: bool = True
       max_graph_hops: int = 2

       # Generation
       llm_model: str = "claude-sonnet-4-20250514"
       llm_temperature: float = 0.1
       max_tokens: int = 4096

       # Embedding
       embedding_model: str = "voyage-law-2"
       embedding_batch_size: int = 128
       embedding_cache_ttl: int = 86400

       # Rate Limits
       rate_limit_per_minute: int = 60
       voyage_requests_per_minute: int = 300
       anthropic_requests_per_minute: int = 60
   ```
3. Use model_config with env_file=".env"
4. Implement @lru_cache decorator for get_settings()
5. Add validation for URLs and positive integers

**Testing**:
- Create tests/test_settings.py
- Test default values load correctly
- Test environment variable override
- Test validation errors for invalid values
- Run: `pytest tests/test_settings.py -v`

**Commit**: Create commit with message: "feat(config): add Pydantic settings management"

**Verification before next prompt**:
- [ ] Settings class created with all fields
- [ ] get_settings() returns cached singleton
- [ ] Environment variables override defaults
- [ ] All tests pass
```

---

### Prompt 3: Docker Environment Setup

```
Continuing the Federal Tax RAG build. Previous step created configuration management.

**Your Task**: Create Docker Compose configuration for local development.

**Enter plan mode first** to design the Docker setup.

**Requirements**:
1. Create docker-compose.yml with services:
   - neo4j:5.15-community
     - Ports: 7474, 7687
     - Auth: neo4j/federal_tax_rag_dev
     - APOC plugin enabled
     - Volume: ./docker-data/neo4j:/data
     - Healthcheck on port 7474
   - weaviate:1.28.2
     - Ports: 8080, 50051
     - Anonymous access enabled
     - No vectorizer (external embeddings)
     - Volume: ./docker-data/weaviate:/var/lib/weaviate
     - Healthcheck on readiness endpoint
   - redis:7-alpine
     - Port: 6379
     - AOF persistence
     - Volume: ./docker-data/redis:/data
     - Healthcheck with redis-cli ping

2. Create shared network: federal_tax_network

3. Create scripts/wait_for_services.py:
   - Check Neo4j connectivity (bolt protocol)
   - Check Weaviate readiness (/v1/.well-known/ready)
   - Check Redis ping
   - Exit with appropriate codes
   - Timeout after 60 seconds with retries

4. Update .gitignore to include docker-data/

**Testing**:
- Run: `docker compose up -d`
- Run: `python scripts/wait_for_services.py`
- Verify all services are healthy: `docker compose ps`

**Commit**: Create commit with message: "infra: add Docker Compose for local development"

**Verification before next prompt**:
- [ ] docker-compose.yml created
- [ ] All services start successfully
- [ ] wait_for_services.py confirms connectivity
- [ ] Data persists in docker-data/ volumes
```

---

### Prompt 4: Observability Foundation

```
Continuing the Federal Tax RAG build. Previous step set up Docker environment.

**Your Task**: Implement the observability layer with structured logging, metrics, and request context.

**Enter plan mode first** to design the observability components.

**Requirements**:
1. Create src/observability/__init__.py exporting main components

2. Create src/observability/logging.py:
   - configure_logging(json_output: bool, log_level: str) function
   - Use structlog with processors:
     - TimeStamper (ISO format)
     - add_log_level
     - PositionalArgumentsFormatter
     - StackInfoRenderer
     - format_exc_info
     - CallsiteParameterAdder (filename, lineno)
   - JSON output for production, colored console for development
   - get_logger(__name__) helper function

3. Create src/observability/context.py:
   - ContextVar for request_id and query_id
   - set_request_context(request_id, query_id) function
   - get_context() -> dict function
   - clear_request_context() function

4. Create src/observability/metrics.py:
   - MetricsCollector class (thread-safe singleton)
   - Track: total_queries, successful_queries, failed_queries
   - Track: latency (total, min, max)
   - Track: errors_by_type dict
   - record_query(latency_ms, success, error_type) method
   - get_stats() -> dict method
   - reset() method for testing

5. Create src/observability/profiler.py:
   - PipelineProfiler class
   - Context manager for profiling stages
   - profile_request(request_id) context manager
   - stage(name, **metadata) context manager
   - get_summary() -> ProfileSummary method

**Testing**:
- Create tests/test_observability.py
- Test logging configuration (JSON and console)
- Test context variable propagation
- Test metrics recording and aggregation
- Test profiler stage timing
- Run: `pytest tests/test_observability.py -v`

**Commit**: Create commit with message: "feat(observability): add logging, metrics, and profiling"

**Verification before next prompt**:
- [ ] Structured logging works in both modes
- [ ] Request context propagates correctly
- [ ] Metrics collect and aggregate properly
- [ ] Profiler times stages accurately
- [ ] All tests pass
```

---

### Prompt 5: Prompt Templates

```
Continuing the Federal Tax RAG build. Previous step created observability layer.

**Your Task**: Create the prompt template library for the RAG system.

**Enter plan mode first** to design the prompt structure.

**Requirements**:
1. Create config/prompts/__init__.py exporting all prompts

2. Create config/prompts/retrieval.py with:
   - QUERY_DECOMPOSITION_PROMPT: System prompt for breaking complex tax queries into sub-queries
     - Include query type classification (definition, computation, procedure, penalty, exemption, etc.)
     - Federal tax specific examples
   - RELEVANCE_SCORING_PROMPT: Score chunk relevance 0-1
   - QUERY_CLASSIFICATION_PROMPT: Classify query complexity

3. Create config/prompts/generation.py with:
   - TAX_ATTORNEY_SYSTEM_PROMPT: Role as experienced federal tax attorney
     - Cite IRC sections, Treasury Regulations, cases
     - Use [Source: citation] format
     - Distinguish binding vs advisory authority
   - ANSWER_SYNTHESIS_PROMPT: Generate answer with citations
   - HALLUCINATION_CHECK_PROMPT: Verify claims against sources
   - CORRECTION_PROMPT: Fix identified issues

4. Create config/prompts/evaluation.py with:
   - LLM_JUDGE_PROMPT: Evaluate answer quality (1-10 scale)
     - Correctness, completeness, clarity, citation accuracy
     - Identify hallucinations
   - FAITHFULNESS_PROMPT: Check faithfulness to sources

5. All prompts should:
   - Be formatted as multi-line strings
   - Include clear instructions
   - Have placeholders for dynamic content ({query}, {context}, etc.)
   - Include federal tax specific terminology and examples

**Testing**:
- Create tests/test_prompts.py
- Verify all prompts are importable
- Verify prompts have required placeholders
- Verify no syntax errors in format strings
- Run: `pytest tests/test_prompts.py -v`

**Commit**: Create commit with message: "feat(prompts): add RAG prompt templates for federal tax"

**Verification before next prompt**:
- [ ] All prompt modules created
- [ ] Prompts use federal tax terminology
- [ ] Placeholders are consistent
- [ ] All tests pass
```

---

### Prompt 6: Base Scraper Infrastructure

```
Continuing the Federal Tax RAG build. Previous step created prompt templates.

**Your Task**: Implement the base scraper infrastructure that all scrapers will inherit from.

**Enter plan mode first** to design the scraper base class.

**Requirements**:
1. Create src/scrapers/__init__.py exporting scraper classes

2. Create src/scrapers/models.py with base models:
   - DocumentType enum: STATUTE, REGULATION, CASE, GUIDANCE
   - RawDocument base class with:
     - id, doc_type, title, text, html, source_url
     - scraped_at, metadata dict
   - ScraperConfig: rate_limit_delay, max_retries, timeout, user_agent

3. Create src/scrapers/base.py with BaseScraper class:
   - __init__(config: ScraperConfig)
   - Async httpx client with connection pooling
   - Rotating User-Agent strings
   - @retry decorator using tenacity:
     - Exponential backoff (1s, 2s, 4s)
     - Max 3 retries
     - Retry on httpx.HTTPStatusError (429, 500, 502, 503)
   - Rate limiting between requests (default 1.0s delay)
   - URL caching with SHA256 hash (avoid re-fetching)
   - fetch_page(url) -> str async method
   - fetch_pdf(url) -> bytes async method
   - parse_html(html, selector) helper using BeautifulSoup
   - extract_text_from_pdf(pdf_bytes) helper using pdfplumber/pypdf
   - Custom exceptions: FetchError, ParseError, ScraperError
   - Context manager support for cleanup

4. Create src/scrapers/utils.py with:
   - Citation extraction patterns for federal tax:
     - IRC pattern: r'(?:IRC|I\.R\.C\.|26 U\.S\.C\.)\s*§?\s*(\d+[a-zA-Z]?)(?:\(([^)]+)\))*'
     - Regulation pattern: r'(?:Treas\.?\s*Reg\.?|26 C\.F\.R\.)\s*§?\s*(\d+\.\d+-\d+)'
     - Case pattern: r'(\d+)\s+(?:T\.C\.|F\.\d+d?|U\.S\.)\s+(\d+)'
   - normalize_citation(citation) function
   - clean_text(text) - remove extra whitespace, normalize unicode

**Testing**:
- Create tests/test_base_scraper.py
- Mock httpx responses
- Test retry logic with mocked failures
- Test rate limiting
- Test PDF extraction
- Test citation patterns
- Run: `pytest tests/test_base_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add base scraper infrastructure"

**Verification before next prompt**:
- [ ] BaseScraper class with all methods
- [ ] Retry logic works correctly
- [ ] Rate limiting enforced
- [ ] Citation patterns match federal formats
- [ ] All tests pass
```

---

### Prompt 7: IRC Scraper

```
Continuing the Federal Tax RAG build. Previous step created base scraper infrastructure.

**Your Task**: Implement the Internal Revenue Code scraper for Title 26 USC.

**Enter plan mode first** to research the uscode.house.gov structure and design the scraper.

**Requirements**:
1. Create src/scrapers/irc.py with IRCScraper class extending BaseScraper

2. Target source: https://uscode.house.gov/browse/prelim@title26

3. Implement scraping for Title 26 structure:
   - Subtitles A-K (focus on A-F for tax provisions)
   - Chapters within each subtitle
   - Sections within each chapter
   - Subsections and paragraphs within sections

4. Create RawStatute model (in models.py):
   - id: "irc:{section}" (e.g., "irc:61")
   - section_number: str (e.g., "61")
   - title: str (e.g., "Gross income defined")
   - subtitle: str
   - chapter: str
   - subchapter: str | None
   - text: str (full section text)
   - html: str (preserved HTML)
   - subsections: list[dict] (structured subsection data)
   - effective_date: date | None
   - amendment_history: list[str]
   - source_url: str
   - StatuteMetadata with computed fields:
     - full_citation: "IRC § {section}"
     - hierarchy_path: "Internal Revenue Code > Subtitle A > Chapter 1 > § 61"

5. Key methods:
   - scrape_title_structure() -> dict (subtitle/chapter/section hierarchy)
   - scrape_section(section_number) -> RawStatute
   - scrape_all_sections(subtitle_filter: list[str] | None) -> list[RawStatute]
   - Parse effective dates from "[Enacted]" and "[Amended]" annotations
   - Handle cross-references to other IRC sections

6. Create scripts/scrape_irc.py:
   - CLI script with argparse
   - Options: --subtitles (filter), --output-dir, --limit, --resume
   - Save to data/raw/irc/{subtitle}/{section}.json
   - Progress logging

**Testing**:
- Create tests/test_scrapers/test_irc_scraper.py
- Mock HTML responses for section pages
- Test hierarchy parsing
- Test section extraction
- Test metadata computation
- Run: `pytest tests/test_scrapers/test_irc_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add Internal Revenue Code scraper"

**Verification before next prompt**:
- [ ] IRCScraper extracts section structure
- [ ] Subsections are properly parsed
- [ ] Citations are correctly formatted
- [ ] Script saves to correct directory
- [ ] All tests pass
```

---

### Prompt 8: Treasury Regulations Scraper

```
Continuing the Federal Tax RAG build. Previous step created IRC scraper.

**Your Task**: Implement the Treasury Regulations scraper using the eCFR API.

**Enter plan mode first** to research the eCFR API and design the scraper.

**Requirements**:
1. Create src/scrapers/treasury_regs.py with TreasuryRegsScraper class

2. Target source: https://www.ecfr.gov/api/versioner/v1
   - API endpoints:
     - GET /titles - List all titles
     - GET /structure/{date}/title-26 - Structure of Title 26
     - GET /full/{date}/title-26/chapter-I/subchapter-A/part-1 - Full content

3. Create RawRegulation model (in models.py):
   - id: "reg:{section}" (e.g., "reg:1.61-1")
   - section_number: str (e.g., "1.61-1")
   - title: str
   - part: str
   - subpart: str | None
   - text: str
   - xml: str (preserved eCFR XML)
   - authority: list[str] (IRC sections that authorize this reg)
   - source_statutes: list[str] (IRC sections implemented)
   - effective_date: date
   - source_url: str
   - RegulationMetadata with computed fields:
     - full_citation: "Treas. Reg. § {section}"
     - hierarchy_path: "Treasury Regulations > Part 1 > § 1.61-1"
     - implements: list of IRC section references

4. Key methods:
   - get_title_structure() -> dict (part/subpart/section hierarchy)
   - get_section(section_number) -> RawRegulation
   - get_all_sections(part_filter: list[str] | None) -> list[RawRegulation]
   - parse_authority_citations(text) -> list[str]
   - extract_implementing_statutes(text) -> list[str]

5. Focus on key parts:
   - Part 1: Income taxes (§§ 1.1 - 1.1563)
   - Part 20: Estate tax (§§ 20.0 - 20.2056)
   - Part 25: Gift tax (§§ 25.0 - 25.2702)
   - Part 31: Employment taxes (§§ 31.0 - 31.6413)
   - Part 301: Procedure and administration

6. Create scripts/scrape_treasury_regs.py:
   - CLI with --parts filter, --output-dir, --limit
   - Save to data/raw/treasury_regs/{part}/{section}.json
   - Handle API rate limits (eCFR is fairly permissive)

**Testing**:
- Create tests/test_scrapers/test_treasury_regs_scraper.py
- Mock eCFR API responses
- Test structure parsing
- Test authority extraction
- Test XML to text conversion
- Run: `pytest tests/test_scrapers/test_treasury_regs_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add Treasury Regulations scraper (eCFR)"

**Verification before next prompt**:
- [ ] eCFR API integration works
- [ ] Regulations properly linked to IRC sections
- [ ] Authority citations extracted
- [ ] All tests pass
```

---

### Prompt 9: Tax Court Scraper

```
Continuing the Federal Tax RAG build. Previous step created Treasury Regulations scraper.

**Your Task**: Implement the U.S. Tax Court case scraper.

**Enter plan mode first** to research the Tax Court website structure and design the scraper.

**Requirements**:
1. Create src/scrapers/tax_court.py with TaxCourtScraper class

2. Target sources:
   - Primary: https://www.ustaxcourt.gov/UstcInOp/OpinionSearch.aspx
   - Alternative: PDF downloads from search results

3. Create RawCase model (in models.py):
   - id: "taxcourt:{docket}" (e.g., "taxcourt:12345-21")
   - docket_number: str
   - case_name: str (e.g., "Smith v. Commissioner")
   - case_type: str (Regular, Memorandum, Summary, Bench)
   - judge: str
   - date_filed: date
   - text: str (opinion text)
   - pdf_path: str | None
   - citations_in_text: list[str] (IRC, regs cited)
   - holding_summary: str | None
   - source_url: str
   - CaseMetadata with computed fields:
     - full_citation: "{case_name}, {docket_number} (T.C. {year})"
     - hierarchy_path: "U.S. Tax Court > {case_type} > {case_name}"

4. Key methods:
   - search_opinions(start_date, end_date, case_type) -> list[dict]
   - download_opinion(result) -> RawCase
   - extract_text_from_opinion_pdf(pdf_bytes) -> str
   - extract_holding(text) -> str | None
   - extract_irc_citations(text) -> list[str]
   - Pagination handling for search results

5. Special handling:
   - Tax Court PDFs can be scanned images - handle OCR fallback
   - Parse structured sections: Facts, Opinion, Decision
   - Handle multi-judge panels

6. Create scripts/scrape_tax_court.py:
   - CLI with --start-date, --end-date, --case-type, --limit
   - Save to data/raw/tax_court/{year}/{docket}.json
   - Store PDFs in data/raw/tax_court/pdfs/

**Testing**:
- Create tests/test_scrapers/test_tax_court_scraper.py
- Mock search results HTML
- Mock PDF downloads
- Test citation extraction
- Test holding extraction
- Run: `pytest tests/test_scrapers/test_tax_court_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add U.S. Tax Court case scraper"

**Verification before next prompt**:
- [ ] Tax Court search works
- [ ] PDFs downloaded and text extracted
- [ ] IRC citations extracted from opinions
- [ ] All tests pass
```

---

### Prompt 10: IRS Guidance Scraper

```
Continuing the Federal Tax RAG build. Previous step created Tax Court scraper.

**Your Task**: Implement the IRS guidance document scraper for PLRs, Revenue Rulings, and other guidance.

**Enter plan mode first** to research IRS.gov structure and design the scraper.

**Requirements**:
1. Create src/scrapers/irs_guidance.py with IRSGuidanceScraper class

2. Target sources:
   - Internal Revenue Bulletins: https://www.irs.gov/irb
   - PLR/TAM index: https://www.irs.gov/privacy-disclosure/private-letter-rulings-technical-advice-memoranda-and-field-service-advice

3. Create RawGuidance model (in models.py):
   - id: "guidance:{type}:{number}" (e.g., "guidance:plr:202401001")
   - guidance_type: GuidanceType enum (PLR, REV_RULING, REV_PROC, TAM, NOTICE)
   - number: str (e.g., "202401001" or "2024-1")
   - title: str
   - issue_date: date
   - text: str
   - pdf_path: str | None
   - irc_sections: list[str] (IRC sections addressed)
   - regulations_cited: list[str]
   - issue_summary: str | None
   - holding: str | None
   - source_url: str
   - GuidanceMetadata with computed fields:
     - full_citation based on type:
       - PLR: "PLR {number}"
       - Rev. Rul.: "Rev. Rul. {number}"
       - Rev. Proc.: "Rev. Proc. {number}"
     - authority_weight: PLR < TAM < Notice < Rev. Proc. < Rev. Rul.

4. Document types to scrape:
   - Private Letter Rulings (PLRs) - taxpayer-specific guidance
   - Revenue Rulings - official IRS interpretations
   - Revenue Procedures - IRS procedural guidance
   - Technical Advice Memoranda (TAMs)
   - Notices - IRS announcements

5. Key methods:
   - scrape_irb_index(year) -> list of bulletin references
   - scrape_bulletin(bulletin_number) -> list[RawGuidance]
   - scrape_plr_index(year) -> list of PLR references
   - download_plr(plr_number) -> RawGuidance
   - extract_issues(text) -> list[str]
   - extract_holding(text) -> str | None

6. Create scripts/scrape_irs_guidance.py:
   - CLI with --types, --year, --limit
   - Save to data/raw/irs_guidance/{type}/{year}/{number}.json

**Testing**:
- Create tests/test_scrapers/test_irs_guidance_scraper.py
- Mock IRS website responses
- Test different guidance types
- Test citation extraction
- Run: `pytest tests/test_scrapers/test_irs_guidance_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add IRS guidance scraper (PLRs, Rev. Rulings)"

**Verification before next prompt**:
- [ ] IRB parsing works
- [ ] PLR downloads work
- [ ] Guidance types correctly identified
- [ ] All tests pass
```

---

### Prompt 11: Circuit Courts Scraper

```
Continuing the Federal Tax RAG build. Previous step created IRS guidance scraper.

**Your Task**: Implement the federal circuit court tax case scraper using CourtListener API.

**Enter plan mode first** to research CourtListener API and design the scraper.

**Requirements**:
1. Create src/scrapers/circuit_courts.py with CircuitCourtScraper class

2. Target source: CourtListener API
   - Base: https://www.courtlistener.com/api/rest/v3/
   - Endpoints:
     - GET /search/?q={query}&type=o (opinion search)
     - GET /opinions/{id}/ (opinion detail)
     - GET /clusters/{id}/ (case cluster)

3. Create RawCircuitCase model (in models.py):
   - id: "circuit:{court}:{cluster_id}"
   - cluster_id: int
   - case_name: str
   - court: str (e.g., "cafc", "ca2", "scotus")
   - docket_number: str
   - date_filed: date
   - judges: list[str]
   - text: str (opinion text)
   - citations: list[str] (reporter citations)
   - irc_sections: list[str]
   - cited_cases: list[int] (cluster IDs of cited cases)
   - source_url: str
   - CircuitCaseMetadata with computed fields:
     - full_citation: "{case_name}, {citation} ({court} {year})"
     - hierarchy_path: "Federal Courts > {court_name} > {case_name}"

4. Search strategy:
   - Search queries: "internal revenue code", "IRC section", "26 USC", "tax court"
   - Filter by courts: scotus, cafc, ca1-ca11, cadc, uscfc
   - Filter by date range

5. Key methods:
   - search_tax_cases(query, courts, start_date, end_date) -> list[dict]
   - get_opinion(opinion_id) -> dict
   - get_cluster(cluster_id) -> RawCircuitCase
   - extract_irc_citations(text) -> list[str]
   - get_citing_opinions(cluster_id) -> list[int]
   - Handle API pagination (next/previous URLs)
   - Respect rate limits (CourtListener: 5000/day free tier)

6. Create scripts/scrape_circuit_courts.py:
   - CLI with --courts, --query, --start-date, --end-date, --limit
   - Save to data/raw/circuit_courts/{court}/{cluster_id}.json

**Testing**:
- Create tests/test_scrapers/test_circuit_courts_scraper.py
- Mock CourtListener API responses
- Test pagination handling
- Test citation extraction
- Run: `pytest tests/test_scrapers/test_circuit_courts_scraper.py -v`

**Commit**: Create commit with message: "feat(scrapers): add federal circuit courts scraper (CourtListener)"

**Verification before next prompt**:
- [ ] CourtListener API integration works
- [ ] Tax cases filtered correctly
- [ ] Citations extracted from opinions
- [ ] All tests pass
```

---

### Prompt 12: Data Models for Ingestion

```
Continuing the Federal Tax RAG build. Previous step completed all scrapers.

**Your Task**: Create the unified data models for the ingestion pipeline.

**Enter plan mode first** to design the data model architecture.

**Requirements**:
1. Create src/ingestion/__init__.py exporting models

2. Create src/ingestion/models.py with:

   DocumentType enum:
   - STATUTE, REGULATION, CASE, GUIDANCE

   ChunkLevel enum:
   - PARENT, CHILD

   LegalDocument (unified document model):
   - id: str (e.g., "irc:61", "reg:1.61-1", "taxcourt:12345-21")
   - doc_type: DocumentType
   - title: str
   - full_citation: str
   - text: str
   - effective_date: date | None
   - source_url: str
   - hierarchy_path: str
   - parent_id: str | None
   - children_ids: list[str]
   - cites_statutes: list[str]
   - cites_regulations: list[str]
   - cites_cases: list[str]
   - cites_guidance: list[str]
   - scraped_at: datetime
   - metadata: dict (type-specific extra fields)

   LegalChunk:
   - id: str (e.g., "chunk:irc:61:0", "chunk:irc:61:1")
   - doc_id: str
   - level: ChunkLevel
   - ancestry: str (context path)
   - subsection_path: str (e.g., "(a)(1)(A)")
   - text: str
   - text_with_ancestry: str (for embedding)
   - parent_chunk_id: str | None
   - child_chunk_ids: list[str]
   - citation: str
   - effective_date: date | None
   - doc_type: DocumentType
   - token_count: int

   Corpus:
   - documents: list[LegalDocument]
   - metadata: CorpusMetadata
     - created_at: datetime
     - version: str
     - counts_by_type: dict[str, int]

   Citation:
   - source_id: str
   - target_id: str
   - citation_text: str
   - relation_type: RelationType (CITES, IMPLEMENTS, INTERPRETS, AMENDS, SUPERSEDES)
   - context: str (surrounding text)

3. All models should be Pydantic BaseModel with:
   - Proper validation
   - JSON serialization support
   - model_config with from_attributes=True

**Testing**:
- Create tests/test_ingestion_models.py
- Test model creation and validation
- Test JSON serialization/deserialization
- Test edge cases (None values, empty lists)
- Run: `pytest tests/test_ingestion_models.py -v`

**Commit**: Create commit with message: "feat(ingestion): add unified data models"

**Verification before next prompt**:
- [ ] All models created with proper types
- [ ] Validation works correctly
- [ ] JSON serialization works
- [ ] All tests pass
```

---

### Prompt 13: Corpus Consolidation

```
Continuing the Federal Tax RAG build. Previous step created data models.

**Your Task**: Implement the corpus consolidation pipeline to transform raw scraped data into unified LegalDocument format.

**Enter plan mode first** to design the consolidation pipeline.

**Requirements**:
1. Create src/ingestion/consolidate.py with:

   DocumentConsolidator class:
   - __init__(raw_data_dir: Path, output_dir: Path)

   Methods for each document type:
   - consolidate_statutes(raw_dir: Path) -> list[LegalDocument]
     - Transform RawStatute -> LegalDocument
     - Extract cross-references to other IRC sections
     - Build hierarchy (subtitle > chapter > section)

   - consolidate_regulations(raw_dir: Path) -> list[LegalDocument]
     - Transform RawRegulation -> LegalDocument
     - Link to implementing IRC sections
     - Preserve authority citations

   - consolidate_cases(raw_dir: Path) -> list[LegalDocument]
     - Consolidate both Tax Court and Circuit Court cases
     - Extract cited authorities
     - Normalize case citations

   - consolidate_guidance(raw_dir: Path) -> list[LegalDocument]
     - Transform RawGuidance -> LegalDocument
     - Categorize by guidance type
     - Extract addressed IRC sections

   Main method:
   - consolidate_all() -> Corpus
     - Process all document types
     - Compute statistics
     - Save to data/processed/corpus.json

2. Handle cross-reference resolution:
   - Normalize all citations to standard format
   - Resolve document references where possible
   - Mark unresolved references for later graph resolution

3. Create scripts/consolidate_corpus.py:
   - CLI with --raw-dir, --output-dir, --types filter
   - Progress logging with document counts
   - Save corpus.json with metadata

**Testing**:
- Create tests/test_consolidate.py
- Test each document type consolidation
- Test cross-reference extraction
- Test corpus statistics
- Use fixture data from conftest.py
- Run: `pytest tests/test_consolidate.py -v`

**Commit**: Create commit with message: "feat(ingestion): add corpus consolidation pipeline"

**Verification before next prompt**:
- [ ] All document types consolidated
- [ ] Cross-references extracted
- [ ] corpus.json generated with metadata
- [ ] All tests pass
```

---

### Prompt 14: Hierarchical Chunking

```
Continuing the Federal Tax RAG build. Previous step created corpus consolidation.

**Your Task**: Implement hierarchical chunking strategy for legal documents.

**Enter plan mode first** to design the chunking strategy for each document type.

**Requirements**:
1. Create src/ingestion/chunking.py with:

   LegalChunker class:
   - __init__(max_chunk_tokens: int = 512, overlap_tokens: int = 50)
   - Use tiktoken for token counting (cl100k_base)

   Document-type specific chunking:

   chunk_statute(doc: LegalDocument) -> list[LegalChunk]:
   - Parent chunk: Full section text
   - Child chunks: Top-level subsections (a), (b), etc.
   - Preserve subsection hierarchy in ancestry
   - Keep nested subsections with parent

   chunk_regulation(doc: LegalDocument) -> list[LegalChunk]:
   - Parent chunk: Full section text
   - Child chunks: Paragraphs (a), (b), etc.
   - Include "Authority:" and "Source:" in parent

   chunk_case(doc: LegalDocument) -> list[LegalChunk]:
   - Parent chunk: Full opinion (if < 2000 tokens)
   - If larger, split into semantic sections:
     - Facts
     - Issues
     - Analysis/Discussion
     - Holding/Conclusion
   - Paragraph-based fallback with ~500 token target

   chunk_guidance(doc: LegalDocument) -> list[LegalChunk]:
   - Parent chunk: Full document (if < 2000 tokens)
   - Child chunks for longer documents:
     - Issue/Facts
     - Law and Analysis
     - Conclusion/Holding

   Main method:
   - chunk_corpus(corpus: Corpus) -> list[LegalChunk]

2. Chunk ID format: "chunk:{doc_type}:{doc_id}:{sequence}"

3. Ancestry format for embeddings:
   - Statute: "Internal Revenue Code > Subtitle A > Chapter 1 > § 61 > (a)(1)"
   - Regulation: "Treasury Regulations > Part 1 > § 1.61-1 > (a)"
   - Case: "U.S. Tax Court > Smith v. Commissioner > Holdings"

4. Create scripts/chunk_corpus.py:
   - CLI with --input (corpus.json), --output (chunks.json)
   - Compute and log statistics:
     - Chunks by level (parent/child)
     - Token distribution
     - Chunks per document type

**Testing**:
- Create tests/test_chunking.py
- Test each document type chunking
- Test token counting
- Test ancestry generation
- Test parent-child relationships
- Run: `pytest tests/test_chunking.py -v`

**Commit**: Create commit with message: "feat(ingestion): add hierarchical chunking for legal documents"

**Verification before next prompt**:
- [ ] All document types chunked correctly
- [ ] Parent-child relationships maintained
- [ ] Ancestry paths generated
- [ ] Token counts accurate
- [ ] All tests pass
```

---

### Prompt 15: Citation Extraction

```
Continuing the Federal Tax RAG build. Previous step created hierarchical chunking.

**Your Task**: Implement comprehensive citation extraction and relationship building.

**Enter plan mode first** to design the citation extraction patterns.

**Requirements**:
1. Create src/ingestion/citation_extractor.py with:

   CitationExtractor class:

   IRC citation patterns:
   - "IRC § 61(a)(1)"
   - "Section 61"
   - "26 U.S.C. § 61"
   - "Internal Revenue Code section 61"
   - Subsection references: "(a)", "(a)(1)", "(a)(1)(A)"

   Regulation citation patterns:
   - "Treas. Reg. § 1.61-1(a)"
   - "26 CFR 1.61-1"
   - "Treasury Regulation section 1.61-1"
   - "Reg. § 1.61-1"

   Case citation patterns:
   - Tax Court: "123 T.C. 456", "T.C. Memo 2024-1"
   - Federal Reporter: "123 F.3d 456 (9th Cir. 2024)"
   - Supreme Court: "123 U.S. 456"
   - Case names: "Smith v. Commissioner"

   Guidance citation patterns:
   - PLR: "PLR 202401001", "Private Letter Ruling 202401001"
   - Rev. Rul.: "Rev. Rul. 2024-1", "Revenue Ruling 2024-1"
   - Rev. Proc.: "Rev. Proc. 2024-1"
   - Notice: "Notice 2024-1"

   Methods:
   - extract_all_citations(text: str) -> list[Citation]
   - extract_irc_citations(text: str) -> list[str]
   - extract_regulation_citations(text: str) -> list[str]
   - extract_case_citations(text: str) -> list[str]
   - extract_guidance_citations(text: str) -> list[str]
   - normalize_citation(citation: str) -> str
   - determine_relation_type(source_type, target_type, context) -> RelationType
   - get_citation_context(text, citation, window=100) -> str

2. RelationType determination:
   - Regulation → IRC: IMPLEMENTS
   - Case → IRC/Regulation: INTERPRETS
   - Guidance → IRC/Regulation: INTERPRETS
   - Any → Any: CITES (default)
   - Amendment references: AMENDS
   - "superseded by": SUPERSEDES

3. Create src/ingestion/build_citation_graph.py:
   - Build list of Citation objects from corpus
   - Resolve citations to document IDs where possible
   - Output: data/processed/citations.json

4. Create scripts/extract_citations.py:
   - CLI with --input (corpus.json), --output (citations.json)
   - Log statistics: citations by type, resolved vs unresolved

**Testing**:
- Create tests/test_citation_extractor.py
- Test each citation pattern with various formats
- Test normalization
- Test relation type detection
- Test context extraction
- Run: `pytest tests/test_citation_extractor.py -v`

**Commit**: Create commit with message: "feat(ingestion): add citation extraction and relationship building"

**Verification before next prompt**:
- [ ] All citation patterns work
- [ ] Normalization consistent
- [ ] Relation types correctly determined
- [ ] citations.json generated
- [ ] All tests pass
```

---

### Prompt 16: Embedding Generation

```
Continuing the Federal Tax RAG build. Previous step created citation extraction.

**Your Task**: Implement the embedding generation pipeline using Voyage AI.

**Enter plan mode first** to design the embedding pipeline.

**Requirements**:
1. Create src/vector/__init__.py exporting components

2. Create src/vector/embeddings.py with:

   VoyageEmbedder class:
   - __init__(api_key: str, model: str = "voyage-law-2")
   - Model produces 1024-dimensional vectors
   - Optimized for legal documents

   Redis caching:
   - Cache key: SHA256 hash of text (first 16 chars)
   - Cache prefix: "emb:"
   - TTL: 30 days (configurable)
   - Check cache before API call
   - Store result after API call

   Methods:
   - embed_single(text: str) -> list[float]
   - embed_batch(texts: list[str], batch_size: int = 128) -> list[list[float]]
   - Uses "text_with_ancestry" field for embedding
   - Retry logic with exponential backoff
   - Rate limiting (300 requests/minute)

   Batch processing:
   - Process chunks in batches of 128
   - Checkpoint progress for resumability
   - Handle partial failures gracefully

3. Create scripts/generate_embeddings.py:
   - CLI with --input (chunks.json), --output (embeddings.npz)
   - Options: --batch-size, --resume, --sample (for testing)
   - Save as NumPy .npz file:
     - chunk_ids: array of chunk ID strings
     - embeddings: 2D array (num_chunks × 1024)
   - Progress bar with estimated time
   - Log cache hit rate

4. Verification step:
   - Verify all chunks have embeddings
   - Check embedding dimensions
   - Sample cosine similarity sanity check

**Testing**:
- Create tests/test_embeddings.py
- Mock Voyage AI API responses
- Test caching behavior
- Test batch processing
- Test checkpoint/resume
- Run: `pytest tests/test_embeddings.py -v`

**Commit**: Create commit with message: "feat(vector): add Voyage AI embedding generation with caching"

**Verification before next prompt**:
- [ ] Embeddings generated for all chunks
- [ ] Caching reduces API calls
- [ ] embeddings.npz file created
- [ ] Dimensions are 1024
- [ ] All tests pass
```

---

### Prompt 17: Weaviate Client and Schema

```
Continuing the Federal Tax RAG build. Previous step created embedding generation.

**Your Task**: Implement the Weaviate client with schema definition and hybrid search.

**Enter plan mode first** to design the Weaviate schema.

**Requirements**:
1. Create src/vector/client.py with:

   WeaviateClient class:
   - __init__(url: str, api_key: str | None, grpc_port: int)
   - Connection management with context manager
   - Health check method

   Collection: "LegalChunk"

   Schema properties:
   - chunk_id (TEXT, FIELD tokenization) - exact match
   - doc_id (TEXT, FIELD) - exact match
   - doc_type (TEXT, WORD) - filterable
   - level (TEXT, FIELD) - parent/child
   - ancestry (TEXT, WORD) - searchable
   - citation (TEXT, WORD) - searchable
   - text (TEXT, WORD) - BM25 keyword search
   - text_with_ancestry (TEXT, WORD) - used for embeddings
   - effective_date (DATE) - filterable
   - token_count (INT)

   Vector configuration:
   - vectorizer: none (external vectors from Voyage)
   - vector_index_type: hnsw
   - distance_metric: cosine

   Methods:
   - create_collection() - idempotent schema creation
   - delete_collection() - for reset
   - upsert_chunk(chunk: LegalChunk, vector: list[float])
   - upsert_batch(chunks: list[LegalChunk], vectors: list[list[float]])
   - hybrid_search(query_vector, query_text, alpha, top_k, filters) -> list[dict]
   - get_chunk(chunk_id) -> dict | None
   - get_chunks_by_doc(doc_id) -> list[dict]
   - count() -> int

   Hybrid search:
   - alpha parameter: 0.25 (keyword-heavy for legal)
   - Combine vector similarity and BM25
   - Support filters: doc_type, level, effective_date

2. Create scripts/load_weaviate.py:
   - CLI with --chunks (chunks.json), --embeddings (embeddings.npz)
   - Options: --batch-size, --reset, --resume
   - Load chunks with their vectors
   - Verify count matches
   - Log progress

3. Create scripts/init_weaviate.py:
   - Create collection schema
   - Verify configuration

**Testing**:
- Create tests/test_vector_client.py
- Test with running Weaviate (mark as integration)
- Test schema creation
- Test upsert and retrieval
- Test hybrid search
- Run: `pytest tests/test_vector_client.py -v -m "not integration"`
- Integration: `pytest tests/test_vector_client.py -v -m integration`

**Commit**: Create commit with message: "feat(vector): add Weaviate client with hybrid search"

**Verification before next prompt**:
- [ ] Schema created correctly
- [ ] Chunks loaded with vectors
- [ ] Hybrid search returns results
- [ ] Filters work correctly
- [ ] All tests pass
```

---

### Prompt 18: Neo4j Client and Schema

```
Continuing the Federal Tax RAG build. Previous step created Weaviate client.

**Your Task**: Implement the Neo4j client with knowledge graph schema.

**Enter plan mode first** to design the graph schema.

**Requirements**:
1. Create src/graph/__init__.py exporting components

2. Create src/graph/client.py with:

   Neo4jClient class:
   - __init__(uri: str, user: str, password: str)
   - Connection pooling (max 50 connections)
   - Context manager for session management
   - health_check() method

   Methods:
   - run_query(cypher: str, params: dict) -> list[dict]
   - run_write(cypher: str, params: dict) -> dict (with counters)
   - session() context manager
   - close() cleanup

3. Create src/graph/schema.py with:

   Node labels:
   - :Document (base for all)
   - :Statute (IRC sections)
   - :Regulation (Treasury Regs)
   - :Case (Tax Court, Circuit Courts)
   - :Guidance (PLRs, Rev. Rulings, etc.)
   - :Chunk (document chunks)

   Relationships:
   - (Regulation)-[:IMPLEMENTS]->(Statute)
   - (Case)-[:INTERPRETS]->(Statute)
   - (Case)-[:INTERPRETS]->(Regulation)
   - (Guidance)-[:INTERPRETS]->(Statute)
   - (Document)-[:CITES]->(Document)
   - (Document)-[:AMENDS]->(Document)
   - (Document)-[:SUPERSEDES]->(Document)
   - (Document)-[:HAS_CHUNK]->(Chunk)
   - (Chunk)-[:CHILD_OF]->(Chunk)

   Constraints (create_constraints() function):
   - Document.id unique
   - Chunk.id unique

   Indexes (create_indexes() function):
   - Document: doc_type, section, citation
   - Chunk: doc_id, level

4. Create src/graph/loader.py with:

   GraphLoader class:
   - load_documents(corpus: Corpus)
   - load_chunks(chunks: list[LegalChunk])
   - load_citations(citations: list[Citation])
   - Use UNWIND for batch operations (500 per batch)
   - Return statistics

5. Create scripts/init_neo4j.py:
   - Create constraints and indexes
   - Verify schema

6. Create scripts/load_neo4j.py:
   - CLI with --corpus, --chunks, --citations
   - Load all data
   - Log statistics

**Testing**:
- Create tests/test_graph_client.py
- Test connection and health check
- Test query execution
- Create tests/test_graph_loader.py (integration)
- Run: `pytest tests/test_graph_client.py -v`

**Commit**: Create commit with message: "feat(graph): add Neo4j client with knowledge graph schema"

**Verification before next prompt**:
- [ ] Schema created with constraints/indexes
- [ ] Documents loaded as nodes
- [ ] Citations loaded as relationships
- [ ] Queries execute correctly
- [ ] All tests pass
```

---

### Prompt 19: Graph Queries

```
Continuing the Federal Tax RAG build. Previous step created Neo4j client.

**Your Task**: Implement graph query functions for citation traversal and interpretation chains.

**Enter plan mode first** to design the query patterns.

**Requirements**:
1. Create src/graph/queries.py with:

   Query functions:

   get_interpretation_chain(statute_id: str) -> dict:
   - For an IRC section, find:
     - Implementing regulations (IMPLEMENTS relationship)
     - Interpreting cases (INTERPRETS relationship)
     - IRS guidance (INTERPRETS relationship)
   - Return structured chain with authority hierarchy
   - Cypher:
     ```cypher
     MATCH (s:Statute {id: $statute_id})
     OPTIONAL MATCH (r:Regulation)-[:IMPLEMENTS]->(s)
     OPTIONAL MATCH (c:Case)-[:INTERPRETS]->(s)
     OPTIONAL MATCH (g:Guidance)-[:INTERPRETS]->(s)
     RETURN s, collect(DISTINCT r) as regulations,
            collect(DISTINCT c) as cases,
            collect(DISTINCT g) as guidance
     ```

   get_citing_documents(doc_id: str, depth: int = 1) -> list[dict]:
   - Find documents that cite the given document
   - Support multi-hop traversal
   - Return with relationship type

   get_cited_documents(doc_id: str, depth: int = 1) -> list[dict]:
   - Find documents cited by the given document
   - Support multi-hop traversal

   get_related_by_statute(statute_id: str) -> list[dict]:
   - Find all documents related to a statute
   - Group by document type
   - Order by authority weight

   get_document_neighborhood(doc_id: str, hops: int = 2) -> dict:
   - Get surrounding citation network
   - Return nodes and edges for visualization

   find_path_between(doc_id_1: str, doc_id_2: str) -> list[dict] | None:
   - Find citation path between two documents
   - Return path if exists

2. Query result models:
   - InterpretationChain
   - CitationResult
   - DocumentNeighborhood

3. Query optimization:
   - Use indexes appropriately
   - Limit result sets
   - Profile queries for performance

**Testing**:
- Create tests/test_graph_queries.py
- Test with mock Neo4j responses
- Test query result parsing
- Test edge cases (no results, circular refs)
- Run: `pytest tests/test_graph_queries.py -v`

**Commit**: Create commit with message: "feat(graph): add citation traversal and interpretation chain queries"

**Verification before next prompt**:
- [ ] Interpretation chain query works
- [ ] Citation traversal works
- [ ] Results properly structured
- [ ] All tests pass
```

---

### Prompt 20: Data Loading Verification

```
Continuing the Federal Tax RAG build. Previous step created graph queries.

**Your Task**: Create verification scripts and run the full data loading pipeline.

**Enter plan mode first** to design the verification approach.

**Requirements**:
1. Create scripts/verify_data_load.py:
   - Check Weaviate:
     - Collection exists
     - Chunk count matches expected
     - Sample hybrid search returns results
     - All doc_types present
   - Check Neo4j:
     - Constraints exist
     - Document count by type
     - Relationship count by type
     - Sample interpretation chain query
   - Output: verification_report.json

2. Create scripts/run_full_pipeline.py:
   - Orchestrate the complete data pipeline:
     1. Wait for services (Docker)
     2. Run all scrapers (or use sample data)
     3. Consolidate corpus
     4. Chunk documents
     5. Extract citations
     6. Generate embeddings
     7. Initialize Weaviate schema
     8. Load Weaviate
     9. Initialize Neo4j schema
     10. Load Neo4j
     11. Run verification
   - Support --sample mode for testing
   - Support --skip-scraping to use existing raw data
   - Comprehensive logging

3. Create sample test data:
   - data/sample/raw/ with small set of each document type
   - Used for CI testing and development

4. Update tests/integration/test_full_pipeline.py:
   - Test complete pipeline with sample data
   - Verify data integrity
   - Mark as integration test

**Testing**:
- Run: `python scripts/run_full_pipeline.py --sample`
- Run: `python scripts/verify_data_load.py`
- Verify all checks pass

**Commit**: Create commit with message: "feat(scripts): add data pipeline orchestration and verification"

**Verification before next prompt**:
- [ ] Full pipeline runs successfully
- [ ] Verification script passes
- [ ] Sample data created
- [ ] Integration test passes
- [ ] All data accessible
```

---

### Prompt 21: Query Decomposition

```
Continuing the Federal Tax RAG build. Previous step completed data loading.

**Your Task**: Implement the query decomposition system for complex tax queries.

**Enter plan mode first** to design the decomposition logic.

**Requirements**:
1. Create src/retrieval/__init__.py exporting components

2. Create src/retrieval/models.py with:
   - QueryType enum: DEFINITION, COMPUTATION, PROCEDURE, EXEMPTION, PENALTY, TEMPORAL, STATUTE_LOOKUP, REGULATION_LOOKUP, GENERAL
   - SubQuery model: text, type, priority (1-5), reasoning
   - DecompositionResult: sub_queries, is_simple, reasoning

3. Create src/retrieval/query_decomposer.py with:

   QueryDecomposer class:
   - __init__(llm_client)

   Methods:
   - decompose(query: str) -> DecompositionResult

   Complexity heuristics:
   - Word count >= 8
   - Contains complexity keywords: "and", "or", "both", "also", "as well as"
   - Contains multiple question words
   - References multiple tax concepts

   LLM decomposition (for complex queries):
   - Use QUERY_DECOMPOSITION_PROMPT
   - Classify query type
   - Break into focused sub-queries
   - Assign priority (1 = most important)
   - Return reasoning

   Query type classification:
   - DEFINITION: "What is...", "Define..."
   - COMPUTATION: "How to calculate...", "What is the amount..."
   - PROCEDURE: "How do I...", "What are the steps..."
   - EXEMPTION: "Is X exempt...", "Does X qualify for exemption..."
   - PENALTY: "What is the penalty...", "What happens if..."
   - TEMPORAL: Year-specific, "In 2024...", effective dates
   - STATUTE_LOOKUP: Direct IRC reference
   - REGULATION_LOOKUP: Direct Reg reference

   Examples:
   - "What is the standard deduction for 2024 and how does it compare to itemized deductions?"
     → SubQuery 1: "What is the standard deduction for 2024?" (COMPUTATION, priority 1)
     → SubQuery 2: "What are itemized deductions?" (DEFINITION, priority 2)
     → SubQuery 3: "When should a taxpayer itemize vs take standard deduction?" (PROCEDURE, priority 2)

**Testing**:
- Create tests/test_query_decomposition.py
- Test complexity heuristics
- Test query type classification
- Test decomposition with mock LLM
- Run: `pytest tests/test_query_decomposition.py -v`

**Commit**: Create commit with message: "feat(retrieval): add query decomposition for complex tax queries"

**Verification before next prompt**:
- [ ] Complexity detection works
- [ ] Query types classified correctly
- [ ] Complex queries decomposed
- [ ] Sub-queries have proper priority
- [ ] All tests pass
```

---

### Prompt 22: Hybrid Retriever

```
Continuing the Federal Tax RAG build. Previous step created query decomposition.

**Your Task**: Implement the hybrid retrieval system combining vector and keyword search.

**Enter plan mode first** to design the retrieval pipeline.

**Requirements**:
1. Create src/retrieval/hybrid.py with:

   HybridRetriever class:
   - __init__(weaviate_client, embedder, settings)

   Methods:
   - retrieve(query: str, top_k: int, alpha: float, filters: dict) -> list[RetrievalResult]

   Retrieval process:
   1. Generate query embedding via Voyage AI
   2. Execute Weaviate hybrid search:
      - Vector similarity (cosine)
      - BM25 keyword search
      - Fusion with alpha=0.25 (keyword-heavy)
   3. Apply filters (doc_type, level, effective_date)
   4. Return ranked results

   RetrievalResult model:
   - chunk_id, doc_id, doc_type
   - text, citation, ancestry
   - score (hybrid), vector_score, bm25_score
   - effective_date

   retrieve_for_subqueries(sub_queries: list[SubQuery], top_k_per: int) -> list[RetrievalResult]:
   - Run retrieval for each sub-query in parallel
   - Deduplicate by chunk_id (keep highest score)
   - Combine results respecting priority

   Filter support:
   - doc_type: ["STATUTE", "REGULATION", ...]
   - level: "PARENT" or "CHILD"
   - effective_date: before/after date

2. Create src/retrieval/multi_retriever.py with:

   MultiRetriever class:
   - Combines multiple retrieval strategies
   - Primary: HybridRetriever
   - Fallback: Pure keyword if vector fails
   - Support for citation-based retrieval

**Testing**:
- Create tests/test_retrieval.py
- Mock Weaviate responses
- Test hybrid scoring
- Test parallel sub-query retrieval
- Test deduplication
- Run: `pytest tests/test_retrieval.py -v`

**Commit**: Create commit with message: "feat(retrieval): add hybrid retrieval with parallel sub-query support"

**Verification before next prompt**:
- [ ] Hybrid search works
- [ ] Alpha parameter affects ranking
- [ ] Filters applied correctly
- [ ] Parallel retrieval efficient
- [ ] All tests pass
```

---

### Prompt 23: Graph Expander

```
Continuing the Federal Tax RAG build. Previous step created hybrid retriever.

**Your Task**: Implement the graph expansion system to enrich retrieval with citation context.

**Enter plan mode first** to design the expansion logic.

**Requirements**:
1. Create src/retrieval/graph_expander.py with:

   GraphExpander class:
   - __init__(neo4j_client, max_hops: int = 2)

   Methods:
   - expand(chunks: list[RetrievalResult]) -> ExpandedContext

   Expansion strategy by document type:

   For Statutes (IRC sections):
   - Find implementing regulations (IMPLEMENTS)
   - Find interpreting cases (INTERPRETS)
   - Find IRS guidance (INTERPRETS)
   - Build interpretation chain

   For Regulations:
   - Find parent statute (IMPLEMENTS inverse)
   - Find interpreting cases
   - Find amending regulations (AMENDS)

   For Cases:
   - Find cited statutes and regulations
   - Find citing cases (precedent chain)
   - Find related IRS guidance

   For Guidance:
   - Find addressed statutes
   - Find related regulations
   - Find superseding guidance (SUPERSEDES)

   ExpandedContext model:
   - original_chunks: list[RetrievalResult]
   - expanded_docs: list[RelatedDocument]
   - interpretation_chains: dict[str, InterpretationChain]
   - citation_graph: dict (nodes, edges for context)

   RelatedDocument model:
   - doc_id, doc_type, title, citation
   - relationship_type (IMPLEMENTS, INTERPRETS, etc.)
   - relationship_direction (TO or FROM)
   - relevance_score (decay by hops)

   Score decay:
   - Hop 1: 0.8 × original score
   - Hop 2: 0.6 × original score

2. Combine with retrieval results:
   - Add expanded documents to context
   - Maintain authority hierarchy
   - Avoid duplicates

**Testing**:
- Create tests/test_graph_expander.py
- Mock Neo4j responses
- Test expansion for each doc type
- Test score decay
- Test duplicate handling
- Run: `pytest tests/test_graph_expander.py -v`

**Commit**: Create commit with message: "feat(retrieval): add graph expansion for citation context"

**Verification before next prompt**:
- [ ] Expansion works for all doc types
- [ ] Interpretation chains built
- [ ] Score decay applied
- [ ] Duplicates handled
- [ ] All tests pass
```

---

### Prompt 24: Legal Reranker

```
Continuing the Federal Tax RAG build. Previous step created graph expander.

**Your Task**: Implement the legal-specific reranking system with authority hierarchy.

**Enter plan mode first** to design the reranking logic.

**Requirements**:
1. Create src/retrieval/reranker.py with:

   LegalReranker class:
   - __init__(settings)

   Authority weights (federal tax hierarchy):
   - STATUTE (IRC): 3.0 (primary binding authority)
   - REGULATION (Treasury): 2.5 (binding interpretive rules)
   - CASE (Tax Court/Circuit): 2.0 (judicial precedent)
   - GUIDANCE (Rev. Rul.): 1.5 (official IRS guidance)
   - GUIDANCE (PLR/TAM): 1.0 (advisory only)

   Methods:
   - rerank(results: list[RetrievalResult], query: str) -> list[RetrievalResult]

   Reranking formula:
   ```
   final_score = base_score × type_weight × recency_boost × authority_boost

   type_weight = AUTHORITY_WEIGHTS[doc_type]

   recency_boost = 1.0 + (0.1 if age <= 2 years else 0.0)

   authority_boost = 1.0 + (0.1 if is_primary_authority else 0.0)
   ```

   Primary authority determination:
   - IRC sections are always primary
   - Final regulations are primary
   - Proposed regulations are secondary
   - Tax Court regular opinions are primary
   - Memorandum opinions are secondary
   - Revenue Rulings are primary guidance
   - PLRs are secondary guidance

   LLM-based relevance scoring (optional):
   - score_relevance(chunks: list, query: str) -> dict[str, float]
   - Use RELEVANCE_SCORING_PROMPT
   - Run in parallel with semaphore (20 concurrent)
   - Use Claude Haiku for speed

2. Combine scoring methods:
   - Hybrid search score (base)
   - Authority weight
   - Recency boost
   - LLM relevance score (if enabled)

**Testing**:
- Create tests/test_reranker.py
- Test authority weights
- Test recency boost
- Test LLM scoring (mocked)
- Test final ranking
- Run: `pytest tests/test_reranker.py -v`

**Commit**: Create commit with message: "feat(retrieval): add legal reranker with authority hierarchy"

**Verification before next prompt**:
- [ ] Authority weights applied
- [ ] Statutes ranked highest
- [ ] Recency affects ranking
- [ ] LLM scoring works
- [ ] All tests pass
```

---

### Prompt 25: Answer Generator

```
Continuing the Federal Tax RAG build. Previous step created legal reranker.

**Your Task**: Implement the answer generation system with citation extraction.

**Enter plan mode first** to design the generation pipeline.

**Requirements**:
1. Create src/generation/__init__.py exporting components

2. Create src/generation/models.py with:
   - ValidatedCitation: citation, source_chunk_id, verified (bool), citation_type
   - GeneratedResponse: answer, citations, chunks_used, confidence, warnings, metadata
   - GenerationConfig: model, temperature, max_tokens, include_reasoning

3. Create src/generation/formatter.py with:
   - format_chunks_for_context(chunks: list, max_chunks: int = 10) -> str
   - Format:
     ```
     [1] IRC § 61 (Statute)
     Gross income defined.—Except as otherwise provided...

     [2] Treas. Reg. § 1.61-1 (Regulation)
     Gross income.—(a) General definition...
     ```
   - Include doc_type indicator
   - Truncate long chunks
   - Number for citation reference

4. Create src/generation/generator.py with:

   TaxLawGenerator class:
   - __init__(anthropic_client, settings)

   Methods:
   - generate(query: str, context_chunks: list, config: GenerationConfig) -> GeneratedResponse

   Generation process:
   1. Format context with format_chunks_for_context()
   2. Build messages with TAX_ATTORNEY_SYSTEM_PROMPT
   3. Call Claude Sonnet
   4. Extract citations from response
   5. Validate citations against provided chunks
   6. Calculate confidence based on citation verification

   Citation extraction patterns:
   - [Source: IRC § 61]
   - [Source: Treas. Reg. § 1.61-1]
   - [Source: Smith v. Commissioner]
   - Inline: "Under IRC § 61..."

   extract_citations(answer: str) -> list[str]
   validate_citations(citations: list, chunks: list) -> list[ValidatedCitation]
   calculate_confidence(validated_citations: list) -> float

   Confidence calculation:
   - Base: 0.7
   - +0.1 if all citations verified
   - +0.1 if >= 3 primary authority citations
   - +0.1 if includes relevant regulation
   - -0.2 if any unverified citations

**Testing**:
- Create tests/test_generation.py
- Mock Claude API responses
- Test context formatting
- Test citation extraction
- Test confidence calculation
- Run: `pytest tests/test_generation.py -v`

**Commit**: Create commit with message: "feat(generation): add answer generator with citation extraction"

**Verification before next prompt**:
- [ ] Context formatted correctly
- [ ] Citations extracted from response
- [ ] Citations validated against chunks
- [ ] Confidence calculated
- [ ] All tests pass
```

---

### Prompt 26: Hallucination Validator

```
Continuing the Federal Tax RAG build. Previous step created answer generator.

**Your Task**: Implement the hallucination detection and response validation system.

**Enter plan mode first** to design the validation logic.

**Requirements**:
1. Create src/generation/validator.py with:

   HallucinationType enum:
   - UNSUPPORTED_CLAIM: Claim not in sources
   - MISQUOTED_TEXT: Incorrect quotation
   - FABRICATED_CITATION: Citation doesn't exist
   - WRONG_ATTRIBUTION: Credited to wrong source
   - OVERGENERALIZATION: Statement too broad
   - OUTDATED_INFO: Information superseded
   - INCORRECT_NUMBER: Wrong numerical value

   HallucinationIssue model:
   - type: HallucinationType
   - severity: float (0-1)
   - claim: str (the problematic claim)
   - expected: str | None (what it should be)
   - source_chunk_id: str | None
   - fix_suggestion: str

   ValidationResult model:
   - passed: bool
   - accuracy_score: float (0-1)
   - issues: list[HallucinationIssue]
   - action: "accept" | "correct" | "regenerate"

   ResponseValidator class:
   - __init__(anthropic_client)

   Methods:
   - validate(answer: str, query: str, chunks: list) -> ValidationResult

   Validation process:
   1. Extract claims from answer
   2. For each claim, check against source chunks:
      - Is the claim supported?
      - Is the citation accurate?
      - Is the scope correct?
   3. Use HALLUCINATION_CHECK_PROMPT for LLM verification
   4. Categorize issues by type
   5. Calculate accuracy score
   6. Determine action

   Action routing:
   - accuracy >= 0.8: "accept"
   - 0.5 <= accuracy < 0.8: "correct"
   - accuracy < 0.5: "regenerate"

   Severity scoring:
   - FABRICATED_CITATION: 0.9 (critical)
   - INCORRECT_NUMBER: 0.8 (high)
   - UNSUPPORTED_CLAIM: 0.7 (high)
   - WRONG_ATTRIBUTION: 0.6 (medium)
   - OVERGENERALIZATION: 0.5 (medium)
   - MISQUOTED_TEXT: 0.4 (low)

**Testing**:
- Create tests/test_validation.py
- Mock LLM responses
- Test hallucination detection
- Test severity scoring
- Test action routing
- Run: `pytest tests/test_validation.py -v`

**Commit**: Create commit with message: "feat(generation): add hallucination detection and validation"

**Verification before next prompt**:
- [ ] Hallucination types detected
- [ ] Severity scored correctly
- [ ] Actions routed properly
- [ ] Issues have fix suggestions
- [ ] All tests pass
```

---

### Prompt 27: Self-Correction

```
Continuing the Federal Tax RAG build. Previous step created hallucination validator.

**Your Task**: Implement the self-correction system to fix identified issues.

**Enter plan mode first** to design the correction strategy.

**Requirements**:
1. Create src/generation/corrector.py with:

   CorrectionResult model:
   - corrected_answer: str
   - corrections_made: list[CorrectionAction]
   - original_confidence: float
   - new_confidence: float
   - correction_type: "simple" | "llm" | "regenerate"

   CorrectionAction model:
   - issue: HallucinationIssue
   - action_taken: str
   - before: str
   - after: str

   ResponseCorrector class:
   - __init__(anthropic_client, generator: TaxLawGenerator)

   Methods:
   - correct(answer: str, validation: ValidationResult, chunks: list) -> CorrectionResult

   Correction strategies:

   Simple corrections (low severity, direct fix):
   - Wrong citation format → fix format
   - Minor text adjustment → find/replace
   - Add disclaimer → append text
   - Implementation: direct string manipulation

   LLM corrections (medium severity, needs rewriting):
   - Unsupported claim → rewrite with qualifier
   - Overgeneralization → add specifics
   - Wrong attribution → correct source
   - Use CORRECTION_PROMPT with specific issues

   Regeneration (high severity or many issues):
   - Route back to generator
   - Include failed answer as negative example
   - Emphasize problematic areas

   Confidence adjustment:
   - Base penalty: -0.2 for any correction
   - Per simple correction: -0.05
   - Per LLM correction: -0.1
   - Per unresolved issue: -0.15
   - Minimum confidence: 0.3

2. Correction tracking:
   - Log all corrections made
   - Track correction effectiveness
   - Store for evaluation metrics

**Testing**:
- Create tests/test_corrector.py
- Test simple corrections
- Test LLM corrections (mocked)
- Test confidence adjustment
- Test regeneration routing
- Run: `pytest tests/test_corrector.py -v`

**Commit**: Create commit with message: "feat(generation): add self-correction system"

**Verification before next prompt**:
- [ ] Simple corrections work
- [ ] LLM corrections work
- [ ] Confidence adjusted correctly
- [ ] Corrections tracked
- [ ] All tests pass
```

---

### Prompt 28: Agent State and Nodes

```
Continuing the Federal Tax RAG build. Previous step created self-correction.

**Your Task**: Implement the LangGraph agent state definition and node functions.

**Enter plan mode first** to design the agent architecture.

**Requirements**:
1. Create src/agent/__init__.py exporting components

2. Create src/agent/state.py with:

   TaxAgentState (TypedDict):
   ```python
   class TaxAgentState(TypedDict, total=False):
       # Input
       original_query: str

       # Decomposition
       sub_queries: list[SubQuery]
       current_sub_query_idx: int
       is_simple_query: bool
       decomposition_reasoning: str

       # Retrieval (accumulates)
       retrieved_chunks: Annotated[list, add]
       current_retrieval_results: list

       # Graph Expansion
       graph_context: list
       interpretation_chains: dict

       # Filtering
       relevance_scores: dict[str, float]
       filtered_chunks: list
       temporally_valid_chunks: list
       query_tax_year: int | None

       # Output
       final_answer: str | None
       citations: list
       confidence: float

       # Validation/Correction
       validation_result: dict | None
       correction_result: dict | None
       validation_passed: bool
       original_answer: str | None
       regeneration_count: int

       # Metadata
       reasoning_steps: Annotated[list, add]
       errors: Annotated[list, add]
       stage_timings: dict
   ```

3. Create src/agent/nodes.py with node functions:

   decompose_query(state: TaxAgentState) -> dict:
   - Use QueryDecomposer
   - Handle simple vs complex queries
   - Fallback on error

   retrieve_for_subquery(state: TaxAgentState) -> dict:
   - Run parallel retrieval for sub-queries
   - Deduplicate results
   - Use asyncio.gather()

   expand_with_graph(state: TaxAgentState) -> dict:
   - Use GraphExpander
   - Build interpretation chains
   - Add expanded context

   score_relevance(state: TaxAgentState) -> dict:
   - Use LLM-based scoring
   - Parallel with semaphore (20)
   - Use Claude Haiku

   filter_irrelevant(state: TaxAgentState) -> dict:
   - Remove chunks below threshold (0.5)
   - Keep minimum 10 chunks

   check_temporal_validity(state: TaxAgentState) -> dict:
   - Extract tax year from query
   - Filter by effective dates

   synthesize_answer(state: TaxAgentState) -> dict:
   - Use TaxLawGenerator
   - Extract citations
   - Calculate confidence

   validate_response(state: TaxAgentState) -> dict:
   - Use ResponseValidator
   - Skip for high-confidence simple queries

   correct_response(state: TaxAgentState) -> dict:
   - Use ResponseCorrector
   - Adjust confidence

   Each node should:
   - Update reasoning_steps
   - Handle errors gracefully
   - Return state updates as dict

**Testing**:
- Create tests/test_agent_nodes.py
- Test each node with mock dependencies
- Test error handling
- Test state updates
- Run: `pytest tests/test_agent_nodes.py -v`

**Commit**: Create commit with message: "feat(agent): add LangGraph state and node functions"

**Verification before next prompt**:
- [ ] State definition complete
- [ ] All 9 nodes implemented
- [ ] Error handling in each node
- [ ] State updates correct
- [ ] All tests pass
```

---

### Prompt 29: Agent Graph Compilation

```
Continuing the Federal Tax RAG build. Previous step created agent nodes.

**Your Task**: Implement the LangGraph agent with conditional edges and graph compilation.

**Enter plan mode first** to design the graph structure.

**Requirements**:
1. Create src/agent/edges.py with conditional edge functions:

   should_expand_graph(state: TaxAgentState) -> str:
   - Return "expand" if:
     - Statutes or regulations found
     - Less than 3 results
   - Otherwise return "score"

   should_continue_retrieval(state: TaxAgentState) -> str:
   - "retrieve_next": More sub-queries remain
   - "decompose_more": Needs more info (temporal issue) and iterations < 3
   - "synthesize": Ready to generate answer

   route_after_validation(state: TaxAgentState) -> str:
   - "accept": validation_passed == True
   - "correct": action == "correct"
   - "regenerate": action == "regenerate" and regeneration_count < 2
   - "accept": regeneration_count >= 2 (give up)

2. Create src/agent/graph.py with:

   build_tax_agent_graph() -> StateGraph:

   Graph structure:
   ```
   START
     │
     ▼
   decompose_query
     │
     ▼
   retrieve_for_subquery ◄──┐
     │                      │
     ▼                      │
   [conditional: expand?]   │
     │         │            │
     ▼         ▼            │
   expand    score          │
     │         │            │
     └────┬────┘            │
          ▼                 │
     filter_irrelevant      │
          │                 │
          ▼                 │
     check_temporal ────────┘
          │        (retrieve_next)
          │
          ▼ (synthesize)
     synthesize_answer
          │
          ▼
     validate_response
          │
          ▼
     [conditional: route]
      │      │       │
      ▼      ▼       ▼
   accept  correct  regenerate
      │      │       │
      │      ▼       └────► synthesize
      │   END
      ▼
     END
   ```

   compile_agent() -> CompiledGraph:
   - Build graph
   - Compile with checkpointing disabled (stateless)
   - Return compiled graph

   AgentRunner class:
   - __init__(graph: CompiledGraph)
   - run(query: str, config: dict) -> TaxAgentState
   - arun(query: str, config: dict) -> TaxAgentState (async)
   - stream(query: str, config: dict) -> AsyncIterator (for SSE)

3. Add profiling integration:
   - Time each node execution
   - Store in state.stage_timings

**Testing**:
- Create tests/test_agent_graph.py
- Test graph compilation
- Test edge routing
- Test full graph execution (integration)
- Run: `pytest tests/test_agent_graph.py -v`

**Commit**: Create commit with message: "feat(agent): add LangGraph compilation and conditional routing"

**Verification before next prompt**:
- [ ] Graph compiles successfully
- [ ] Conditional edges route correctly
- [ ] Agent runs end-to-end
- [ ] Streaming works
- [ ] All tests pass
```

---

### Prompt 30: FastAPI Application

```
Continuing the Federal Tax RAG build. Previous step completed agent graph.

**Your Task**: Implement the FastAPI application with all endpoints.

**Enter plan mode first** to design the API structure.

**Requirements**:
1. Create src/api/__init__.py

2. Create src/api/models.py with request/response schemas:
   - QueryRequest: query, tax_year, timeout, doc_type_filter
   - QueryResponse: answer, citations, sources, confidence, validation_passed, processing_time_ms, stage_timings, warnings
   - SourceDetail: chunk_id, citation, doc_type, text, relevance_score, effective_date
   - HealthResponse: status, services, timestamp
   - ErrorResponse: error, detail, request_id

3. Create src/api/errors.py with:
   - TaxRAGError (base), RateLimitError, QueryTimeoutError, RetrievalError, NotFoundError
   - Exception handlers

4. Create src/api/dependencies.py with:
   - get_settings() - cached settings
   - get_neo4j_client() - cached client
   - get_weaviate_client() - cached client
   - get_agent_graph() - compiled graph
   - get_request_id() - from header or generate

5. Create src/api/middleware.py with:
   - RateLimitMiddleware (per-IP, configurable)
   - RequestLoggingMiddleware (structured logging)

6. Create src/api/cache.py with:
   - Redis-backed query result cache
   - 1-hour TTL
   - Cache key: hash(query + options)

7. Create src/api/routes.py with:
   - POST /api/v1/query - Execute query
   - POST /api/v1/query/stream - Stream results (SSE)
   - GET /api/v1/sources/{chunk_id} - Get chunk details
   - GET /api/v1/statute/{section} - Get statute with interpretation chain
   - GET /api/v1/health - Health check
   - GET /api/v1/metrics - API metrics
   - GET /api/v1/ping - Liveness check

8. Create src/api/main.py with:
   - create_app() factory function
   - Lifespan manager for startup/shutdown
   - Middleware stack (CORS, rate limit, logging)
   - Include router

**Testing**:
- Create tests/test_api.py
- Test each endpoint with mocked dependencies
- Test error handling
- Test rate limiting
- Run: `pytest tests/test_api.py -v`

**Commit**: Create commit with message: "feat(api): add FastAPI application with all endpoints"

**Verification before next prompt**:
- [ ] All endpoints working
- [ ] Error handling correct
- [ ] Rate limiting functional
- [ ] Health check passes
- [ ] All tests pass
```

---

### Prompt 31: Streamlit Frontend

```
Continuing the Federal Tax RAG build. Previous step created FastAPI application.

**Your Task**: Implement the Streamlit frontend for the tax law assistant.

**Enter plan mode first** to design the UI layout.

**Requirements**:
1. Create streamlit_app/app.py with:

   Configuration:
   - API_URL = os.getenv("API_URL", "http://localhost:8000")
   - Page config: title="Federal Tax Law Assistant", layout="wide"

   Sidebar:
   - Document type filters (checkboxes): Statutes, Regulations, Cases, Guidance
   - Tax year input (numeric, default current year)
   - Advanced options expander:
     - Show reasoning steps toggle
   - API health status indicator:
     - Green/red status
     - Service latencies

   Main content:
   - Title: "Federal Tax Law Assistant"
   - Subtitle: "Ask questions about federal tax law..."
   - Text input area for query (100px height)
   - "Search" button
   - Example questions (clickable):
     - "What is gross income under IRC § 61?"
     - "How are capital gains taxed?"
     - "What are the requirements for a § 1031 exchange?"
     - "What is the standard deduction for 2024?"

   Results display:
   - Answer section with markdown formatting
   - Confidence badge (High/Medium/Low with colors)
   - Processing time
   - Validation status indicator
   - Warnings (if any)

   Sources section:
   - Expandable list of source documents
   - Document type badges (colored)
   - Relevance scores
   - Truncated text preview (expandable)
   - Citation format

   Performance section (collapsible):
   - Stage timing breakdown
   - Bar chart visualization

   Helper functions:
   - check_health() -> dict
   - query_tax_law(question, options) -> dict
   - get_source_details(chunk_id) -> dict
   - display_confidence_badge(confidence) - visual badge

2. Error handling:
   - Connection error display
   - Timeout handling
   - User-friendly error messages

3. Legal disclaimer footer:
   - "For informational purposes only. Not legal advice."

**Testing**:
- Manual testing: `streamlit run streamlit_app/app.py`
- Verify UI elements render
- Test query submission
- Test error states

**Commit**: Create commit with message: "feat(frontend): add Streamlit UI for tax law assistant"

**Verification before next prompt**:
- [ ] UI renders correctly
- [ ] Query submission works
- [ ] Results display properly
- [ ] Health status shows
- [ ] Error handling works
```

---

### Prompt 32: Production Deployment

```
Continuing the Federal Tax RAG build. Previous step created Streamlit frontend.

**Your Task**: Create production deployment configuration.

**Enter plan mode first** to design the deployment architecture.

**Requirements**:
1. Create Dockerfile:
   - Base: python:3.11-slim
   - Install dependencies from pyproject.toml
   - Non-root user "appuser"
   - Health check: curl /api/v1/ping
   - Start uvicorn on $PORT

2. Create docker-compose.prod.yml:
   - API service:
     - Build from Dockerfile
     - Port: 8000
     - Memory limits: 2GB
     - Logging: json-file with rotation
     - Health check
   - Environment variables from .env
   - Assumes external managed services:
     - Neo4j Aura
     - Weaviate Cloud
     - ElastiCache Redis

3. Create railway.json:
   - Builder: DOCKERFILE
   - Health check: /api/v1/ping
   - Timeout: 120s
   - Restart: ON_FAILURE

4. Create .env.production.example:
   - All required environment variables
   - Comments explaining each
   - Placeholder values

5. Create scripts/deploy_check.py:
   - Pre-deployment verification
   - Check all required env vars set
   - Test database connections
   - Verify data loaded
   - Output deployment readiness report

6. Update README.md with:
   - Project overview
   - Quick start (local development)
   - Production deployment instructions
   - API documentation summary
   - Environment variables reference

7. Create GitHub Actions workflow .github/workflows/deploy.yml:
   - Trigger on push to main
   - Build Docker image
   - Run tests
   - Deploy to Railway (if configured)

**Testing**:
- Build Docker image: `docker build -t federal_tax_rag .`
- Run container: `docker run -p 8000:8000 federal_tax_rag`
- Test health endpoint
- Run deploy check script

**Commit**: Create commit with message: "infra: add production deployment configuration"

**Verification - Final**:
- [ ] Docker image builds
- [ ] Container runs successfully
- [ ] Railway config valid
- [ ] Deploy check script passes
- [ ] README complete
- [ ] All tests pass
- [ ] System ready for production!
```

---

## Appendix A: Key Differences from Florida System

| Aspect | Florida Tax RAG | Federal Tax RAG |
|--------|-----------------|-----------------|
| **Primary Law** | Florida Statutes Ch. 192-220 | Internal Revenue Code (Title 26 USC) |
| **Regulations** | Florida Admin Code 12A-12D | Treasury Regulations (26 CFR) |
| **Cases** | Florida courts, DCA | Tax Court, Federal Circuit Courts |
| **Guidance** | Technical Assistance Advisements | PLRs, Rev. Rulings, Rev. Procs |
| **Authority Hierarchy** | State-specific | Federal (IRC → Regs → Cases → Guidance) |
| **Citation Formats** | Fla. Stat. §, F.A.C. | IRC §, Treas. Reg. §, T.C. |
| **Tax Types** | Sales/use, corporate, property | Income, estate, gift, excise, employment |
| **Scope** | Single state | National |

## Appendix B: Estimated Implementation Timeline

| Phase | Prompts | Focus |
|-------|---------|-------|
| 1. Foundation | 1-5 | Setup, config, Docker, observability |
| 2. Scrapers | 6-11 | Data collection from all sources |
| 3. Processing | 12-16 | Models, consolidation, chunking, embeddings |
| 4. Storage | 17-20 | Weaviate, Neo4j, data loading |
| 5. Retrieval | 21-24 | Query decomposition, hybrid search, reranking |
| 6. Generation | 25-27 | Answer synthesis, validation, correction |
| 7. Agent | 28-29 | LangGraph orchestration |
| 8. Deployment | 30-32 | API, UI, production config |

## Appendix C: Testing Strategy

Each prompt includes specific testing requirements. The overall testing strategy:

1. **Unit Tests**: Test individual components in isolation with mocks
2. **Integration Tests**: Test component interactions with real services (Docker)
3. **End-to-End Tests**: Test complete query flow
4. **Evaluation Tests**: Test answer quality against golden dataset

Run all tests: `pytest tests/ -v --cov=src`
Run unit only: `pytest tests/ -v -m "not integration"`
Run integration: `pytest tests/ -v -m integration`

---

**Document Version**: 1.0
**Created**: Based on florida_tax_rag architecture
**Target**: federal_tax_rag repository
