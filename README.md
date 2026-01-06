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
│   │   └── multi_retriever.py  # Multi-query parallel retrieval
│   ├── agent/              # LangGraph agentic workflow
│   │   ├── state.py        # TaxAgentState TypedDict
│   │   ├── nodes.py        # 9 node functions (decompose, retrieve, validate, etc.)
│   │   ├── edges.py        # Conditional routing logic
│   │   └── graph.py        # StateGraph definition
│   ├── generation/         # LLM response generation with citations
│   │   ├── formatter.py    # Chunk formatting for context
│   │   ├── generator.py    # TaxLawGenerator class
│   │   ├── validator.py    # ResponseValidator (hallucination detection)
│   │   ├── corrector.py    # ResponseCorrector (self-correction)
│   │   └── models.py       # GeneratedResponse, ValidationResult, etc.
│   ├── api/                # FastAPI REST API
│   │   ├── main.py         # FastAPI app with lifespan
│   │   ├── routes.py       # API endpoints (7 routes)
│   │   ├── models.py       # Request/response Pydantic models
│   │   ├── dependencies.py # Dependency injection (singletons)
│   │   ├── errors.py       # Custom exception hierarchy
│   │   └── middleware.py   # Request logging, rate limiting
│   ├── observability/      # Logging, metrics, tracing
│   │   ├── logging.py      # Centralized structlog configuration
│   │   ├── metrics.py      # Thread-safe metrics collection
│   │   └── context.py      # Request context propagation
│   └── evaluation/         # RAG quality evaluation
│       ├── models.py       # EvalQuestion, EvalResult, JudgmentResult
│       ├── metrics.py      # Citation precision/recall, F1 score
│       ├── llm_judge.py    # GPT-4 based answer evaluation
│       ├── runner.py       # EvaluationRunner orchestration
│       ├── report.py       # Report models + markdown generation
│       ├── retrieval_metrics.py  # MRR, NDCG, Recall@k metrics
│       └── retrieval_analysis.py # RetrievalAnalyzer class
├── config/
│   ├── settings.py         # Pydantic settings with validation
│   ├── development.env     # Development environment defaults
│   ├── staging.env         # Staging environment defaults
│   ├── production.env.template  # Production template (no secrets)
│   └── prompts/            # Centralized LLM prompts
│       ├── retrieval.py    # Query decomposition, classification, relevance
│       ├── generation.py   # Response generation, hallucination detection
│       └── evaluation.py   # LLM judge prompts
├── docs/
│   └── configuration.md    # Configuration guide
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
│   ├── visualize_agent.py  # Visualize LangGraph agent workflow
│   ├── test_load.py        # Load testing script
│   ├── run_evaluation.py   # Run evaluation pipeline
│   ├── analyze_retrieval.py # Retrieval analysis and alpha tuning
│   └── validate_config.py  # Configuration validation script
├── data/
│   ├── raw/                # Raw scraped data
│   │   ├── statutes/       # 742 statute sections
│   │   ├── admin_code/     # 101 administrative rules
│   │   ├── taa/            # Technical Assistance Advisements + PDFs
│   │   └── case_law/       # 308 Florida Supreme Court cases
│   ├── processed/          # Processed data
│   │   ├── corpus.json     # Unified document corpus (4.16 MB)
│   │   ├── chunks.json     # Hierarchical chunks (11.18 MB)
│   │   ├── citation_graph.json # Citation relationships (670 KB)
│   │   ├── embeddings.npz  # Voyage AI embeddings (11 MB)
│   │   └── statistics.json # Consolidation metrics
│   └── evaluation/         # Evaluation datasets
│       ├── golden_dataset.json # 20 seed evaluation questions
│       ├── README.md       # Evaluation methodology
│       └── reports/        # Evaluation run reports
│           └── analysis.md # Baseline analysis
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

The system uses a LangGraph StateGraph to orchestrate multi-step reasoning over tax law queries. The agent processes queries through 9 specialized nodes that progressively refine results.

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
   │ synthesize_answer │ ◄── LLM generation with citations
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ validate_response │ ◄── Hallucination detection
   └────────┬─────────┘
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
   accept  correct  regenerate
     │      │          │
     │      ▼          │
     │  ┌────────┐     │
     │  │ correct │     │
     │  │ response│     │
     │  └────┬───┘     │
     │       │         │
     ▼       ▼         │
   ┌─────────────┐     │
   │     END     │◄────┘ (back to synthesize, max 2x)
   └─────────────┘
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
| `synthesize_answer` | Generate final answer | TaxLawGenerator: LLM call, citation extraction & validation |
| `validate_response` | Detect hallucinations | LLM-based semantic verification against sources |
| `correct_response` | Fix hallucinated content | Text replacement or LLM rewriting |

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

# Validation state (hallucination detection)
validation_result: dict      # ValidationResult with detected hallucinations
correction_result: dict      # CorrectionResult with corrections made
regeneration_count: int      # Number of regeneration attempts (max 2)
validation_passed: bool      # Whether response passed validation
original_answer: str         # Original answer before corrections
```

### Visualizing the Agent

```bash
# Generate workflow diagram
python scripts/visualize_agent.py
```

## Answer Generation

The `TaxLawGenerator` produces legally-accurate responses with validated citations using Claude LLM.

### Generation Pipeline

```
temporally_valid_chunks → TaxLawGenerator → GeneratedResponse
                              │
                              ├── format_chunks_for_context()
                              ├── LLM call with SYSTEM_PROMPT
                              ├── extract_citations()
                              └── validate_citations()
```

### System Prompt

The generator uses a specialized **Florida Tax Attorney** persona with 12 critical rules:
1. Answer using ONLY provided legal context
2. Cite specific Florida Statute (§) or Rule (F.A.C.) for EVERY claim
3. Format citations as `[Source: § 212.05(1)]` or `[Source: Rule 12A-1.005]`
4. State when law is ambiguous or context insufficient
5. Present multiple interpretations with sources
6. Note effective dates of cited provisions
7. Distinguish between Statutes, Rules, Cases, and TAAs

### Citation Validation

Citations extracted from LLM responses are validated against source chunks:

```python
from src.generation import TaxLawGenerator

generator = TaxLawGenerator()

# Generate response with validated citations
response = await generator.generate(
    query="What is the sales tax rate in Florida?",
    chunks=retrieved_chunks[:10],
)

print(f"Answer: {response.answer[:200]}...")
print(f"Confidence: {response.confidence:.2f}")
print(f"Citations: {len(response.citations)}")

for citation in response.citations:
    status = "✓" if citation.verified else "⚠ UNVERIFIED"
    print(f"  {status} {citation.citation_text}")
```

### Confidence Scoring

Confidence is calculated based on source quality and citation verification:

| Factor | Weight |
|--------|--------|
| Source quality (top 5 chunks) | 60% |
| Citation verification rate | 40% |

**Source type weights:**
| Document Type | Weight |
|--------------|--------|
| Statute | 1.0 |
| Rule | 0.9 |
| Case | 0.7 |
| TAA | 0.6 |

### Hallucination Detection & Self-Correction

The system includes comprehensive hallucination detection and self-correction using the `ResponseValidator` and `ResponseCorrector` classes.

#### Hallucination Types Detected

| Type | Description | Severity |
|------|-------------|----------|
| `unsupported_claim` | Claim not supported by any source | 0.7-0.9 |
| `misquoted_text` | Inaccurate quotation of source material | 0.5-0.7 |
| `fabricated_citation` | Citation to non-existent law | 0.9 |
| `outdated_info` | Superseded or repealed provisions | 0.6-0.8 |
| `misattributed` | Correct info attributed to wrong source | 0.4-0.6 |
| `overgeneralization` | Claim too broad for source support | 0.3-0.5 |

#### Validation Pipeline

```python
from src.generation import ResponseValidator, ResponseCorrector

validator = ResponseValidator()  # Uses Claude Haiku (fast, cheap)
corrector = ResponseCorrector()  # Uses Claude Sonnet (higher quality)

# Validate response against source chunks
result = await validator.validate_response(
    response_text=answer,
    query=original_query,
    chunks=retrieved_chunks,
)

print(f"Accuracy: {result.overall_accuracy:.1%}")
print(f"Hallucinations: {len(result.hallucinations)}")
print(f"Verified claims: {len(result.verified_claims)}")

if result.needs_correction:
    correction = await corrector.correct(answer, result, chunks)
    print(f"Corrected answer: {correction.corrected_answer[:200]}...")
    print(f"Confidence adjustment: {correction.confidence_adjustment:.0%}")
```

#### Routing Thresholds

| Metric | Regenerate | Correct | Accept |
|--------|------------|---------|--------|
| Max severity | >= 0.9 | < 0.9 | - |
| Avg severity | >= 0.7 | >= 0.3 | < 0.3 |
| Accuracy | < 0.5 | < 0.8 | >= 0.8 |

- **Regenerate**: Severe hallucinations require complete re-synthesis (max 2 attempts)
- **Correct**: Moderate issues can be patched via text replacement or LLM rewriting
- **Accept**: Response passes validation, proceed to output

#### Confidence Adjustment

After corrections, confidence is reduced to reflect uncertainty:

| Correction Type | Adjustment |
|-----------------|------------|
| Simple text replacement | -10% per correction (max -30%) |
| LLM-based rewriting | -20% base + severity-based (max -50%) |

#### Simple Citation Warnings

The generator also flags citations not found in provided context:

```python
# Warnings indicate potential hallucinations
if response.warnings:
    print("Warnings:")
    for w in response.warnings:
        print(f"  ⚠ {w}")
# Example: "2 citation(s) could not be verified against provided sources"
```

## REST API

The system exposes a FastAPI REST API for querying Florida tax law.

### Starting the API

```bash
# Development (with hot reload)
make dev

# Production (with 4 workers)
make serve
```

The API will be available at `http://localhost:8000`. Swagger docs at `/docs`.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Execute tax law query through the agent |
| `/api/v1/query/stream` | POST | Stream query results via SSE |
| `/api/v1/sources/{chunk_id}` | GET | Get full chunk by ID |
| `/api/v1/statute/{section}` | GET | Get statute with implementing rules |
| `/api/v1/graph/{doc_id}/related` | GET | Get related documents via citations |
| `/api/v1/metrics` | GET | Get API metrics (queries, latency, errors) |
| `/api/v1/health` | GET | Check Neo4j and Weaviate health |

### Query Example

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Florida sales tax rate?"}'
```

Response:
```json
{
  "request_id": "abc123",
  "answer": "The Florida sales tax rate is 6%...",
  "citations": [
    {"doc_id": "statute:212.05", "citation": "Fla. Stat. § 212.05", "doc_type": "statute"}
  ],
  "sources": [...],
  "confidence": 0.85,
  "validation_passed": true,
  "processing_time_ms": 3200
}
```

### Query Options

```json
{
  "query": "What is the sales tax rate?",
  "options": {
    "tax_year": 2024,
    "include_reasoning": true,
    "timeout_seconds": 60
  }
}
```

### Streaming Example

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the sales tax rate?"}'
```

Streams SSE events: `status`, `reasoning`, `chunk`, `answer`, `complete`.

### Metrics Endpoint

```bash
curl http://localhost:8000/api/v1/metrics
```

Response:
```json
{
  "total_queries": 150,
  "successful_queries": 142,
  "failed_queries": 8,
  "success_rate_percent": 94.67,
  "latency_ms": {
    "avg": 3250.5,
    "min": 1200.0,
    "max": 8500.0
  },
  "errors_by_type": {
    "TIMEOUT": 5,
    "RETRIEVAL_ERROR": 3
  },
  "uptime_seconds": 3600,
  "started_at": "2024-01-15T10:00:00"
}
```

## Observability & Error Handling

The system includes production-grade observability features for monitoring, debugging, and performance analysis.

### Structured Logging

All components use structured logging via `structlog` with automatic request context propagation:

```python
from src.observability.logging import get_logger

logger = get_logger(__name__)
logger.info("query_started", query=query[:50], timeout=60)
```

**Log Output (JSON in production):**
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "event": "query_started",
  "request_id": "abc-123",
  "query_id": "q-xyz",
  "query": "What is the Florida sales tax...",
  "timeout": 60
}
```

**Configuration:**
- `ENVIRONMENT=production`: JSON output for log aggregation
- `ENVIRONMENT=development`: Colored console output
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default: INFO)

### Request Tracing

Every request is assigned tracing IDs that propagate through all logs:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (from client or generated) |
| `X-Query-ID` | Short 8-character ID for query tracing |

Both IDs are automatically included in all log messages and returned in response headers.

### Rate Limiting

In-memory sliding window rate limiter protects the API:

```bash
# Default: 60 requests per minute per IP
# Configure via environment variable
RATE_LIMIT_PER_MINUTE=100
```

**Excluded paths** (not rate limited):
- `/api/v1/health`
- `/api/v1/metrics`
- `/docs`, `/redoc`, `/`

**Rate limit response:**
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded: 60 requests per minute",
  "details": {
    "limit": 60,
    "window_seconds": 60,
    "retry_after_seconds": 60
  }
}
```

### Custom Exceptions

The API uses a structured exception hierarchy for consistent error responses:

| Exception | Error Code | HTTP Status | Use Case |
|-----------|------------|-------------|----------|
| `TaxRAGError` | `TAX_RAG_ERROR` | 500 | Base exception |
| `RetrievalError` | `RETRIEVAL_ERROR` | 503 | Neo4j/Weaviate failures |
| `GenerationError` | `GENERATION_ERROR` | 502 | Claude API failures |
| `ValidationError` | `VALIDATION_ERROR` | 400 | Invalid input |
| `RateLimitError` | `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `QueryTimeoutError` | `TIMEOUT` | 408 | Query exceeded timeout |
| `NotFoundError` | `NOT_FOUND` | 404 | Resource not found |

**Error response format:**
```json
{
  "request_id": "abc-123",
  "error": "RETRIEVAL_ERROR",
  "message": "Neo4j connection failed",
  "details": {"error_type": "ConnectionError"},
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Metrics Collection

Thread-safe in-memory metrics collector tracks API performance:

```python
from src.observability.metrics import get_metrics_collector

metrics = get_metrics_collector()
stats = metrics.get_stats()

print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['success_rate_percent']}%")
print(f"Avg latency: {stats['latency_ms']['avg']}ms")
```

**Tracked metrics:**
- Query counts (total, successful, failed)
- Latency statistics (avg, min, max)
- Error breakdown by type
- Uptime tracking

### Load Testing

Test API performance with the included load testing script:

```bash
# Basic load test (50 requests, 5 concurrent)
python scripts/test_load.py

# Custom parameters
python scripts/test_load.py --num-requests 100 --concurrency 10

# With think time between requests
python scripts/test_load.py -n 50 -c 5 --think-time 0.5

# Custom endpoint
python scripts/test_load.py --url http://localhost:8000/api/v1/query
```

**Sample output:**
```
============================================================
LOAD TEST RESULTS
============================================================

Summary:
  Total Requests:     50
  Successful:         47
  Failed:             3
  Success Rate:       94.0%
  Duration:           45.23s
  Requests/Second:    1.11

Latency (ms):
  Average:            3250
  P50:                2800
  P95:                6500
  P99:                8200
  Min:                1200
  Max:                8500

Status Codes:
  200: 47
  408: 2
  429: 1

Errors:
  TIMEOUT: 2
  RATE_LIMIT_EXCEEDED: 1
============================================================
```

## Evaluation Framework

The system includes a comprehensive evaluation framework for measuring RAG quality with citation accuracy metrics and LLM-based judging.

### Golden Dataset

A seed dataset of 20 Florida tax law questions with expected answers:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 5 | Direct statutory answers (sales tax rate, basic exemptions) |
| Medium | 10 | Requires connecting statute + rule |
| Hard | 5 | Nuanced "it depends" answers, case law interpretation |

**Categories:** sales_tax, property_tax, corporate_tax, exemptions, procedures

### Evaluation Metrics

**Citation Metrics:**
- **Precision** = Correct Citations / Generated Citations
- **Recall** = Found Citations / Expected Citations
- **F1 Score** = Harmonic mean of precision and recall

**Answer Quality (LLM Judge, 0-10 scale):**
- Correctness, Completeness, Clarity, Citation Accuracy

**Pass Criteria:** Overall score ≥ 7, no hallucinations

### Running Evaluations

```bash
# Run full evaluation (all 20 questions with GPT-4 judge)
make eval

# Run quick evaluation (5 questions, no judge - for testing)
make eval-quick

# Run without LLM judge (faster, citation metrics only)
make eval-no-judge

# Custom options
python scripts/run_evaluation.py --limit 10 --output data/evaluation/reports
```

**CLI Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to golden dataset JSON | `data/evaluation/golden_dataset.json` |
| `--output` | Output directory for reports | `data/evaluation/reports` |
| `--limit` | Limit to N questions | All questions |
| `--no-judge` | Skip GPT-4 LLM judge | Enabled |
| `--timeout` | Timeout per question (seconds) | 120 |

### Evaluation Reports

Each run generates two report files:

```
data/evaluation/reports/
├── eval_{run_id}.json    # Machine-readable results
├── eval_{run_id}.md      # Human-readable markdown report
└── analysis.md           # Baseline analysis and findings
```

**Markdown Report Contents:**
- Summary table (pass rate, precision, recall, F1, latency)
- Results by category (sales_tax, exemptions, etc.)
- Results by difficulty (easy, medium, hard)
- Worst performing questions
- Individual question results

### Baseline Metrics (v1.0)

Initial evaluation results establish the baseline:

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Citation Precision | 85-90% | 90% | Generated citations are accurate |
| Citation Recall | 40-50% | 80% | Some expected citations missed |
| Citation F1 | 55-60% | 85% | Balanced measure |
| Hallucinations | 0 | 0 | No fabricated citations |
| Avg Latency | 35-70s | 10s | Includes retrieval + generation |

**Strengths:**
- High precision: generated citations reference real statutes
- Zero hallucinations: no fabricated legal references
- Comprehensive answers with exceptions and caveats

**Areas for Improvement:**
- Citation recall: retrieve more expected sources
- Latency: optimize context size and caching

### Using the Evaluation Module

```python
from src.evaluation import (
    EvalDataset,
    EvaluationRunner,
    LLMJudge,
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
)
from src.agent import create_tax_agent_graph

# Load golden dataset
import json
with open("data/evaluation/golden_dataset.json") as f:
    dataset = EvalDataset(**json.load(f))

# Initialize components
graph = create_tax_agent_graph()
judge = LLMJudge(api_key="your-openai-key")

# Create runner and execute evaluation
runner = EvaluationRunner(agent=graph, judge=judge, dataset_path="data/evaluation/golden_dataset.json")

# Run evaluation
report = await runner.run_all(limit=5, progress_callback=lambda i, n: print(f"{i}/{n}"))

print(f"Pass Rate: {report.pass_rate:.1%}")
print(f"Citation F1: {report.avg_citation_f1:.1%}")
print(f"Avg Latency: {report.avg_latency_ms:.0f}ms")

# Generate markdown report
from src.evaluation import generate_markdown_report
markdown = generate_markdown_report(report)
```

See [data/evaluation/README.md](./data/evaluation/README.md) for full methodology and [data/evaluation/reports/analysis.md](./data/evaluation/reports/analysis.md) for detailed findings.

### Retrieval Analysis

The system includes a dedicated retrieval analysis module for measuring and optimizing search quality independent of answer generation.

**Retrieval Metrics:**
- **MRR (Mean Reciprocal Rank)** - How high the first relevant doc ranks
- **NDCG@k** - Normalized Discounted Cumulative Gain at k
- **Recall@k** - Fraction of relevant docs in top-k
- **Precision@k** - Fraction of top-k that are relevant

**Running Retrieval Analysis:**

```bash
# Full analysis with method comparison
python scripts/analyze_retrieval.py

# Alpha tuning (find optimal vector vs keyword balance)
python scripts/analyze_retrieval.py --tune-alpha

# Analyze single question
python scripts/analyze_retrieval.py --question eval_001

# Debug a specific query
python scripts/analyze_retrieval.py --debug "What is the Florida sales tax rate?"

# Limit to N questions
python scripts/analyze_retrieval.py --limit 5
```

**Alpha Tuning Results:**

The alpha parameter controls the balance between vector similarity (1.0) and keyword/BM25 matching (0.0):

| Alpha | MRR | Recall@10 | Interpretation |
|-------|-----|-----------|----------------|
| **0.25** | **0.61** | **0.60** | **Best - keyword-heavy hybrid** |
| 0.00 | 0.42 | 0.50 | Pure keyword |
| 0.50 | 0.51 | 0.60 | Balanced |
| 1.00 | 0.25 | 0.50 | Pure vector (worst) |

**Key Finding:** Keyword-weighted hybrid search (alpha=0.25) significantly outperforms pure vector or pure keyword approaches for legal document retrieval. Legal queries benefit from exact term matching (statute numbers, legal terminology).

**Using the Retrieval Analysis Module:**

```python
from src.evaluation import (
    RetrievalAnalyzer,
    debug_retrieval,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.retrieval import create_retriever

# Create retriever and analyzer
retriever = create_retriever()
analyzer = RetrievalAnalyzer(retriever, dataset)

# Compare retrieval methods for a question
comparison = analyzer.compare_retrieval_methods(question, top_k=20)
print(f"Best method: {comparison.best_method} (MRR={comparison.best_mrr:.4f})")

# Tune alpha parameter
results = analyzer.tune_alpha(alphas=[0.0, 0.25, 0.5, 0.75, 1.0])
optimal = max(results, key=lambda r: r.mrr)
print(f"Optimal alpha: {optimal.alpha}")

# Debug single query
debug_info = debug_retrieval(retriever, "What is the sales tax rate?")
```

See [RETRIEVAL_ANALYSIS.md](./RETRIEVAL_ANALYSIS.md) for detailed findings and recommendations.

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
- `openai` - GPT-4 evaluation judge

### Scraping
- `httpx` - Async HTTP client
- `beautifulsoup4` / `lxml` - HTML parsing
- `pdfplumber` / `pypdf` - PDF text extraction
- `tenacity` - Retry logic

### Data Validation
- `pydantic` - Data models and validation

## Configuration

### Quick Start

```bash
# Copy example configuration
cp .env.example .env

# Fill in API keys and run validation
python scripts/validate_config.py
```

### Environment Variables

Create a `.env` file (see `.env.example` for full documentation):

```env
# Environment
ENV=development               # development | staging | production
DEBUG=true
LOG_LEVEL=INFO                # DEBUG | INFO | WARNING | ERROR

# Required API Keys
VOYAGE_API_KEY=your_key       # Voyage AI (legal embeddings)
ANTHROPIC_API_KEY=your_key    # Anthropic (Claude LLM)
NEO4J_PASSWORD=your_password

# Optional API Keys
OPENAI_API_KEY=your_key       # GPT-4 evaluation judge (optional)
WEAVIATE_API_KEY=your_key     # Weaviate Cloud (optional)

# Database Connections
NEO4J_URI=bolt://localhost:7687
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379/0

# Retrieval Settings
HYBRID_ALPHA=0.25             # Optimal: keyword-heavy hybrid (0=keyword, 1=vector)
RETRIEVAL_TOP_K=20

# Generation Settings
LLM_MODEL=claude-sonnet-4-20250514
LLM_TEMPERATURE=0.1
MAX_TOKENS=4096

# Rate Limits
RATE_LIMIT_PER_MINUTE=60
```

### Environment-Specific Configuration

| Environment | File | Key Settings |
|-------------|------|--------------|
| Development | `config/development.env` | `DEBUG=true`, `LOG_LEVEL=DEBUG`, relaxed rate limits |
| Staging | `config/staging.env` | `DEBUG=false`, `LOG_LEVEL=INFO` |
| Production | `config/production.env.template` | `LOG_LEVEL=WARNING`, fail-fast startup |

### Configuration Validation

```bash
# Full validation (settings + service connections)
python scripts/validate_config.py

# Quick validation (settings only)
python scripts/validate_config.py --quick

# Test specific service
python scripts/validate_config.py --service neo4j

# Show masked API keys
python scripts/validate_config.py --verbose
```

**Production Behavior:** In production (`ENV=production`), the API fails fast if Neo4j or Weaviate are unavailable.

See [docs/configuration.md](./docs/configuration.md) for complete configuration documentation.

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

# API
make dev                # Start API with hot reload (development)
make serve              # Start API in production mode

# Evaluation
make eval               # Run full evaluation (all questions + GPT-4 judge)
make eval-quick         # Run quick evaluation (5 questions, no judge)
make eval-no-judge      # Run all questions without LLM judge

# Load Testing
python scripts/test_load.py           # Run load test (50 req, 5 concurrent)
python scripts/test_load.py -n 100    # Custom request count

# Configuration
python scripts/validate_config.py     # Validate settings and test connections
python scripts/validate_config.py --quick  # Validate settings only

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
  - [x] Unit tests (377 tests passing)
  - [x] Integration tests

- [x] **Phase 6: Answer Generation & API**
  - [x] Full answer synthesis with Claude (TaxLawGenerator)
  - [x] Citation extraction and validation
  - [x] Hallucination detection (LLM-based semantic verification)
  - [x] Self-correction (ResponseValidator + ResponseCorrector)
  - [x] Confidence scoring (source quality + verification rate)
  - [x] FastAPI REST endpoint (7 endpoints)
  - [x] Streaming responses (SSE)

- [x] **Phase 7: Observability & Error Handling**
  - [x] Custom exception hierarchy (`TaxRAGError`, `RetrievalError`, etc.)
  - [x] Structured logging with structlog (JSON/console output)
  - [x] Request tracing (`request_id`, `query_id` propagation)
  - [x] Request logging middleware
  - [x] Rate limiting middleware (sliding window)
  - [x] Metrics collection (`/api/v1/metrics` endpoint)
  - [x] Load testing script (`scripts/test_load.py`)
  - [x] Observability tests (27 tests)

- [x] **Phase 8: Evaluation Framework**
  - [x] Evaluation models (`EvalQuestion`, `EvalResult`, `JudgmentResult`)
  - [x] Citation metrics (precision, recall, F1 score)
  - [x] LLM judge with GPT-4 (`LLMJudge` class)
  - [x] Golden dataset (20 questions: 5 easy, 10 medium, 5 hard)
  - [x] Evaluation methodology documentation
  - [x] Evaluation tests (42 tests)
  - [x] Evaluation runner (`EvaluationRunner` class)
  - [x] CLI script (`scripts/run_evaluation.py`)
  - [x] Report generation (JSON + Markdown)
  - [x] Baseline metrics established (v1.0)

- [x] **Phase 9: Configuration Management**
  - [x] Comprehensive Settings class with validators
  - [x] Environment-specific configs (development, staging, production)
  - [x] Centralized prompts in `config/prompts/`
  - [x] Configuration validation script (`scripts/validate_config.py`)
  - [x] Production fail-fast startup behavior
  - [x] Configuration documentation (`docs/configuration.md`)

## Documentation

- [docs/configuration.md](./docs/configuration.md) - Configuration guide (environment variables, validation)
- [SCRAPING_NOTES.md](./SCRAPING_NOTES.md) - Detailed scraping documentation
- [RETRIEVAL_ANALYSIS.md](./RETRIEVAL_ANALYSIS.md) - Retrieval analysis and alpha tuning results
- [data/processed/README.md](./data/processed/README.md) - Unified corpus schema documentation
- [data/evaluation/README.md](./data/evaluation/README.md) - Evaluation methodology and metrics

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [CourtListener](https://www.courtlistener.com/) - Free legal research API
- [Florida Legislature](https://www.leg.state.fl.us/) - Official statutes
- [Florida Administrative Code](https://www.flrules.org/) - Administrative rules
- [Florida DOR](https://floridarevenue.com/) - Tax guidance documents
