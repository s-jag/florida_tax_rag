# System Architecture

This document provides a comprehensive overview of the Florida Tax RAG system architecture, including component diagrams, data flow, and database schemas.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Knowledge Graph Schema](#knowledge-graph-schema)
- [Vector Store Schema](#vector-store-schema)
- [Retrieval Architecture](#retrieval-architecture)
- [Agent Workflow](#agent-workflow)
- [Generation Pipeline](#generation-pipeline)

---

## System Overview

The Florida Tax RAG system is a Hybrid Agentic GraphRAG system that combines:

- **Vector Search** (Weaviate) - Semantic similarity using legal-specific embeddings
- **Keyword Search** (Weaviate BM25) - Traditional keyword matching
- **Knowledge Graph** (Neo4j) - Citation relationships and document hierarchy
- **Multi-Agent Orchestration** (LangGraph) - Stateful query processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Florida Tax RAG System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Scrapers   │───▶│  Ingestion   │───▶│   Storage    │                   │
│  │              │    │              │    │              │                   │
│  │ • Statutes   │    │ • Chunking   │    │ • Neo4j      │                   │
│  │ • Admin Code │    │ • Citations  │    │ • Weaviate   │                   │
│  │ • TAAs       │    │ • Embeddings │    │ • Redis      │                   │
│  │ • Case Law   │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                                                  ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Query Processing                               │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐  │   │
│  │  │   Query    │──▶│  Hybrid    │──▶│   Agent    │──▶│ Generation │  │   │
│  │  │Decomposer  │   │ Retrieval  │   │  Workflow  │   │  Pipeline  │  │   │
│  │  └────────────┘   └────────────┘   └────────────┘   └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                  │                           │
│                                                  ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          REST API (FastAPI)                           │   │
│  │  /query  /query/stream  /sources  /statute  /graph  /health  /metrics│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Data Collection Layer (`src/scrapers/`)

Responsible for collecting legal documents from official sources.

| Component | Source | Output |
|-----------|--------|--------|
| `statutes.py` | Florida Legislature | Chapters 192-220 (Tax & Finance) |
| `admin_code.py` | Florida Administrative Code | Chapter 12A (DOR Rules) |
| `taa.py` | Florida DOR | Technical Assistance Advisements |
| `case_law.py` | CourtListener API | Florida Supreme Court cases |

**Features:**
- Rate limiting with configurable delays
- Exponential backoff retry logic
- Request caching to avoid redundant fetches
- Structured logging

### Ingestion Layer (`src/ingestion/`)

Processes raw documents into the unified format.

```
Raw Data                    Processed Data
─────────                   ──────────────
statutes/*.json    ─┐
admin_code/*.json   ├──▶  corpus.json ──▶ chunks.json ──▶ embeddings.npz
taa/*.json          │                            │
case_law/*.json   ─┘                            ▼
                                         citation_graph.json
```

| Component | Purpose |
|-----------|---------|
| `models.py` | Unified `LegalDocument` model |
| `consolidate.py` | Merge all sources into corpus |
| `chunking.py` | Hierarchical chunking (parent/child) |
| `tokenizer.py` | Token counting with tiktoken |
| `citation_extractor.py` | Extract cross-references |
| `build_citation_graph.py` | Build citation relationships |

### Storage Layer

#### Neo4j Knowledge Graph (`src/graph/`)

Stores document metadata and citation relationships.

| Component | Purpose |
|-----------|---------|
| `schema.py` | Node labels, relationships, constraints |
| `client.py` | `Neo4jClient` with connection pooling |
| `loader.py` | Data loading functions |
| `queries.py` | Graph traversal queries |

#### Weaviate Vector Store (`src/vector/`)

Stores chunks with embeddings for hybrid search.

| Component | Purpose |
|-----------|---------|
| `schema.py` | `LegalChunk` collection definition |
| `client.py` | `WeaviateClient` with hybrid search |
| `embeddings.py` | `VoyageEmbedder` + Redis cache |

#### Redis Cache

Caches computed embeddings to reduce API calls to Voyage AI.

### Retrieval Layer (`src/retrieval/`)

Combines multiple retrieval strategies.

| Component | Purpose |
|-----------|---------|
| `models.py` | `RetrievalResult`, `CitationContext` |
| `hybrid.py` | `HybridRetriever` (vector + keyword + graph) |
| `graph_expander.py` | Neo4j graph expansion |
| `reranker.py` | Legal-specific reranking |
| `query_decomposer.py` | Claude-powered query decomposition |
| `multi_retriever.py` | Multi-query parallel retrieval |

### Agent Layer (`src/agent/`)

LangGraph-based stateful query processing.

| Component | Purpose |
|-----------|---------|
| `state.py` | `TaxAgentState` TypedDict |
| `nodes.py` | 9 node functions |
| `edges.py` | Conditional routing logic |
| `graph.py` | StateGraph definition |

### Generation Layer (`src/generation/`)

LLM-based response generation with validation.

| Component | Purpose |
|-----------|---------|
| `formatter.py` | Chunk formatting for context |
| `generator.py` | `TaxLawGenerator` class |
| `validator.py` | `ResponseValidator` (hallucination detection) |
| `corrector.py` | `ResponseCorrector` (self-correction) |
| `models.py` | `GeneratedResponse`, `ValidationResult` |

### API Layer (`src/api/`)

FastAPI REST API.

| Component | Purpose |
|-----------|---------|
| `main.py` | FastAPI app with lifespan |
| `routes.py` | 8 API endpoints |
| `models.py` | Request/response Pydantic models |
| `dependencies.py` | Dependency injection |
| `errors.py` | Custom exception hierarchy |
| `middleware.py` | Logging, rate limiting |

---

## Data Flow

### Query Processing Flow

```
User Query
    │
    ▼
┌───────────────────┐
│ Query Decomposer  │  Claude LLM analyzes complexity
└─────────┬─────────┘  Returns: is_simple, sub_queries[]
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
Simple       Complex
Query        Query
    │           │
    │     ┌─────┴─────┐
    │     │ For each  │
    │     │ sub-query │
    │     └─────┬─────┘
    │           │
    └─────┬─────┘
          │
          ▼
┌───────────────────┐
│ Hybrid Retriever  │
├───────────────────┤
│ 1. Embed query    │  Voyage AI voyage-law-2
│ 2. Vector search  │  Weaviate cosine similarity
│ 3. Keyword search │  Weaviate BM25
│ 4. Merge & score  │  Alpha-weighted combination
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Graph Expander   │  Neo4j traversal
├───────────────────┤
│ • Implementing    │  Statute → Rules
│   rules           │
│ • Interpreting    │  Statute → Cases
│   cases           │
│ • Related TAAs    │  Any → TAAs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Relevance Filter  │  LLM scoring (0-10)
├───────────────────┤
│ Threshold: 5.0    │
│ Minimum: 10 chunks│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Temporal Filter   │  Check effective dates
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Answer Synthesis │  Claude LLM generation
├───────────────────┤
│ • Format context  │
│ • Generate answer │
│ • Extract cites   │
│ • Validate cites  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Validation     │  Hallucination detection
├───────────────────┤
│ If accuracy < 0.8 │─▶ Correction or Regeneration
└─────────┬─────────┘
          │
          ▼
     Final Answer
```

### Document Ingestion Flow

```
Raw Document (JSON)
       │
       ▼
┌─────────────────┐
│  Consolidation  │  Normalize to LegalDocument
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  Hierarchical parent/child
└────────┬────────┘
         │
         ├──────────────┐
         ▼              ▼
┌─────────────────┐ ┌─────────────────┐
│   Embeddings    │ │ Citation Graph  │
│   (Voyage AI)   │ │   Extraction    │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│    Weaviate     │ │     Neo4j       │
│  (chunks +      │ │  (documents +   │
│   vectors)      │ │   citations)    │
└─────────────────┘ └─────────────────┘
```

---

## Knowledge Graph Schema

### Neo4j Database

#### Node Labels

```
┌──────────────────────────────────────────────────────────────┐
│                        Node Labels                            │
├──────────────┬───────────────────────────────────────────────┤
│ Document     │ Root container for all document types         │
│ Statute      │ Florida statutes (Chapter 192-220)            │
│ Rule         │ Administrative rules (12A-1.xxx)              │
│ Case         │ Court decisions                               │
│ TAA          │ Technical Assistance Advisements              │
│ Chunk        │ Text segments for retrieval                   │
└──────────────┴───────────────────────────────────────────────┘
```

#### Relationship Types

```
Citation Relationships:
─────────────────────
(Document)-[:CITES]->(Document)        # Direct citation
(Rule)-[:IMPLEMENTS]->(Statute)        # Rule implements statute
(Document)-[:AUTHORITY]->(Document)    # Authority reference
(Case)-[:INTERPRETS]->(Statute)        # Case interprets statute
(Document)-[:AMENDS]->(Document)       # Amendment
(Document)-[:SUPERSEDES]->(Document)   # Supersession

Structural Relationships:
────────────────────────
(Document)-[:HAS_CHUNK]->(Chunk)       # Document contains chunk
(Chunk)-[:CHILD_OF]->(Chunk)           # Child to parent hierarchy
```

#### Schema Diagram

```
                    ┌─────────────┐
                    │   Statute   │
                    │             │
                    │ • doc_id    │
                    │ • section   │
                    │ • chapter   │
                    │ • title     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
     [:IMPLEMENTS]   [:INTERPRETS]    [:CITES]
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Rule   │    │   Case   │    │   TAA    │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
    [:HAS_CHUNK]    [:HAS_CHUNK]    [:HAS_CHUNK]
         │               │               │
         ▼               ▼               ▼
    ┌─────────────────────────────────────────┐
    │                 Chunk                    │
    │                                          │
    │ • chunk_id (unique)                      │
    │ • doc_id (indexed)                       │
    │ • level (parent/child)                   │
    │ • text                                   │
    │ • token_count                            │
    └─────────────────────────────────────────┘
              │
         [:CHILD_OF]
              │
              ▼
         Parent Chunk
```

#### Constraints and Indexes

```cypher
-- Constraints
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

-- Indexes
CREATE INDEX doc_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type);
CREATE INDEX doc_section_idx IF NOT EXISTS FOR (d:Document) ON (d.section);
CREATE INDEX doc_chapter_idx IF NOT EXISTS FOR (d:Document) ON (d.chapter);
CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id);
CREATE INDEX chunk_level_idx IF NOT EXISTS FOR (c:Chunk) ON (c.level);
CREATE INDEX chunk_doc_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_type);
```

---

## Vector Store Schema

### Weaviate Collection: `LegalChunk`

#### Properties

| Property | Type | Tokenization | Description |
|----------|------|--------------|-------------|
| `chunk_id` | TEXT | FIELD | Unique identifier (e.g., `chunk:statute:212.05:0`) |
| `doc_id` | TEXT | FIELD | Parent document ID (e.g., `statute:212.05`) |
| `doc_type` | TEXT | WORD | Document type: `statute`, `rule`, `case`, `taa` |
| `level` | TEXT | FIELD | Hierarchy level: `parent` or `child` |
| `ancestry` | TEXT | WORD | Hierarchical path (for BM25) |
| `citation` | TEXT | WORD | Legal citation (e.g., `Fla. Stat. § 212.05`) |
| `text` | TEXT | WORD | Raw chunk text (for BM25) |
| `text_with_ancestry` | TEXT | WORD | Text with ancestry prefix (for embedding) |
| `effective_date` | DATE | - | Effective date of legal document |
| `token_count` | INT | - | Token count in chunk |

#### Vector Configuration

```yaml
vectorizer: none  # External vectors from Voyage AI
vector_index_type: hnsw
vector_index_config:
  distance_metric: cosine
  ef_construction: 128
  max_connections: 64
  ef: -1  # dynamic
vector_dimension: 1024  # voyage-law-2
```

#### BM25 Configuration

```yaml
bm25:
  b: 0.75   # Document length normalization
  k1: 1.2   # Term frequency saturation
```

#### Hybrid Search

Weaviate combines vector and keyword search:

```
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score

alpha = 0.25 (optimal for legal queries - keyword-heavy)
```

---

## Retrieval Architecture

### Hybrid Retrieval Pipeline

```
Query: "What is the sales tax rate in Florida?"
                    │
                    ▼
         ┌──────────────────┐
         │  Query Embedding │  Voyage AI voyage-law-2
         │  dim: 1024       │
         └────────┬─────────┘
                  │
     ┌────────────┴────────────┐
     │                         │
     ▼                         ▼
┌─────────────┐         ┌─────────────┐
│   Vector    │         │   Keyword   │
│   Search    │         │   Search    │
│  (Cosine)   │         │   (BM25)    │
└──────┬──────┘         └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Hybrid Merge   │  alpha = 0.25
         │  Score Fusion   │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Graph Expansion │  Neo4j traversal
         │                 │
         │ statute:212.05  │──▶ rule:12A-1.001
         │                 │──▶ case:123456
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Legal Reranker  │
         │                 │
         │ Weights:        │
         │ • Statute: 1.0  │
         │ • Rule:    0.9  │
         │ • Case:    0.8  │
         │ • TAA:     0.7  │
         └────────┬────────┘
                  │
                  ▼
         Top-K Results
```

### Reranking Formula

```
final_score = base_score * type_weight * recency_boost * authority_boost

where:
- type_weight: Document type importance (statute > rule > case > taa)
- recency_boost: Newer documents slightly preferred
- authority_boost: Primary authority documents boosted
```

---

## Agent Workflow

### LangGraph StateGraph

```
                    ┌─────────────────┐
                    │  decompose_query │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │retrieve_for_subquery│◄────┐
                    └────────┬────────────┘     │
                             │                  │
                    ┌────────┴────────┐         │
                    ▼                 ▼         │
             ┌───────────┐    ┌─────────────┐   │
             │expand_with│    │   score_    │   │
             │  _graph   │    │ relevance   │   │
             └─────┬─────┘    └──────┬──────┘   │
                   │                 │          │
                   └────────┬────────┘          │
                            ▼                   │
                   ┌────────────────┐           │
                   │filter_irrelevant│          │
                   └───────┬────────┘           │
                           │                    │
                           ▼                    │
                   ┌───────────────┐            │
                   │ more_subqueries│───Yes────▶┘
                   └───────┬───────┘
                           │ No
                           ▼
              ┌──────────────────────┐
              │check_temporal_validity│
              └───────────┬──────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ synthesize_answer │◄────────────┐
                └────────┬─────────┘              │
                         │                        │
                         ▼                        │
                ┌──────────────────┐              │
                │ validate_response │             │
                └────────┬─────────┘              │
                         │                        │
              ┌──────────┼──────────┐             │
              ▼          ▼          ▼             │
           accept     correct   regenerate       │
              │          │          │             │
              │          ▼          │             │
              │    ┌──────────┐     │             │
              │    │ correct_ │     │             │
              │    │ response │     └─────────────┘
              │    └────┬─────┘          (max 2x)
              │         │
              ▼         ▼
           ┌─────────────────┐
           │       END       │
           └─────────────────┘
```

### Agent State

```python
class TaxAgentState(TypedDict):
    # Input
    original_query: str

    # Query Processing
    sub_queries: list[SubQuery]
    current_subquery_index: int
    is_simple_query: bool
    decomposition_reasoning: str

    # Retrieval (Annotated with add operator for accumulation)
    retrieved_chunks: Annotated[list[dict], add]
    current_retrieval_results: list[dict]

    # Filtering
    filtered_chunks: list[dict]
    temporally_valid_chunks: list[dict]
    relevance_scores: dict[str, float]

    # Output
    citations: list[Citation]
    answer: str
    confidence: float

    # Validation
    validation_result: dict
    correction_result: dict
    regeneration_count: int
    validation_passed: bool
    original_answer: str

    # Metadata
    reasoning_steps: Annotated[list[str], add]
    errors: Annotated[list[str], add]
```

---

## Generation Pipeline

### Answer Synthesis

```
Filtered Chunks (top 10-20)
           │
           ▼
┌──────────────────────────┐
│   Format for Context     │
│                          │
│ [1] Fla. Stat. § 212.05  │
│ The state sales tax...   │
│                          │
│ [2] Rule 12A-1.001       │
│ Implementation of...     │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Claude LLM Call        │
│                          │
│ System: Tax Attorney     │
│ Context: Formatted chunks│
│ Query: User question     │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Citation Extraction     │
│                          │
│ Pattern: [Source: ...]   │
│ Extract: § numbers, rules│
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Citation Validation     │
│                          │
│ Verify against context   │
│ Flag unverified cites    │
└───────────┬──────────────┘
            │
            ▼
     GeneratedResponse
```

### Hallucination Detection

```
Response + Context
       │
       ▼
┌──────────────────────────┐
│   LLM Fact Checker       │
│   (Claude Haiku)         │
│                          │
│ For each claim:          │
│ • Is it in sources?      │
│ • Is citation correct?   │
│ • Is scope accurate?     │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Hallucination Types    │
│                          │
│ • unsupported_claim      │
│ • misquoted_text         │
│ • fabricated_citation    │
│ • outdated_info          │
│ • misattributed          │
│ • overgeneralization     │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Routing Decision       │
│                          │
│ accuracy >= 0.8 → Accept │
│ severity < 0.7  → Correct│
│ severity >= 0.7 → Regen  │
└──────────────────────────┘
```

### Self-Correction

```
Detected Hallucinations
         │
         ▼
┌────────────────────┐
│  Simple Correction │  Text replacement
│                    │  for minor issues
└────────┬───────────┘
         │
         ▼ (if complex)
┌────────────────────┐
│   LLM Correction   │  Claude Sonnet
│                    │  rewrites sections
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Confidence Adjust  │
│                    │
│ -10% per simple    │
│ -20% base for LLM  │
│ + severity penalty │
└────────────────────┘
```

---

## Service Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Voyage AI  │  │  Anthropic  │  │   OpenAI    │         │
│  │             │  │             │  │  (optional) │         │
│  │ voyage-law-2│  │   Claude    │  │    GPT-4    │         │
│  │ embeddings  │  │ generation  │  │  eval judge │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Docker Services                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Neo4j    │  │  Weaviate   │  │    Redis    │         │
│  │             │  │             │  │             │         │
│  │ :7474/:7687 │  │ :8080/:50051│  │    :6379    │         │
│  │ graph DB    │  │ vector DB   │  │   cache     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## See Also

- [API Documentation](./api.md) - REST API reference
- [Deployment Guide](./deployment.md) - Deployment instructions
- [Configuration Guide](./configuration.md) - Environment variables
- [Evaluation Guide](./evaluation.md) - Quality metrics
