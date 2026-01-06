# Changelog

All notable changes to the Florida Tax RAG system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-06

### Added

#### Data Collection
- Florida Statutes scraper for tax chapters (192-220)
- Florida Administrative Code scraper (Chapter 12A)
- Technical Assistance Advisement scraper with PDF extraction
- CourtListener case law scraper for Florida Supreme Court decisions
- Data quality audit script (`scripts/audit_raw_data.py`)

#### Knowledge Base
- Neo4j knowledge graph with document nodes, chunk nodes, and citation edges
- Weaviate vector store with hybrid search (vector + BM25)
- Voyage AI voyage-law-2 embeddings (1,024 dimensions)
- Redis caching for embeddings and query results
- Citation graph with 1,126 legal reference relationships

#### Document Processing
- Unified LegalDocument model with Pydantic validation
- Hierarchical chunking (parent/child with 500-token target)
- Citation extraction for statutes, rules, cases, and TAAs
- Token counting with tiktoken

#### Retrieval System
- Hybrid retrieval combining vector similarity and keyword matching
- Query decomposition using Claude LLM for complex queries
- Multi-query parallel retrieval with result merging
- Graph expansion via Neo4j for related documents
- Legal-specific reranking (primary authority boost)
- Optimal alpha tuning (0.25 for keyword-heavy hybrid)

#### Agent Workflow (LangGraph)
- 9-node StateGraph workflow:
  - decompose_query: Break complex queries into sub-queries
  - retrieve_for_subquery: Hybrid search execution
  - expand_with_graph: Neo4j related document discovery
  - score_relevance: LLM-based relevance scoring (parallel)
  - filter_irrelevant: Quality threshold filtering
  - check_temporal_validity: Effective date verification
  - synthesize_answer: Claude-powered response generation
  - validate_response: Hallucination detection
  - correct_response: Self-correction for issues

#### Answer Generation
- TaxLawGenerator with Florida Tax Attorney persona
- Citation extraction and validation against sources
- Confidence scoring based on source quality
- Hallucination detection (6 types: unsupported claims, misquotes, fabricated citations, etc.)
- Self-correction with text replacement and LLM rewriting

#### REST API
- FastAPI application with 7 endpoints
- Query endpoint (`POST /api/v1/query`)
- Streaming endpoint with SSE (`POST /api/v1/query/stream`)
- Source lookup (`GET /api/v1/sources/{chunk_id}`)
- Statute with rules (`GET /api/v1/statute/{section}`)
- Related documents (`GET /api/v1/graph/{doc_id}/related`)
- Metrics endpoint (`GET /api/v1/metrics`)
- Health check (`GET /api/v1/health`)

#### Observability
- Structured logging with structlog (JSON in production)
- Request tracing with X-Request-ID propagation
- Rate limiting middleware (sliding window)
- Metrics collection (queries, latency, errors)
- Pipeline profiler with per-stage timing
- Load testing script (`scripts/test_load.py`)

#### Evaluation Framework
- Golden dataset with 20 Florida tax law questions
- Citation precision, recall, and F1 metrics
- LLM judge using GPT-4 for answer quality
- Retrieval analysis with MRR, NDCG, Recall@k
- Evaluation runner with markdown report generation
- Baseline metrics established

#### Configuration
- Pydantic Settings with environment-specific configs
- Centralized prompts in `config/prompts/`
- Configuration validation script
- Production fail-fast startup behavior

#### Performance Optimization
- Query result caching with 1-hour TTL
- Parallel sub-query retrieval with asyncio.gather
- Neo4j index optimization (6 indexes)
- Comprehensive benchmark script
- Performance documentation

#### Documentation
- README.md with full system documentation
- Architecture guide (docs/architecture.md)
- API reference (docs/api.md)
- Configuration guide (docs/configuration.md)
- Deployment guide (docs/deployment.md)
- Development guide (docs/development.md)
- Evaluation guide (docs/evaluation.md)
- Performance guide (docs/performance.md)
- Troubleshooting guide (docs/troubleshooting.md)

### Data Statistics

| Category | Count |
|----------|-------|
| Florida Statutes | 742 sections |
| Administrative Rules | 101 rules |
| Supreme Court Cases | 308 cases |
| TAAs | 1 document |
| Total Documents | 1,152 |
| Total Chunks | 3,022 |
| Vector Embeddings | 3,022 |
| Citation Edges | 1,126 |

### Known Issues

- TAA coverage is limited to 1 document (historical TAAs require URL enumeration)
- Case law opinion text is truncated due to CourtListener API limitations
- Complex queries have high latency (30-70 seconds) due to multiple LLM calls
- Citation recall is approximately 50% (room for improvement in retrieval)

## [Unreleased]

### Planned
- Streaming token generation for lower time-to-first-byte
- Expanded TAA collection from historical archives
- Query classification for fast-path simple queries
- Read replicas for Neo4j scaling
- Response compression for large results
