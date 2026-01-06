# Performance Optimization Guide

This document describes the performance optimizations implemented in the Florida Tax RAG system and provides guidance for monitoring and tuning performance.

## Architecture Overview

The query pipeline consists of the following stages:

```
Query → Decompose → Retrieve → Expand Graph → Score Relevance → Filter → Temporal Check → Synthesize → Validate → [Correct] → Response
```

Each stage can be profiled individually using the built-in pipeline profiler.

## Optimizations Implemented

### 1. Pipeline Profiler

**Location:** `src/observability/profiler.py`

The pipeline profiler provides per-stage timing visibility for every request.

**Usage:**
```python
from src.observability.profiler import profile_request

with profile_request(request_id) as profiler:
    with profiler.stage("decompose"):
        result = await decomposer.decompose(query)
    with profiler.stage("retrieve"):
        results = await retriever.retrieve(query)

    summary = profiler.get_summary()
    # {"request_id": "...", "total_ms": 1234, "stages": {"decompose": 150, "retrieve": 800, ...}}
```

**API Response:**
Responses now include `stage_timings` showing milliseconds spent in each stage:
```json
{
  "answer": "...",
  "processing_time_ms": 3500,
  "stage_timings": {
    "decompose": 250,
    "retrieve": 800,
    "expand_graph": 150,
    "score_relevance": 1200,
    "synthesize": 900,
    "validate": 200
  }
}
```

### 2. Query Result Caching

**Location:** `src/api/cache.py`

Redis-backed cache for query responses with 1-hour TTL.

**Benefits:**
- Repeated identical queries return in <10ms
- Reduces LLM API costs
- Improves consistency for common queries

**Cache Key Generation:**
- Normalized query text (lowercase, stripped)
- Relevant options (tax_year, include_reasoning)
- SHA256 hash for key

**Cache Skip Conditions:**
- Low confidence responses (<0.3)
- Failed responses
- Responses without answers

**Monitoring:**
```python
from src.api.cache import get_query_cache

cache = get_query_cache()
stats = cache.get_stats()
# {"hits": 150, "misses": 50, "hit_rate": 0.75}
```

### 3. Parallel Sub-Query Retrieval

**Location:** `src/agent/nodes.py:retrieve_for_subquery`

Sub-queries from decomposition now execute in parallel using `asyncio.gather()`.

**Before (sequential):**
```
Query decomposes to 3 sub-queries
  → Retrieve sub-query 1: 800ms
  → Retrieve sub-query 2: 750ms
  → Retrieve sub-query 3: 820ms
Total: 2370ms
```

**After (parallel):**
```
Query decomposes to 3 sub-queries
  → Retrieve all in parallel: 850ms
Total: 850ms
```

**Deduplication:**
Results are deduplicated by `chunk_id` to avoid processing the same chunk multiple times.

### 4. Neo4j Query Optimization

**Location:** `src/graph/schema.py`

Indexes added for common query patterns:

| Index | Property | Purpose |
|-------|----------|---------|
| `doc_type_idx` | `Document.doc_type` | Filter by statute/rule/case/taa |
| `doc_section_idx` | `Document.section` | Lookup by section number |
| `doc_chapter_idx` | `Document.chapter` | Filter by chapter |
| `doc_citation_idx` | `Document.full_citation` | Citation text lookup |
| `chunk_doc_id_idx` | `Chunk.doc_id` | Join chunks to documents |
| `chunk_citation_idx` | `Chunk.citation` | Citation text lookup |

**Analyze Queries:**
```bash
python scripts/analyze_neo4j.py --explain --update-indexes
```

### 5. Existing Optimizations

The following were already present in the codebase:

| Feature | Location | Description |
|---------|----------|-------------|
| Embedding Cache | `src/vector/embeddings.py` | Redis cache with 30-day TTL |
| Neo4j Connection Pool | `src/graph/client.py` | 50-connection pool |
| Client Singletons | `src/api/dependencies.py` | `@lru_cache` pattern |
| Streaming Endpoint | `src/api/routes.py` | SSE for real-time updates |
| Parallel LLM Scoring | `src/agent/nodes.py` | Semaphore(5) for relevance scoring |

## Benchmarking

### Quick Load Test
```bash
python scripts/test_load.py --num-requests 50 --concurrency 5
```

### Comprehensive Benchmark
```bash
python scripts/benchmark.py --num-queries 50 --output report.md

# Compare before/after optimization
python scripts/benchmark.py --save-json before.json
# ... apply changes ...
python scripts/benchmark.py --compare-before before.json --output comparison.md
```

### Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| P50 Latency | <2000ms | Median query time |
| P95 Latency | <5000ms | 95th percentile |
| P99 Latency | <10000ms | 99th percentile |
| Success Rate | >95% | Queries returning valid answers |
| Cache Hit Rate | >30% | For repeated queries |

## Stage-by-Stage Timing Breakdown

Typical timing distribution for a complex query:

| Stage | Typical Time | Notes |
|-------|-------------|-------|
| decompose | 200-400ms | LLM call to analyze query |
| retrieve | 500-1000ms | Vector search + reranking |
| expand_graph | 50-200ms | Neo4j graph traversal |
| score_relevance | 500-1500ms | Parallel LLM scoring (5 concurrent) |
| filter | <10ms | CPU-bound filtering |
| temporal_check | <10ms | Date validation |
| synthesize | 800-2000ms | Main LLM generation |
| validate | 200-500ms | LLM validation call |
| correct | 0-500ms | Only if corrections needed |

**Bottlenecks:**
1. LLM API calls (decompose, score, synthesize, validate)
2. Vector search with reranking
3. Graph expansion for complex documents

## Tuning Recommendations

### For Lower Latency

1. **Reduce retrieval count**: Adjust `top_k` in retrieval (default: 20)
2. **Skip validation for high-confidence**: Add confidence threshold
3. **Use cache aggressively**: Increase TTL for stable queries
4. **Limit graph expansion**: Cap related documents at 5

### For Higher Quality

1. **Increase retrieval count**: More chunks for better coverage
2. **Lower relevance threshold**: Include more marginal results
3. **Enable full validation**: Always run validation step
4. **Expand graph depth**: Include 2-hop relationships

### For Cost Reduction

1. **Use cache**: 30%+ hit rate significantly reduces LLM costs
2. **Batch similar queries**: Process in groups
3. **Skip redundant validation**: For high-confidence responses
4. **Use smaller model for scoring**: Haiku vs Sonnet for relevance

## Monitoring

### Logs

Structured logs include timing information:
```json
{"event": "request_profile", "request_id": "abc123", "total_ms": 3500, "stages": {...}}
{"event": "stage_completed", "stage": "retrieve", "duration_ms": 850}
```

### Metrics Endpoint

```bash
curl http://localhost:8000/api/v1/metrics
```

Returns:
```json
{
  "total_queries": 1000,
  "success_rate_percent": 97.5,
  "latency_ms": {
    "avg": 2500,
    "p50": 2000,
    "p95": 4500,
    "p99": 8000
  }
}
```

## Future Improvements

1. **Streaming Generation**: Stream tokens as they're generated to reduce time-to-first-byte
2. **Precomputed Embeddings**: Cache common query embeddings
3. **Read Replicas**: Add Neo4j read replicas for scaling
4. **Query Classification**: Fast-path simple queries without full pipeline
5. **Response Compression**: Gzip for large responses

---

## See Also

- [Architecture](./architecture.md) - System architecture overview
- [API Reference](./api.md) - REST API documentation
- [Deployment](./deployment.md) - Deployment and scaling
- [Troubleshooting](./troubleshooting.md) - Common issues and debugging
