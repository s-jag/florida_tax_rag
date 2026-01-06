# API Reference

Complete REST API reference for the Florida Tax RAG system.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Request Headers](#request-headers)
- [Endpoints](#endpoints)
  - [Query Endpoints](#query-endpoints)
  - [Source Endpoints](#source-endpoints)
  - [Graph Endpoints](#graph-endpoints)
  - [System Endpoints](#system-endpoints)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Overview

The Florida Tax RAG API provides access to an intelligent retrieval-augmented generation system for Florida tax law questions. The API combines vector search, knowledge graphs, and LLM-based generation to provide accurate, cited answers.

**OpenAPI Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Base URL

```
Development: http://localhost:8000
Production:  https://your-domain.com
```

All API endpoints are prefixed with `/api/v1/`.

---

## Authentication

Currently, the API does not require authentication. Rate limiting is applied per IP address.

> **Note:** Authentication will be added in a future release.

---

## Rate Limiting

| Limit | Value |
|-------|-------|
| Requests per minute | 60 (configurable via `RATE_LIMIT_PER_MINUTE`) |
| Window | 60 seconds (sliding) |

**Excluded Paths:** `/`, `/docs`, `/redoc`, `/api/v1/health`, `/api/v1/metrics`

**Rate Limit Response:**
```json
{
  "request_id": "abc-123",
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded: 60 requests per minute",
  "details": {
    "limit": 60,
    "window_seconds": 60,
    "retry_after_seconds": 60
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Headers:**
- `Retry-After: 60` (seconds until rate limit resets)

---

## Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes (POST) | Must be `application/json` |
| `X-Request-ID` | No | Custom request ID for tracing |

**Response Headers:**
| Header | Description |
|--------|-------------|
| `X-Request-ID` | Request identifier (echoed or generated) |
| `X-Query-ID` | Short query ID for tracing |

---

## Endpoints

### Query Endpoints

#### POST /api/v1/query

Execute a tax law query through the RAG agent.

**Request Body:**

```json
{
  "query": "What is the Florida sales tax rate?",
  "options": {
    "doc_types": ["statute", "rule"],
    "tax_year": 2024,
    "expand_graph": true,
    "include_reasoning": false,
    "timeout_seconds": 60
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Tax law question (3-2000 chars) |
| `options.doc_types` | array | No | Filter by document types: `statute`, `rule`, `case`, `taa` |
| `options.tax_year` | integer | No | Specific tax year (1990-2030) |
| `options.expand_graph` | boolean | No | Expand results via graph traversal (default: true) |
| `options.include_reasoning` | boolean | No | Include reasoning steps (default: false) |
| `options.timeout_seconds` | integer | No | Request timeout 5-300 (default: 60) |

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "The Florida state sales tax rate is 6%. This is established by Florida Statutes § 212.05, which imposes a tax of 6% on retail sales of tangible personal property...\n\n[Source: § 212.05(1)]",
  "citations": [
    {
      "doc_id": "statute:212.05",
      "citation": "Fla. Stat. § 212.05",
      "doc_type": "statute",
      "text_snippet": "There is levied on each taxable transaction..."
    }
  ],
  "sources": [
    {
      "chunk_id": "chunk:statute:212.05:0",
      "doc_id": "statute:212.05",
      "doc_type": "statute",
      "citation": "Fla. Stat. § 212.05",
      "text": "212.05 Sales, storage, use tax.—(1) There is levied on each taxable transaction...",
      "effective_date": "2023-07-01T00:00:00",
      "relevance_score": 0.95
    }
  ],
  "confidence": 0.92,
  "warnings": [],
  "reasoning_steps": [],
  "validation_passed": true,
  "processing_time_ms": 3250,
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

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `answer` | string | Generated answer with inline citations |
| `citations` | array | Citations referenced in the answer |
| `sources` | array | Source chunks used (max 10, truncated to 500 chars) |
| `confidence` | float | Confidence score (0-1) |
| `warnings` | array | Warnings about the response |
| `reasoning_steps` | array | Agent reasoning steps (if requested) |
| `validation_passed` | boolean | Whether response passed validation |
| `processing_time_ms` | integer | Total processing time |
| `stage_timings` | object | Per-stage timing breakdown in milliseconds |

**Error Responses:**
- `408 Request Timeout` - Query exceeded timeout
- `500 Internal Server Error` - Processing error

---

#### POST /api/v1/query/stream

Stream query results via Server-Sent Events (SSE).

**Request Body:** Same as `/api/v1/query`

**Response:** SSE stream with the following event types:

```
event: status
data: {"message": "Entering node: decompose_query"}

event: reasoning
data: {"step_number": 1, "node": "decompose_query", "description": "Query decomposed into 2 sub-queries"}

event: chunk
data: {"chunks": [{"chunk_id": "...", "citation": "§ 212.05", "relevance_score": 0.95}]}

event: answer
data: {"answer": "The Florida sales tax rate is 6%..."}

event: complete
data: {"request_id": "...", "confidence": 0.92, "processing_time_ms": 3250}

event: error
data: {"error": "TIMEOUT", "message": "Query timed out after 60 seconds"}
```

**Event Types:**

| Event | Description | Data Fields |
|-------|-------------|-------------|
| `status` | Processing status update | `message` |
| `reasoning` | Reasoning step completed | `step_number`, `node`, `description` |
| `chunk` | Source chunks retrieved | `chunks[]` (max 5 per event) |
| `answer` | Final answer generated | `answer` |
| `complete` | Processing complete | `request_id`, `confidence`, `processing_time_ms` |
| `error` | Error occurred | `error`, `message` |

**Response Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Request-ID: {request_id}
```

---

### Source Endpoints

#### GET /api/v1/sources/{chunk_id}

Get full details for a specific chunk.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk_id` | string | Chunk identifier (e.g., `chunk:statute:212.05:0`) |

**Response:**

```json
{
  "chunk_id": "chunk:statute:212.05:0",
  "doc_id": "statute:212.05",
  "doc_type": "statute",
  "level": "parent",
  "text": "212.05 Sales, storage, use tax.—(1) There is levied on each taxable transaction...",
  "text_with_ancestry": "Chapter 212 > Section 212.05\n\n212.05 Sales, storage, use tax...",
  "ancestry": "Chapter 212 > Section 212.05",
  "citation": "Fla. Stat. § 212.05",
  "effective_date": "2023-07-01T00:00:00",
  "token_count": 450,
  "parent_chunk_id": null,
  "child_chunk_ids": ["chunk:statute:212.05:1", "chunk:statute:212.05:2"],
  "related_doc_ids": ["rule:12A-1.001", "rule:12A-1.005", "case:123456"]
}
```

**Error Responses:**
- `404 Not Found` - Chunk not found

---

### Graph Endpoints

#### GET /api/v1/statute/{section}

Get a statute with its implementing rules and interpreting documents.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `section` | string | Statute section number (e.g., `212.05`) |

**Response:**

```json
{
  "statute": {
    "doc_id": "statute:212.05",
    "title": "Sales, storage, use tax",
    "text": "...",
    "effective_date": "2023-07-01"
  },
  "implementing_rules": [
    {
      "doc_id": "rule:12A-1.001",
      "title": "Specific Application of Tax",
      "citation": "Fla. Admin. Code R. 12A-1.001"
    }
  ],
  "interpreting_cases": [
    {
      "doc_id": "case:123456",
      "title": "Department of Revenue v. ABC Corp",
      "citation": "123 So. 3d 456"
    }
  ],
  "interpreting_taas": [
    {
      "doc_id": "taa:25A-001",
      "title": "TAA 25A-001",
      "citation": "Fla. DOR TAA 25A-001"
    }
  ]
}
```

**Error Responses:**
- `404 Not Found` - Statute not found

---

#### GET /api/v1/graph/{doc_id}/related

Get documents related via citations.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | string | Document identifier (e.g., `statute:212.05`) |

**Response:**

```json
{
  "doc_id": "statute:212.05",
  "citing_documents": [
    {
      "doc_id": "case:123456",
      "doc_type": "case",
      "citation": "123 So. 3d 456"
    }
  ],
  "cited_documents": [
    {
      "doc_id": "statute:212.02",
      "doc_type": "statute",
      "citation": "Fla. Stat. § 212.02"
    }
  ],
  "interpretation_chain": {
    "implementing_rules": [...],
    "interpreting_cases": [...],
    "interpreting_taas": [...]
  }
}
```

> **Note:** `interpretation_chain` is only included for statute documents.

---

### System Endpoints

#### GET /api/v1/health

Check health of all backend services.

**Response:**

```json
{
  "status": "healthy",
  "services": [
    {
      "name": "neo4j",
      "healthy": true,
      "latency_ms": 12.5,
      "error": null
    },
    {
      "name": "weaviate",
      "healthy": true,
      "latency_ms": 8.3,
      "error": null
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

| Status | Description |
|--------|-------------|
| `healthy` | All services healthy |
| `degraded` | Some services healthy |
| `unhealthy` | No services healthy |

---

#### GET /api/v1/metrics

Get API metrics and statistics.

**Response:**

```json
{
  "total_queries": 1500,
  "successful_queries": 1425,
  "failed_queries": 75,
  "success_rate_percent": 95.0,
  "latency_ms": {
    "avg": 3250.5,
    "min": 1200.0,
    "max": 8500.0
  },
  "errors_by_type": {
    "TIMEOUT": 50,
    "RETRIEVAL_ERROR": 15,
    "GENERATION_ERROR": 10
  },
  "uptime_seconds": 86400.0,
  "started_at": "2024-01-14T10:30:00.000Z"
}
```

---

#### GET /

Root endpoint with API info.

**Response:**

```json
{
  "name": "Florida Tax RAG API",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

---

## Error Handling

### Error Response Format

All errors return a consistent JSON structure:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {},
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `TIMEOUT` | 408 | Request timed out |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `GENERATION_ERROR` | 502 | LLM generation failed |
| `RETRIEVAL_ERROR` | 503 | Retrieval service error |
| `SERVICE_UNAVAILABLE` | 503 | Backend service unavailable |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Examples

### curl Examples

**Basic Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Florida sales tax rate?"}'
```

**Query with Options:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Are groceries exempt from sales tax?",
    "options": {
      "doc_types": ["statute", "rule"],
      "include_reasoning": true
    }
  }'
```

**Streaming Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "What is the sales tax rate?"}'
```

**Get Chunk Details:**
```bash
curl http://localhost:8000/api/v1/sources/chunk:statute:212.05:0
```

**Get Statute with Rules:**
```bash
curl http://localhost:8000/api/v1/statute/212.05
```

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

---

### Python Example

```python
import httpx
import asyncio

async def query_tax_law():
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "query": "What is the Florida sales tax rate?",
                "options": {
                    "doc_types": ["statute", "rule"],
                    "include_reasoning": True
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Citations: {len(result['citations'])}")

        for citation in result['citations']:
            print(f"  - {citation['citation']}")

asyncio.run(query_tax_law())
```

**Streaming Example:**

```python
import httpx
import json

def stream_query():
    with httpx.stream(
        "POST",
        "http://localhost:8000/api/v1/query/stream",
        json={"query": "What is the sales tax rate?"},
        timeout=120.0
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data:
                    print(f"Answer: {data['answer'][:100]}...")
                elif "message" in data:
                    print(f"Status: {data['message']}")

stream_query()
```

---

### JavaScript/TypeScript Example

```typescript
interface QueryResponse {
  request_id: string;
  answer: string;
  citations: Array<{
    doc_id: string;
    citation: string;
    doc_type: string;
  }>;
  confidence: number;
  processing_time_ms: number;
}

async function queryTaxLaw(query: string): Promise<QueryResponse> {
  const response = await fetch("http://localhost:8000/api/v1/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      options: {
        doc_types: ["statute", "rule"],
        timeout_seconds: 60,
      },
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`${error.error}: ${error.message}`);
  }

  return response.json();
}

// Usage
queryTaxLaw("What is the Florida sales tax rate?")
  .then((result) => {
    console.log(`Answer: ${result.answer}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`Processing time: ${result.processing_time_ms}ms`);
  })
  .catch(console.error);
```

**Streaming Example:**

```typescript
async function streamQuery(query: string): Promise<void> {
  const response = await fetch("http://localhost:8000/api/v1/query/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ query }),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) throw new Error("No response body");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        if (data.answer) {
          console.log("Answer:", data.answer);
        } else if (data.message) {
          console.log("Status:", data.message);
        }
      }
    }
  }
}

streamQuery("What is the sales tax rate?");
```

---

## Performance Features

### Query Result Caching

Identical queries are cached in Redis with a 1-hour TTL. Cache hits return in <10ms.

**Cache Behavior:**
- Cache key is based on normalized query text + options (doc_types, tax_year, include_reasoning)
- Low confidence responses (<0.3) are not cached
- Failed responses are not cached

### Stage Timings

API responses include a `stage_timings` object showing milliseconds spent in each pipeline stage:

| Stage | Description |
|-------|-------------|
| `decompose` | Query analysis and decomposition |
| `retrieve` | Vector + keyword search |
| `expand_graph` | Neo4j graph traversal |
| `score_relevance` | LLM-based relevance scoring |
| `filter` | Relevance filtering |
| `temporal_check` | Temporal validity check |
| `synthesize` | Answer generation |
| `validate` | Hallucination detection |
| `correct` | Self-correction (if needed) |

See [Performance Guide](./performance.md) for optimization details.

---

## See Also

- [Architecture](./architecture.md) - System architecture overview
- [Deployment](./deployment.md) - Deployment instructions
- [Configuration](./configuration.md) - Environment variables
- [Performance](./performance.md) - Optimization and profiling
- [Troubleshooting](./troubleshooting.md) - Common issues
