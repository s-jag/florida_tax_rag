# Deployment Guide

This guide covers deploying the Florida Tax RAG system in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Database Initialization](#database-initialization)
- [Running the API](#running-the-api)
- [Production Configuration](#production-configuration)
- [Scaling Considerations](#scaling-considerations)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Service orchestration |
| Python | 3.9+ | Application runtime |

### Required API Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Voyage AI | `VOYAGE_API_KEY` | Legal document embeddings |
| Anthropic | `ANTHROPIC_API_KEY` | Claude LLM generation |

### Optional API Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | GPT-4 evaluation judge |
| Weaviate | `WEAVIATE_API_KEY` | Weaviate Cloud (if not self-hosted) |

---

## Docker Compose Deployment

### Service Overview

The system runs three Docker services:

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| Neo4j | `neo4j:5.15-community` | 7474, 7687 | Knowledge graph |
| Weaviate | `weaviate:1.28.2` | 8080, 50051 | Vector store |
| Redis | `redis:7-alpine` | 6379 | Embedding cache |

### Starting Services

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready
make docker-wait

# Check service status
docker-compose ps
```

### Service Health Checks

```bash
# Neo4j
curl http://localhost:7474

# Weaviate
curl http://localhost:8080/v1/.well-known/ready

# Redis
docker exec florida_tax_redis redis-cli ping
```

### Stopping Services

```bash
# Stop services (preserve data)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (DESTRUCTIVE)
docker-compose down -v
```

### Data Persistence

Data is persisted in `./docker-data/`:

```
docker-data/
├── neo4j/
│   ├── data/      # Graph database files
│   ├── logs/      # Neo4j logs
│   └── plugins/   # APOC plugin
├── weaviate/      # Vector store data
└── redis/         # Cache data (AOF persistence)
```

---

## Database Initialization

### Step 1: Initialize Neo4j

```bash
# Create schema and load documents
python scripts/init_neo4j.py --verify

# With verbose output
python scripts/init_neo4j.py --verify --verbose

# Clear and reload (DESTRUCTIVE)
python scripts/init_neo4j.py --clear --verify
```

**Expected Output:**
```
Neo4j Schema:
  Constraints: 2
  Indexes: 6

Loading Data:
  Documents: 1,152
  Chunks: 3,022
  HAS_CHUNK edges: 3,022
  CHILD_OF edges: 1,870
  Citation edges: 1,126
```

### Step 2: Initialize Weaviate

```bash
# Create LegalChunk collection
python scripts/init_weaviate.py --verify

# Delete and recreate (DESTRUCTIVE)
python scripts/init_weaviate.py --delete --verify
```

### Step 3: Generate Embeddings

```bash
# Generate all embeddings (takes ~10-20 minutes)
python scripts/generate_embeddings.py --verify

# Test with sample first
python scripts/generate_embeddings.py --sample 10 --verify

# Resume interrupted run
python scripts/generate_embeddings.py --resume
```

**Progress Output:**
```
Generating embeddings...
  Progress: 3022/3022 chunks
  Batches: 24
  API calls: 24
  Cache hits: 0
  Time: 12m 34s
```

### Step 4: Load Weaviate

```bash
# Load chunks and embeddings
python scripts/load_weaviate.py

# Verify loaded data
python scripts/verify_vector_store.py
```

### All-in-One Initialization

```bash
# Full initialization sequence
make docker-up
make docker-wait
python scripts/init_neo4j.py --verify
python scripts/init_weaviate.py --verify
python scripts/generate_embeddings.py --verify
python scripts/load_weaviate.py
python scripts/verify_vector_store.py
```

---

## Running the API

### Development Mode

```bash
# With hot reload
make dev

# Or directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# With Gunicorn (4 workers)
make serve

# Or directly
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Verify API is Running

```bash
# Root endpoint
curl http://localhost:8000/

# Health check
curl http://localhost:8000/api/v1/health

# Test query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the sales tax rate?"}'
```

---

## Production Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp config/production.env.template .env
# Edit .env with production values
```

**Critical Production Settings:**

```env
# Environment
ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# API Keys (use secrets management)
VOYAGE_API_KEY=<from-secrets-manager>
ANTHROPIC_API_KEY=<from-secrets-manager>
NEO4J_PASSWORD=<strong-password>

# Service URLs (update for production)
NEO4J_URI=bolt://prod-neo4j:7687
WEAVIATE_URL=http://prod-weaviate:8080
REDIS_URL=redis://prod-redis:6379/0

# Rate limits
RATE_LIMIT_PER_MINUTE=60

# Retrieval (optimized)
HYBRID_ALPHA=0.25
RETRIEVAL_TOP_K=20
```

### Validate Configuration

```bash
# Full validation
python scripts/validate_config.py

# Quick validation (settings only)
python scripts/validate_config.py --quick
```

### Production Startup Behavior

In production (`ENV=production`), the API will:
- **Fail fast** if Neo4j or Weaviate are unavailable
- Use JSON-formatted logs for aggregation
- Set `LOG_LEVEL=WARNING` by default

---

## Scaling Considerations

### Connection Pooling

**Neo4j:**
```env
NEO4J_MAX_CONNECTION_POOL_SIZE=100  # Increase for higher concurrency
NEO4J_CONNECTION_TIMEOUT=30
```

**Weaviate:**
- Uses persistent HTTP connections
- Configure `WEAVIATE_TIMEOUT` for long queries

### API Workers

Scale Gunicorn workers based on CPU cores:

```bash
# Formula: (2 * CPU cores) + 1
gunicorn src.api.main:app --workers 9  # For 4-core machine
```

### Embedding Cache

Redis caches Voyage AI embeddings to reduce API costs:

```env
EMBEDDING_CACHE_TTL=86400  # 24 hours (default)
```

Cache hit rate should be high after initial embedding generation.

### Rate Limiting

Adjust per-client rate limits:

```env
RATE_LIMIT_PER_MINUTE=60   # Default
RATE_LIMIT_PER_MINUTE=120  # Higher for trusted clients
```

### Memory Requirements

| Service | Minimum | Recommended |
|---------|---------|-------------|
| Neo4j | 2GB | 4GB |
| Weaviate | 2GB | 4GB |
| Redis | 256MB | 512MB |
| API | 1GB | 2GB |

---

## Monitoring

### Health Endpoint

```bash
# Check service health
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "services": [
    {"name": "neo4j", "healthy": true, "latency_ms": 12.5},
    {"name": "weaviate", "healthy": true, "latency_ms": 8.3}
  ]
}
```

### Metrics Endpoint

```bash
# Get API metrics
curl http://localhost:8000/api/v1/metrics
```

```json
{
  "total_queries": 1500,
  "successful_queries": 1425,
  "success_rate_percent": 95.0,
  "latency_ms": {"avg": 3250.5, "min": 1200, "max": 8500},
  "errors_by_type": {"TIMEOUT": 50, "RETRIEVAL_ERROR": 15},
  "uptime_seconds": 86400
}
```

### Structured Logging

In production, logs are JSON-formatted for aggregation:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "event": "query_completed",
  "request_id": "abc-123",
  "latency_ms": 3250,
  "confidence": 0.92
}
```

### Log Aggregation

Recommended tools:
- **Elasticsearch + Kibana** - Log storage and visualization
- **Datadog** - APM and log management
- **CloudWatch** - AWS native logging

### Docker Logs

```bash
# View all service logs
docker-compose logs -f

# View specific service
docker-compose logs -f neo4j
docker-compose logs -f weaviate
docker-compose logs -f redis
```

---

## Backup and Recovery

### Neo4j Backup

```bash
# Stop Neo4j first
docker-compose stop neo4j

# Backup data directory
tar -czvf neo4j-backup-$(date +%Y%m%d).tar.gz docker-data/neo4j/data

# Restart Neo4j
docker-compose start neo4j
```

### Weaviate Backup

```bash
# Using Weaviate backup API
curl -X POST http://localhost:8080/v1/backups/filesystem \
  -H "Content-Type: application/json" \
  -d '{"id": "backup-2024-01-15", "include": ["LegalChunk"]}'
```

### Full System Backup

```bash
# Stop all services
docker-compose down

# Backup all data
tar -czvf florida-tax-rag-backup-$(date +%Y%m%d).tar.gz \
  docker-data/ \
  data/processed/ \
  .env

# Restart services
docker-compose up -d
```

### Recovery

```bash
# Stop services
docker-compose down

# Restore backup
tar -xzvf florida-tax-rag-backup-20240115.tar.gz

# Restart services
docker-compose up -d

# Verify
python scripts/validate_config.py
```

### Re-initialization

If you need to rebuild from scratch:

```bash
# Remove all data
docker-compose down -v
rm -rf docker-data/

# Reinitialize
docker-compose up -d
make docker-wait
python scripts/init_neo4j.py --verify
python scripts/init_weaviate.py --verify
python scripts/generate_embeddings.py
python scripts/load_weaviate.py
```

---

## Troubleshooting

See [Troubleshooting Guide](./troubleshooting.md) for common issues and solutions.

---

## See Also

- [Configuration Guide](./configuration.md) - Environment variables
- [Architecture](./architecture.md) - System overview
- [API Reference](./api.md) - REST API documentation
