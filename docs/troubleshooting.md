# Troubleshooting Guide

This guide covers common issues and their solutions when working with the Florida Tax RAG system.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Configuration Issues](#configuration-issues)
- [Connection Failures](#connection-failures)
- [Retrieval Issues](#retrieval-issues)
- [Generation Issues](#generation-issues)
- [Performance Issues](#performance-issues)
- [Docker Issues](#docker-issues)
- [Error Reference](#error-reference)
- [FAQ](#faq)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

### Validate Configuration

The first step for any issue is to run the configuration validator:

```bash
python scripts/validate_config.py
```

**Expected output:**

```
Configuration Validation
========================
  Settings loaded: OK
  Neo4j connection: OK
  Weaviate connection: OK
  Redis connection: OK
  Voyage API key: OK (valid)
  Anthropic API key: OK (valid)

All checks passed!
```

### Check Service Health

```bash
# API health check
curl http://localhost:8000/api/v1/health

# Docker services status
docker-compose ps

# Service logs
docker-compose logs -f
```

### Verify Data

```bash
# Check Weaviate data
python scripts/verify_vector_store.py

# Check Neo4j data
python scripts/init_neo4j.py --verify
```

---

## Configuration Issues

### Missing Environment Variables

**Symptom:**

```
pydantic_settings.ValidationError: 1 validation error for Settings
VOYAGE_API_KEY
  field required
```

**Solution:**

1. Copy environment template:
   ```bash
   cp .env.example .env
   ```

2. Fill in required variables:
   ```env
   VOYAGE_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   NEO4J_PASSWORD=your-password
   ```

### Invalid API Key

**Symptom:**

```
API key validation failed: 401 Unauthorized
```

**Solution:**

1. Verify your API key is correct
2. Check for extra whitespace or quotes:
   ```env
   # Wrong
   VOYAGE_API_KEY="voyage-abc123"

   # Correct
   VOYAGE_API_KEY=voyage-abc123
   ```

3. Test the key directly:
   ```bash
   python scripts/validate_config.py --check-apis
   ```

### Wrong Environment

**Symptom:**

API fails immediately in production mode.

**Solution:**

Check your `ENV` setting:

```env
# Development (lenient startup)
ENV=development

# Production (fail-fast startup)
ENV=production
```

In production, the API fails fast if Neo4j or Weaviate are unavailable.

---

## Connection Failures

### Neo4j Connection Failed

**Symptom:**

```
Neo4j connection failed: ServiceUnavailable: Connection refused
```

**Solutions:**

1. **Check Neo4j is running:**
   ```bash
   docker-compose ps | grep neo4j
   ```

2. **Check Neo4j logs:**
   ```bash
   docker-compose logs neo4j
   ```

3. **Wait for startup:**
   ```bash
   make docker-wait
   ```

4. **Verify credentials:**
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   ```

5. **Restart Neo4j:**
   ```bash
   docker-compose restart neo4j
   ```

### Weaviate Connection Failed

**Symptom:**

```
Weaviate connection failed: Connection refused
```

**Solutions:**

1. **Check Weaviate is running:**
   ```bash
   curl http://localhost:8080/v1/.well-known/ready
   ```

2. **Check logs:**
   ```bash
   docker-compose logs weaviate
   ```

3. **Verify URL:**
   ```env
   WEAVIATE_URL=http://localhost:8080
   ```

4. **Restart Weaviate:**
   ```bash
   docker-compose restart weaviate
   ```

### Redis Connection Failed

**Symptom:**

```
Redis connection failed: Connection refused
```

**Solutions:**

1. **Check Redis is running:**
   ```bash
   docker exec florida_tax_redis redis-cli ping
   ```

2. **Verify URL:**
   ```env
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Redis is optional** - the system works without it (no embedding cache)

---

## Retrieval Issues

### No Results Returned

**Symptom:**

Query returns empty results.

**Solutions:**

1. **Verify Weaviate has data:**
   ```bash
   python scripts/verify_vector_store.py
   ```

2. **Check collection exists:**
   ```python
   import weaviate
   client = weaviate.connect_to_local()
   print(client.collections.exists("LegalChunk"))
   ```

3. **Reload data if needed:**
   ```bash
   python scripts/init_weaviate.py --delete --verify
   python scripts/load_weaviate.py
   ```

4. **Debug specific query:**
   ```bash
   python scripts/analyze_retrieval.py --debug "your query"
   ```

### Wrong Results Returned

**Symptom:**

Retrieval returns irrelevant documents.

**Solutions:**

1. **Check alpha parameter:**
   ```env
   HYBRID_ALPHA=0.25  # Keyword-heavy works best for legal queries
   ```

2. **Analyze retrieval quality:**
   ```bash
   python scripts/analyze_retrieval.py --question eval_001
   ```

3. **Try different alpha values:**
   ```bash
   python scripts/analyze_retrieval.py --tune-alpha
   ```

### Missing Citations

**Symptom:**

Expected documents not in results.

**Solutions:**

1. **Increase top_k:**
   ```env
   RETRIEVAL_TOP_K=30  # Default is 20
   ```

2. **Enable graph expansion:**
   ```python
   results = await retriever.retrieve(
       query=query,
       use_graph_expansion=True,
   )
   ```

3. **Check document exists:**
   ```python
   from src.graph.client import GraphClient

   client = GraphClient()
   doc = client.get_document_by_id("statute:212.05")
   print(doc)
   ```

---

## Generation Issues

### Hallucination Detected

**Symptom:**

```
Hallucination detected: Answer contains fabricated citations
```

**Solutions:**

1. This is expected behavior - the system catches hallucinations
2. The self-correction loop should fix it automatically
3. If persistent, check retrieval quality (relevant docs may not be found)

### Incomplete Answers

**Symptom:**

Answer doesn't address all aspects of the question.

**Solutions:**

1. **Check retrieval found relevant docs:**
   ```bash
   python scripts/analyze_retrieval.py --debug "your query"
   ```

2. **Increase generation context:**
   ```env
   GENERATION_MAX_CONTEXT_CHUNKS=15  # Default is 10
   ```

3. **Check token limits:**
   ```env
   GENERATION_MAX_TOKENS=4096
   ```

### API Rate Limits

**Symptom:**

```
anthropic.RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Add retry logic** (already built-in with tenacity)

2. **Reduce concurrent requests:**
   ```env
   RATE_LIMIT_PER_MINUTE=30
   ```

3. **Wait and retry** - typically resolves in 60 seconds

---

## Performance Issues

### Slow Response Times

**Symptom:**

Queries take >10 seconds.

**Solutions:**

1. **Check Redis cache is working:**
   ```bash
   docker exec florida_tax_redis redis-cli info | grep keyspace
   ```

2. **Profile the query:**
   ```python
   import time
   start = time.time()
   # ... run query ...
   print(f"Total: {time.time() - start:.2f}s")
   ```

3. **Reduce retrieval scope:**
   ```env
   RETRIEVAL_TOP_K=15  # Reduce from 20
   ```

4. **Check service resources:**
   ```bash
   docker stats
   ```

### High Memory Usage

**Symptom:**

Services consuming excessive memory.

**Solutions:**

1. **Check container limits:**
   ```bash
   docker stats --no-stream
   ```

2. **Restart services:**
   ```bash
   docker-compose restart
   ```

3. **Adjust Neo4j heap:**
   ```yaml
   # docker-compose.yml
   environment:
     NEO4J_dbms_memory_heap_initial__size: 1G
     NEO4J_dbms_memory_heap_max__size: 2G
   ```

### Embedding Generation Slow

**Symptom:**

`generate_embeddings.py` takes very long.

**Solutions:**

1. **Use batch processing** (already default)

2. **Resume interrupted runs:**
   ```bash
   python scripts/generate_embeddings.py --resume
   ```

3. **Test with sample first:**
   ```bash
   python scripts/generate_embeddings.py --sample 10 --verify
   ```

---

## Docker Issues

### Container Won't Start

**Symptom:**

```
ERROR: Container exited with code 1
```

**Solutions:**

1. **Check logs:**
   ```bash
   docker-compose logs [service_name]
   ```

2. **Check port conflicts:**
   ```bash
   lsof -i :7474  # Neo4j
   lsof -i :8080  # Weaviate
   lsof -i :6379  # Redis
   ```

3. **Reset and restart:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Volume Permission Issues

**Symptom:**

```
Permission denied: ./docker-data/neo4j
```

**Solutions:**

1. **Fix permissions:**
   ```bash
   sudo chown -R $(whoami) docker-data/
   ```

2. **Reset volumes:**
   ```bash
   docker-compose down -v
   rm -rf docker-data/
   docker-compose up -d
   ```

### Out of Disk Space

**Symptom:**

```
No space left on device
```

**Solutions:**

1. **Clean Docker:**
   ```bash
   docker system prune -a
   docker volume prune
   ```

2. **Check disk usage:**
   ```bash
   du -sh docker-data/*
   ```

---

## Error Reference

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `CONFIG_ERROR` | Invalid configuration | Check `.env` file |
| `NEO4J_UNAVAILABLE` | Can't connect to Neo4j | Start Docker, check credentials |
| `WEAVIATE_UNAVAILABLE` | Can't connect to Weaviate | Start Docker, check URL |
| `EMBEDDING_ERROR` | Voyage API failed | Check API key, rate limits |
| `GENERATION_ERROR` | Claude API failed | Check API key, rate limits |
| `RETRIEVAL_ERROR` | Search failed | Check Weaviate has data |
| `TIMEOUT` | Operation timed out | Increase timeout, check resources |
| `VALIDATION_ERROR` | Invalid request | Check request format |

### Common Error Messages

**"No LegalChunk collection found"**

```bash
python scripts/init_weaviate.py --verify
```

**"Neo4j authentication failed"**

```bash
# Check password matches docker-compose.yml
grep NEO4J_AUTH docker-compose.yml
grep NEO4J_PASSWORD .env
```

**"Embedding dimension mismatch"**

```bash
# Re-generate embeddings with correct model
python scripts/init_weaviate.py --delete
python scripts/generate_embeddings.py --verify
python scripts/load_weaviate.py
```

---

## FAQ

### How do I reset everything?

```bash
# Stop services and remove data
docker-compose down -v
rm -rf docker-data/

# Restart and reinitialize
docker-compose up -d
make docker-wait
python scripts/init_neo4j.py --verify
python scripts/init_weaviate.py --verify
python scripts/generate_embeddings.py
python scripts/load_weaviate.py
```

### How do I re-embed all documents?

```bash
# Clear embedding cache
docker exec florida_tax_redis redis-cli FLUSHDB

# Re-generate embeddings
python scripts/generate_embeddings.py --verify
python scripts/load_weaviate.py
```

### Why is retrieval slow first time?

The first query warms up caches:
- Redis embedding cache (subsequent queries are faster)
- Weaviate index (loads into memory)
- Connection pools (established on first use)

### Why are some citations missing?

1. **Document not in corpus** - Check if the statute/rule was scraped
2. **Alpha too high** - Try lowering to 0.25 (more keyword matching)
3. **Top_k too low** - Increase `RETRIEVAL_TOP_K`

### How do I add new documents?

1. Add to `data/raw/` directory
2. Re-run ingestion: `python -m src.ingestion.run`
3. Re-generate embeddings: `python scripts/generate_embeddings.py`
4. Reload Weaviate: `python scripts/load_weaviate.py`

### Can I use a different embedding model?

Yes, configure in settings:

```env
EMBEDDING_MODEL=voyage-law-2    # Default (recommended for legal)
EMBEDDING_MODEL=voyage-large-2  # Alternative
```

Note: Changing models requires re-generating all embeddings.

### How do I run without Docker?

You can run Neo4j, Weaviate, and Redis natively, but Docker is recommended.

```env
# Point to your native instances
NEO4J_URI=bolt://localhost:7687
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379/0
```

---

## Getting Help

### Logs to Collect

When reporting issues, include:

1. **Configuration validation output:**
   ```bash
   python scripts/validate_config.py 2>&1
   ```

2. **Service logs:**
   ```bash
   docker-compose logs --tail=100
   ```

3. **Error traceback** (full stack trace)

4. **Environment info:**
   ```bash
   python --version
   docker --version
   docker-compose --version
   ```

### Reporting Issues

1. Check existing issues in GitHub
2. Include logs and configuration (redact API keys)
3. Describe steps to reproduce
4. Include expected vs actual behavior

### Resources

- [Configuration Guide](./configuration.md)
- [Deployment Guide](./deployment.md)
- [Development Guide](./development.md)
- [GitHub Issues](https://github.com/your-org/florida-tax-rag/issues)
