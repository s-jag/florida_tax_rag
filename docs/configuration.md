# Configuration Guide

This document describes how to configure the Florida Tax RAG system for different environments.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in your API keys:
   ```bash
   VOYAGE_API_KEY=your_voyage_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   NEO4J_PASSWORD=your_neo4j_password
   ```

3. Validate your configuration:
   ```bash
   python scripts/validate_config.py
   ```

## Environment Variables

### Required API Keys

| Variable | Description | How to Obtain |
|----------|-------------|---------------|
| `VOYAGE_API_KEY` | Voyage AI API key for legal embeddings | [voyageai.com](https://www.voyageai.com/) |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude LLM | [console.anthropic.com](https://console.anthropic.com/) |
| `NEO4J_PASSWORD` | Neo4j database password | Set during Neo4j setup |

### Optional API Keys

| Variable | Description | When Needed |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 evaluation judge | Running evaluations |
| `WEAVIATE_API_KEY` | Weaviate API key | Weaviate Cloud deployments |

### Environment Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment: `development`, `staging`, `production` |
| `DEBUG` | `true` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### Database Configuration

#### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | (required) | Neo4j password |
| `NEO4J_CONNECTION_TIMEOUT` | `30` | Connection timeout (seconds) |
| `NEO4J_MAX_CONNECTION_POOL_SIZE` | `50` | Max connection pool size |

#### Weaviate

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate server URL |
| `WEAVIATE_TIMEOUT` | `30` | Request timeout (seconds) |
| `WEAVIATE_API_KEY` | (optional) | API key for cloud deployments |

#### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (preferred) |
| `REDIS_HOST` | `localhost` | Redis host (legacy) |
| `REDIS_PORT` | `6379` | Redis port (legacy) |
| `REDIS_DB` | `0` | Redis database number (legacy) |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `20` | Number of documents to retrieve |
| `HYBRID_ALPHA` | `0.25` | Hybrid search alpha (0.0=keyword, 1.0=vector) |
| `EXPAND_GRAPH` | `true` | Enable graph expansion |
| `MAX_GRAPH_HOPS` | `2` | Maximum graph traversal depth |

The `HYBRID_ALPHA` value of 0.25 was determined through retrieval analysis to be optimal for legal document search. See `RETRIEVAL_ANALYSIS.md` for details.

### Generation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for generation |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (0.0-2.0) |
| `MAX_TOKENS` | `4096` | Maximum response tokens |
| `LLM_TIMEOUT` | `60` | LLM request timeout (seconds) |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `voyage-law-2` | Voyage embedding model |
| `EMBEDDING_BATCH_SIZE` | `128` | Batch size for embeddings |
| `EMBEDDING_CACHE_TTL` | `86400` | Cache TTL (seconds, default 24h) |

### Rate Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit per client |
| `VOYAGE_REQUESTS_PER_MINUTE` | `300` | Voyage AI rate limit |
| `ANTHROPIC_REQUESTS_PER_MINUTE` | `60` | Anthropic rate limit |

## Environment-Specific Configuration

### Development

Use `config/development.env` as a reference for development settings:

```bash
# Inherit from development.env
source config/development.env
```

Key differences from production:
- `DEBUG=true`
- `LOG_LEVEL=DEBUG`
- Higher rate limits (`RATE_LIMIT_PER_MINUTE=100`)
- Local service URLs

### Staging

Use `config/staging.env` as a reference:
- `DEBUG=false`
- `LOG_LEVEL=INFO`
- Standard rate limits

### Production

Use `config/production.env.template` as a starting point:

```bash
cp config/production.env.template .env
# Edit .env with production values
```

**Important production considerations:**

1. **Never commit secrets** - Use environment variables or secrets management
2. **Set `ENV=production`** - Enables fail-fast behavior for critical services
3. **Use `LOG_LEVEL=WARNING`** - Reduces log volume
4. **Secure service connections** - Use TLS for Neo4j (`neo4j+s://`) and Weaviate (`https://`)

## Prompt Customization

LLM prompts are centralized in `config/prompts/`:

```
config/prompts/
├── __init__.py      # Exports all prompts
├── retrieval.py     # Query decomposition, classification, relevance
├── generation.py    # Response generation, hallucination detection
└── evaluation.py    # LLM judge prompts
```

### Available Prompts

| Prompt | File | Description |
|--------|------|-------------|
| `DECOMPOSITION_PROMPT` | `retrieval.py` | Breaks complex queries into sub-queries |
| `CLASSIFICATION_PROMPT` | `retrieval.py` | Classifies query type |
| `RETRIEVAL_SYSTEM_MESSAGE` | `retrieval.py` | System context for decomposition |
| `RELEVANCE_PROMPT` | `retrieval.py` | Scores chunk relevance |
| `GENERATION_SYSTEM_PROMPT` | `generation.py` | System prompt for response generation |
| `CONTEXT_TEMPLATE` | `generation.py` | Template for legal context |
| `HALLUCINATION_DETECTION_PROMPT` | `generation.py` | Detects hallucinations |
| `CORRECTION_PROMPT` | `generation.py` | Corrects hallucinated content |
| `JUDGE_PROMPT` | `evaluation.py` | Evaluates answer quality |

### Customizing Prompts

```python
# Import prompts
from config.prompts import GENERATION_SYSTEM_PROMPT, DECOMPOSITION_PROMPT

# Use in your code
response = client.messages.create(
    system=GENERATION_SYSTEM_PROMPT,
    ...
)
```

## Configuration Validation

### Using the Validation Script

```bash
# Full validation (settings + all services)
python scripts/validate_config.py

# Quick validation (settings only)
python scripts/validate_config.py --quick

# Test specific service
python scripts/validate_config.py --service neo4j
python scripts/validate_config.py --service weaviate
python scripts/validate_config.py --service redis

# Skip API key tests (faster)
python scripts/validate_config.py --skip-api-tests

# Show detailed settings (including masked keys)
python scripts/validate_config.py --verbose
```

### Validation Checks

| Check | Required | Description |
|-------|----------|-------------|
| Settings load | Yes | Pydantic validates all fields |
| Neo4j connection | Yes | Connect and health check |
| Weaviate connection | Yes | Health check endpoint |
| Redis connection | No | Ping test |
| Voyage API key | No | Embedding test |
| Anthropic API key | No | Completion test |

### Startup Validation

In production (`ENV=production`), the API will fail to start if:
- Neo4j is unavailable
- Weaviate is unavailable
- Any required settings are missing

In development/staging, the API will start with warnings but continue.

## Service Connection Requirements

### Neo4j

Requirements:
- Neo4j 4.4+ or Neo4j 5.x
- APOC plugin installed
- GDS plugin (for graph algorithms, optional)

Connection URI formats:
- Local: `bolt://localhost:7687`
- Cloud/TLS: `neo4j+s://your-instance.databases.neo4j.io`

### Weaviate

Requirements:
- Weaviate 1.19+
- `text2vec-transformers` module (or external vectorizer)

Connection URL formats:
- Local: `http://localhost:8080`
- Cloud: `https://your-cluster.weaviate.network`

### Redis

Requirements:
- Redis 6.0+ (recommended)
- Used for embedding cache

Connection URL format:
- `redis://host:port/db`
- `redis://:password@host:port/db` (with auth)

## Troubleshooting

### Settings Won't Load

```bash
# Check for syntax errors in .env
python -c "from config.settings import get_settings; print(get_settings())"
```

### Neo4j Connection Failed

1. Check Neo4j is running: `docker-compose ps`
2. Verify URI and credentials
3. Check network connectivity
4. Review Neo4j logs: `docker-compose logs neo4j`

### Weaviate Connection Failed

1. Check Weaviate is running: `curl http://localhost:8080/v1/.well-known/ready`
2. Verify URL is correct
3. Check for schema issues

### Redis Connection Failed

Redis is optional. If unavailable:
- Embedding cache will be disabled
- System will still function (slower)

### API Key Invalid

```bash
# Test Voyage API key
python scripts/validate_config.py --service voyage

# Test Anthropic API key
python scripts/validate_config.py --service anthropic
```

## Security Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use secrets management** in production (AWS Secrets Manager, HashiCorp Vault, etc.)
3. **Rotate API keys** regularly
4. **Use read-only credentials** where possible
5. **Enable TLS** for all production connections
6. **Audit log access** to sensitive configuration
