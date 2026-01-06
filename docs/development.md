# Development Guide

This guide covers setting up a development environment and contributing to the Florida Tax RAG project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Code Style](#code-style)
- [Testing](#testing)
- [Adding Features](#adding-features)
- [Debugging](#debugging)
- [Contributing](#contributing)

---

## Prerequisites

### Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| Python | 3.11+ | `brew install python@3.11` (macOS) |
| Docker | 20.10+ | [Docker Desktop](https://docker.com/products/docker-desktop) |
| Docker Compose | 2.0+ | Included with Docker Desktop |
| Git | 2.0+ | `brew install git` (macOS) |

### Required API Keys

| Service | Purpose | Sign Up |
|---------|---------|---------|
| Voyage AI | Legal embeddings | [voyageai.com](https://www.voyageai.com/) |
| Anthropic | Claude LLM | [console.anthropic.com](https://console.anthropic.com/) |

### Optional API Keys

| Service | Purpose | When Needed |
|---------|---------|-------------|
| OpenAI | GPT-4 evaluation judge | Running full evaluations |
| Weaviate Cloud | Cloud vector store | Production deployment |

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/florida-tax-rag.git
cd florida-tax-rag
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or using Make
make install
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
vim .env
```

**Minimum required variables:**

```env
VOYAGE_API_KEY=your-voyage-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
NEO4J_PASSWORD=your-secure-password
```

### 5. Start Docker Services

```bash
# Start Neo4j, Weaviate, and Redis
make docker-up

# Wait for services to be ready
make docker-wait
```

### 6. Initialize Databases

```bash
# Initialize Neo4j schema and load data
python scripts/init_neo4j.py --verify

# Initialize Weaviate schema
python scripts/init_weaviate.py --verify

# Generate embeddings (takes ~10-20 minutes)
python scripts/generate_embeddings.py --verify

# Load embeddings into Weaviate
python scripts/load_weaviate.py
```

### 7. Validate Setup

```bash
# Run full configuration validation
python scripts/validate_config.py
```

Expected output:

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

---

## Project Structure

```
florida-tax-rag/
├── config/                 # Configuration
│   ├── settings.py         # Pydantic settings
│   ├── prompts/            # LLM prompts
│   ├── development.env     # Dev environment
│   └── production.env.template
├── data/
│   ├── raw/                # Scraped documents
│   ├── processed/          # Chunked documents
│   └── evaluation/         # Golden dataset
├── docs/                   # Documentation
├── scripts/                # CLI scripts
│   ├── init_neo4j.py       # Database init
│   ├── generate_embeddings.py
│   ├── run_evaluation.py
│   └── validate_config.py
├── src/
│   ├── agent/              # LangGraph agent
│   │   ├── graph.py        # StateGraph definition
│   │   ├── nodes.py        # Agent nodes
│   │   └── state.py        # Agent state
│   ├── api/                # FastAPI application
│   │   ├── main.py         # App entry point
│   │   ├── routes.py       # Endpoints
│   │   └── models.py       # Request/response models
│   ├── evaluation/         # Evaluation framework
│   │   ├── metrics.py      # Citation metrics
│   │   ├── llm_judge.py    # GPT-4 judge
│   │   └── runner.py       # Evaluation runner
│   ├── generation/         # Answer generation
│   │   ├── synthesizer.py  # Answer synthesis
│   │   └── validator.py    # Hallucination detection
│   ├── graph/              # Neo4j knowledge graph
│   │   ├── client.py       # Database client
│   │   ├── schema.py       # Schema definition
│   │   └── queries.py      # Cypher queries
│   ├── ingestion/          # Document processing
│   │   ├── chunker.py      # Text chunking
│   │   └── citation_extractor.py
│   ├── observability/      # Logging and metrics
│   ├── retrieval/          # Hybrid retrieval
│   │   ├── hybrid.py       # HybridRetriever
│   │   └── decomposition.py
│   ├── scrapers/           # Data collection
│   └── vector/             # Weaviate client
├── tests/                  # Test suite
├── docker-compose.yml      # Service definitions
├── Makefile                # Common commands
└── pyproject.toml          # Project metadata
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/agent/` | LangGraph agent orchestrating query processing |
| `src/retrieval/` | Hybrid vector + keyword search with reranking |
| `src/generation/` | Answer synthesis with hallucination detection |
| `src/evaluation/` | Evaluation framework with LLM judge |
| `src/graph/` | Neo4j knowledge graph operations |
| `src/vector/` | Weaviate vector store operations |

---

## Running Locally

### Development Server

```bash
# Start with hot reload
make dev

# Or directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000

### Production Mode

```bash
# Start without hot reload
make serve

# Or with Gunicorn (4 workers)
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Services

```bash
# Start services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down

# Reset (DESTRUCTIVE - wipes all data)
make docker-reset
```

### Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| API | http://localhost:8000 | REST API |
| Neo4j Browser | http://localhost:7474 | Graph visualization |
| Weaviate | http://localhost:8080 | Vector store |

---

## Code Style

### Linting with Ruff

```bash
# Check for issues
make lint

# Auto-fix issues
make format
```

### Ruff Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

| Rule Set | Description |
|----------|-------------|
| E | pycodestyle errors |
| F | Pyflakes (unused imports, etc.) |
| I | isort (import sorting) |
| N | pep8-naming |
| W | pycodestyle warnings |
| UP | pyupgrade (Python 3.11+ syntax) |

### Type Checking with mypy

```bash
mypy src/
```

Configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
strict = true
```

### Code Style Guidelines

1. **Type hints** - All functions should have type annotations
2. **Docstrings** - Use Google-style docstrings
3. **Line length** - Max 100 characters
4. **Imports** - Sorted with isort rules
5. **Async** - Prefer async functions for I/O operations

```python
async def retrieve_documents(
    query: str,
    top_k: int = 20,
    alpha: float = 0.25,
) -> list[RetrievalResult]:
    """Retrieve documents using hybrid search.

    Args:
        query: The search query.
        top_k: Maximum results to return.
        alpha: Balance between vector (1.0) and keyword (0.0).

    Returns:
        List of retrieval results with scores.
    """
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Or directly with pytest
pytest tests/ -v

# Run specific test file
pytest tests/test_retrieval.py -v

# Run specific test
pytest tests/test_retrieval.py::test_hybrid_search -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Test Categories

| File | Tests |
|------|-------|
| `test_api.py` | API endpoints, request/response |
| `test_retrieval.py` | Hybrid search, reranking |
| `test_generation.py` | Answer synthesis, validation |
| `test_evaluation.py` | Metrics, LLM judge |
| `test_agent_graph.py` | LangGraph workflow |
| `test_chunking.py` | Document chunking |
| `test_graph_*.py` | Neo4j operations |
| `test_vector_*.py` | Weaviate operations |

### Writing Tests

```python
import pytest
from src.retrieval.hybrid import HybridRetriever

@pytest.fixture
def retriever():
    """Create a retriever instance for testing."""
    return HybridRetriever()

@pytest.mark.asyncio
async def test_hybrid_search(retriever):
    """Test hybrid search returns results."""
    results = await retriever.retrieve(
        query="What is the sales tax rate?",
        top_k=10,
        alpha=0.25,
    )

    assert len(results) > 0
    assert all(r.score >= 0 for r in results)
```

### Test Markers

```python
@pytest.mark.asyncio      # Async test
@pytest.mark.slow         # Long-running test
@pytest.mark.integration  # Requires external services
```

---

## Adding Features

### Adding a New Scraper

1. Create scraper in `src/scrapers/`:

```python
# src/scrapers/new_source.py
from .base import BaseScraper

class NewSourceScraper(BaseScraper):
    """Scraper for [New Source]."""

    async def scrape(self) -> list[Document]:
        """Scrape documents from source."""
        ...
```

2. Register in `src/scrapers/run.py`

3. Add tests in `tests/test_new_source_scraper.py`

### Adding a New Retrieval Method

1. Implement in `src/retrieval/`:

```python
# src/retrieval/new_method.py
async def new_retrieval_method(
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    ...
```

2. Integrate into `HybridRetriever`

3. Add evaluation in `scripts/analyze_retrieval.py`

### Adding a New API Endpoint

1. Add route in `src/api/routes.py`:

```python
@router.get("/api/v1/new-endpoint")
async def new_endpoint(
    param: str = Query(..., description="Parameter description"),
) -> ResponseModel:
    """Endpoint description."""
    ...
```

2. Add request/response models in `src/api/models.py`

3. Add tests in `tests/test_api.py`

### Modifying the Agent

1. Add/modify nodes in `src/agent/nodes.py`

2. Update state in `src/agent/state.py`

3. Update graph in `src/agent/graph.py`

4. Add tests in `tests/test_agent_*.py`

---

## Debugging

### Logging Configuration

Set log level in `.env`:

```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "query_executed",
    query=query,
    results_count=len(results),
    latency_ms=elapsed,
)
```

### Debug Endpoints

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Get metrics
curl http://localhost:8000/api/v1/metrics
```

### Common Debug Scenarios

**Service Connection Failed:**

```bash
# Check Docker services
docker-compose ps

# Check service logs
docker-compose logs neo4j
docker-compose logs weaviate
```

**Retrieval Returns No Results:**

```bash
# Debug specific query
python scripts/analyze_retrieval.py --debug "your query here"

# Check Weaviate has data
python scripts/verify_vector_store.py
```

**Slow Response Times:**

```python
# Enable detailed timing in agent
from src.agent.graph import create_tax_agent_graph

graph = create_tax_agent_graph(debug=True)
```

### Using the Validation Script

```bash
# Full validation with connectivity tests
python scripts/validate_config.py

# Quick settings-only validation
python scripts/validate_config.py --quick
```

---

## Contributing

### Branch Naming

```
feature/add-new-scraper
fix/retrieval-empty-results
docs/update-api-reference
refactor/simplify-chunking
```

### Commit Messages

```
Add TAA document scraper

- Implement TAAScaper class
- Add TAA-specific parsing logic
- Update run.py to include new scraper
```

### Pull Request Process

1. **Create feature branch** from `main`
2. **Make changes** with tests
3. **Run checks:**
   ```bash
   make lint
   make test
   ```
4. **Push and create PR**
5. **Address review feedback**
6. **Squash and merge**

### PR Template

```markdown
## Summary
Brief description of changes.

## Changes
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] README updated (if needed)
- [ ] Docstrings added
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Type hints are complete
- [ ] Tests cover new functionality
- [ ] No sensitive data (API keys, etc.)
- [ ] Documentation updated

---

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make test` | Run test suite |
| `make lint` | Check code style |
| `make format` | Auto-fix code style |
| `make dev` | Start development server |
| `make serve` | Start production server |
| `make docker-up` | Start Docker services |
| `make docker-down` | Stop Docker services |
| `make docker-logs` | View service logs |
| `make docker-reset` | Reset all data |
| `make docker-wait` | Wait for services |
| `make generate-embeddings` | Generate Voyage embeddings |
| `make init-weaviate` | Initialize Weaviate schema |
| `make load-weaviate` | Load data into Weaviate |
| `make verify-weaviate` | Verify Weaviate data |
| `make eval` | Run full evaluation |
| `make eval-quick` | Run quick evaluation |
| `make eval-no-judge` | Run evaluation without judge |
| `make clean` | Remove generated files |

---

## See Also

- [Architecture](./architecture.md) - System design
- [Configuration](./configuration.md) - Environment variables
- [Deployment](./deployment.md) - Production setup
- [Evaluation](./evaluation.md) - Testing quality
- [API Reference](./api.md) - Endpoint documentation
