.PHONY: install test lint scrape ingest serve dev clean \
       docker-up docker-down docker-logs docker-reset docker-wait \
       generate-embeddings load-weaviate verify-weaviate init-weaviate \
       eval eval-quick eval-no-judge

# Install dependencies
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run linting and type checking
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

# Format code
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Run scrapers to collect Florida tax law data
scrape:
	python -m src.scrapers.run

# Ingest and process scraped data into vector store and knowledge graph
ingest:
	python -m src.ingestion.run

# Start the FastAPI server (development with hot reload)
dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start the FastAPI server (production mode)
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

# =============================================================================
# Docker Commands
# =============================================================================

# Start all Docker services (Neo4j, Weaviate, Redis)
docker-up:
	docker-compose up -d
	@echo "Services starting... Run 'make docker-wait' to wait for readiness"

# Stop all Docker services
docker-down:
	docker-compose down

# Tail logs from all services
docker-logs:
	docker-compose logs -f

# Wipe all Docker volumes and restart fresh
docker-reset:
	docker-compose down -v
	rm -rf ./docker-data
	docker-compose up -d
	@echo "Services reset and starting fresh..."

# Wait for all services to be ready
docker-wait:
	python scripts/wait_for_services.py

# =============================================================================
# Vector Store Commands
# =============================================================================

# Generate embeddings for all chunks using Voyage AI
generate-embeddings:
	python scripts/generate_embeddings.py --verify

# Initialize Weaviate schema
init-weaviate:
	python scripts/init_weaviate.py

# Load chunks and embeddings into Weaviate
load-weaviate:
	python scripts/load_weaviate.py

# Verify Weaviate vector store
verify-weaviate:
	python scripts/verify_vector_store.py

# =============================================================================
# Evaluation Commands
# =============================================================================

# Run full evaluation (all 20 questions with GPT-4 judge)
eval:
	python scripts/run_evaluation.py

# Run quick evaluation (5 questions, no judge - for testing)
eval-quick:
	python scripts/run_evaluation.py --limit 5 --no-judge

# Run evaluation without LLM judge (metrics only)
eval-no-judge:
	python scripts/run_evaluation.py --no-judge
