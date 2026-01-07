FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    langchain>=0.3.0 \
    langgraph>=0.2.0 \
    weaviate-client>=4.0.0 \
    neo4j>=5.0.0 \
    voyageai>=0.3.0 \
    anthropic>=0.40.0 \
    openai>=1.0.0 \
    beautifulsoup4>=4.12.0 \
    httpx>=0.27.0 \
    lxml>=5.0.0 \
    pypdf>=4.0.0 \
    pdfplumber>=0.11.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    python-dotenv>=1.0.0 \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.32.0" \
    tenacity>=9.0.0 \
    structlog>=24.0.0 \
    redis>=5.0.0

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/processed/ ./data/processed/

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the API with 2 workers (suitable for t3.medium)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
