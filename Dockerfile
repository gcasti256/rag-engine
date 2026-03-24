FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install
COPY pyproject.toml ./
COPY rag_engine/ ./rag_engine/
RUN pip install --no-cache-dir .

FROM python:3.11-slim AS production

WORKDIR /app

# Copy installed packages from build stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY rag_engine/ ./rag_engine/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "-m", "rag_engine.cli"]
