# RAG Engine

Production-grade Retrieval-Augmented Generation pipeline with pgvector, hybrid search, and source citations. Built for financial data patterns.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Multi-format ingestion** — PDF, Markdown, plain text, and CSV with automatic text extraction and metadata parsing
- **Recursive chunking** — Configurable chunk size and overlap with intelligent boundary detection (paragraphs → sentences → words)
- **Hybrid search** — Combines pgvector cosine similarity with BM25 keyword search using Reciprocal Rank Fusion
- **Source citations** — Every answer includes cited sources with page numbers, sections, and relevance scores
- **Confidence scoring** — Transparent confidence metrics based on retrieval relevance
- **Streaming responses** — Server-sent token streaming for real-time answer generation
- **Multi-tenant** — Namespace isolation for separate document collections
- **Evaluation framework** — LLM-as-judge metrics: answer relevance, faithfulness, context precision
- **Query UI** — React + TypeScript frontend with dark theme

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      FastAPI REST API                     │
│  POST /ingest  ·  POST /query  ·  GET /documents         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │   Ingestion      │    │   Query Pipeline              │ │
│  │   Pipeline       │    │                              │ │
│  │                  │    │  Question                    │ │
│  │  Extract → Chunk │    │    ↓                         │ │
│  │    ↓             │    │  Hybrid Retrieval            │ │
│  │  Embed → Store   │    │  (Vector + BM25 + RRF)      │ │
│  │                  │    │    ↓                         │ │
│  │  Extractors:     │    │  Context Assembly            │ │
│  │  · PDF (PyMuPDF) │    │  (token-aware truncation)    │ │
│  │  · Markdown      │    │    ↓                         │ │
│  │  · Plain text    │    │  LLM Generation              │ │
│  │  · CSV           │    │  (with source citations)     │ │
│  └────────┬─────────┘    └──────────┬───────────────────┘ │
│           │                         │                     │
│  ┌────────▼─────────────────────────▼───────────────────┐ │
│  │              Vector Store (pgvector)                   │ │
│  │  · Cosine similarity search                           │ │
│  │  · BM25 keyword search                                │ │
│  │  · Reciprocal Rank Fusion                             │ │
│  │  · Namespace isolation                                │ │
│  └───────────────────────┬──────────────────────────────┘ │
│                          │                                │
├──────────────────────────▼────────────────────────────────┤
│  PostgreSQL 16 + pgvector  ·  OpenAI Embeddings + GPT     │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Docker Compose (recommended)

```bash
# Clone
git clone https://github.com/gcasti256/rag-engine.git
cd rag-engine

# Configure
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Start
docker compose up -d

# API is now running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Local Development

```bash
# Prerequisites: Python 3.11+, PostgreSQL 16 with pgvector

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your settings

# Initialize database
python -c "import asyncio; from rag_engine.database import init_db; asyncio.run(init_db())"

# Run API
rag-engine
# Or: uvicorn rag_engine.api.app:app --reload

# Frontend (optional)
cd frontend && npm install && npm run dev
```

## API Usage

### Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@quarterly_report.pdf" \
  -F "namespace=financial" \
  -F "title=Q3 2024 Earnings Report"
```

Response:
```json
{
  "document_id": "a1b2c3d4-...",
  "filename": "quarterly_report.pdf",
  "chunk_count": 47,
  "total_tokens": 12830,
  "namespace": "financial"
}
```

### Query Documents

```bash
curl -X POST http://localhost:8000/query \
  -F "question=What was the net revenue in Q3 2024?" \
  -F "search_method=hybrid" \
  -F "namespace=financial"
```

Response:
```json
{
  "answer": "According to the Q3 2024 earnings report, net revenue was $15.4 billion, representing a 12% increase year-over-year [Source 1].",
  "citations": [
    {
      "chunk_id": "...",
      "source": "quarterly_report.pdf",
      "content": "Net revenue of $15.4 billion in Q3 2024...",
      "page_number": 3,
      "section": "Financial Highlights",
      "relevance_score": 0.94
    }
  ],
  "confidence": 0.92,
  "query": "What was the net revenue in Q3 2024?",
  "model": "gpt-4o-mini",
  "total_tokens_used": 847,
  "retrieval_time_ms": 45.2,
  "generation_time_ms": 1230.5
}
```

### Stream Response

```bash
curl -X POST http://localhost:8000/query/stream \
  -F "question=Summarize the key financial metrics" \
  -F "namespace=financial"
```

### List Documents

```bash
curl http://localhost:8000/documents?namespace=financial
```

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Evaluation

Run batch evaluation against your RAG pipeline:

```bash
# Create an eval file (see docs/eval_sample.json)
rag-eval docs/eval_sample.json financial

# Output:
# [1/5] What was American Express net revenue in Q3 2024?
#   Relevance: 0.95 | Faithfulness: 0.90 | Precision: 0.88 | Overall: 0.91
# ...
# AGGREGATE RESULTS
#   Avg Relevance:    0.92
#   Avg Faithfulness:  0.88
#   Avg Precision:     0.85
#   Avg Overall:       0.88
```

Metrics:
- **Answer Relevance** — Does the answer address the question?
- **Faithfulness** — Is every claim grounded in the retrieved context?
- **Context Precision** — Did retrieval find the right chunks?

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_COMPLETION_MODEL` | `gpt-4o-mini` | Completion model |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `DEFAULT_TOP_K` | `5` | Default retrieval count |
| `SIMILARITY_THRESHOLD` | `0.7` | Min cosine similarity |
| `MAX_CONTEXT_TOKENS` | `4096` | Max context window for generation |

## Tech Stack

- **Python 3.11+** — Type-annotated, async-first
- **FastAPI** — High-performance async API framework
- **PostgreSQL 16 + pgvector** — Vector similarity search at scale
- **OpenAI API** — Embeddings (text-embedding-3-small) and completions (GPT-4o-mini)
- **SQLAlchemy 2.0** — Async ORM with pgvector integration
- **BM25 (rank-bm25)** — Keyword search for hybrid retrieval
- **Reciprocal Rank Fusion** — Combines vector and keyword results
- **PyMuPDF** — PDF text extraction
- **tiktoken** — Token counting for chunking
- **structlog** — Structured JSON logging
- **React 19 + TypeScript** — Query interface frontend
- **Tailwind CSS v4** — Utility-first styling
- **Docker Compose** — One-command deployment
- **pytest** — Unit and integration testing
- **GitHub Actions** — CI/CD pipeline

## Project Structure

```
rag-engine/
├── rag_engine/
│   ├── api/              # FastAPI routes and app factory
│   │   ├── app.py        # Application factory with lifespan
│   │   ├── routes.py     # REST API endpoints
│   │   └── schemas.py    # Request/response schemas
│   ├── ingestion/        # Document processing pipeline
│   │   ├── extractors.py # PDF, Markdown, Text, CSV extractors
│   │   ├── chunker.py    # Recursive text chunking
│   │   └── pipeline.py   # End-to-end ingestion orchestration
│   ├── storage/          # Vector storage and search
│   │   ├── embeddings.py # OpenAI embedding generation
│   │   └── vector_store.py # pgvector + BM25 hybrid search
│   ├── query/            # RAG query pipeline
│   │   └── pipeline.py   # Retrieval → augmentation → generation
│   ├── evaluation/       # Quality assessment framework
│   │   ├── metrics.py    # Relevance, faithfulness, precision
│   │   └── cli.py        # Batch evaluation CLI
│   ├── config.py         # Pydantic settings
│   ├── database.py       # SQLAlchemy + pgvector schema
│   ├── models.py         # Pydantic models
│   └── logging.py        # Structured logging
├── frontend/             # React query interface
│   └── src/
│       ├── App.tsx
│       ├── QueryPanel.tsx
│       ├── DocumentsPanel.tsx
│       └── UploadPanel.tsx
├── tests/
│   ├── unit/             # Unit tests (no external deps)
│   └── integration/      # Integration tests (DB + API)
├── docs/
│   └── eval_sample.json  # Sample evaluation dataset
├── docker-compose.yml    # Full stack deployment
├── Dockerfile            # Production container
└── .github/workflows/ci.yml
```

## License

MIT
