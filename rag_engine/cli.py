"""CLI entry point for the RAG engine server."""

from __future__ import annotations

import uvicorn

from rag_engine.config import settings


def main() -> None:
    """Start the RAG engine API server."""
    uvicorn.run(
        "rag_engine.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
