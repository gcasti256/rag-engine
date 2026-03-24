"""Tests for configuration."""

from __future__ import annotations

from rag_engine.config import Settings


class TestSettings:
    def test_defaults(self) -> None:
        """Settings should have reasonable defaults."""
        s = Settings(openai_api_key="test-key")
        assert s.chunk_size == 512
        assert s.chunk_overlap == 64
        assert s.default_top_k == 5
        assert s.similarity_threshold == 0.7
        assert s.embedding_dimensions == 1536

    def test_custom_values(self) -> None:
        """Settings should accept custom values."""
        s = Settings(
            openai_api_key="test-key",
            chunk_size=256,
            default_top_k=10,
        )
        assert s.chunk_size == 256
        assert s.default_top_k == 10
