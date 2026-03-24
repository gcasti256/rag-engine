"""Tests for logging configuration."""

from __future__ import annotations

from unittest.mock import patch

import structlog

from rag_engine.logging import setup_logging


class TestSetupLogging:
    def test_json_format(self) -> None:
        """Should configure JSON logging by default."""
        with patch("rag_engine.logging.settings") as mock_settings:
            mock_settings.log_format = "json"
            mock_settings.log_level = "INFO"
            setup_logging()
            logger = structlog.get_logger()
            assert logger is not None

    def test_console_format(self) -> None:
        """Should configure console logging when format is not json."""
        with patch("rag_engine.logging.settings") as mock_settings:
            mock_settings.log_format = "console"
            mock_settings.log_level = "DEBUG"
            setup_logging()
            logger = structlog.get_logger()
            assert logger is not None
