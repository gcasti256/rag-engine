"""Shared test fixtures."""

from __future__ import annotations

import pytest

from rag_engine.ingestion.chunker import RecursiveChunker
from rag_engine.ingestion.extractors import ExtractedPage


@pytest.fixture
def chunker() -> RecursiveChunker:
    """Create a chunker with small sizes for testing."""
    return RecursiveChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def sample_pages() -> list[ExtractedPage]:
    """Sample extracted pages for testing."""
    return [
        ExtractedPage(
            content="The Federal Reserve raised interest rates by 25 basis points in Q3 2024. "
            "This decision was driven by persistent inflationary pressures in the US economy. "
            "Consumer spending remained robust despite higher borrowing costs.",
            page_number=1,
            section="Monetary Policy",
        ),
        ExtractedPage(
            content="Credit card delinquency rates rose to 2.98% in Q3, up from 2.77% in Q2. "
            "American Express reported a 12% increase in net revenue, driven by strong "
            "consumer spending in travel and entertainment categories.",
            page_number=2,
            section="Financial Services",
        ),
    ]


@pytest.fixture
def sample_markdown() -> bytes:
    """Sample markdown document."""
    return b"""# Financial Report Q3 2024

## Executive Summary

Revenue grew 15% year-over-year to $4.2 billion.
Operating margins improved to 22.3%.

## Revenue Breakdown

- Card services: $2.1B (+18%)
- Merchant services: $1.4B (+12%)
- Other: $0.7B (+8%)

## Risk Assessment

Credit losses remained within expected ranges.
Net charge-off rate: 1.8% (vs 1.5% prior year).
"""


@pytest.fixture
def sample_csv() -> bytes:
    """Sample CSV financial data."""
    return b"""date,category,amount,description
2024-01-15,Revenue,1250000,Card processing fees
2024-01-15,Revenue,890000,Annual card fees
2024-01-15,Expense,340000,Card rewards program
2024-02-15,Revenue,1310000,Card processing fees
2024-02-15,Revenue,920000,Annual card fees
2024-02-15,Expense,355000,Card rewards program
"""
