"""Tests for evaluation metrics (mocked OpenAI)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from rag_engine.evaluation.metrics import (
    answer_relevance,
    context_precision,
    evaluate_response,
    faithfulness,
)
from rag_engine.models import Citation, QueryResult


def _mock_openai_json(score: float, reasoning: str = "test") -> AsyncMock:
    """Create a mock OpenAI client that returns a JSON score."""
    client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = (
        f'{{"score": {score}, "reasoning": "{reasoning}"}}'
    )
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


class TestAnswerRelevance:
    async def test_returns_score(self) -> None:
        """Should return relevance score from LLM judge."""
        client = _mock_openai_json(0.95, "highly relevant")
        result = await answer_relevance("What is X?", "X is Y.", client=client)
        assert result.metric == "answer_relevance"
        assert result.score == 0.95

    async def test_clamps_score(self) -> None:
        """Should clamp score to [0, 1]."""
        client = _mock_openai_json(1.5)
        result = await answer_relevance("Q?", "A.", client=client)
        assert result.score == 1.0

    async def test_handles_negative_score(self) -> None:
        """Should clamp negative scores to 0."""
        client = _mock_openai_json(-0.5)
        result = await answer_relevance("Q?", "A.", client=client)
        assert result.score == 0.0


class TestFaithfulness:
    async def test_returns_score(self) -> None:
        """Should return faithfulness score."""
        client = _mock_openai_json(0.88, "well grounded")
        result = await faithfulness("Answer text", "Context text", client=client)
        assert result.metric == "faithfulness"
        assert result.score == 0.88


class TestContextPrecision:
    async def test_returns_score(self) -> None:
        """Should return context precision score."""
        client = _mock_openai_json(0.92)
        result = await context_precision(
            "What is X?", ["Context 1", "Context 2"], client=client
        )
        assert result.metric == "context_precision"
        assert result.score == 0.92


class TestEvaluateResponse:
    async def test_full_evaluation(self) -> None:
        """Should run all three metrics and compute overall score."""
        query_result = QueryResult(
            answer="Revenue was $4.2B",
            citations=[
                Citation(
                    chunk_id="c1",
                    source="report.pdf",
                    content="Revenue: $4.2B",
                    relevance_score=0.9,
                )
            ],
            confidence=0.85,
            query="What was revenue?",
            model="gpt-4o-mini",
        )

        mock_client = _mock_openai_json(0.9)
        with patch(
            "rag_engine.evaluation.metrics.AsyncOpenAI", return_value=mock_client
        ):
            result = await evaluate_response(query_result)

        assert result.question == "What was revenue?"
        assert result.relevance.score == 0.9
        assert result.faithfulness.score == 0.9
        assert result.context_precision.score == 0.9
        assert result.overall_score == 0.9
