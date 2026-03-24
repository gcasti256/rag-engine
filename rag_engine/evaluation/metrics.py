"""RAG evaluation metrics: relevance, faithfulness, and context precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from rag_engine.config import settings

if TYPE_CHECKING:
    from rag_engine.models import QueryResult


@dataclass
class EvalResult:
    """Result from a single evaluation metric."""

    metric: str
    score: float
    reasoning: str


@dataclass
class FullEvalResult:
    """Complete evaluation of a RAG response."""

    question: str
    answer: str
    relevance: EvalResult
    faithfulness: EvalResult
    context_precision: EvalResult
    overall_score: float


async def answer_relevance(
    question: str,
    answer: str,
    client: AsyncOpenAI | None = None,
) -> EvalResult:
    """Score how relevant the answer is to the question (0-1).

    Uses LLM-as-judge to assess whether the answer addresses the question.
    """
    client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    prompt = f"""Rate how well the following answer addresses the question on a scale of 0 to 1.
0 = completely irrelevant, 1 = perfectly addresses the question.

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    response = await client.chat.completions.create(
        model=settings.openai_completion_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    import json

    content = response.choices[0].message.content or "{}"
    result = json.loads(content)

    return EvalResult(
        metric="answer_relevance",
        score=min(max(float(result.get("score", 0)), 0), 1),
        reasoning=result.get("reasoning", ""),
    )


async def faithfulness(
    answer: str,
    context: str,
    client: AsyncOpenAI | None = None,
) -> EvalResult:
    """Score how grounded the answer is in the provided context (0-1).

    Checks whether all claims in the answer can be traced to the context.
    """
    client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    prompt = f"""Rate how faithful the following answer is to the provided context \
on a scale of 0 to 1.
0 = answer contains claims not supported by context, 1 = every claim is grounded in context.

Context: {context[:3000]}

Answer: {answer}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    response = await client.chat.completions.create(
        model=settings.openai_completion_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    import json

    content = response.choices[0].message.content or "{}"
    result = json.loads(content)

    return EvalResult(
        metric="faithfulness",
        score=min(max(float(result.get("score", 0)), 0), 1),
        reasoning=result.get("reasoning", ""),
    )


async def context_precision(
    question: str,
    contexts: list[str],
    client: AsyncOpenAI | None = None,
) -> EvalResult:
    """Score whether the retrieved contexts are relevant to the question (0-1).

    Checks if the top-retrieved chunks actually contain useful information.
    """
    client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    context_list = "\n".join(
        f"Context {i + 1}: {ctx[:500]}" for i, ctx in enumerate(contexts[:5])
    )

    prompt = f"""Rate how relevant the retrieved contexts are to the question on a scale of 0 to 1.
0 = none of the contexts are relevant, 1 = all contexts contain information needed to answer.

Question: {question}

Retrieved Contexts:
{context_list}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    response = await client.chat.completions.create(
        model=settings.openai_completion_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    import json

    content = response.choices[0].message.content or "{}"
    result = json.loads(content)

    return EvalResult(
        metric="context_precision",
        score=min(max(float(result.get("score", 0)), 0), 1),
        reasoning=result.get("reasoning", ""),
    )


async def evaluate_response(
    query_result: QueryResult,
) -> FullEvalResult:
    """Run all evaluation metrics on a query result."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    context_texts = [c.content for c in query_result.citations]
    full_context = "\n\n".join(context_texts)

    rel = await answer_relevance(query_result.query, query_result.answer, client)
    faith = await faithfulness(query_result.answer, full_context, client)
    prec = await context_precision(query_result.query, context_texts, client)

    overall = (rel.score + faith.score + prec.score) / 3

    return FullEvalResult(
        question=query_result.query,
        answer=query_result.answer,
        relevance=rel,
        faithfulness=faith,
        context_precision=prec,
        overall_score=round(overall, 3),
    )
