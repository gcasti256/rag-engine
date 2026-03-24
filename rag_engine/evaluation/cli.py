"""CLI for batch evaluation of RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rag_engine.evaluation.metrics import evaluate_response
from rag_engine.query.pipeline import QueryPipeline
from rag_engine.storage import VectorStore


async def run_eval(eval_file: str, namespace: str = "default") -> None:
    """Run batch evaluation from a JSON file.

    Expected format:
    [
        {"question": "What is X?", "expected_answer": "X is..."},
        ...
    ]
    """
    path = Path(eval_file)
    if not path.exists():
        print(f"Error: {eval_file} not found")
        sys.exit(1)

    with open(path) as f:
        eval_data = json.load(f)

    vector_store = VectorStore()
    pipeline = QueryPipeline(vector_store=vector_store)

    results: list[dict[str, Any]] = []
    total_relevance = 0.0
    total_faithfulness = 0.0
    total_precision = 0.0

    print(f"\nRunning evaluation on {len(eval_data)} questions...\n")
    print("-" * 70)

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"[{i}/{len(eval_data)}] {question}")

        # Run query
        query_result = await pipeline.query(
            question=question,
            namespace=namespace,
        )

        # Evaluate
        eval_result = await evaluate_response(query_result)

        total_relevance += eval_result.relevance.score
        total_faithfulness += eval_result.faithfulness.score
        total_precision += eval_result.context_precision.score

        results.append({
            "question": question,
            "answer": query_result.answer[:200],
            "relevance": eval_result.relevance.score,
            "faithfulness": eval_result.faithfulness.score,
            "context_precision": eval_result.context_precision.score,
            "overall": eval_result.overall_score,
            "citations": len(query_result.citations),
            "confidence": query_result.confidence,
        })

        print(f"  Relevance: {eval_result.relevance.score:.2f} | "
              f"Faithfulness: {eval_result.faithfulness.score:.2f} | "
              f"Precision: {eval_result.context_precision.score:.2f} | "
              f"Overall: {eval_result.overall_score:.2f}")

    n = len(eval_data)
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Questions evaluated: {n}")
    print(f"  Avg Relevance:      {total_relevance / n:.3f}")
    print(f"  Avg Faithfulness:   {total_faithfulness / n:.3f}")
    print(f"  Avg Precision:      {total_precision / n:.3f}")
    avg_overall = (total_relevance + total_faithfulness + total_precision) / (3 * n)
    print(f"  Avg Overall:        {avg_overall:.3f}")

    # Write results to file
    output_path = path.with_suffix(".results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to {output_path}")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: rag-eval <eval_file.json> [namespace]")
        print("\nEval file format:")
        print('[{"question": "What is X?", "expected_answer": "X is..."}]')
        sys.exit(1)

    eval_file = sys.argv[1]
    namespace = sys.argv[2] if len(sys.argv) > 2 else "default"
    asyncio.run(run_eval(eval_file, namespace))


if __name__ == "__main__":
    main()
