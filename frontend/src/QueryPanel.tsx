import { useState, useRef } from "react";
import type { QueryResult } from "./api";
import { queryDocuments } from "./api";

export function QueryPanel() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchMethod, setSearchMethod] = useState("hybrid");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await queryDocuments(question, { search_method: searchMethod });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Query form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            className="mb-2 block text-sm font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Ask a question about your documents
          </label>
          <textarea
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="e.g., What was the net revenue in Q3 2024?"
            rows={3}
            className="w-full resize-none rounded-lg border px-4 py-3 text-sm outline-none transition-colors focus:border-[var(--accent)]"
            style={{
              background: "var(--bg-elevated)",
              borderColor: "var(--border)",
              color: "var(--text-primary)",
            }}
          />
        </div>

        <div className="flex items-center gap-4">
          <select
            value={searchMethod}
            onChange={(e) => setSearchMethod(e.target.value)}
            className="rounded-lg border px-3 py-2 text-sm outline-none"
            style={{
              background: "var(--bg-elevated)",
              borderColor: "var(--border)",
              color: "var(--text-primary)",
            }}
          >
            <option value="hybrid">Hybrid Search</option>
            <option value="vector">Vector Only</option>
            <option value="keyword">Keyword Only</option>
          </select>

          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="ml-auto rounded-lg px-6 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50"
            style={{ background: loading ? "var(--text-muted)" : "var(--accent)" }}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div
          className="rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: "var(--error)", color: "var(--error)", background: "#ef44441a" }}
        >
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Answer */}
          <div
            className="rounded-lg border p-6"
            style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
          >
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                Answer
              </h3>
              <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-muted)" }}>
                <span>
                  Confidence:{" "}
                  <span
                    style={{
                      color: result.confidence > 0.7 ? "var(--success)" : result.confidence > 0.4 ? "var(--warning)" : "var(--error)",
                    }}
                  >
                    {(result.confidence * 100).toFixed(0)}%
                  </span>
                </span>
                <span>·</span>
                <span>{result.retrieval_time_ms.toFixed(0)}ms retrieval</span>
                <span>·</span>
                <span>{result.generation_time_ms.toFixed(0)}ms generation</span>
              </div>
            </div>
            <p className="whitespace-pre-wrap text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
              {result.answer}
            </p>
          </div>

          {/* Citations */}
          {result.citations.length > 0 && (
            <div>
              <h3
                className="mb-3 text-sm font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                Sources ({result.citations.length})
              </h3>
              <div className="space-y-2">
                {result.citations.map((citation, i) => (
                  <div
                    key={citation.chunk_id}
                    className="rounded-lg border p-4"
                    style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}
                  >
                    <div className="mb-2 flex items-center gap-2">
                      <span
                        className="rounded px-2 py-0.5 text-xs font-medium text-white"
                        style={{ background: "var(--accent)" }}
                      >
                        Source {i + 1}
                      </span>
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                        {citation.source}
                        {citation.page_number && ` · Page ${citation.page_number}`}
                        {citation.section && ` · ${citation.section}`}
                      </span>
                      <span
                        className="ml-auto text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {(citation.relevance_score * 100).toFixed(0)}% relevant
                      </span>
                    </div>
                    <p
                      className="text-xs leading-relaxed"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {citation.content}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Meta */}
          <div
            className="flex gap-4 text-xs"
            style={{ color: "var(--text-muted)" }}
          >
            <span>Model: {result.model}</span>
            <span>Tokens: {result.total_tokens_used}</span>
          </div>
        </div>
      )}
    </div>
  );
}
