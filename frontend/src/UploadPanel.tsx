import { useState, useRef } from "react";
import type { IngestResponse } from "./api";
import { ingestDocument } from "./api";

const ACCEPTED_TYPES = ".pdf,.md,.markdown,.txt,.text,.csv";

export function UploadPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [namespace, setNamespace] = useState("default");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await ingestDocument(file, namespace, title);
      setResult(res);
      setFile(null);
      setTitle("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* File drop zone */}
        <div
          className="cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors"
          style={{ borderColor: file ? "var(--accent)" : "var(--border)" }}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED_TYPES}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="hidden"
          />
          {file ? (
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {file.name}
              </p>
              <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                {(file.size / 1024).toFixed(1)} KB · Click to change
              </p>
            </div>
          ) : (
            <div>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Click to select a file
              </p>
              <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                Supports PDF, Markdown, TXT, CSV
              </p>
            </div>
          )}
        </div>

        {/* Options */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--text-muted)" }}>
              Namespace
            </label>
            <input
              type="text"
              value={namespace}
              onChange={(e) => setNamespace(e.target.value)}
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none"
              style={{
                background: "var(--bg-elevated)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--text-muted)" }}>
              Title (optional)
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Auto-detected from content"
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none"
              style={{
                background: "var(--bg-elevated)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full rounded-lg py-2.5 text-sm font-medium text-white transition-colors disabled:opacity-50"
          style={{ background: loading ? "var(--text-muted)" : "var(--accent)" }}
        >
          {loading ? "Processing..." : "Upload & Ingest"}
        </button>
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

      {/* Success */}
      {result && (
        <div
          className="rounded-lg border p-4"
          style={{ borderColor: "var(--success)", background: "#22c55e1a" }}
        >
          <p className="text-sm font-medium" style={{ color: "var(--success)" }}>
            Document ingested successfully
          </p>
          <div className="mt-2 space-y-1 text-xs" style={{ color: "var(--text-secondary)" }}>
            <p>File: {result.filename}</p>
            <p>Chunks: {result.chunk_count}</p>
            <p>Tokens: {result.total_tokens.toLocaleString()}</p>
            <p>Namespace: {result.namespace}</p>
            <p className="font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
              ID: {result.document_id}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
