import { useState, useEffect } from "react";
import type { Document } from "./api";
import { listDocuments, deleteDocument } from "./api";

export function DocumentsPanel() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function fetchDocuments() {
    setLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchDocuments();
  }, []);

  async function handleDelete(id: string, filename: string) {
    if (!confirm(`Delete "${filename}" and all its chunks?`)) return;
    try {
      await deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  if (loading) {
    return (
      <div className="py-12 text-center text-sm" style={{ color: "var(--text-muted)" }}>
        Loading documents...
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div
        className="rounded-lg border py-16 text-center"
        style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
      >
        <p className="text-lg" style={{ color: "var(--text-secondary)" }}>No documents ingested</p>
        <p className="mt-1 text-sm" style={{ color: "var(--text-muted)" }}>
          Upload documents in the Upload tab to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <div
          className="rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: "var(--error)", color: "var(--error)" }}
        >
          {error}
        </div>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
          {documents.length} document{documents.length !== 1 && "s"} ingested
        </h2>
        <button
          onClick={fetchDocuments}
          className="rounded px-3 py-1 text-xs transition-colors"
          style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)" }}
        >
          Refresh
        </button>
      </div>

      <div className="space-y-2">
        {documents.map((doc) => (
          <div
            key={doc.id}
            className="flex items-center justify-between rounded-lg border p-4"
            style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
          >
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {doc.title || doc.filename}
              </p>
              <div className="mt-1 flex gap-3 text-xs" style={{ color: "var(--text-muted)" }}>
                <span>{doc.filename}</span>
                <span>·</span>
                <span>{doc.chunk_count} chunks</span>
                <span>·</span>
                <span>{doc.total_tokens.toLocaleString()} tokens</span>
                <span>·</span>
                <span>{doc.namespace}</span>
              </div>
            </div>
            <button
              onClick={() => handleDelete(doc.id, doc.filename)}
              className="rounded px-3 py-1 text-xs transition-colors hover:bg-[var(--error)]"
              style={{ color: "var(--error)" }}
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
