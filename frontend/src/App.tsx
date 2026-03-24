import { useState } from "react";
import { QueryPanel } from "./QueryPanel";
import { DocumentsPanel } from "./DocumentsPanel";
import { UploadPanel } from "./UploadPanel";

type Tab = "query" | "documents" | "upload";

export function App() {
  const [tab, setTab] = useState<Tab>("query");

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-base)" }}>
      {/* Header */}
      <header
        className="border-b px-6 py-4"
        style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
      >
        <div className="mx-auto flex max-w-4xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg text-sm font-bold text-white"
              style={{ background: "var(--accent)" }}
            >
              R
            </div>
            <div>
              <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
                RAG Engine
              </h1>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Retrieval-Augmented Generation
              </p>
            </div>
          </div>

          {/* Tab nav */}
          <nav className="flex gap-1 rounded-lg p-1" style={{ background: "var(--bg-elevated)" }}>
            {(["query", "documents", "upload"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="rounded-md px-4 py-1.5 text-sm font-medium capitalize transition-colors"
                style={{
                  background: tab === t ? "var(--accent)" : "transparent",
                  color: tab === t ? "white" : "var(--text-secondary)",
                }}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="mx-auto max-w-4xl px-6 py-8">
        {tab === "query" && <QueryPanel />}
        {tab === "documents" && <DocumentsPanel />}
        {tab === "upload" && <UploadPanel />}
      </main>
    </div>
  );
}
