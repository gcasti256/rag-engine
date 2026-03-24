const BASE = "/api";

export interface Citation {
  chunk_id: string;
  source: string;
  content: string;
  page_number: number | null;
  section: string;
  relevance_score: number;
}

export interface QueryResult {
  answer: string;
  citations: Citation[];
  confidence: number;
  query: string;
  model: string;
  total_tokens_used: number;
  retrieval_time_ms: number;
  generation_time_ms: number;
}

export interface Document {
  id: string;
  filename: string;
  document_type: string;
  title: string;
  namespace: string;
  chunk_count: number;
  total_tokens: number;
  created_at: string;
}

export interface IngestResponse {
  document_id: string;
  filename: string;
  chunk_count: number;
  total_tokens: number;
  namespace: string;
}

export async function queryDocuments(
  question: string,
  options: { top_k?: number; namespace?: string; search_method?: string } = {}
): Promise<QueryResult> {
  const form = new FormData();
  form.append("question", question);
  if (options.top_k) form.append("top_k", String(options.top_k));
  if (options.namespace) form.append("namespace", options.namespace);
  if (options.search_method) form.append("search_method", options.search_method);

  const res = await fetch(`${BASE}/query`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function queryStream(
  question: string,
  options: { top_k?: number; namespace?: string; search_method?: string } = {}
): Promise<Response> {
  const form = new FormData();
  form.append("question", question);
  if (options.top_k) form.append("top_k", String(options.top_k));
  if (options.namespace) form.append("namespace", options.namespace);
  if (options.search_method) form.append("search_method", options.search_method);

  const res = await fetch(`${BASE}/query/stream`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res;
}

export async function listDocuments(namespace?: string): Promise<Document[]> {
  const params = namespace ? `?namespace=${namespace}` : "";
  const res = await fetch(`${BASE}/documents${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function ingestDocument(
  file: File,
  namespace: string = "default",
  title: string = ""
): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("namespace", namespace);
  if (title) form.append("title", title);

  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
