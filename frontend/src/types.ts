/**
 * The shapes the backend actually returns.
 *
 * These mirror `verity/api/documents.py` and `verity/api/sessions.py`. The
 * frontend does not invent a friendlier shape here: a mismatch between these
 * types and the API is a bug that TypeScript cannot see through fetch().
 */

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentRecord {
  id: number;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  chunk_count: number;
  created_at: string;
  processed_at: string | null;
}

export interface UploadResult {
  document_id: number;
  status: string;
  duplicate: boolean;
  message: string;
}

export interface Citation {
  /** Position in the chunk list shown to the model (live) or the DB chunk id (history). */
  chunk_id: number;
  /** 1-based order to display. Live answers set this; history may omit it. */
  ordinal?: number;
  rel_path: string;
  location: string;
  text?: string;
}

export interface SearchedChunk {
  rel_path: string;
  location: string;
}

export interface TraceChunk {
  rel_path: string;
  location: string;
  score: number;
  rank: number;
  sources: Array<[string, number]>;
  matched_terms: string[];
  preview: string;
}

export type AgentAction = "search" | "answer" | "clarification" | "refusal";

export interface TraceStep {
  index: number;
  /** The query this step actually searched. */
  query: string;
  /** Set when a follow-up was rewritten: what the user typed. */
  original_query?: string;
  /** Set when a rewrite was attempted, whatever it produced. */
  rewrite_attempted?: boolean;
  /** Set when the rewrite was attempted and failed; the literal query ran. */
  rewrite_failed?: boolean;
  action: AgentAction;
  results: number;
  retrieve_ms: number;
  decide_ms: number;
  cited?: number[];
  top: TraceChunk[];
}

export interface Trace {
  retrievers: string[];
  /** True when the vector half did not run; the fused scores are lexical only. */
  degraded: boolean;
  duration_ms: number;
  steps: TraceStep[];
}

export type AgentKind = "answer" | "clarification" | "refusal";

export interface AgentResponse {
  type: AgentKind;
  text: string;
  citations: Citation[];
  searched: SearchedChunk[];
  iterations: number;
  /** Absent on the early "no documents" refusal. */
  trace?: Trace;
}

export interface HistoryMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  kind: AgentKind;
  created_at: string;
  citations: Citation[];
}

export interface HistoryResponse {
  session_id: number;
  messages: HistoryMessage[];
}

export type Turn =
  | {
      key: string;
      role: "user";
      text: string;
      at: string | null;
    }
  | {
      key: string;
      role: "assistant";
      response: AgentResponse;
      at: string | null;
    };

export function hasTrace(response: AgentResponse): response is AgentResponse & {
  trace: Trace;
} {
  return Boolean(response.trace && response.trace.steps?.length);
}
