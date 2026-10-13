/**
 * Client-side session state.
 *
 * Deliberately small: the *authoritative* conversation state lives in the
 * LangGraph `MemorySaver` checkpointer on the server, keyed on `threadId`.
 * What is kept here is only what the browser needs to redraw itself — the
 * transcript it has already displayed, plus the latest turn's telemetry for
 * the observability panel.
 *
 * Nothing here is persisted. Reloading the page starts a new thread, which
 * matches what "New conversation" does and keeps the demo predictable.
 */

import type { AppConfig, Customer, TurnResponse } from "./api.js";

export interface ChatMessage {
  role: "user" | "assistant";
  /** Text as displayed (offer marker already stripped by the server). */
  content: string;
  /** Wall-clock time the message entered the transcript. */
  at: Date;
  /**
   * LangSmith root run for this turn. Only set on assistant messages, and
   * only when tracing is on — thumbs buttons are hidden without it, since
   * there would be nothing to attach the feedback to.
   */
  runId: string | null;
  /** Set when the turn failed; rendered as an error bubble instead. */
  isError?: boolean;
}

export interface SessionState {
  threadId: string;
  customerId: number | null;
  history: ChatMessage[];
  /** Telemetry from the most recent successful turn. */
  lastResult: TurnResponse | null;
  customers: Customer[];
  config: AppConfig | null;
  /** True while a turn is in flight; gates the composer against double-sends. */
  pending: boolean;
}

function newThreadId(): string {
  // crypto.randomUUID needs a secure context. CAI serves the app over HTTPS
  // and localhost counts as secure, but fall back anyway so a plain-HTTP
  // port-forward during development does not break the app outright.
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `thread-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

export const state: SessionState = {
  threadId: newThreadId(),
  customerId: null,
  history: [],
  lastResult: null,
  customers: [],
  config: null,
  pending: false,
};

/**
 * Start a fresh conversation.
 *
 * A new `threadId` is the whole mechanism: the server's checkpointer has no
 * entry for it, so the graph starts from an empty state. The selected
 * customer deliberately survives, so you can re-run the same profile.
 */
export function resetConversation(): void {
  state.threadId = newThreadId();
  state.history = [];
  state.lastResult = null;
  state.pending = false;
}

/** Build the LangSmith Threads deep-link for the current conversation. */
export function langsmithThreadUrl(): string | null {
  const config = state.config;
  if (!config || !config.langsmith_links_enabled) {
    return null;
  }
  return (
    `https://smith.langchain.com/o/${encodeURIComponent(config.langsmith_org)}` +
    `/projects/p/${encodeURIComponent(config.langsmith_project)}` +
    `/t/${encodeURIComponent(state.threadId)}`
  );
}
