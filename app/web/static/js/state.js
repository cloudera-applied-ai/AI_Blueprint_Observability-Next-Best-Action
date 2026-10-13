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
function newThreadId() {
    // crypto.randomUUID needs a secure context. CAI serves the app over HTTPS
    // and localhost counts as secure, but fall back anyway so a plain-HTTP
    // port-forward during development does not break the app outright.
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `thread-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}
export const state = {
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
export function resetConversation() {
    state.threadId = newThreadId();
    state.history = [];
    state.lastResult = null;
    state.pending = false;
}
/** Build the LangSmith Threads deep-link for the current conversation. */
export function langsmithThreadUrl() {
    const config = state.config;
    if (!config || !config.langsmith_links_enabled) {
        return null;
    }
    return (`https://smith.langchain.com/o/${encodeURIComponent(config.langsmith_org)}` +
        `/projects/p/${encodeURIComponent(config.langsmith_project)}` +
        `/t/${encodeURIComponent(state.threadId)}`);
}
//# sourceMappingURL=state.js.map