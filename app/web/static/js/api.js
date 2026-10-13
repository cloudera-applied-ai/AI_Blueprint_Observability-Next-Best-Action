/**
 * Typed wrappers over the FastAPI layer in `app/server.py`.
 *
 * The interfaces here mirror the Pydantic models on the server one-for-one.
 * If you change a model there, change it here — `tsc --strict` is the only
 * thing keeping the two in sync, since there is no codegen step.
 */
/** Thrown for any non-2xx response, carrying the server's `detail` when present. */
export class ApiError extends Error {
    status;
    constructor(status, message) {
        super(message);
        this.name = "ApiError";
        this.status = status;
    }
}
async function request(path, init) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...init,
    });
    if (!response.ok) {
        // FastAPI puts the message in `detail`; fall back to the status text when
        // the body is not JSON at all (a proxy error page, say).
        let detail = response.statusText;
        try {
            const body = (await response.json());
            if (typeof body.detail === "string") {
                detail = body.detail;
            }
        }
        catch {
            /* non-JSON body — keep statusText */
        }
        throw new ApiError(response.status, detail);
    }
    return (await response.json());
}
export function fetchConfig() {
    return request("/api/config");
}
export function fetchCustomers() {
    return request("/api/customers");
}
/**
 * Run one turn. This is the call that becomes an `nba_turn` root run in
 * LangSmith, so one invocation here == one traced turn there.
 */
export function postTurn(message, threadId, customerId) {
    return request("/api/turn", {
        method: "POST",
        body: JSON.stringify({
            message,
            thread_id: threadId,
            customer_id: customerId,
        }),
    });
}
/** Attach thumbs feedback to one turn's root run. `score`: 1 up, 0 down. */
export function postFeedback(runId, score) {
    return request("/api/feedback", {
        method: "POST",
        body: JSON.stringify({ run_id: runId, score }),
    });
}
//# sourceMappingURL=api.js.map