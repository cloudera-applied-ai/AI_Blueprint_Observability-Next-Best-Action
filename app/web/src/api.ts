/**
 * Typed wrappers over the FastAPI layer in `app/server.py`.
 *
 * The interfaces here mirror the Pydantic models on the server one-for-one.
 * If you change a model there, change it here — `tsc --strict` is the only
 * thing keeping the two in sync, since there is no codegen step.
 */

/** Mirrors `server.AppConfig`. */
export interface AppConfig {
  langsmith_org: string;
  langsmith_project: string;
  max_clarify_turns: number;
  /** Guardrail decision boundary, 0-1. Drawn as the tick on the risk meter. */
  high_risk_threshold: number;
  langsmith_links_enabled: boolean;
}

/** Mirrors `server.Customer`. */
export interface Customer {
  customer_id: number;
  full_name: string;
  risk_tier: string;
  annual_income: number;
  employment_status: string | null;
  existing_debt: number | null;
}

/** A credit-card offer row from the SQLite catalog (see `app/db.py`). */
export interface Offer {
  offer_id: string;
  offer_name: string;
  apr_range?: string;
  rewards_summary?: string;
  marketing_hook?: string;
  cta_text?: string;
  [key: string]: unknown;
}

/** Mirrors `server.TurnResponse`. */
export interface TurnResponse {
  run_id: string | null;
  /** Raw text as stored in `state.messages`, `[OFFER PRESENTED: ...]` and all. */
  assistant_message: string;
  /** Same text with the marker stripped — this is what gets rendered. */
  display_message: string;
  selected_offer: Offer | null;
  candidate_offers: Offer[] | null;
  high_risk_probability: number | null;
  risk_blocked: boolean;
  conversation_stage: string | null;
  accumulated_features: Record<string, unknown>;
  clarify_count: number | null;
}

/** Mirrors `server.FeedbackResponse`. */
export interface FeedbackResponse {
  ok: boolean;
  error: string | null;
}

/** Thrown for any non-2xx response, carrying the server's `detail` when present. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    // FastAPI puts the message in `detail`; fall back to the status text when
    // the body is not JSON at all (a proxy error page, say).
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      /* non-JSON body — keep statusText */
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export function fetchConfig(): Promise<AppConfig> {
  return request<AppConfig>("/api/config");
}

export function fetchCustomers(): Promise<Customer[]> {
  return request<Customer[]>("/api/customers");
}

/**
 * Run one turn. This is the call that becomes an `nba_turn` root run in
 * LangSmith, so one invocation here == one traced turn there.
 */
export function postTurn(
  message: string,
  threadId: string,
  customerId: number | null,
): Promise<TurnResponse> {
  return request<TurnResponse>("/api/turn", {
    method: "POST",
    body: JSON.stringify({
      message,
      thread_id: threadId,
      customer_id: customerId,
    }),
  });
}

/** Attach thumbs feedback to one turn's root run. `score`: 1 up, 0 down. */
export function postFeedback(
  runId: string,
  score: 0 | 1,
): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ run_id: runId, score }),
  });
}
