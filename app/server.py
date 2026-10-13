#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""FastAPI layer for the NBA chatbot.

Serves two things off a single port, because a Cloudera AI Application runs
one script bound to one port (``$CDSW_READONLY_PORT``):

* a small JSON API over the LangGraph agent in :mod:`nba_app`, and
* the compiled TypeScript client under ``web/static``.

Same origin for both, so there is no CORS configuration to get wrong.

The agent itself lives in :mod:`nba_app` and is imported flat (``from nba_app
import ...``), matching how ``nba_experiments.ipynb`` imports it after putting
``app/`` on ``sys.path``.  Run with::

    uvicorn server:app --app-dir app --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langsmith import Client
from pydantic import BaseModel, Field

from nba_app import (
    HIGH_RISK_THRESHOLD,
    LANGSMITH_ORG,
    LANGSMITH_PROJECT,
    MAX_CLARIFY_TURNS,
    demo_customer_picks,
    run_turn,
    strip_offer_marker,
)

log = logging.getLogger("nba.server")

_STATIC_DIR = Path(__file__).parent / "web" / "static"


# ----------------------------------------------------------------------
# Wire models
# ----------------------------------------------------------------------


class AppConfig(BaseModel):
    """Bootstrap payload the client fetches once on load."""

    langsmith_org: str = ""
    langsmith_project: str = ""
    max_clarify_turns: int = 3
    # The guardrail's decision boundary. The UI draws it as a tick on the
    # risk meter - without it the probability is a number with no reference
    # point, and you cannot see how close a customer came to being declined.
    high_risk_threshold: float = 0.5
    # False when LANGSMITH_ORG is unset - the client hides the thread link
    # rather than rendering one that 404s.
    langsmith_links_enabled: bool = False


class Customer(BaseModel):
    customer_id: int
    full_name: str
    risk_tier: str
    annual_income: float
    employment_status: Optional[str] = None
    existing_debt: Optional[float] = None


class TurnRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    # Only strictly needed on the first turn of a thread; LangGraph's
    # checkpointer carries it forward. The client sends it every time anyway,
    # which is harmless and keeps the client ignorant of that rule.
    customer_id: Optional[int] = None


class TurnResponse(BaseModel):
    """Mirrors ``run_turn``'s return dict, minus ``messages``.

    ``messages`` holds raw ``BaseMessage`` objects that are not JSON
    serializable - and the client keeps its own transcript anyway, so the
    server would just be echoing it back. Authoritative conversation state
    lives in the graph's checkpointer, keyed on ``thread_id``.
    """

    run_id: Optional[str] = None
    # Raw text as stored in state.messages, marker and all.
    assistant_message: str = ""
    # Same text with the [OFFER PRESENTED: ...] marker stripped, for display.
    display_message: str = ""
    selected_offer: Optional[Dict[str, Any]] = None
    candidate_offers: Optional[List[Dict[str, Any]]] = None
    high_risk_probability: Optional[float] = None
    risk_blocked: bool = False
    conversation_stage: Optional[str] = None
    accumulated_features: Dict[str, Any] = Field(default_factory=dict)
    clarify_count: Optional[int] = None


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1)
    # 1 = thumbs up, 0 = thumbs down. Same score the Streamlit UI wrote.
    score: int = Field(ge=0, le=1)


class FeedbackResponse(BaseModel):
    ok: bool
    # Populated instead of raising when LangSmith is unreachable - a failed
    # thumbs click should surface as a toast, not a broken conversation.
    error: Optional[str] = None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = FastAPI(
    title="NBA Chatbot API",
    description="Next-Best-Action credit-card chatbot on Cloudera AI.",
    version="1.0.0",
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=AppConfig)
def get_config() -> AppConfig:
    return AppConfig(
        langsmith_org=LANGSMITH_ORG,
        langsmith_project=LANGSMITH_PROJECT,
        max_clarify_turns=MAX_CLARIFY_TURNS,
        high_risk_threshold=HIGH_RISK_THRESHOLD,
        langsmith_links_enabled=bool(LANGSMITH_ORG),
    )


@app.get("/api/customers", response_model=List[Customer])
def get_customers() -> List[Customer]:
    """Demo customer slate, ordered LOW -> MED -> HIGH.

    Returns ``[]`` rather than erroring when the DB has no customers yet, so
    a fresh clone renders a "seed the database" hint instead of a stack trace.
    """
    try:
        rows = demo_customer_picks()
    except Exception as exc:  # noqa: BLE001 - unseeded or missing DB file
        log.warning("customer slate unavailable: %s", exc)
        return []
    return [Customer(**row) for row in rows]


@app.post("/api/turn", response_model=TurnResponse)
def post_turn(req: TurnRequest) -> TurnResponse:
    """Run one turn of the conversation.

    Each call is one ``nba_turn`` root run in LangSmith, tagged with
    ``thread_id`` so the Threads tab folds the conversation back together.
    """
    result = run_turn(req.message, req.thread_id, req.customer_id)
    raw = result.get("assistant_message", "") or ""
    return TurnResponse(
        run_id=result.get("run_id"),
        assistant_message=raw,
        display_message=strip_offer_marker(raw),
        selected_offer=result.get("selected_offer"),
        candidate_offers=result.get("candidate_offers"),
        high_risk_probability=result.get("high_risk_probability"),
        risk_blocked=bool(result.get("risk_blocked", False)),
        conversation_stage=result.get("conversation_stage"),
        accumulated_features=result.get("accumulated_features") or {},
        clarify_count=result.get("clarify_count"),
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Attach thumbs feedback to one turn's root run.

    ``run_id`` is the per-turn root, not the thread - that per-turn precision
    is what makes the signal usable by downstream evaluators.
    """
    try:
        Client().create_feedback(req.run_id, key="user_thumbs", score=req.score)
    except Exception as exc:  # noqa: BLE001 - no API key, offline, bad run id
        log.warning("LangSmith feedback failed for run %s: %s", req.run_id, exc)
        return FeedbackResponse(ok=False, error=str(exc))
    return FeedbackResponse(ok=True)


@app.exception_handler(Exception)
async def _unhandled(request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    """Return JSON on unexpected failures so the client can show a message.

    Without this the client's ``response.json()`` would choke on an HTML
    traceback page and the chat would hang on the typing indicator forever.
    """
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Mounted last: this catch-all would otherwise shadow the /api routes above.
# html=True makes StaticFiles serve index.html for "/".
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="web")
else:  # pragma: no cover - only hit when the UI has not been built
    log.warning(
        "UI assets missing at %s - run `npm install && npm run build` in app/web. "
        "The API still works.",
        _STATIC_DIR,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("NBA_HOST", "127.0.0.1"),
        port=int(
            os.environ.get("CDSW_READONLY_PORT", os.environ.get("NBA_PORT", "8000"))
        ),
    )
