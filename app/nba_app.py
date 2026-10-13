#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""Next-Best-Action multi-turn chatbot demo.

A LangGraph pipeline that guides a bank customer through a short conversation,
applies a customer-risk guardrail, consults a local SQLite rule engine of
credit-card offers, and presents the best-fit offer in-chat.  Fully traced
through LangSmith with a per-thread ``thread_id`` so the entire conversation
appears as one thread in the LangSmith Threads tab.

The risk guardrail is currently a **rules-based placeholder** that reads
``state["customer_record"]`` and applies a few if/else checks (see
``_rules_based_risk_score``).  An ML-backed guardrail (XGBoost or PyTorch,
served via Cloudera AI Inference Service) is scaffolded under
``xgboost/`` and ``pytorch_train/`` for future re-enablement — swap the
call in ``risk_guardrail_node`` back to the model endpoint when ready.

Runs as a Cloudera AI Application via ``launch_app.py``.
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from db import get_conn, init_schema, seed_offers
from offer_rules import select_offers


# ----------------------------------------------------------------------
# Environment / configuration
# ----------------------------------------------------------------------

# Enable LangSmith tracing by default so every turn shows up in the UI.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "nba-demo")

# LLM (Nemotron on Cloudera AI Inference Service)
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nemotron")
LLM_ENDPOINT_BASE_URL = os.environ.get("LLM_ENDPOINT_BASE_URL", "")
LLM_CDP_TOKEN = os.environ.get("LLM_CDP_TOKEN", "")

# Classifier (currently disabled — rules-based guardrail is used instead).
# These env vars are read but unused; the ML-backed guardrail can be re-enabled
# by pointing risk_guardrail_node back at the endpoint call in the git history
# (see xgboost/ and pytorch_train/ for the trained artifacts).
CLF_MODEL_ID = os.environ.get("CLF_MODEL_ID", "nba-risk-onnx-xgboost")
CLF_ENDPOINT_BASE_URL = os.environ.get("CLF_ENDPOINT_BASE_URL", "")
CLF_CDP_TOKEN = os.environ.get("CLF_CDP_TOKEN", "")

# Threshold for the customer-risk guardrail.  FRAUD_THRESHOLD is kept as an
# alias so existing deploy scripts continue to work.
HIGH_RISK_THRESHOLD = float(
    os.environ.get("HIGH_RISK_THRESHOLD", os.environ.get("FRAUD_THRESHOLD", "0.5"))
)
NBA_MOCK = os.environ.get("NBA_MOCK", "") == "1"

# Hard cap on clarifying questions before the graph is forced to present an
# offer.  Prevents the intent router from looping on NEED_MORE_INFO when the
# feature schema can't capture every detail the customer volunteers.
MAX_CLARIFY_TURNS = int(os.environ.get("NBA_MAX_CLARIFY_TURNS", "3"))

# LangSmith URLs (for UI links from the sidebar)
LANGSMITH_ORG = os.environ.get("LANGSMITH_ORG", "")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "nba-demo")


# ----------------------------------------------------------------------
# LLM setup
# ----------------------------------------------------------------------

_llm: Optional[ChatOpenAI] = None


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Cached ChatOpenAI client pointed at the Nemotron endpoint.

    We defer construction so importing this module doesn't require the env
    vars to be set (evaluators can point at a different LLM).
    """
    global _llm
    if _llm is None:
        # ``timeout`` and ``max_retries`` matter: Nemotron on CAI Inference
        # accepts our request but returns an empty ``tool_calls`` list when
        # the OpenAI function-calling schema is sent, so downstream
        # ``with_structured_output`` calls use ``method="json_mode"`` (see
        # call sites) and we bound how long any single HTTP call can wait.
        _llm = ChatOpenAI(
            model=LLM_MODEL_ID,
            base_url=LLM_ENDPOINT_BASE_URL or None,
            api_key=LLM_CDP_TOKEN or "unused",
            temperature=temperature,
            timeout=60,
            max_retries=1,
        )
    return _llm


# ----------------------------------------------------------------------
# Structured-output schemas
# ----------------------------------------------------------------------


class Features(BaseModel):
    """Features extracted from the customer's chat messages.

    Used by the clarifying-question prompt and passed to the offer rule
    engine for personalization.  The XGBoost risk guardrail no longer reads
    from here — it looks up the customer row directly from SQLite.
    """

    age: int = Field(default=0)
    annual_income: float = Field(default=0)
    requested_credit_line: float = Field(default=0)
    existing_debt: float = Field(default=0)
    employment_status: Literal[
        "EMPLOYED", "SELF_EMPLOYED", "STUDENT", "UNEMPLOYED", "RETIRED", "UNKNOWN"
    ] = "UNKNOWN"
    monthly_spend: float = Field(default=0)
    stated_goal: str = Field(default="")


class Intent(BaseModel):
    """Router decision emitted after every user turn."""

    stage: Literal["NEED_MORE_INFO", "READY_FOR_OFFER", "POST_OFFER_CHAT"]
    reasoning: str = ""


# ML-guardrail feature order.  Currently unused (the rules-based guardrail
# reads customer_record fields directly), but kept as documentation of the
# training contract in xgboost/01_train_xgboost_onnx.ipynb and
# pytorch_train/01_train_pytorch_onnx.ipynb.  Restore the endpoint call in
# risk_guardrail_node and this list becomes the payload's column order.
_EMP_STATUSES = ["EMPLOYED", "SELF_EMPLOYED", "STUDENT", "UNEMPLOYED", "RETIRED"]
FEATURE_ORDER = [
    "age",
    "annual_income",
    "existing_debt",
] + [f"emp_{s}" for s in _EMP_STATUSES]


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

feature_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a feature-extraction engine.  Respond with a single
JSON object with the following keys:
- age (int)
- annual_income (float)
- requested_credit_line (float)
- existing_debt (float)
- employment_status ("EMPLOYED"|"SELF_EMPLOYED"|"STUDENT"|"UNEMPLOYED"|"RETIRED"|"UNKNOWN")
- monthly_spend (float)
- stated_goal (short free-text summary of what the customer wants)

Rules:
- If a value is not stated, use 0 (or "UNKNOWN" for employment_status, "" for stated_goal).
- Do NOT infer values that were not explicitly stated.
- Only use numeric values for numeric fields.
- Return ONLY the JSON object, nothing else.
""",
        ),
        ("human", "{input}"),
    ]
)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the intent router for a credit-card recommendation chatbot.

Classify the conversation state given the full message history and the
features already gathered.  Respond with a single JSON object with keys:
- stage: one of "NEED_MORE_INFO", "READY_FOR_OFFER", "POST_OFFER_CHAT"
- reasoning: one short sentence explaining the choice

Stage semantics:
- NEED_MORE_INFO   — we don't yet know enough about the customer's goal,
                     income, or debt situation to make a specific recommendation.
- READY_FOR_OFFER  — the customer has stated a clear goal AND we have at
                     least an approximate income OR employment status.  The
                     next turn should present a specific offer.
- POST_OFFER_CHAT  — an offer has ALREADY been presented in a prior assistant
                     turn (look for `[OFFER PRESENTED: ...]` markers).  The
                     customer is asking follow-up questions or negotiating.

Be conservative: prefer NEED_MORE_INFO if the goal is vague.  Prefer
POST_OFFER_CHAT whenever an offer has already been presented, even for
questions that could arguably reset the flow.

IMPORTANT: If the assistant has already asked 3 or more clarifying
questions in the transcript, choose READY_FOR_OFFER — even if some
fields are still missing.  The rule engine can pick a best-guess offer
from partial info; a fourth clarifying question is worse than a
best-guess pitch.

Return ONLY the JSON object, nothing else.
""",
        ),
        (
            "human",
            "Conversation so far:\n{transcript}\n\n"
            "Accumulated features (JSON):\n{features}\n\n"
            "Offer already presented? {offer_presented}\n\n"
            "What is the current stage?",
        ),
    ]
)

clarify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant helping a customer find
the best credit-card offer.  Ask ONE short, specific clarifying question to
close the biggest gap in what you know so far.

Guidelines:
- NEVER re-ask a question already in "Questions already asked" below, even
  if you paraphrase it.  Pick a different topic.
- NEVER re-ask for a field that is already populated in the features JSON.
- Prefer questions that would materially change which card to recommend
  (primary use / goal, approximate income, credit-building need).  Do NOT
  drill into granular details (specific merchants, exact monthly spend,
  merchant categories) — you have enough to recommend after 2-3 questions.
- Keep it under 40 words.
- Sound conversational, not like a form.
- If the customer mentioned a goal (e.g. travel, rebuilding credit,
  consolidating debt), acknowledge it briefly first.
""",
        ),
        (
            "human",
            "Conversation so far:\n{transcript}\n\n"
            "Accumulated features (JSON):\n{features}\n\n"
            "Questions already asked (do NOT repeat any of these):\n{prior_questions}\n\n"
            "Ask your next clarifying question.",
        ),
    ]
)

offer_pitch_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant presenting a credit-card
offer to a customer in a chat.  Do NOT write an email.  Write a warm, concise
chat message (under 120 words) that:

1. Names the offer.
2. Highlights the two most relevant benefits given what the customer said.
3. Mentions the APR range.
4. Ends with a soft, open call-to-action question.

Prefix your message with `[OFFER PRESENTED: <offer_id>]` on its own first
line so downstream nodes can detect that an offer has already been made.
Then leave a blank line and start the customer-visible message.
""",
        ),
        (
            "human",
            "Customer profile:\n"
            "- Name: {full_name}\n"
            "- Risk tier: {risk_tier}\n"
            "- Employment: {employment_status}\n\n"
            "Conversation so far:\n{transcript}\n\n"
            "Recommended offer:\n"
            "- offer_id: {offer_id}\n"
            "- offer_name: {offer_name}\n"
            "- marketing_hook: {marketing_hook}\n"
            "- rewards_summary: {rewards_summary}\n"
            "- apr_range: {apr_range}\n"
            "- cta_text: {cta_text}\n\n"
            "Write your pitch.",
        ),
    ]
)

post_offer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant. The customer has already
been shown a specific credit-card offer (details below). Answer their next
question grounded ONLY on the offer details and general facts about credit
cards.  If they ask something you don't know, say so and offer to connect
them with a specialist.  Keep it under 120 words.
""",
        ),
        (
            "human",
            "Presented offer:\n{offer_json}\n\n"
            "Conversation so far:\n{transcript}\n\n"
            "Respond to the customer's latest message.",
        ),
    ]
)

decline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a banking assistant.  Our risk guardrail flagged this
session as high-risk, so we cannot make card recommendations or answer
account-specific questions in this chat.

Reply politely with a short chat message (under 80 words) that:
- Acknowledges the customer's message without repeating their question back
  verbatim.
- Explains that for account-specific questions (fees, balances, limits,
  APR on their current card, etc.) they need to check their online banking
  or call the support line.
- Does NOT reveal the risk score or that a guardrail fired.
- Does NOT quote any specific dollar amounts, percentages, dates, account
  numbers, or card names.  In particular NEVER write placeholders like
  "$XX", "X%", or "your account".  If a specific figure would be needed to
  answer, say you can't share it here and point them to support.
- Chat tone, warm but firm.
""",
        ),
        ("human", "Conversation so far:\n{transcript}"),
    ]
)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    customer_id: Optional[int]
    accumulated_features: Dict[str, Any]
    high_risk_probability: Optional[float]
    risk_blocked: bool
    customer_record: Optional[Dict[str, Any]]
    selected_offer: Optional[Dict[str, Any]]
    candidate_offers: Optional[List[Dict[str, Any]]]
    conversation_stage: Literal["gathering", "offer_presented", "post_offer"]
    # Transient hop from intent_router -> stage_router. Must be declared on
    # the TypedDict or LangGraph drops it as an unknown key when persisting.
    _next_stage: Optional[str]
    # Clarify-cap telemetry — how many clarifying questions have been asked
    # so far (surfaced to the LangSmith node output and the web UI's
    # observability panel so the cap is visible without opening the
    # transcript).
    clarify_count: Optional[int]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _transcript(messages: List[BaseMessage]) -> str:
    """Render message list as a compact transcript for prompt injection."""
    lines = []
    for m in messages:
        role = "CUSTOMER" if isinstance(m, HumanMessage) else "ASSISTANT"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(no messages yet)"


def _count_clarify_turns(messages: List[BaseMessage]) -> int:
    """Number of assistant clarifying questions already asked (pre-offer).

    Stops counting the moment we hit an ``[OFFER PRESENTED: ...]`` marker so
    that post-offer assistant chat never inflates the clarify budget.
    """
    n = 0
    for m in messages or []:
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else ""
            if content.startswith("[OFFER PRESENTED:"):
                return n
            n += 1
    return n


def _prior_clarify_questions(messages: List[BaseMessage]) -> List[str]:
    """Verbatim list of prior assistant clarifying questions.

    Fed to the clarify prompt so the LLM can be told, in-band, exactly which
    questions are off-limits on this turn.
    """
    out: List[str] = []
    for m in messages or []:
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else ""
            if content.startswith("[OFFER PRESENTED:"):
                break
            out.append(content)
    return out


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Strip Nemotron-style chain-of-thought from an LLM response.

    Nemotron 3 Super wraps its reasoning in ``<think>...</think>`` tags
    before emitting the final answer.  Depending on the token stream, we can
    end up with the full paired tags, only the closing ``</think>`` (opening
    tag consumed by the tokenizer), or neither.  Drop the reasoning in all
    three cases before showing the message to the customer / storing it in
    state.messages / feeding it to the conversation-quality judge — the raw
    LLM output is still visible on the individual LLM span in LangSmith for
    debugging.
    """
    if not text:
        return text
    # Full paired tags anywhere in the string.
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Orphan closing tag: keep everything after the LAST ``</think>``.
    lower = cleaned.lower()
    idx = lower.rfind("</think>")
    if idx != -1:
        cleaned = cleaned[idx + len("</think>"):]
    return cleaned.strip()


def strip_offer_marker(text: str) -> str:
    """Hide the ``[OFFER PRESENTED: <offer_id>]`` marker from the customer.

    ``offer_presentation_node`` prefixes its reply with that marker so later
    turns can detect an offer is already on the table (see
    ``intent_router_node``).  It belongs in ``state.messages``, not on screen,
    so the UI renders the text *after* the marker line.  Returns ``text``
    unchanged when no marker is present, or when stripping would leave
    nothing behind.
    """
    if not text or not text.startswith("[OFFER PRESENTED:"):
        return text
    return "\n".join(text.splitlines()[2:]) or text


def _merge_features(current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge freshly extracted features into the accumulator.

    Only overwrite when the new value is meaningful (non-zero, non-UNKNOWN,
    non-empty).  Preserves earlier information across turns.
    """
    merged = dict(current or {})
    for key, val in new.items():
        if val in (0, 0.0, "", None, "UNKNOWN"):
            continue
        merged[key] = val
    return merged


def _load_customer(customer_id: Optional[int]) -> Dict[str, Any]:
    """Fetch a customer row from SQLite.  Falls back to a random row."""
    conn = get_conn()
    try:
        init_schema(conn)
        if customer_id is None:
            max_id_row = conn.execute("SELECT COALESCE(MAX(customer_id), 0) FROM customers").fetchone()
            max_id = max_id_row[0] if max_id_row else 0
            if max_id == 0:
                # No customers seeded — return a synthetic default so the app
                # can still run (e.g. during ad-hoc dev).
                return {
                    "customer_id": 0,
                    "full_name": "Demo Customer",
                    "email": "demo@example.com",
                    "risk_tier": "MED",
                    "age": 35,
                    "annual_income": 60_000,
                    "employment_status": "EMPLOYED",
                    "existing_debt": 5_000,
                }
            customer_id = random.randint(1, max_id)
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return {"customer_id": customer_id, "full_name": "Unknown", "risk_tier": "MED"}
        return {k: row[k] for k in row.keys()}
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Rules-based customer-risk guardrail (placeholder for the ML model)
# ----------------------------------------------------------------------


@traceable(run_type="tool", name="rules_based_risk_score")
def _rules_based_risk_score(customer: Dict[str, Any]) -> float:
    """Return a pseudo-probability of "high risk" from a couple of if/else rules.

    Placeholder until the XGBoost/PyTorch endpoint under ``xgboost/`` or
    ``pytorch_train/`` is re-enabled.  Reads the same fields the ML guardrail
    would consume (age, annual_income, existing_debt, employment_status),
    plus ``risk_tier`` (already computed in ``pii_datagen``).

    Rules (any match → 0.9, else scaled by debt-to-income):
    - risk_tier == "HIGH"
    - debt-to-income ratio > 0.6
    - UNEMPLOYED with existing_debt > 5_000
    """
    risk_tier = str(customer.get("risk_tier", "") or "").upper()
    income = float(customer.get("annual_income", 0) or 0)
    debt = float(customer.get("existing_debt", 0) or 0)
    employment = str(customer.get("employment_status", "") or "").upper()
    dti = (debt / income) if income > 0 else 0.0

    if risk_tier == "HIGH":
        return 0.9
    if dti > 0.6:
        return 0.9
    if employment == "UNEMPLOYED" and debt > 5_000:
        return 0.9

    # Below the blocking threshold: return a small DTI-scaled score so the
    # UI's "high-risk probability" panel still shows something meaningful.
    return min(0.3, dti / 2.0)


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="intake_node")
def intake_node(state: GraphState) -> Dict[str, Any]:
    """First node every turn: extract offer-personalization features from the
    latest user message and merge them into the running feature dict.  Also
    lazy-loads the customer record on the first turn of the thread.

    Note: the fields extracted here feed the clarifying-question prompt and
    the offer rule engine.  The XGBoost risk guardrail reads directly from
    ``state["customer_record"]`` and does not depend on this node.
    """
    latest_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            latest_user = m.content
            break

    if NBA_MOCK:
        # Skip a live LLM call in mock mode; return a fixed chat-derived
        # profile.  The customer risk (and thus the decline path) is driven
        # by the looked-up ``customer_record``, not by anything here.
        extracted = Features(
            age=42,
            annual_income=180_000,
            requested_credit_line=25_000,
            existing_debt=5_000,
            employment_status="EMPLOYED",
            monthly_spend=4_000,
            stated_goal=latest_user[:80],
        )
    else:
        features_chain = feature_prompt | get_llm().with_structured_output(
            Features, method="json_mode"
        )
        extracted = features_chain.invoke({"input": latest_user})

    new_features = extracted.dict() if hasattr(extracted, "dict") else dict(extracted)
    merged = _merge_features(state.get("accumulated_features", {}), new_features)

    updates: Dict[str, Any] = {"accumulated_features": merged}

    # Lazy customer lookup on first turn.
    if state.get("customer_record") is None:
        updates["customer_record"] = _load_customer(state.get("customer_id"))

    # Default conversation_stage.
    if state.get("conversation_stage") is None:
        updates["conversation_stage"] = "gathering"

    return updates


@traceable(run_type="chain", name="risk_guardrail_node")
def risk_guardrail_node(state: GraphState) -> Dict[str, Any]:
    """Score the customer once per thread using the rules-based guardrail.

    Reads ``state["customer_record"]`` (loaded by ``_load_customer``) and
    passes the profile fields to ``_rules_based_risk_score``.  If the
    customer_id is unknown and no row was loaded, skip scoring and pass
    through as low-risk.

    The ML-backed guardrail (XGBoost/PyTorch endpoint) is temporarily
    disabled; re-enable by swapping ``_rules_based_risk_score`` back for the
    endpoint call.
    """
    if state.get("high_risk_probability") is not None:
        return {}

    customer = state.get("customer_record") or {}
    if not customer:
        # No customer looked up — nothing to score.  Assume low risk so the
        # graph continues to the intent router.
        return {"high_risk_probability": 0.0, "risk_blocked": False}

    proba = _rules_based_risk_score(customer)
    return {
        "high_risk_probability": proba,
        "risk_blocked": proba >= HIGH_RISK_THRESHOLD,
    }


def risk_router(state: GraphState) -> Literal["decline", "continue"]:
    return "decline" if state.get("risk_blocked") else "continue"


@traceable(run_type="chain", name="intent_router_node")
def intent_router_node(state: GraphState) -> Dict[str, Any]:
    """LLM-classified conversation stage router.

    Writes ``conversation_stage`` in state so downstream nodes and the
    web UI can inspect it.
    """
    transcript = _transcript(state.get("messages", []))
    features = state.get("accumulated_features", {}) or {}
    offer_presented = state.get("selected_offer") is not None

    if NBA_MOCK:
        # Deterministic staging for the mock path: first turn asks a
        # clarifying question, second turn presents an offer, subsequent
        # turns are post-offer chat.
        n_human = sum(1 for m in state.get("messages", []) if isinstance(m, HumanMessage))
        if offer_presented:
            stage = "POST_OFFER_CHAT"
        elif n_human >= 2:
            stage = "READY_FOR_OFFER"
        else:
            stage = "NEED_MORE_INFO"
    else:
        chain = intent_prompt | get_llm().with_structured_output(
            Intent, method="json_mode"
        )
        decision = chain.invoke(
            {
                "transcript": transcript,
                "features": json.dumps(features, default=str),
                "offer_presented": "yes" if offer_presented else "no",
            }
        )
        stage = decision.stage

    # Hard cap: if we've already asked MAX_CLARIFY_TURNS clarifying questions
    # and no offer has been presented yet, force the graph to commit to an
    # offer regardless of what the LLM router said.  The rule engine can
    # always pick *something* from the offers table given the customer_record
    # + partial features, so we'd rather present a best-guess offer than loop
    # forever.  Deliberately unconditional on ``stage`` so a mis-typed LLM
    # response can't slip past the cap.
    #
    # Source of truth: the explicit ``clarify_count`` state field that
    # ``clarify_node`` increments every time it asks a question.  We *also*
    # scan the transcript as a belt-and-suspenders fallback in case the
    # counter was lost across a checkpointer round-trip — some LangGraph
    # versions deserialize BaseMessage subclasses as dicts, which would
    # silently zero out a pure ``_count_clarify_turns``-based counter (this
    # is the bug that let the chatbot loop for 8 turns previously).
    stored = state.get("clarify_count") or 0
    scanned = _count_clarify_turns(state.get("messages", []))
    clarify_count = max(int(stored), int(scanned))
    msgs = state.get("messages", []) or []
    ai_types = [type(m).__name__ for m in msgs if not isinstance(m, HumanMessage)]
    print(
        f"[intent_router] messages={len(msgs)} stored_count={stored} "
        f"scanned_count={scanned} → clarify_count={clarify_count} "
        f"ai_message_types={ai_types}",
        flush=True,
    )
    cap_hit = (not offer_presented) and (clarify_count >= MAX_CLARIFY_TURNS)
    if cap_hit:
        stage = "READY_FOR_OFFER"
        print(
            f"[intent_router] clarify cap hit: clarify_count={clarify_count} "
            f"MAX_CLARIFY_TURNS={MAX_CLARIFY_TURNS} → forcing READY_FOR_OFFER",
            flush=True,
        )
        rt = get_current_run_tree()
        if rt is not None:
            try:
                rt.add_tags(["clarify_cap_hit"])
            except Exception:
                pass

    stage_map = {
        "NEED_MORE_INFO": "gathering",
        "READY_FOR_OFFER": "offer_presented",  # will be set for real by presentation node
        "POST_OFFER_CHAT": "post_offer",
    }
    return {
        "_next_stage": stage,
        "conversation_stage": stage_map[stage],
        # Surface the cap counter so it shows up in the LangSmith node output.
        "clarify_count": clarify_count,
    }


def stage_router(state: GraphState) -> Literal["clarify", "select_offer", "post_offer_chat"]:
    stage = state.get("_next_stage", "NEED_MORE_INFO")
    if stage == "READY_FOR_OFFER":
        return "select_offer"
    if stage == "POST_OFFER_CHAT":
        return "post_offer_chat"
    return "clarify"


@traceable(run_type="chain", name="clarify_node")
def clarify_node(state: GraphState) -> Dict[str, Any]:
    transcript = _transcript(state.get("messages", []))
    features = state.get("accumulated_features", {}) or {}

    if NBA_MOCK:
        content = (
            "Thanks for reaching out! To point you at the best card, could you tell me a bit "
            "about what you're looking for — travel rewards, everyday cashback, help with debt, "
            "or something else — and roughly what your annual income is?"
        )
    else:
        prior_qs = _prior_clarify_questions(state.get("messages", []))
        prior_qs_str = (
            "\n".join(f"- {q}" for q in prior_qs) if prior_qs else "(none)"
        )
        chain = clarify_prompt | get_llm()
        resp = chain.invoke(
            {
                "transcript": transcript,
                "features": json.dumps(features, default=str),
                "prior_questions": prior_qs_str,
            }
        )
        content = _strip_reasoning(resp.content)

    # Increment the clarify counter explicitly.  This is the authoritative
    # signal the intent router reads — do not rely on scanning messages
    # after checkpointer restore (message subclasses may deserialize as
    # dicts, silently zeroing an ``isinstance``-based counter).
    prev = state.get("clarify_count") or 0
    new_count = int(prev) + 1
    print(
        f"[clarify_node] asking Q#{new_count} (prev clarify_count={prev})",
        flush=True,
    )
    return {
        "messages": [AIMessage(content=content)],
        "clarify_count": new_count,
    }


@traceable(run_type="retriever", name="offer_selection_node")
def offer_selection_node(state: GraphState) -> Dict[str, Any]:
    conn = get_conn()
    try:
        init_schema(conn)
        seed_offers(conn)  # idempotent — cheap safety net
        candidates = select_offers(
            conn,
            customer=state.get("customer_record") or {},
            features=state.get("accumulated_features", {}),
            top_k=3,
        )
    finally:
        conn.close()

    selected = candidates[0] if candidates else None
    return {"selected_offer": selected, "candidate_offers": candidates}


@traceable(run_type="chain", name="offer_presentation_node")
def offer_presentation_node(state: GraphState) -> Dict[str, Any]:
    offer = state.get("selected_offer")
    customer = state.get("customer_record") or {}
    features = state.get("accumulated_features", {}) or {}
    transcript = _transcript(state.get("messages", []))

    if offer is None:
        # No offer available — apologise gracefully; still ends the turn.
        content = (
            "Thanks for the details — based on what you've shared I don't have a card that's a "
            "great fit right now, but I can connect you with a specialist who can look at "
            "options in more depth."
        )
        return {"messages": [AIMessage(content=content)]}

    if NBA_MOCK:
        content = (
            f"[OFFER PRESENTED: {offer['offer_id']}]\n\n"
            f"Based on what you've told me, I'd recommend our **{offer['offer_name']}**. "
            f"{offer['marketing_hook']} APR range is {offer['apr_range']}. "
            f"{offer['cta_text']}"
        )
    else:
        chain = offer_pitch_prompt | get_llm()
        resp = chain.invoke(
            {
                "full_name": customer.get("full_name", "there"),
                "risk_tier": customer.get("risk_tier", "MED"),
                "employment_status": features.get("employment_status", "UNKNOWN"),
                "transcript": transcript,
                "offer_id": offer["offer_id"],
                "offer_name": offer["offer_name"],
                "marketing_hook": offer["marketing_hook"],
                "rewards_summary": offer["rewards_summary"],
                "apr_range": offer["apr_range"],
                "cta_text": offer["cta_text"],
            }
        )
        content = _strip_reasoning(resp.content)

    return {
        "messages": [AIMessage(content=content)],
        "conversation_stage": "offer_presented",
    }


@traceable(run_type="chain", name="post_offer_chat_node")
def post_offer_chat_node(state: GraphState) -> Dict[str, Any]:
    offer = state.get("selected_offer") or {}
    transcript = _transcript(state.get("messages", []))

    if NBA_MOCK:
        content = (
            f"Happy to answer more about {offer.get('offer_name', 'the card')}. "
            "In a live demo this would use Nemotron to answer questions grounded on the offer's "
            "APR, rewards, and fees."
        )
    else:
        chain = post_offer_prompt | get_llm()
        resp = chain.invoke(
            {
                "offer_json": json.dumps(offer, default=str),
                "transcript": transcript,
            }
        )
        content = _strip_reasoning(resp.content)

    return {"messages": [AIMessage(content=content)], "conversation_stage": "post_offer"}


@traceable(run_type="chain", name="decline_node")
def decline_node(state: GraphState) -> Dict[str, Any]:
    transcript = _transcript(state.get("messages", []))

    if NBA_MOCK:
        content = (
            "Thanks for reaching out. For your protection I'm unable to continue this session "
            "in chat — please verify recent activity in your banking app or contact our support "
            "line to continue."
        )
    else:
        chain = decline_prompt | get_llm()
        content = _strip_reasoning(chain.invoke({"transcript": transcript}).content)

    return {"messages": [AIMessage(content=content)], "risk_blocked": True}


# ----------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------


def _build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("intake", intake_node)
    builder.add_node("risk_guardrail", risk_guardrail_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("select_offer", offer_selection_node)
    builder.add_node("offer_presentation", offer_presentation_node)
    builder.add_node("post_offer_chat", post_offer_chat_node)
    builder.add_node("decline", decline_node)

    builder.set_entry_point("intake")
    builder.add_edge("intake", "risk_guardrail")
    builder.add_conditional_edges(
        "risk_guardrail",
        risk_router,
        {"decline": "decline", "continue": "intent_router"},
    )
    builder.add_conditional_edges(
        "intent_router",
        stage_router,
        {
            "clarify": "clarify",
            "select_offer": "select_offer",
            "post_offer_chat": "post_offer_chat",
        },
    )
    builder.add_edge("select_offer", "offer_presentation")
    builder.add_edge("clarify", END)
    builder.add_edge("offer_presentation", END)
    builder.add_edge("post_offer_chat", END)
    builder.add_edge("decline", END)

    # MemorySaver keeps thread state alive within a single process; swap for
    # SqliteSaver to persist across app restarts.
    return builder.compile(checkpointer=MemorySaver())


# Compiled graph singleton.  ``MemorySaver`` holds every thread's checkpoint
# in the graph object itself, so all callers in a process must share one
# instance — rebuild it per request and no conversation would survive past its
# first turn.  One uvicorn worker therefore == one graph == one set of live
# threads; the evaluators and tests that import this module get the same
# singleton for free.
_graph_singleton = None


def get_graph():
    """Return the compiled LangGraph, built once per process.

    The ``MemorySaver`` checkpointer lives inside the compiled graph, so this
    single module-level singleton is what keeps every thread's conversation
    state alive for the lifetime of the uvicorn worker.  Restart the app to
    clear it.
    """
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = _build_graph()
    return _graph_singleton


# NOTE: no eager ``graph = get_graph()`` at module top level — building the
# graph is deferred so importers that only want the helpers (or that set
# ``NBA_MOCK`` after import, as the eval notebooks do) don't pay for it.  All
# callers go through ``get_graph()`` lazily.


# ----------------------------------------------------------------------
# Per-turn entry point (LangSmith root trace)
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="nba_turn")
def run_turn(
    user_input: str,
    thread_id: str,
    customer_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one turn of the chatbot.

    The @traceable decorator makes this the root of the per-turn trace in
    LangSmith.  The ``thread_id`` metadata is what LangSmith Threads groups
    on; ``configurable.thread_id`` is what LangGraph's checkpointer keys on.
    Setting both is required.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id, "customer_id": customer_id},
    }

    input_state = {"messages": [HumanMessage(content=user_input)]}
    # Seed customer_id only on the first invocation; the checkpointer keeps
    # it in subsequent turns.
    if customer_id is not None:
        input_state["customer_id"] = customer_id

    # Always resolve via ``get_graph()`` so every turn hits the same compiled
    # graph — and therefore the same MemorySaver — for the life of the
    # process.
    final_state = get_graph().invoke(input_state, config=config)

    rt = get_current_run_tree()
    run_id = str(rt.id) if rt else None

    return {
        "run_id": run_id,
        "assistant_message": final_state["messages"][-1].content
        if final_state.get("messages")
        else "",
        "messages": final_state.get("messages", []),
        "selected_offer": final_state.get("selected_offer"),
        "candidate_offers": final_state.get("candidate_offers"),
        "high_risk_probability": final_state.get("high_risk_probability"),
        "risk_blocked": final_state.get("risk_blocked", False),
        "conversation_stage": final_state.get("conversation_stage"),
        "accumulated_features": final_state.get("accumulated_features", {}),
        "customer_record": final_state.get("customer_record"),
        # Surfaced so the observability panel can confirm the cap counter is
        # advancing — if this stays at None across turns, the deployed app is
        # running a build older than commit 4968e00.
        "clarify_count": final_state.get("clarify_count"),
    }


# ----------------------------------------------------------------------
# Demo customer slate (served to the web UI by app/server.py)
# ----------------------------------------------------------------------


def demo_customer_picks(per_tier: int = 2) -> List[Dict[str, Any]]:
    """Return a small deterministic slate of customers spanning every risk tier.

    The web UI builds its customer dropdown from this, so demo users can hit
    the offer path (LOW/MED tiers) or the risk-decline path (HIGH tier) on
    purpose, rather than rolling dice against a random ``customer_id``.
    """
    conn = get_conn()
    try:
        init_schema(conn)
        rows = conn.execute(
            """
            SELECT customer_id, full_name, risk_tier, annual_income,
                   employment_status, existing_debt
            FROM customers
            WHERE risk_tier IN ('LOW','MED','HIGH')
            ORDER BY risk_tier, customer_id
            """
        ).fetchall()
    finally:
        conn.close()

    picks: Dict[str, List[Dict[str, Any]]] = {"LOW": [], "MED": [], "HIGH": []}
    for r in rows:
        tier = r["risk_tier"]
        if tier in picks and len(picks[tier]) < per_tier:
            picks[tier].append({k: r[k] for k in r.keys()})
    # Order: LOW → MED → HIGH so the dropdown flows from safest to riskiest.
    return [r for tier in ("LOW", "MED", "HIGH") for r in picks[tier]]
