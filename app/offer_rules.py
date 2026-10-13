#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
#  All rights reserved.
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""Rule engine for Next-Best-Action credit-card offer selection.

The engine is deliberately simple: a parameterized SQL filter over the
``offers`` table using customer + feature attributes, then a small Python
re-scoring pass that expresses the "why this offer is a good fit"
heuristic.  Kept as pure functions so it is trivially unit-testable and
usable from both the LangGraph app and the offline evaluation notebooks.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional


_FALLBACK_OFFER_ID = "CASHBACK_EVERYDAY"

# Rough credit-line "sweet spot" per offer, used to nudge scoring when a
# customer names a requested credit line.  Lower/upper are inclusive.
_CREDIT_LINE_BANDS: Dict[str, tuple] = {
    "PLAT_TRAVEL": (10_000, 100_000),
    "CASHBACK_EVERYDAY": (2_000, 25_000),
    "BALANCE_TRANSFER": (3_000, 30_000),
    "SECURED_BUILDER": (200, 3_000),
    "STUDENT_STARTER": (500, 5_000),
}

_REWARDS_HEAVY = {"PLAT_TRAVEL", "CASHBACK_EVERYDAY"}


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _base_query() -> str:
    return """
        SELECT * FROM offers
        WHERE (target_risk_tier = :risk_tier OR target_risk_tier = 'ANY')
          AND :age    >= min_age
          AND :age    <= max_age
          AND :income >= min_income
        ORDER BY priority ASC
    """


def _score(offer: Dict[str, Any], customer: Dict[str, Any], features: Dict[str, Any]) -> int:
    """Score a candidate offer against customer + accumulated features."""
    score = 100 - int(offer.get("priority", 100))

    # Exact tier match beats an ANY-tier catch-all.
    if offer["target_risk_tier"] == customer.get("risk_tier"):
        score += 30

    # Requested credit-line falls within the offer's typical band.
    requested = float(features.get("requested_credit_line", 0) or 0)
    if requested > 0:
        lo, hi = _CREDIT_LINE_BANDS.get(offer["offer_id"], (0, 10 ** 9))
        if lo <= requested <= hi:
            score += 10

    # Student-specific nudge.
    if (
        offer["offer_id"] == "STUDENT_STARTER"
        and str(features.get("employment_status", "")).upper() == "STUDENT"
    ):
        score += 5

    # High debt-to-income → discourage rewards-heavy cards.
    income = float(features.get("annual_income", 0) or customer.get("annual_income", 0) or 0)
    debt = float(features.get("existing_debt", 0) or customer.get("existing_debt", 0) or 0)
    if income > 0 and debt / income > 0.4 and offer["offer_id"] in _REWARDS_HEAVY:
        score -= 20

    return score


def select_offers(
    conn: sqlite3.Connection,
    customer: Dict[str, Any],
    features: Optional[Dict[str, Any]] = None,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """Return up to ``top_k`` offers ranked best-fit-first.

    Parameters
    ----------
    conn: open SQLite connection (see :func:`db.get_conn`).
    customer: row from the ``customers`` table (or a compatible dict).
    features: accumulated features extracted from the conversation.
    top_k: how many candidates to return.
    """
    features = features or {}

    # Prefer the LLM-extracted age/income if present; fall back to PII values.
    age = int(features.get("age") or customer.get("age") or 0)
    income = float(features.get("annual_income") or customer.get("annual_income") or 0)
    risk_tier = customer.get("risk_tier") or "MED"

    params = {"risk_tier": risk_tier, "age": age, "income": income}
    rows = conn.execute(_base_query(), params).fetchall()

    if not rows:
        # Fall back to the universal offer regardless of filter.
        fallback = conn.execute(
            "SELECT * FROM offers WHERE offer_id = ?", (_FALLBACK_OFFER_ID,)
        ).fetchone()
        return [_row_to_dict(fallback)] if fallback else []

    ranked = [
        (_score(_row_to_dict(r), customer, features), _row_to_dict(r)) for r in rows
    ]
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [offer for _score_val, offer in ranked[: max(1, top_k)]]


if __name__ == "__main__":
    # Smoke test: seed if needed and print top-3 for a synthetic profile.
    import json

    from db import get_conn, init_schema, seed_offers

    with get_conn() as conn:
        init_schema(conn)
        seed_offers(conn)
        customer = {"risk_tier": "LOW", "age": 42, "annual_income": 180_000, "existing_debt": 5_000}
        features = {"requested_credit_line": 25_000, "employment_status": "EMPLOYED"}
        picks = select_offers(conn, customer, features, top_k=3)
        print(json.dumps(picks, indent=2))
