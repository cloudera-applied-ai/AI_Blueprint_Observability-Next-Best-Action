#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""SQLite backing store for the NBA demo.

Provides the DB path, connection helper, schema initializer, and idempotent
seeders for the ``customers`` and ``offers`` tables.  The offer catalog is
fixed (5 rows with stable IDs) so ``INSERT OR IGNORE`` makes re-seeding a
no-op.  Customer seeding is delegated to :mod:`pii_datagen`.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional


# The SQLite file lives at the project root (``/home/cdsw/nba_demo.db`` on
# Cloudera AI) so both the app code under ``app/`` and the eval / training
# notebooks at the repo root read from the same store.  Override with
# ``NBA_DB_PATH`` if you need a different location.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get(
    "NBA_DB_PATH",
    os.path.join(_PROJECT_ROOT, "nba_demo.db"),
)


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a connection with ``row_factory`` set for dict-like access."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Safe to call every launch."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id       INTEGER PRIMARY KEY,
            full_name         TEXT NOT NULL,
            email             TEXT NOT NULL,
            phone_number      TEXT,
            street_address    TEXT,
            city              TEXT,
            state             TEXT,
            zip_code          TEXT,
            company           TEXT,
            job_title         TEXT,
            age               INTEGER,
            annual_income     REAL,
            employment_status TEXT,
            existing_debt     REAL,
            risk_tier         TEXT CHECK(risk_tier IN ('LOW','MED','HIGH'))
        );

        CREATE TABLE IF NOT EXISTS offers (
            offer_id         TEXT PRIMARY KEY,
            offer_name       TEXT NOT NULL,
            target_risk_tier TEXT NOT NULL,
            min_age          INTEGER DEFAULT 0,
            max_age          INTEGER DEFAULT 200,
            min_income       REAL    DEFAULT 0,
            apr_range        TEXT,
            rewards_summary  TEXT,
            marketing_hook   TEXT,
            cta_text         TEXT,
            priority         INTEGER DEFAULT 100
        );

        CREATE INDEX IF NOT EXISTS idx_customers_risk ON customers(risk_tier);
        """
    )
    conn.commit()


# Fixed offer catalog. offer_id is stable so re-seeding is a no-op.
_OFFERS = [
    dict(
        offer_id="PLAT_TRAVEL",
        offer_name="Platinum Travel Rewards",
        target_risk_tier="LOW",
        min_age=25,
        max_age=200,
        min_income=100_000,
        apr_range="16.99% - 22.99%",
        rewards_summary="3x points on flights, hotels and dining; annual travel credit; airport lounge access.",
        marketing_hook="Earn 3x points on flights, hotels, and dining, plus lounge access worldwide.",
        cta_text="Ready to unlock premium travel perks?",
        priority=10,
    ),
    dict(
        offer_id="CASHBACK_EVERYDAY",
        offer_name="Cashback Everyday",
        target_risk_tier="ANY",
        min_age=21,
        max_age=200,
        min_income=30_000,
        apr_range="18.99% - 25.99%",
        rewards_summary="Flat 2% cash back on every purchase, no rotating categories.",
        marketing_hook="2% cash back on every purchase, every day, no exceptions.",
        cta_text="Want a card that pays you back on everything?",
        priority=30,
    ),
    dict(
        offer_id="BALANCE_TRANSFER",
        offer_name="Balance Transfer Saver",
        target_risk_tier="MED",
        min_age=21,
        max_age=200,
        min_income=40_000,
        apr_range="0% intro for 15 months, then 19.99%",
        rewards_summary="0% intro APR on balance transfers for 15 months; no annual fee.",
        marketing_hook="Consolidate existing debt with 0% APR for 15 months and save on interest.",
        cta_text="Want to see how much you could save consolidating balances?",
        priority=20,
    ),
    dict(
        offer_id="SECURED_BUILDER",
        offer_name="Secured Credit Builder",
        target_risk_tier="HIGH",
        min_age=18,
        max_age=200,
        min_income=0,
        apr_range="24.99%",
        rewards_summary="Refundable security deposit sets your limit; credit-bureau reporting to help rebuild.",
        marketing_hook="Rebuild your credit with a refundable deposit and monthly bureau reporting.",
        cta_text="Would you like help getting started on rebuilding your credit?",
        priority=40,
    ),
    dict(
        offer_id="STUDENT_STARTER",
        offer_name="Student Starter",
        target_risk_tier="ANY",
        min_age=18,
        max_age=26,
        min_income=0,
        apr_range="21.99%",
        rewards_summary="No annual fee, 1% cash back on all purchases, credit-building tools.",
        marketing_hook="Build credit while you study — no annual fee and 1% cash back.",
        cta_text="Want a card designed for building credit as a student?",
        priority=25,
    ),
]


def seed_offers(conn: sqlite3.Connection) -> int:
    """Insert the fixed offer catalog. Idempotent via ``INSERT OR IGNORE``.

    Returns the count of rows *after* seeding.
    """
    cols = list(_OFFERS[0].keys())
    placeholders = ",".join(f":{c}" for c in cols)
    sql = f"INSERT OR IGNORE INTO offers ({','.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, _OFFERS)
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0]


def seed_customers(conn: sqlite3.Connection, n: int = 10_000, seed: int = 42) -> int:
    """Delegated to :mod:`pii_datagen` so this module stays Faker-free.

    Keeping the seeding logic there means ``db`` can be imported anywhere
    (including inside the LangGraph app) without pulling Faker onto the
    critical path.
    """
    from pii_datagen import seed_customers as _impl
    return _impl(conn, n=n, seed=seed)


if __name__ == "__main__":
    # Convenience: `python db.py` prints DB path and table counts.
    with get_conn() as _c:
        init_schema(_c)
        seed_offers(_c)
        offers = _c.execute("SELECT COUNT(*) FROM offers").fetchone()[0]
        customers = _c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        print(f"DB path : {DB_PATH}")
        print(f"offers  : {offers}")
        print(f"customers: {customers}")
