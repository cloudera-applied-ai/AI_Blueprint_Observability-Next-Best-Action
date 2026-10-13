#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""SQLite-backed customer + offer seeder for the NBA demo.

Replaces the previous Spark / dbldatagen implementation.  Uses ``Faker`` to
generate synthetic PII directly into a local SQLite file so the demo runs
inside a single Cloudera AI container without a Spark connection.

Usage:
    python pii_datagen.py --rows 500
    python pii_datagen.py --ensure          # skip if DB already seeded

Idempotency:
    * ``init_schema`` uses CREATE TABLE IF NOT EXISTS.
    * ``seed_offers`` uses INSERT OR IGNORE on stable offer_ids.
    * ``seed_customers`` short-circuits when the table already has >= n rows.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
from typing import Iterable

from faker import Faker

from db import DB_PATH, get_conn, init_schema, seed_offers


# ----------------------------------------------------------------------
# Customer generation
# ----------------------------------------------------------------------

# Risk-tier weights preserved from the original Spark generator.
_RISK_TIERS = ["LOW", "MED", "HIGH"]
_RISK_WEIGHTS = [70, 20, 10]

_EMP_STATUSES = ["EMPLOYED", "SELF_EMPLOYED", "STUDENT", "UNEMPLOYED", "RETIRED"]
_EMP_WEIGHTS = [60, 15, 10, 5, 10]


def _sample_income(tier: str, rng: random.Random) -> float:
    """Sample an annual income conditioned on the customer's risk tier."""
    # LOW risk = healthier balance sheet = higher income band.
    if tier == "LOW":
        return round(rng.uniform(60_000, 250_000), 2)
    if tier == "MED":
        return round(rng.uniform(35_000, 80_000), 2)
    return round(rng.uniform(0, 40_000), 2)


def _sample_age(emp: str, rng: random.Random) -> int:
    return rng.randint(18, 26) if emp == "STUDENT" else rng.randint(18, 80)


def _customer_rows(
    start_id: int, end_id: int, seed: int
) -> Iterable[tuple]:
    """Yield tuples matching the ``customers`` column order."""
    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)

    for i in range(start_id, end_id + 1):
        tier = rng.choices(_RISK_TIERS, weights=_RISK_WEIGHTS, k=1)[0]
        emp = rng.choices(_EMP_STATUSES, weights=_EMP_WEIGHTS, k=1)[0]
        age = _sample_age(emp, rng)
        income = _sample_income(tier, rng)
        existing_debt = round(rng.uniform(0, max(income * 0.6, 1)), 2)

        yield (
            i,
            fake.name(),
            fake.email(),
            fake.phone_number(),
            fake.street_address(),
            fake.city(),
            fake.state_abbr(),
            fake.postcode(),
            fake.company(),
            fake.job(),
            age,
            income,
            emp,
            existing_debt,
            tier,
        )


def seed_customers(conn: sqlite3.Connection, n: int = 10_000, seed: int = 42) -> int:
    """Seed the ``customers`` table up to ``n`` rows. Idempotent.

    Returns the row count after seeding.
    """
    existing = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if existing >= n:
        print(f"[pii_datagen] customers already has {existing} rows (>= {n}); skipping seed.")
        return existing

    print(f"[pii_datagen] seeding customers {existing + 1}..{n} ...")
    rows = list(_customer_rows(existing + 1, n, seed=seed))
    conn.executemany(
        """
        INSERT INTO customers (
            customer_id, full_name, email, phone_number, street_address,
            city, state, zip_code, company, job_title,
            age, annual_income, employment_status, existing_debt, risk_tier
        ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    print(f"[pii_datagen] customers total after seed: {total}")
    return total


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the NBA demo SQLite database.")
    parser.add_argument(
        "--rows",
        type=int,
        default=int(os.environ.get("NBA_CUSTOMER_ROWS", "10000")),
        help="Target number of customer rows (default 10000 or $NBA_CUSTOMER_ROWS).",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Only create schema and seed if missing; safe to run on every app launch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Faker/RNG seed.")
    args = parser.parse_args()

    with get_conn() as conn:
        init_schema(conn)
        n_offers = seed_offers(conn)
        n_customers = seed_customers(conn, n=args.rows, seed=args.seed)
        print(
            f"[pii_datagen] DB ready at {DB_PATH} "
            f"(offers={n_offers}, customers={n_customers})"
        )


if __name__ == "__main__":
    main()
