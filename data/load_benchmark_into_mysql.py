"""
Aletheia — data/load_benchmark_into_mysql.py

Load one benchmark split into raw.*.

Usage:
    python data/load_benchmark_into_mysql.py --split dev
    python data/load_benchmark_into_mysql.py --split holdout

Default data root: data/generated

This is an explicit full-reload ingestion path. By default it TRUNCATEs the raw
benchmark tables first so repeated benchmark runs cannot silently duplicate data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

# Allow direct execution from the repository root:
#     python data\load_benchmark_into_mysql.py --split dev
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import database  # noqa: E402


TABLES = [
    (
        "sales.csv",
        "raw.sales",
        [
            "sale_date", "region", "product_id", "units_sold", "list_price",
            "unit_cost", "discount_pct", "net_revenue", "inventory_units",
            "reorder_point", "source_loaded_at",
        ],
    ),
    (
        "marketing.csv",
        "raw.marketing",
        [
            "activity_date", "region", "channel", "spend", "impressions",
            "sessions", "orders", "new_customers", "source_loaded_at",
        ],
    ),
    (
        "market_context.csv",
        "raw.market_context",
        [
            "week_start", "region", "competitor_price_index", "weather_index",
            "event_type", "source_type", "source_loaded_at",
        ],
    ),
    (
        "customer_success.csv",
        "raw.customer_success",
        [
            "activity_date", "region", "active_customers", "churned_customers",
            "support_tickets", "platform_uptime_pct", "source_loaded_at",
        ],
    ),
    (
        "business_calendar.csv",
        "raw.business_calendar",
        [
            "calendar_date", "region", "event_name", "event_type",
            "expected_direction", "is_known_in_advance", "notes",
            "source_loaded_at",
        ],
    ),
    (
        "metric_registry.csv",
        "raw.metric_registry",
        [
            "metric_name", "definition_version", "effective_start",
            "effective_end", "formula_text", "grain", "owner_team",
            "is_current", "source_loaded_at",
        ],
    ),
    (
        "source_health.csv",
        "raw.source_health",
        [
            "health_date", "source_name", "region",
            "expected_cadence_hours", "observed_lag_hours",
            "completeness_pct", "schema_version", "status", "notes",
            "source_loaded_at",
        ],
    ),
]


def none_if_nan(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def truncate_tables() -> None:
    # Child/metadata tables have no FKs, but truncate in reverse conceptual order.
    for _, table, _ in reversed(TABLES):
        database.execute(f"TRUNCATE TABLE {table}")


def load_csv(base: Path, csv_name: str, table: str, columns: list[str]) -> int:
    path = base / csv_name
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} missing columns: {missing}")

    df = df[columns]
    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

    rows = [
        tuple(none_if_nan(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    if rows:
        database.executemany(sql, rows)
    return len(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "holdout"], required=True)
    p.add_argument("--data-root", default="data/generated")
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate raw tables before load. Not recommended for benchmark runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.data_root) / args.split

    if not args.no_truncate:
        truncate_tables()

    total = 0
    for csv_name, table, columns in TABLES:
        n = load_csv(base, csv_name, table, columns)
        total += n
        print(f"Loaded {n:>6} rows -> {table}")

    print(f"Loaded {total} total rows for split={args.split}")
    print("Verify:")
    print("  SELECT COUNT(*) FROM mart.daily_kpi_evidence;")
    print("  SELECT * FROM mart.source_health_latest ORDER BY source_name, region;")


if __name__ == "__main__":
    main()
