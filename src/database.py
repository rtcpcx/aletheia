"""
Aletheia â€” src/database.py

Single shared MySQL connection layer. Every other src module reads/writes
through here so that raw-table immutability and analysis-table write
boundaries can be enforced and audited in one place.

Credentials come from environment variables set in Section 1.4 of the
build plan:
    ALETHEIA_DB_HOST
    ALETHEIA_DB_PORT
    ALETHEIA_DB_USER
    ALETHEIA_DB_PASSWORD
"""

from __future__ import annotations

import decimal
import os
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import mysql.connector
import pandas as pd
from mysql.connector import pooling

_POOL: pooling.MySQLConnectionPool | None = None

# Tables written only by the benchmark ingestion layer.
# Production analysis code must never INSERT/UPDATE/DELETE raw business data.
RAW_TABLES = frozenset(
    {
        "raw.sales",
        "raw.marketing",
        "raw.market_context",
        "raw.customer_success",
        "raw.business_calendar",
        "raw.metric_registry",
        "raw.source_health"
    }
)

# The only table LLM retrieval is permitted to write to.
RETRIEVAL_WRITE_TABLE = "analysis.retrieved_context"


def _build_pool() -> pooling.MySQLConnectionPool:
    return pooling.MySQLConnectionPool(
        pool_name="aletheia_pool",
        pool_size=5,
        host=os.environ.get("ALETHEIA_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("ALETHEIA_DB_PORT", "3306")),
        user=os.environ.get("ALETHEIA_DB_USER", "root"),
        password=os.environ.get("ALETHEIA_DB_PASSWORD", ""),
        autocommit=False,
    )


def get_pool() -> pooling.MySQLConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()
    return _POOL


@contextmanager
def get_connection() -> Iterator[mysql.connector.MySQLConnection]:
    conn = get_pool().get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(dictionary: bool = True) -> Iterator[Any]:
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=dictionary)
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def _coerce_decimals(row: dict[str, Any]) -> dict[str, Any]:
    """
    mysql-connector-python returns DECIMAL/NUMERIC columns as
    decimal.Decimal, not float. When a LEFT JOIN leaves a DECIMAL cell
    NULL, that becomes NaN (a float) in the same pandas column, mixing
    Decimal and float in one object-dtype series. Python's decimal module
    deliberately refuses to mix the two in arithmetic (e.g. Series.diff()),
    which is exactly the TypeError this fixes. Only Decimal values are
    touched; strings, dates, and None pass through untouched.
    """
    return {
        k: (float(v) if isinstance(v, decimal.Decimal) else v)
        for k, v in row.items()
    }


def query_df(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    """
    Run a read query and return a pandas DataFrame.

    Deliberately does NOT use pd.read_sql(): pandas only documents that
    function as tested against a SQLAlchemy engine/connection or a sqlite3
    DBAPI2 connection. Passing it a raw mysql-connector connection works in
    practice but is explicitly undocumented/untested by pandas and emits a
    UserWarning on every call. Fetching rows via the cursor and building the
    DataFrame directly avoids that dependency entirely.
    """
    with get_cursor(dictionary=True) as cursor:
        cursor.execute(sql, params or ())
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

    if not rows:
        return pd.DataFrame(columns=columns)
    rows = [_coerce_decimals(r) for r in rows]
    return pd.DataFrame(rows, columns=columns)


def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    """
    Run a single write statement (INSERT/UPDATE/DELETE/REPLACE).
    Returns the number of affected rows.
    """
    with get_cursor(dictionary=False) as cursor:
        cursor.execute(sql, params or ())
        return cursor.rowcount


def executemany(sql: str, rows: Sequence[Sequence[Any]]) -> int:
    """Bulk insert/update. Returns the number of affected rows."""
    if not rows:
        return 0
    with get_cursor(dictionary=False) as cursor:
        cursor.executemany(sql, rows)
        return cursor.rowcount


def check_connection() -> dict:
    """
    Quick diagnostic: verifies the pool can actually reach MySQL and that
    every expected schema exists. Safe to call from a Python shell:

        python3 -c "from src.database import check_connection; print(check_connection())"
    """
    result: dict[str, Any] = {"connected": False, "schemas": {}, "error": None}
    try:
        with get_cursor(dictionary=True) as cursor:
            cursor.execute("SELECT 1 AS ok")
            cursor.fetchone()
        result["connected"] = True

        for schema in ("raw", "mart", "analysis", "app"):
            with get_cursor(dictionary=True) as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS n FROM information_schema.tables "
                    "WHERE table_schema = %s",
                    (schema,),
                )
                row = cursor.fetchone()
                result["schemas"][schema] = int(row["n"]) if row else 0
    except Exception as exc:  # noqa: BLE001 - diagnostic helper, report anything
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def describe_pool() -> dict:
    """Returns basic pool config for troubleshooting (never includes the password)."""
    return {
        "host": os.environ.get("ALETHEIA_DB_HOST", "127.0.0.1"),
        "port": os.environ.get("ALETHEIA_DB_PORT", "3306"),
        "user": os.environ.get("ALETHEIA_DB_USER", "root"),
        "password_set": bool(os.environ.get("ALETHEIA_DB_PASSWORD")),
        "pool_size": 5,
    }


def assert_not_raw_write(target_table: str) -> None:
    """
    Guardrail helper: call this at the top of any write path that is not
    the demo data generator / production ingestion loader, to make it
    impossible to accidentally mutate raw.* from analysis code.
    """
    if target_table.lower() in RAW_TABLES:
        raise PermissionError(
            f"Refusing to write to '{target_table}' from analysis code. "
            f"raw.* tables are immutable outside the data ingestion loader."
        )


def table_checksum(table: str) -> str:
    """
    Cheap deterministic checksum of a table's contents, used by the
    retrieval-isolation test (eval/validate_against_truth.py) to prove that
    running orchestration + retrieval never mutates mart.* or raw.*.
    """
    sql = f"CHECKSUM TABLE {table}"
    with get_cursor(dictionary=True) as cursor:
        cursor.execute(sql)
        row = cursor.fetchone()
        return str(row["Checksum"]) if row else "EMPTY"

