"""
SQLite-backed metrics store for persistent request tracking.
"""
import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

DB_PATH = Path("metrics.db")

_local = threading.local()


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            query       TEXT NOT NULL,
            answer      TEXT,
            sources     TEXT,          -- JSON list of source strings
            -- latency breakdown (ms)
            latency_retrieval_ms  REAL,
            latency_generation_ms REAL,
            latency_total_ms      REAL,
            -- tokens
            tokens_prompt     INTEGER,
            tokens_completion INTEGER,
            tokens_total      INTEGER,
            -- cost
            cost_usd   REAL,
            -- quality (0-1)
            faithfulness        REAL,
            answer_relevancy    REAL,
            context_precision   REAL,
            quality_avg         REAL,
            -- trace id
            trace_id   TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            run_label   TEXT,
            avg_faithfulness     REAL,
            avg_answer_relevancy REAL,
            avg_context_precision REAL,
            avg_quality          REAL,
            p50_latency_ms       REAL,
            p95_latency_ms       REAL,
            passed               INTEGER,  -- 1=pass, 0=fail
            report_json          TEXT
        );
        """)


def log_request(record: dict) -> int:
    """Insert a request record and return its id."""
    cols = ", ".join(record.keys())
    placeholders = ", ".join("?" for _ in record)
    vals = list(record.values())
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO requests ({cols}) VALUES ({placeholders})", vals
        )
        return cur.lastrowid


def get_recent_requests(n: int = 200) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_percentile_latency(window: int = 1000) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT latency_total_ms FROM requests "
            "WHERE latency_total_ms IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?",
            (window,),
        ).fetchall()
    if not rows:
        return {"p50": 0, "p95": 0}
    vals = sorted([r[0] for r in rows])
    n = len(vals)
    p50 = vals[int(n * 0.50)]
    p95 = vals[min(int(n * 0.95), n - 1)]
    return {"p50": round(p50, 1), "p95": round(p95, 1)}


def get_summary_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_requests,
                AVG(latency_total_ms) as avg_latency_ms,
                SUM(cost_usd) as total_cost_usd,
                AVG(quality_avg) as avg_quality,
                AVG(faithfulness) as avg_faithfulness,
                AVG(answer_relevancy) as avg_relevancy,
                AVG(context_precision) as avg_precision
            FROM requests
        """).fetchone()
    return dict(row) if row else {}


def log_eval_run(record: dict) -> int:
    cols = ", ".join(record.keys())
    placeholders = ", ".join("?" for _ in record)
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO eval_runs ({cols}) VALUES ({placeholders})",
            list(record.values()),
        )
        return cur.lastrowid


def get_eval_history(n: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]