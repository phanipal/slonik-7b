from __future__ import annotations

import multiprocessing as mp
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlglot


_DESTRUCTIVE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|attach|detach)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ExecResult:
    ok: bool
    rows: list[tuple] | None
    error: str = ""
    timed_out: bool = False


def _is_safe(sql: str) -> bool:
    return _DESTRUCTIVE.search(sql) is None


def _run_sqlite(db_path: str, sql: str, queue: mp.Queue) -> None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(sql)
        rows = cur.fetchmany(1000)
        conn.close()
        queue.put(ExecResult(ok=True, rows=rows))
    except Exception as e:
        queue.put(ExecResult(ok=False, rows=None, error=str(e)))


def execute_sqlite(db_path: str | Path, sql: str, timeout: float = 5.0) -> ExecResult:
    if not _is_safe(sql):
        return ExecResult(ok=False, rows=None, error="destructive statement blocked")

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_sqlite, args=(str(db_path), sql, queue), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExecResult(ok=False, rows=None, error="timeout", timed_out=True)

    return queue.get() if not queue.empty() else ExecResult(ok=False, rows=None, error="no result")


def execute_postgres(conn_str: str, sql: str, timeout: float = 5.0) -> ExecResult:
    if not _is_safe(sql):
        return ExecResult(ok=False, rows=None, error="destructive statement blocked")
    try:
        import psycopg
    except ImportError:
        return ExecResult(ok=False, rows=None, error="psycopg not installed")

    try:
        with psycopg.connect(conn_str, connect_timeout=int(timeout)) as conn:
            conn.execute(f"SET statement_timeout = {int(timeout * 1000)}")
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchmany(1000)
        return ExecResult(ok=True, rows=rows)
    except Exception as e:
        msg = str(e).lower()
        return ExecResult(ok=False, rows=None, error=str(e), timed_out="timeout" in msg)


def results_equal(a: list[tuple] | None, b: list[tuple] | None, order_matters: bool = False) -> bool:
    if a is None or b is None:
        return False
    if order_matters:
        return a == b
    return sorted(map(str, a)) == sorted(map(str, b))


def is_valid_syntax(sql: str, dialect: str = "postgres") -> bool:
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True
    except Exception:
        return False
