from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from slonik.training.exec_sandbox import execute_sqlite, is_valid_syntax, results_equal


@pytest.fixture()
def tiny_db():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript("CREATE TABLE t(id INT, name TEXT); INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');")
        conn.commit()
        conn.close()
        yield path


def test_select_runs(tiny_db):
    res = execute_sqlite(tiny_db, "SELECT id FROM t ORDER BY id")
    assert res.ok
    assert res.rows == [(1,), (2,), (3,)]


def test_destructive_blocked(tiny_db):
    res = execute_sqlite(tiny_db, "DROP TABLE t")
    assert not res.ok
    assert "destructive" in res.error


def test_timeout_kills(tiny_db):
    res = execute_sqlite(tiny_db, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT * FROM c", timeout=0.5)
    assert not res.ok
    assert res.timed_out or "error" in res.error.lower()


def test_results_equal_unordered():
    assert results_equal([(1,), (2,)], [(2,), (1,)])
    assert not results_equal([(1,), (2,)], [(1,), (3,)])


def test_syntax_validator():
    assert is_valid_syntax("SELECT 1")
    assert not is_valid_syntax("SELEC 1 FROM")
