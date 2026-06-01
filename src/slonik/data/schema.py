from __future__ import annotations

import re
import sqlite3
from pathlib import Path


_CREATE_RE = re.compile(r"CREATE\s+TABLE\s+.*?;", re.IGNORECASE | re.DOTALL)


def from_sqlite(db_path: str | Path) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        stmts = [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()

    return "\n\n".join(s.strip().rstrip(";") + ";" for s in stmts)


def from_ddl_string(ddl: str) -> str:
    statements = _CREATE_RE.findall(ddl)
    return "\n\n".join(s.strip() for s in statements) if statements else ddl.strip()


def sample_rows_block(db_path: str | Path, n: int = 3) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        for t in tables:
            try:
                rows = conn.execute(f'SELECT * FROM "{t}" LIMIT {n}').fetchall()
                cols = [d[0] for d in conn.execute(f'SELECT * FROM "{t}" LIMIT 1').description]
            except sqlite3.Error:
                continue
            if not rows:
                continue
            out.append(f"-- {t} sample rows")
            out.append("-- " + " | ".join(cols))
            for r in rows:
                out.append("-- " + " | ".join(str(v)[:40] for v in r))
            out.append("")
    finally:
        conn.close()
    return "\n".join(out)


def linearize(db_path: str | Path, with_samples: bool = True) -> str:
    schema = from_sqlite(db_path)
    if not with_samples:
        return schema
    samples = sample_rows_block(db_path, n=3)
    return f"{schema}\n\n{samples}" if samples else schema
