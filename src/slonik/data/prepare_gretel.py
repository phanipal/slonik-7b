from __future__ import annotations

import json
import re
from pathlib import Path

import click
from datasets import load_dataset
from loguru import logger


_PG_PATTERNS = [
    re.compile(r"\bjsonb?\b", re.I),
    re.compile(r"\bilike\b", re.I),
    re.compile(r"\bgenerate_series\b", re.I),
    re.compile(r"\barray\[", re.I),
    re.compile(r"\bcte\b|\bwith\s+\w+\s+as\b", re.I),
    re.compile(r"\bover\s*\(", re.I),
    re.compile(r"\bnow\(\)|current_timestamp", re.I),
]


def _is_pg_flavored(sql: str, prompt: str) -> bool:
    blob = f"{prompt}\n{sql}".lower()
    if "sqlite" in blob or "mysql" in blob or "tsql" in blob:
        return False
    return any(p.search(blob) for p in _PG_PATTERNS)


@click.command()
@click.option("--out", default="data/processed/gretel_pg.jsonl", type=click.Path())
@click.option("--max-rows", default=15000, type=int)
def main(out: str, max_rows: int) -> None:
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            sql = row.get("sql") or row.get("output", "")
            prompt = row.get("sql_prompt") or row.get("instruction", "")
            schema = row.get("sql_context") or row.get("input", "")
            if not (sql and prompt):
                continue
            if not _is_pg_flavored(sql, prompt):
                continue
            f.write(json.dumps({
                "schema": schema,
                "question": prompt,
                "sql": sql,
                "evidence": "",
                "db_id": "gretel",
                "source": "gretel",
            }) + "\n")
            written += 1
            if written >= max_rows:
                break
    logger.info(f"Wrote {written} PG-flavored examples → {out}")


if __name__ == "__main__":
    main()
