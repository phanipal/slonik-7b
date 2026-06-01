from __future__ import annotations

import json
from pathlib import Path

import click
import sqlglot
from datasets import load_dataset
from loguru import logger


def _build_schema_hint(db_id: str, query: str) -> str:
    return f"-- database: {db_id}\n-- See Spider schema files for full DDL."


def _to_postgres(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite", write="postgres")[0]
    except Exception:
        return sql


@click.command()
@click.option("--out", default="data/processed/spider.jsonl", type=click.Path())
@click.option("--splits", default="train,validation")
def main(out: str, splits: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split in splits.split(","):
            ds = load_dataset("xlangai/spider", split=split.strip())
            for row in ds:
                f.write(json.dumps({
                    "schema": _build_schema_hint(row["db_id"], row["query"]),
                    "question": row["question"],
                    "sql": _to_postgres(row["query"]),
                    "evidence": "",
                    "db_id": row["db_id"],
                    "source": f"spider/{split}",
                }) + "\n")
                written += 1
    logger.info(f"Wrote {written} examples → {out}")


if __name__ == "__main__":
    main()
