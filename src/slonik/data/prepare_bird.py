from __future__ import annotations

import json
import zipfile
from pathlib import Path

import click
import sqlglot
import yaml
from datasets import load_dataset
from loguru import logger

from slonik.data.chatml import Example
from slonik.data.schema import linearize


def _to_postgres(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite", write="postgres")[0]
    except Exception:
        return sql


def _row_to_example(row: dict, db_root: Path) -> Example | None:
    db_path = db_root / row["db_id"] / f"{row['db_id']}.sqlite"
    if not db_path.exists():
        return None
    schema = linearize(db_path, with_samples=True)
    return Example(
        schema=schema,
        question=row["question"],
        sql=_to_postgres(row["SQL"]),
        evidence=row.get("evidence", ""),
        db_id=row["db_id"],
    )


def _ensure_databases(zip_url: str | None, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        return
    target.mkdir(parents=True, exist_ok=True)
    archive = target.parent / "bird_train.zip"
    if not archive.exists() and zip_url:
        import httpx
        logger.info(f"Downloading {zip_url}")
        with httpx.stream("GET", zip_url, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=2**20):
                    f.write(chunk)
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)


@click.command()
@click.option("--config", default="configs/datasets.yaml", type=click.Path(exists=True))
@click.option("--out", default="data/processed/bird_train.jsonl", type=click.Path())
@click.option("--limit", default=0, type=int, help="0 = all rows")
def main(config: str, out: str, limit: int) -> None:
    cfg = yaml.safe_load(Path(config).read_text())["bird"]
    db_root = Path("data/raw/bird/train_databases/train/train_databases")
    if not db_root.exists():
        db_root = Path("data/raw/bird/train_databases")
    _ensure_databases(cfg.get("train_databases_url"), db_root)

    ds = load_dataset(cfg["train_id"], split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            ex = _row_to_example(row, db_root)
            if ex is None:
                continue
            f.write(json.dumps({
                "schema": ex.schema,
                "question": ex.question,
                "sql": ex.sql,
                "evidence": ex.evidence,
                "db_id": ex.db_id,
                "source": "bird",
            }) + "\n")
            written += 1

    logger.info(f"Wrote {written} examples → {out}")


if __name__ == "__main__":
    main()
