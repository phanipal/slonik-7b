from __future__ import annotations

import json
from pathlib import Path

import click
from datasets import load_dataset
from loguru import logger
from rich.console import Console
from tqdm import tqdm

from slonik.data.chatml import extract_sql
from slonik.eval.bird_runner import _DummyTok, _serve_completion
from slonik.training.exec_sandbox import execute_sqlite, results_equal


@click.command()
@click.option("--server", default="http://localhost:8000")
@click.option("--model", default="slonik-7b")
@click.option("--out", default="outputs/spider_eval.json", type=click.Path())
@click.option("--db-root", default="data/raw/spider/database", type=click.Path())
@click.option("--limit", default=0, type=int)
def main(server: str, model: str, out: str, db_root: str, limit: int) -> None:
    ds = load_dataset("xlangai/spider", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    rows: list[dict] = []
    tok = _DummyTok()

    for ex in tqdm(ds, desc="Spider eval"):
        db_path = Path(db_root) / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            continue

        from slonik.data.schema import linearize
        from slonik.data.chatml import to_prompt

        prompt = to_prompt(linearize(db_path, with_samples=False), ex["question"], tok)
        try:
            completion = _serve_completion(prompt, server, model)
        except Exception as e:
            logger.error(f"Server error on {ex['db_id']}: {e}")
            continue

        pred_sql = extract_sql(completion)
        gold_res = execute_sqlite(db_path, ex["query"])
        pred_res = execute_sqlite(db_path, pred_sql)
        is_correct = pred_res.ok and gold_res.ok and results_equal(pred_res.rows, gold_res.rows)

        correct += int(is_correct)
        total += 1
        rows.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold": ex["query"],
            "pred": pred_sql,
            "correct": is_correct,
        })

    accuracy = correct / total if total else 0.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model": model,
        "n_total": total,
        "n_correct": correct,
        "execution_accuracy": accuracy,
        "rows": rows,
    }, indent=2))

    Console().print(f"Spider: {correct}/{total} = {accuracy:.1%}")


if __name__ == "__main__":
    main()
