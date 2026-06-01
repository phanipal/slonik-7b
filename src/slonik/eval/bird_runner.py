from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click
import yaml
from datasets import load_dataset
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from slonik.data.chatml import extract_sql, to_prompt
from slonik.data.schema import linearize
from slonik.training.exec_sandbox import execute_sqlite, results_equal


def _serve_completion(prompt: str, base_url: str, model: str) -> str:
    import httpx
    r = httpx.post(
        f"{base_url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 512, "temperature": 0.0, "stop": ["###", "<|im_end|>"]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def _difficulty_buckets(rows: list[dict]) -> dict[str, list[int]]:
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r.get("difficulty", "unknown")].append(i)
    return buckets


@click.command()
@click.option("--config", default="configs/datasets.yaml", type=click.Path(exists=True))
@click.option("--server", default="http://localhost:8000", help="vLLM OpenAI-compatible URL")
@click.option("--model", default="slonik-7b", help="Served model name")
@click.option("--out", default="outputs/bird_eval.json", type=click.Path())
@click.option("--limit", default=0, type=int)
@click.option("--dialect", default="sqlite", type=click.Choice(["sqlite", "postgresql"]))
def main(config: str, server: str, model: str, out: str, limit: int, dialect: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())["bird"]
    ds = load_dataset(cfg["dev_id"], split="dev")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    db_root = Path(f"data/raw/bird/dev_databases_{dialect}")
    if not db_root.exists():
        logger.warning(f"Database root not found: {db_root}. Download mini-dev DBs first.")

    correct = 0
    total = 0
    per_db: dict[str, list[bool]] = defaultdict(list)
    rows: list[dict] = []

    for ex in tqdm(ds, desc="BIRD eval"):
        db_id = ex["db_id"]
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue
        schema = linearize(db_path, with_samples=False)
        prompt = to_prompt(schema, ex["question"], _DummyTok(), ex.get("evidence", ""))

        try:
            completion = _serve_completion(prompt, server, model)
        except Exception as e:
            logger.error(f"Server error on {db_id}: {e}")
            continue

        pred_sql = extract_sql(completion)
        gold_sql = ex["SQL"]
        gold_res = execute_sqlite(db_path, gold_sql)
        pred_res = execute_sqlite(db_path, pred_sql)
        is_correct = pred_res.ok and gold_res.ok and results_equal(pred_res.rows, gold_res.rows)

        per_db[db_id].append(is_correct)
        correct += int(is_correct)
        total += 1
        rows.append({
            "db_id": db_id,
            "question": ex["question"],
            "gold": gold_sql,
            "pred": pred_sql,
            "correct": is_correct,
            "error": pred_res.error,
        })

    accuracy = correct / total if total else 0.0
    summary = {
        "model": model,
        "n_total": total,
        "n_correct": correct,
        "execution_accuracy": accuracy,
        "per_db": {k: sum(v) / len(v) for k, v in per_db.items()},
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    console = Console()
    table = Table(title=f"BIRD-SQL Eval — {model}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total", str(total))
    table.add_row("Correct", str(correct))
    table.add_row("Execution accuracy", f"{accuracy:.1%}")
    console.print(table)


class _DummyTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


if __name__ == "__main__":
    main()
