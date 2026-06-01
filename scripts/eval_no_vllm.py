"""
Slonik-7B eval via transformers + sqlglot equivalence.

Two-mode comparison:
  - When a SQLite database is available for db_id, executes both SQLs and compares results.
  - Otherwise, falls back to sqlglot-based structural normalization comparison.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import torch
from loguru import logger


def _resolve_db_path(db_id: str) -> Path | None:
    candidates = [
        Path(f"data/raw/bird/train_databases/train/train_databases/{db_id}/{db_id}.sqlite"),
        Path(f"data/raw/bird/train_databases/train_databases/{db_id}/{db_id}.sqlite"),
        Path(f"data/raw/bird/train_databases/{db_id}/{db_id}.sqlite"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _execute(db_path: Path, sql: str, timeout_s: float = 5.0):
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout_s)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, [f"{type(e).__name__}: {str(e)[:160]}"]


def _normalize_rows(rows: list) -> set:
    try:
        return {tuple(str(c) for c in r) for r in rows}
    except Exception:
        return {str(rows)}


def _sqlglot_equivalent(pred_sql: str, gold_sql: str) -> tuple[bool, str]:
    try:
        import sqlglot
    except ImportError:
        return False, "sqlglot not installed"
    try:
        p = sqlglot.parse_one(pred_sql, dialect="postgres")
        g = sqlglot.parse_one(gold_sql, dialect="postgres")
        if p is None or g is None:
            return False, "parse failure"
        p_norm = p.sql(dialect="postgres", normalize=True, comments=False)
        g_norm = g.sql(dialect="postgres", normalize=True, comments=False)
        if p_norm == g_norm:
            return True, "exact"
        p_tokens = set(str(t).lower() for t in p.walk())
        g_tokens = set(str(t).lower() for t in g.walk())
        overlap = len(p_tokens & g_tokens) / max(len(g_tokens), 1)
        return overlap >= 0.85, f"token_overlap={overlap:.2f}"
    except Exception as e:
        return False, f"err: {type(e).__name__}"


def _extract_sql(text: str) -> str:
    if "```sql" in text:
        sql = text.split("```sql")[1].split("```")[0]
    elif "```" in text:
        sql = text.split("```")[1].split("```")[0]
    else:
        sql = text
    return sql.strip().rstrip(";")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft-merged")
    ap.add_argument("--eval-file", default="data/processed/eval.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="outputs/eval_results.jsonl")
    args = ap.parse_args()

    examples = []
    with Path(args.eval_file).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if args.limit > 0:
        examples = examples[: args.limit]
    logger.info(f"Loaded {len(examples)} examples from {args.eval_file}")

    logger.info(f"Loading model from {args.model}")
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": 0, "exec_correct": 0, "exec_wrong": 0, "exec_error": 0,
        "sqlglot_equiv": 0, "sqlglot_diff": 0, "sqlglot_unparseable": 0,
    }
    t_start = time.time()

    with Path(args.out).open("w") as fout:
        for i, ex in enumerate(examples, 1):
            stats["total"] += 1
            question = ex.get("question", "")
            schema = ex.get("schema", "")
            gold_sql = (ex.get("sql") or "").strip().rstrip(";")
            db_id = ex.get("db_id", "")

            prompt = tok.apply_chat_template(
                [{"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
                )
            full = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            pred_sql = _extract_sql(full)

            row = {
                "i": i, "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "pred_sql": pred_sql,
            }

            db_path = _resolve_db_path(db_id)
            if db_path is not None:
                gold_ok, gold_rows = _execute(db_path, gold_sql + ";")
                pred_ok, pred_rows = _execute(db_path, pred_sql + ";")
                row["mode"] = "exec"
                if not pred_ok:
                    row["status"] = "exec_error"; row["error"] = pred_rows[0]
                    stats["exec_error"] += 1
                elif _normalize_rows(pred_rows) == _normalize_rows(gold_rows):
                    row["status"] = "exec_correct"
                    stats["exec_correct"] += 1
                else:
                    row["status"] = "exec_wrong"
                    stats["exec_wrong"] += 1
            else:
                equiv, reason = _sqlglot_equivalent(pred_sql, gold_sql)
                row["mode"] = "sqlglot"; row["reason"] = reason
                if reason == "parse failure":
                    row["status"] = "unparseable"
                    stats["sqlglot_unparseable"] += 1
                elif equiv:
                    row["status"] = "sqlglot_equiv"
                    stats["sqlglot_equiv"] += 1
                else:
                    row["status"] = "sqlglot_diff"
                    stats["sqlglot_diff"] += 1

            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if i % 10 == 0 or i == len(examples):
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (len(examples) - i) / max(rate, 1e-9)
                exec_total = stats["exec_correct"] + stats["exec_wrong"] + stats["exec_error"]
                sqlglot_total = stats["sqlglot_equiv"] + stats["sqlglot_diff"] + stats["sqlglot_unparseable"]
                exec_acc = stats["exec_correct"] / max(exec_total, 1) * 100
                sqlglot_acc = stats["sqlglot_equiv"] / max(sqlglot_total, 1) * 100
                logger.info(
                    f"  {i}/{len(examples)}  exec_acc={exec_acc:.1f}% ({stats['exec_correct']}/{exec_total})  "
                    f"sqlglot_acc={sqlglot_acc:.1f}% ({stats['sqlglot_equiv']}/{sqlglot_total})  "
                    f"rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                )

    elapsed = time.time() - t_start
    exec_total = stats["exec_correct"] + stats["exec_wrong"] + stats["exec_error"]
    sqlglot_total = stats["sqlglot_equiv"] + stats["sqlglot_diff"] + stats["sqlglot_unparseable"]
    logger.info("\n=== RESULTS ===")
    logger.info(f"  Total: {stats['total']}")
    if exec_total > 0:
        logger.info(
            f"  Execution mode ({exec_total} examples): "
            f"correct={stats['exec_correct']} ({stats['exec_correct']/exec_total*100:.1f}%)  "
            f"wrong={stats['exec_wrong']}  error={stats['exec_error']}"
        )
    if sqlglot_total > 0:
        logger.info(
            f"  Sqlglot mode ({sqlglot_total} examples): "
            f"equiv={stats['sqlglot_equiv']} ({stats['sqlglot_equiv']/sqlglot_total*100:.1f}%)  "
            f"diff={stats['sqlglot_diff']}  unparseable={stats['sqlglot_unparseable']}"
        )
    logger.info(f"  Elapsed: {elapsed/60:.1f} min")
    logger.info(f"  Output: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
