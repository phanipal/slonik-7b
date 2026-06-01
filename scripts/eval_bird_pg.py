"""
BIRD Mini-Dev PostgreSQL benchmark for Slonik-7B.

For each of the 500 questions:
  1. Look up db_id in dev_tables.json → build CREATE TABLE schema string.
  2. Send schema + question + evidence to the model.
  3. Extract SQL from model output.
  4. Execute predicted SQL and gold SQL against the loaded Postgres.
  5. Compare result sets (order-independent unless ORDER BY in either).

Reports execution accuracy overall and by difficulty.

Prereqs:
  - Postgres running with all BIRD tables loaded (see setup steps).
  - `pip install psycopg2-binary` (or psycopg2).
  - dev_tables.json + mini_dev_pg.jsonl downloaded.

Usage:
  python scripts/eval_bird_pg.py
  python scripts/eval_bird_pg.py --limit 20      # smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import psycopg2
import torch
from loguru import logger

PG_DSN = "host=localhost port=5432 user=postgres password=slonik dbname=postgres"

# Postgres reserved words that BIRD uses as table/column names.
RESERVED = {"order", "user", "table", "group", "select", "from", "where", "join"}


def _quote_ident(name: str) -> str:
    if name.lower() in RESERVED or not re.match(r"^[a-z_][a-z0-9_]*$", name):
        return f'"{name}"'
    return name


def build_schema_string(table_info: dict) -> str:
    """Build CREATE TABLE statements for one BIRD db_id from dev_tables.json entry."""
    tables = table_info["table_names_original"]
    columns = table_info["column_names_original"]  # [(table_idx, col_name)]
    types = table_info["column_types"]  # parallel to columns
    pks = table_info.get("primary_keys", [])
    fks = table_info.get("foreign_keys", [])

    # Group columns by table_idx
    by_table: dict[int, list[tuple[str, str]]] = {}
    for i, (t_idx, col_name) in enumerate(columns):
        if t_idx == -1:  # the '*' placeholder
            continue
        by_table.setdefault(t_idx, []).append((col_name, types[i]))

    # PK columns indexed
    pk_cols_by_table: dict[int, list[str]] = {}
    for pk in pks:
        if isinstance(pk, list):  # composite key
            t_idx = columns[pk[0]][0]
            pk_cols_by_table.setdefault(t_idx, []).extend(columns[c][1] for c in pk)
        else:
            t_idx = columns[pk][0]
            pk_cols_by_table.setdefault(t_idx, []).append(columns[pk][1])

    out = []
    for t_idx, tname in enumerate(tables):
        cols = by_table.get(t_idx, [])
        if not cols:
            continue
        col_strs = []
        for cname, ctype in cols:
            pg_type = {
                "text": "TEXT", "number": "NUMERIC", "integer": "INTEGER",
                "real": "REAL", "date": "DATE", "time": "TIME",
                "boolean": "BOOLEAN", "others": "TEXT",
            }.get(ctype.lower(), "TEXT")
            col_strs.append(f"  {_quote_ident(cname)} {pg_type}")
        pk_list = pk_cols_by_table.get(t_idx, [])
        if pk_list:
            col_strs.append(f"  PRIMARY KEY ({', '.join(_quote_ident(c) for c in pk_list)})")
        out.append(f"CREATE TABLE {_quote_ident(tname)} (\n" + ",\n".join(col_strs) + "\n);")

    # Add foreign keys as comments (model gets the hint without invalid SQL)
    fk_strs = []
    for fk in fks:
        from_idx, to_idx = fk
        from_table = tables[columns[from_idx][0]]
        from_col = columns[from_idx][1]
        to_table = tables[columns[to_idx][0]]
        to_col = columns[to_idx][1]
        fk_strs.append(f"-- FK: {from_table}.{from_col} -> {to_table}.{to_col}")
    if fk_strs:
        out.append("\n".join(fk_strs))

    return "\n\n".join(out)


def execute_sql(conn, sql: str, timeout_ms: int = 8000):
    """Execute SQL with timeout. Returns (success, rows_or_error)."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {timeout_ms};")
            cur.execute(sql)
            try:
                rows = cur.fetchall()
            except psycopg2.ProgrammingError:
                rows = []  # query had no results to fetch
            conn.commit()
            return True, rows
    except Exception as e:
        conn.rollback()
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def normalize_rows(rows) -> set:
    """Order-independent comparison of result rows. Handles None, decimal, etc."""
    if not rows:
        return set()
    try:
        return {tuple(str(c) if c is not None else "NULL" for c in r) for r in rows}
    except Exception:
        return {str(rows)}


def extract_sql(text: str) -> str:
    """Extract SQL from model output, handling code fences and prose."""
    if "```sql" in text.lower():
        m = re.search(r"```sql\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(";")
    if "```" in text:
        m = re.search(r"```\s*(.+?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip().rstrip(";")
    # Look for SELECT / WITH starting the response
    m = re.search(r"\b(SELECT|WITH)\b.+?(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip().rstrip(";")
    return text.strip().rstrip(";")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft-merged")
    ap.add_argument("--questions", default="data/raw/bird_mini_dev/mini_dev_pg.jsonl")
    ap.add_argument(
        "--tables",
        default="data/raw/minidev_pg/minidev/MINIDEV/dev_tables.json",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="outputs/bird_pg_results.jsonl")
    args = ap.parse_args()

    # Load schemas
    with open(args.tables) as f:
        all_tables = json.load(f)
    schemas: dict[str, str] = {}
    for t in all_tables:
        schemas[t["db_id"]] = build_schema_string(t)
    logger.info(f"Built schemas for {len(schemas)} databases")

    # Load questions
    questions = []
    with open(args.questions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[: args.limit]
    logger.info(f"Loaded {len(questions)} questions")

    # Load model
    logger.info(f"Loading model from {args.model}")
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded.")

    # Connect to Postgres
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    logger.info("Connected to Postgres.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "correct": 0, "wrong": 0, "exec_error": 0, "gold_error": 0,
        "no_schema": 0,
    }
    by_difficulty: dict[str, dict[str, int]] = {}
    t_start = time.time()

    with open(args.out, "w") as fout:
        for i, ex in enumerate(questions, 1):
            stats["total"] += 1
            db_id = ex["db_id"]
            question = ex["question"]
            evidence = ex.get("evidence", "")
            gold_sql = ex["SQL"].strip().rstrip(";")
            difficulty = ex.get("difficulty", "unknown")
            by_difficulty.setdefault(difficulty, {"total": 0, "correct": 0})
            by_difficulty[difficulty]["total"] += 1

            schema = schemas.get(db_id)
            if not schema:
                stats["no_schema"] += 1
                continue

            user_msg = (
                f"You are a PostgreSQL expert. Given the schema below, write a single SQL query that answers the question.\n\n"
                f"### Schema:\n{schema}\n\n"
                f"### Question:\n{question}\n"
            )
            if evidence:
                user_msg += f"\n### Hint:\n{evidence}\n"
            user_msg += "\nReturn ONLY the SQL query, no explanation."

            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=0.0,
                )
            full = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            pred_sql = extract_sql(full)

            row = {
                "question_id": ex["question_id"], "db_id": db_id,
                "difficulty": difficulty, "question": question,
                "gold_sql": gold_sql, "pred_sql": pred_sql,
            }

            # Execute gold first (sanity)
            gold_ok, gold_result = execute_sql(conn, gold_sql)
            if not gold_ok:
                row["status"] = "gold_error"
                row["gold_error"] = gold_result
                stats["gold_error"] += 1
            else:
                pred_ok, pred_result = execute_sql(conn, pred_sql)
                if not pred_ok:
                    row["status"] = "exec_error"
                    row["error"] = pred_result
                    stats["exec_error"] += 1
                elif normalize_rows(pred_result) == normalize_rows(gold_result):
                    row["status"] = "correct"
                    stats["correct"] += 1
                    by_difficulty[difficulty]["correct"] += 1
                else:
                    row["status"] = "wrong"
                    row["pred_rows"] = len(pred_result)
                    row["gold_rows"] = len(gold_result)
                    stats["wrong"] += 1

            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if i % 10 == 0 or i == len(questions):
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (len(questions) - i) / max(rate, 1e-9)
                exec_total = stats["correct"] + stats["wrong"] + stats["exec_error"]
                acc = stats["correct"] / max(exec_total, 1) * 100
                logger.info(
                    f"  {i}/{len(questions)}  acc={acc:.1f}%  "
                    f"correct={stats['correct']} wrong={stats['wrong']} "
                    f"exec_err={stats['exec_error']} gold_err={stats['gold_error']}  "
                    f"rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                )

    elapsed = time.time() - t_start
    exec_total = stats["correct"] + stats["wrong"] + stats["exec_error"]
    acc = stats["correct"] / max(exec_total, 1) * 100

    logger.info("\n=== BIRD-PG MINI-DEV RESULTS ===")
    logger.info(f"  Total examples:    {stats['total']}")
    logger.info(f"  Eligible (gold OK): {exec_total}")
    logger.info(f"  CORRECT:           {stats['correct']:3d}  ({acc:.1f}%)  ← headline accuracy")
    logger.info(f"  WRONG:             {stats['wrong']:3d}")
    logger.info(f"  EXEC ERROR:        {stats['exec_error']:3d}")
    logger.info(f"  GOLD ERROR:        {stats['gold_error']:3d}")
    logger.info(f"  NO SCHEMA:         {stats['no_schema']:3d}")
    logger.info(f"")
    logger.info(f"  By difficulty:")
    for diff in ["simple", "moderate", "challenging"]:
        d = by_difficulty.get(diff)
        if d:
            d_acc = d["correct"] / max(d["total"], 1) * 100
            logger.info(f"    {diff:12s}: {d['correct']:3d}/{d['total']:3d}  ({d_acc:.1f}%)")
    logger.info(f"")
    logger.info(f"  Elapsed:   {elapsed/60:.1f} min")
    logger.info(f"  Output:    {args.out}")
    logger.info(f"")
    logger.info(f"  COMPARISON (BIRD Mini-Dev PG, from BIRD docs):")
    logger.info(f"    Qwen2.5-Coder-7B (base):     12.22%")
    logger.info(f"    Codestral 22B:               21.11%")
    logger.info(f"    Qwen2.5-Coder-32B:           22.96%")
    logger.info(f"    Slonik-7B (this run):        {acc:.2f}%  ← you")
    logger.info(f"    GPT-4o:                      34.44%")
    logger.info(f"    Claude 3.7 Sonnet:           39.26%")
    logger.info(f"    o3-mini:                     47.78%")

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
