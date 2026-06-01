"""BIRD Mini-Dev SQLite eval — same as PG eval but executes against per-db SQLite files."""
import argparse, json, sqlite3, sys, time, re
from pathlib import Path
import torch
from loguru import logger

# Reuse most code from eval_bird_pg.py — just change the execute function
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("epg", "scripts/eval_bird_pg.py")
epg = importlib.util.module_from_spec(spec); spec.loader.exec_module(epg)

DB_ROOT = Path("data/raw/minidev_pg/minidev/MINIDEV/dev_databases")

def execute_sqlite(db_path, sql, timeout_ms=8000):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_ms/1000)
        cur = conn.cursor(); cur.execute(sql); rows = cur.fetchall(); conn.close()
        return True, rows
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/grpo-merged")
    ap.add_argument("--questions", default="data/raw/bird_mini_dev/mini_dev_sqlite.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/bird_sqlite_results.jsonl")
    args = ap.parse_args()

    with open("data/raw/minidev_pg/minidev/MINIDEV/dev_tables.json") as f:
        all_tables = json.load(f)
    schemas = {t["db_id"]: epg.build_schema_string(t) for t in all_tables}

    questions = []
    with open(args.questions) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]
    logger.info(f"Loaded {len(questions)} questions")

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    stats = {"total":0,"correct":0,"wrong":0,"exec_error":0,"gold_error":0,"no_db":0}
    by_diff = {}
    t_start = time.time()

    with open(args.out, "w") as fout:
        for i, ex in enumerate(questions, 1):
            stats["total"] += 1
            db_id = ex["db_id"]; question = ex["question"]; evidence = ex.get("evidence", "")
            gold_sql = ex["SQL"].strip().rstrip(";"); difficulty = ex.get("difficulty","?")
            by_diff.setdefault(difficulty, {"total":0,"correct":0})
            by_diff[difficulty]["total"] += 1

            db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                stats["no_db"] += 1
                continue

            schema = schemas.get(db_id, "")
            user_msg = (f"You are a SQLite expert. Given the schema below, write a single SQL query.\n\n"
                        f"### Schema:\n{schema}\n\n### Question:\n{question}\n")
            if evidence: user_msg += f"\n### Hint:\n{evidence}\n"
            user_msg += "\nReturn ONLY the SQL query, no explanation."

            prompt = tok.apply_chat_template(
                [{"role":"user","content":user_msg}], tokenize=False, add_generation_prompt=True
            )
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=512, do_sample=False, temperature=0.0)
            full = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            pred_sql = epg.extract_sql(full)

            row = {"question_id": ex["question_id"], "db_id": db_id, "difficulty": difficulty,
                   "question": question, "gold_sql": gold_sql, "pred_sql": pred_sql}

            gold_ok, gold_rows = execute_sqlite(db_path, gold_sql)
            if not gold_ok:
                row["status"] = "gold_error"; row["gold_error"] = gold_rows; stats["gold_error"] += 1
            else:
                pred_ok, pred_rows = execute_sqlite(db_path, pred_sql)
                if not pred_ok:
                    row["status"] = "exec_error"; row["error"] = pred_rows; stats["exec_error"] += 1
                elif epg.normalize_rows(pred_rows) == epg.normalize_rows(gold_rows):
                    row["status"] = "correct"; stats["correct"] += 1
                    by_diff[difficulty]["correct"] += 1
                else:
                    row["status"] = "wrong"; stats["wrong"] += 1

            fout.write(json.dumps(row) + "\n"); fout.flush()
            if i % 10 == 0:
                elapsed = time.time() - t_start
                exec_t = stats["correct"] + stats["wrong"] + stats["exec_error"]
                acc = stats["correct"] / max(exec_t,1) * 100
                eta = (len(questions) - i) / (i/elapsed) / 60
                logger.info(f"  {i}/{len(questions)}  acc={acc:.1f}%  ETA={eta:.1f}min")

    exec_t = stats["correct"] + stats["wrong"] + stats["exec_error"]
    acc = stats["correct"] / max(exec_t,1) * 100
    logger.info(f"\n=== BIRD-SQLITE RESULTS ===")
    logger.info(f"  CORRECT: {stats['correct']} ({acc:.1f}%)")
    logger.info(f"  WRONG:   {stats['wrong']}")
    logger.info(f"  ERROR:   {stats['exec_error']}")
    logger.info(f"  By difficulty:")
    for d in ["simple","moderate","challenging"]:
        if d in by_diff:
            dacc = by_diff[d]["correct"] / max(by_diff[d]["total"],1) * 100
            logger.info(f"    {d:12s}: {by_diff[d]['correct']}/{by_diff[d]['total']} ({dacc:.1f}%)")
    logger.info(f"\n  PG comparison: Slonik-PG=34.60%, Slonik-SQLite={acc:.2f}%")

if __name__ == "__main__":
    sys.exit(main())
