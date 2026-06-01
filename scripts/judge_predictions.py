"""
LLM-as-judge scoring of Slonik-7B predictions.
Reads eval_results.jsonl, asks DeepSeek if each (gold, pred) pair is semantically equivalent,
writes verdicts back and reports accuracy.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from loguru import logger

JUDGE_PROMPT = """You are evaluating a text-to-SQL model. Given the question, gold SQL, and predicted SQL, classify the prediction into EXACTLY ONE of three buckets.

EQUIVALENT — Predicted SQL would return the same logical rows as the gold SQL on realistic data. Different aliases, JOIN order, equivalent expressions are all fine.

PARTIAL — Predicted SQL captures the main intent and uses correct operators/tables, but misses 1-2 secondary constraints (e.g. one missing WHERE filter, missing PARTITION BY, slightly wrong output columns). Most of the work is right.

NOT_EQUIVALENT — Predicted SQL is fundamentally wrong: wrong tables, wrong aggregation type, hallucinated columns, completely different structure, or so many missing constraints that the result would be unrecognizable.

Be FAIR. Most realistic LLM predictions land in PARTIAL — it should be your most common verdict, not NOT_EQUIVALENT. Only use NOT_EQUIVALENT when the prediction is genuinely off-target.

QUESTION:
{question}

GOLD SQL:
{gold}

PREDICTED SQL:
{pred}

Respond with EXACTLY this format:
VERDICT: <EQUIVALENT | PARTIAL | NOT_EQUIVALENT>
REASON: <one sentence>
"""


def judge_one(client: Anthropic, model: str, ex: dict, max_retries: int = 3) -> dict:
    """Score one prediction. Returns ex augmented with judge verdict."""
    prompt = JUDGE_PROMPT.format(
        question=ex.get("question", "")[:1500],
        schema=ex.get("schema", "")[:2000],
        gold=ex.get("gold_sql", "")[:2000],
        pred=ex.get("pred_sql", "")[:2000],
    )
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
                    break
            lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
            verdict = lines[0].replace("VERDICT:", "").strip() if lines else "PARSE_ERROR"
            reason = lines[1].replace("REASON:", "").strip() if len(lines) > 1 else ""
            ex["judge_verdict"] = verdict
            ex["judge_reason"] = reason
            return ex
        except Exception as e:
            if attempt == max_retries - 1:
                ex["judge_verdict"] = "ERROR"
                ex["judge_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
                return ex
            time.sleep(2 ** attempt)


def main():
    input_path = Path("outputs/eval_results.jsonl")
    output_path = Path("outputs/eval_judged.jsonl")
    model = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))

    # Load examples and schema/question from original eval file
    eval_examples = {}
    with Path("data/processed/eval.jsonl").open() as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            eval_examples[i] = {"schema": obj.get("schema", ""), "question": obj.get("question", "")}

    examples = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Backfill schema and question from original eval file
            idx = ex.get("i")
            if idx and idx in eval_examples:
                ex["schema"] = eval_examples[idx]["schema"]
                ex["question"] = eval_examples[idx]["question"]
            examples.append(ex)
    logger.info(f"Loaded {len(examples)} predictions to judge")

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    )

    results = []
    stats = {"EQUIVALENT": 0, "PARTIAL": 0, "NOT_EQUIVALENT": 0, "ERROR": 0, "PARSE_ERROR": 0}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, client, model, ex): ex for ex in examples}
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            v = r.get("judge_verdict", "ERROR")
            stats[v] = stats.get(v, 0) + 1
            with open("outputs/eval_judged.jsonl", "a") as _fout:
                _fout.write(__import__("json").dumps(r) + "\n")
            if n % 25 == 0 or n == len(examples):
                elapsed = time.time() - t_start
                rate = n / elapsed
                eta = (len(examples) - n) / max(rate, 1e-9)
                eq = stats["EQUIVALENT"]
                pa = stats["PARTIAL"]
                ne = stats["NOT_EQUIVALENT"]
                total_scored = eq + pa + ne
                acc = eq / max(total_scored, 1) * 100
                acc_lenient = (eq + pa) / max(total_scored, 1) * 100
                logger.info(
                    f"  {n}/{len(examples)}  strict={acc:.1f}%  lenient={acc_lenient:.1f}%  "
                    f"EQ={eq} PARTIAL={pa} NE={ne} ERR={stats['ERROR']}  "
                    f"rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                )

    # Write all results
    results.sort(key=lambda r: r.get("i", 0))
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    eq = stats["EQUIVALENT"]
    pa = stats["PARTIAL"]
    ne = stats["NOT_EQUIVALENT"]
    total = eq + pa + ne
    logger.info("\n=== FINAL JUDGMENT ===")
    logger.info(f"  Total scored:     {total}/{len(examples)}")
    logger.info(f"  EQUIVALENT:       {eq:3d}  ({eq/max(total,1)*100:.1f}%)  ← strict accuracy")
    logger.info(f"  PARTIAL:          {pa:3d}  ({pa/max(total,1)*100:.1f}%)")
    logger.info(f"  NOT_EQUIVALENT:   {ne:3d}  ({ne/max(total,1)*100:.1f}%)")
    logger.info(f"  ERRORS/PARSE:     {stats['ERROR']+stats['PARSE_ERROR']:3d}")
    logger.info(f"  Lenient accuracy (EQ+PARTIAL): {(eq+pa)/max(total,1)*100:.1f}%")
    logger.info(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")
    logger.info(f"  Per-example verdicts saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
