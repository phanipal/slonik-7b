from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import click
import yaml
from anthropic import AsyncAnthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


GENERATOR_MODEL = os.environ.get("SYNTH_MODEL", "deepseek-v4-pro")

PROMPT = """Generate ONE realistic PostgreSQL text-to-SQL training example using the feature: {feature}.

Schema must be 2-5 tables, realistic domain (not generic "users/products"). Use modern Postgres types where natural (JSONB, TIMESTAMPTZ, TEXT[], TSVECTOR, vector for pgvector).

Output strict JSON only:
{{
  "schema": "<CREATE TABLE statements>",
  "question": "<natural language question, 1-2 sentences>",
  "sql": "<single valid PostgreSQL query using the {feature} feature>",
  "evidence": "<optional short hint or empty string>"
}}

Feature requirements:
- cte: at least one WITH clause, possibly recursive
- window: at least one window function (ROW_NUMBER, LAG, SUM OVER, etc.)
- jsonb: use ->, ->>, @>, jsonb_path_query, or jsonb_agg
- pgvector: use <-> or <=> distance operator on a vector column
- fulltext: use to_tsvector / to_tsquery / @@ or websearch_to_tsquery
- array_ops: use unnest, ANY, ALL, array_agg, or [] indexing

Generate something a senior data engineer would actually write."""


def _pick_feature(mix: dict[str, float]) -> str:
    keys = list(mix.keys())
    weights = [mix[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _generate_one(client: AsyncAnthropic, feature: str, stats: dict) -> dict | None:
    msg = await client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT.format(feature=feature)}],
    )
    if msg.usage:
        stats["input_tokens"] += msg.usage.input_tokens
        stats["output_tokens"] += msg.usage.output_tokens
    text = ""
    for block in msg.content:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            text = block.text.strip()
            break
    if not text:
        stats["empty"] += 1
        stop_reason = getattr(msg, "stop_reason", "?")
        if stats["empty"] <= 3:
            block_summary = ", ".join(f"{b.type}:{len(getattr(b, 'text', '') or getattr(b, 'thinking', ''))}" for b in msg.content)
            logger.warning(f"  empty text  stop_reason={stop_reason}  blocks=[{block_summary}]  out_tokens={msg.usage.output_tokens if msg.usage else '?'}")
        return None
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        stats["parse_fail"] += 1
        if stats["parse_fail"] <= 3:
            logger.warning(f"  parse fail at pos {e.pos}: text[-200:]={text[-200:]!r}")
        return None
    if not all(k in obj for k in ("schema", "question", "sql")):
        stats["missing_keys"] += 1
        return None
    obj["feature"] = feature
    obj["db_id"] = "pg_modern"
    obj["source"] = "pg_modern_synth"
    return obj


async def _run(n: int, mix: dict, concurrency: int, out_path: Path, force: bool) -> int:
    existing = 0
    if out_path.exists() and not force:
        with out_path.open("r", encoding="utf-8") as fh:
            existing = sum(1 for line in fh if line.strip())
        if existing >= n:
            logger.info(f"  output already complete: {existing}/{n} examples in {out_path}. Use --force to regenerate.")
            return existing
        if existing > 0:
            logger.info(f"  RESUMING: {existing} existing examples in file, generating {n - existing} more")
    elif out_path.exists() and force:
        logger.info(f"  FORCE: overwriting {out_path}")

    remaining = n - existing
    file_mode = "a" if existing > 0 else "w"

    client = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        timeout=600.0,
    )
    sem = asyncio.Semaphore(concurrency)
    stats = {"input_tokens": 0, "output_tokens": 0, "empty": 0, "parse_fail": 0, "missing_keys": 0}
    call_counter = {"started": 0, "finished": 0}
    verbose = remaining <= 20

    async def task() -> dict | None:
        async with sem:
            call_counter["started"] += 1
            idx = call_counter["started"]
            feat = _pick_feature(mix)
            if verbose:
                logger.info(f"  [{idx}/{remaining}] start  feature={feat}")
            t0 = asyncio.get_event_loop().time()
            try:
                result = await _generate_one(client, feat, stats)
            except Exception as e:
                logger.warning(f"  [{idx}/{remaining}] FAIL after retries: {type(e).__name__}: {str(e)[:120]}")
                stats["exception"] = stats.get("exception", 0) + 1
                return None
            call_counter["finished"] += 1
            if verbose:
                dt = asyncio.get_event_loop().time() - t0
                ok = "OK " if result else "DROP"
                logger.info(f"  [{idx}/{remaining}] {ok}   feature={feat}  took={dt:.1f}s")
            return result

    written_this_run = 0
    with out_path.open(file_mode, encoding="utf-8") as f:
        coros = [task() for _ in range(remaining)]
        log_every = max(1, remaining // 20) if not verbose else 1_000_000
        for i, fut in enumerate(asyncio.as_completed(coros), 1):
            result = await fut
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
                written_this_run += 1
            if not verbose and (i % log_every == 0 or i == remaining):
                total = existing + written_this_run
                logger.info(f"  progress {i}/{remaining}  total_in_file={total}/{n}  empty={stats['empty']}  parse_fail={stats['parse_fail']}  fails={stats.get('exception', 0)}  tokens(in/out)={stats['input_tokens']}/{stats['output_tokens']}")
    logger.info(f"Final stats this run: {stats}")
    if "deepseek" in (os.environ.get("ANTHROPIC_BASE_URL") or "").lower():
        cost = stats["input_tokens"] / 1_000_000 * 0.435 + stats["output_tokens"] / 1_000_000 * 0.87
        logger.info(f"Estimated DeepSeek V4 Pro cost: ${cost:.4f} (this run only)")
    return existing + written_this_run


@click.command()
@click.option("--config", default="configs/datasets.yaml", type=click.Path(exists=True))
@click.option("--split", type=click.Choice(["train", "eval"]), required=True)
@click.option("--concurrency", default=8, type=int)
@click.option("--dry-run", is_flag=True, help="Generate only 5 examples to validate the pipeline")
@click.option("--force", is_flag=True, help="Discard existing output file and regenerate from scratch")
def main(config: str, split: str, concurrency: int, dry_run: bool, force: bool) -> None:
    cfg = yaml.safe_load(Path(config).read_text())["pg_modern"]
    n = 5 if dry_run else (cfg["n_train"] if split == "train" else cfg["n_eval"])
    out_path = Path(cfg[f"output_{split}"])
    if dry_run:
        out_path = out_path.with_name(out_path.stem + ".dryrun.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(3407 if split == "train" else 4242)
    logger.info(f"Generating {n} {split} examples with mix={cfg['feature_mix']}  model={GENERATOR_MODEL}")
    if dry_run:
        logger.info("DRY-RUN: 5 examples only, output → " + str(out_path))
    written = asyncio.run(_run(n, cfg["feature_mix"], concurrency, out_path, force))
    logger.info(f"Total in file: {written}/{n} → {out_path}")


if __name__ == "__main__":
    main()