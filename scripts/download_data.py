from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def run(args: list[str], label: str) -> None:
    console.rule(f"[cyan]{label}")
    result = subprocess.run([sys.executable, "-m", *args], cwd=ROOT)
    if result.returncode != 0:
        console.print(f"[red]{label} failed (exit {result.returncode})")
        sys.exit(result.returncode)


def merge(paths: list[Path], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with target.open("w", encoding="utf-8") as out:
        for p in paths:
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n += 1
    return n


def main() -> None:
    processed = ROOT / "data" / "processed"
    synthetic = ROOT / "data" / "synthetic"
    processed.mkdir(parents=True, exist_ok=True)
    synthetic.mkdir(parents=True, exist_ok=True)

    run(["slonik.data.prepare_bird", "--out", str(processed / "bird_train.jsonl")], "BIRD-SQL train")
    run(["slonik.data.prepare_spider", "--out", str(processed / "spider.jsonl"), "--splits", "train,validation"], "Spider")
    run(["slonik.data.prepare_gretel", "--out", str(processed / "gretel_pg.jsonl"), "--max-rows", "15000"], "Gretel PG-filtered")

    if os.getenv("ANTHROPIC_API_KEY"):
        run(["slonik.data.synthesize_pg_modern", "--split", "train", "--concurrency", "8"], "PG-Modern train (synthetic)")
        run(["slonik.data.synthesize_pg_modern", "--split", "eval", "--concurrency", "8"], "PG-Modern eval (synthetic)")
    else:
        console.print("[yellow]ANTHROPIC_API_KEY not set, skipping synthetic PG-modern generation.")

    console.rule("[cyan]Merging train splits")
    train_n = merge(
        [
            processed / "bird_train.jsonl",
            processed / "spider.jsonl",
            processed / "gretel_pg.jsonl",
            synthetic / "pg_modern_train.jsonl",
        ],
        processed / "train.jsonl",
    )
    console.print(f"train.jsonl: [green]{train_n}[/] examples")

    eval_src = synthetic / "pg_modern_eval.jsonl"
    eval_target = processed / "eval.jsonl"
    if eval_src.exists():
        lines = eval_src.read_text(encoding="utf-8").splitlines()[:500]
        eval_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"eval.jsonl: [green]{len(lines)}[/] examples")
    else:
        console.print("[yellow]No PG-modern eval file found; skipping eval.jsonl.")


if __name__ == "__main__":
    main()
