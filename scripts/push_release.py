from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def run(args: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None, help="defaults to <HF_USERNAME>/Slonik-7B")
    parser.add_argument("--model-dir", default="checkpoints/grpo-merged")
    parser.add_argument("--gguf", action="store_true", help="also convert to GGUF and push as <repo>-GGUF")
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        console.print("[red]HF_TOKEN not set in .env")
        sys.exit(1)

    repo = args.repo or f"{os.environ.get('HF_USERNAME', 'phaneendra')}/Slonik-7B"
    model_dir = (ROOT / args.model_dir).resolve()
    if not (model_dir / "config.json").exists():
        console.print(f"[red]Model dir not found: {model_dir}")
        sys.exit(1)

    rc = run(["slonik.publish.push_to_hub", "--model-dir", str(model_dir), "--repo", repo])
    if rc != 0:
        sys.exit(rc)

    if args.gguf:
        gguf_dir = ROOT / "outputs" / "gguf"
        console.rule("[cyan]Building GGUF quants")
        run(["slonik.publish.convert_gguf", "--model", str(model_dir), "--out-dir", str(gguf_dir)])
        console.rule(f"[cyan]Pushing {repo}-GGUF")
        run(["slonik.publish.push_to_hub", "--model-dir", str(gguf_dir), "--repo", f"{repo}-GGUF"])


if __name__ == "__main__":
    main()
