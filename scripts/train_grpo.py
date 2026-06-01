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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

sft_merged = ROOT / "checkpoints" / "sft-merged" / "config.json"
if not sft_merged.exists():
    console.print("[red]SFT merged checkpoint not found. Run train_sft.py first.")
    sys.exit(1)

bird_train = ROOT / "data" / "processed" / "bird_train.jsonl"
grpo_prompts = ROOT / "data" / "processed" / "grpo_prompts.jsonl"
if not grpo_prompts.exists() and bird_train.exists():
    console.print("[cyan]Building GRPO prompt set from BIRD train (first 3000 examples)...")
    lines = bird_train.read_text(encoding="utf-8").splitlines()[:3000]
    grpo_prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")

cmd = [sys.executable, "-m", "slonik.training.grpo", "--config", "configs/grpo.yaml", *sys.argv[1:]]
sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)
