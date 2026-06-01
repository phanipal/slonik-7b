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


def _resolve_model(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if (p / "config.json").exists():
            return p
        console.print(f"[red]No config.json at {p}")
        sys.exit(1)

    grpo = ROOT / "checkpoints" / "grpo-merged"
    sft = ROOT / "checkpoints" / "sft-merged"
    if (grpo / "config.json").exists():
        return grpo
    if (sft / "config.json").exists():
        console.print("[yellow]GRPO checkpoint not found, falling back to SFT.")
        return sft
    console.print("[red]No merged checkpoint found. Train SFT first.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--served-name", default="slonik-7b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    model_path = _resolve_model(args.model_path)
    console.print(f"[cyan]Serving [bold]{model_path}[/] as '{args.served_name}' on :{args.port}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", args.served_name,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
