from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

server = os.getenv("SLONIK_SERVER", "http://localhost:8000")
model = os.getenv("SLONIK_MODEL", "slonik-7b")

console.print(f"[cyan]Probing server {server} ...")
try:
    httpx.get(f"{server}/v1/models", timeout=5).raise_for_status()
except Exception as e:
    console.print(f"[red]vLLM server not reachable at {server}: {e}")
    console.print("Start it with: [bold]python scripts/serve_local.py")
    sys.exit(1)


def run(args: list[str], label: str) -> None:
    console.rule(f"[cyan]{label}")
    rc = subprocess.run([sys.executable, "-m", *args], cwd=ROOT).returncode
    if rc != 0:
        console.print(f"[yellow]{label} exited {rc}")


outputs = ROOT / "outputs"
outputs.mkdir(exist_ok=True)

run(
    ["slonik.eval.bird_runner",
     "--server", server, "--model", model,
     "--out", str(outputs / "bird_eval.json"),
     "--dialect", "postgresql"],
    "BIRD mini-dev (PostgreSQL)",
)

run(
    ["slonik.eval.spider_runner",
     "--server", server, "--model", model,
     "--out", str(outputs / "spider_eval.json")],
    "Spider validation",
)

pg_eval = ROOT / "data" / "synthetic" / "pg_modern_eval.jsonl"
if pg_eval.exists():
    run(
        ["slonik.eval.bird_runner",
         "--server", server, "--model", model,
         "--out", str(outputs / "pg_modern_eval.json"),
         "--limit", "500"],
        "PG-Modern eval",
    )

console.rule("[green]Done")
console.print(f"Results: [bold]{outputs}")
