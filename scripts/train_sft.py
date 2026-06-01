from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

cmd = [sys.executable, "-m", "slonik.training.sft", "--config", "configs/sft_qlora.yaml", *sys.argv[1:]]
sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)
