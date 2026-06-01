#!/bin/bash
# Compare local sft-merged files vs what's on HF. List missing.

REPO="Phanii1/Slonik-7B-SFT"
LOCAL_DIR="/mnt/d/AI/Projects/slonik-7b/checkpoints/sft-merged"

echo "════════════════════════════════════════════════════════"
echo "  HF UPLOAD STATUS: $REPO"
echo "════════════════════════════════════════════════════════"
echo ""

python << PYEOF
import os
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
try:
    files_remote = set(api.list_repo_files("$REPO", repo_type="model", token=os.environ.get("HF_TOKEN")))
except Exception as e:
    print(f"✗ Cannot reach HF: {e}")
    exit(1)

local_dir = Path("$LOCAL_DIR")
files_local = {p.name for p in local_dir.iterdir() if p.is_file()}

# Important files we expect to upload (excluding .cache and similar)
expected = sorted([f for f in files_local if not f.startswith(".")])

uploaded = []
missing = []
for f in expected:
    if f in files_remote:
        size = (local_dir / f).stat().st_size
        uploaded.append((f, size))
    else:
        size = (local_dir / f).stat().st_size
        missing.append((f, size))

print(f"✓ Uploaded ({len(uploaded)} files):")
for f, sz in uploaded:
    print(f"   {sz/(1024**3):>6.2f} GB   {f}")

total_remaining_gb = sum(sz for _, sz in missing) / (1024**3)
print(f"\n✗ Missing ({len(missing)} files, {total_remaining_gb:.2f} GB remaining):")
for f, sz in missing:
    print(f"   {sz/(1024**3):>6.2f} GB   {f}")

# Print one-liner upload commands for the missing files
if missing:
    print(f"\n────────────────────────────────────────")
    print("Commands to upload remaining files:")
    print("────────────────────────────────────────")
    for f, _ in missing:
        print(f"hf upload $REPO $LOCAL_DIR/{f} {f} --repo-type=model")
PYEOF
