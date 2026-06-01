#!/bin/bash
# Wait for GRPO, then run BOTH PG and SQLite evals.

cd /mnt/d/AI/Projects/slonik-7b
exec > >(tee -a logs/pipeline-2k.log) 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "════════════════════════════════════════"
log "  GRPO-2000 → DUAL EVAL PIPELINE"
log "════════════════════════════════════════"

# Wait for GRPO
log "▸ Phase 1: Watching GRPO..."
while pgrep -f "slonik.training.grpo" > /dev/null; do
    STEP=$(grep -ao "[0-9]\+/[0-9]\+ \[" logs/grpo_2k.log 2>/dev/null | tail -1 | tr -d ' [')
    log "  ... running (step: ${STEP:-?})"
    sleep 600  # check every 10 min
done
log "  ✓ GRPO ended. Waiting 60s for merge..."
sleep 60

# Verify merge
log ""
log "▸ Phase 2: Verifying merged checkpoint..."
GRPO_DIR="checkpoints/grpo-merged"
SHARDS=$(ls $GRPO_DIR/model-*.safetensors 2>/dev/null | wc -l)
if [ "$SHARDS" -lt 4 ]; then
    log "  ✗ Merge failed ($SHARDS shards). Trying manual merge from latest checkpoint..."
    LATEST=$(ls -d checkpoints/grpo/checkpoint-* 2>/dev/null | sort -V | tail -1)
    log "  Manual merging from: $LATEST"
    python << PYEOF
import torch
from unsloth import FastLanguageModel
m, t = FastLanguageModel.from_pretrained(
    model_name="$LATEST", max_seq_length=3072,
    dtype=torch.bfloat16, load_in_4bit=True
)
m.save_pretrained_merged("$GRPO_DIR", t, save_method="merged_16bit")
print("Manual merge done")
PYEOF
fi
log "  ✓ $GRPO_DIR ready ($SHARDS shards)"

# Start Postgres
log ""
log "▸ Phase 3: Starting Postgres..."
docker start slonik-pg 2>&1 | sed 's/^/    /'
sleep 10

# Run PG eval
log ""
log "▸ Phase 4: BIRD-PG eval (17 min)..."
python -u scripts/eval_bird_pg.py \
    --model $GRPO_DIR \
    --out outputs/bird_pg_results_grpo2k.jsonl \
    > logs/bird_pg_2k.log 2>&1
log "  ✓ PG eval done"

# Run SQLite eval
log ""
log "▸ Phase 5: BIRD-SQLite eval (25 min)..."
python -u scripts/eval_bird_sqlite.py \
    --model $GRPO_DIR \
    --out outputs/bird_sqlite_results_2k.jsonl \
    > logs/bird_sqlite_2k.log 2>&1
log "  ✓ SQLite eval done"

# Final summary
log ""
log "════════════════════════════════════════"
log "  FINAL RESULTS"
log "════════════════════════════════════════"
{
    echo "════════════════════════════════════════"
    echo "  SLONIK-7B-GRPO (2000 steps) — RESULTS"
    echo "  Completed: $(date)"
    echo "════════════════════════════════════════"
    echo ""
    echo "▸ BIRD Mini-Dev PostgreSQL:"
    grep -A 20 "BIRD-PG MINI-DEV RESULTS" logs/bird_pg_2k.log | head -25
    echo ""
    echo "▸ BIRD Mini-Dev SQLite:"
    grep -A 15 "BIRD-SQLITE RESULTS" logs/bird_sqlite_2k.log | head -20
    echo ""
    echo "▸ COMPARISON vs 500-step run:"
    echo "    500 steps  → PG 34.60%, SQLite 42.40%"
    echo "    2000 steps → PG <see above>, SQLite <see above>"
} > outputs/GRPO_2K_RESULTS.txt

cat outputs/GRPO_2K_RESULTS.txt
log ""
log "✓ Pipeline complete. Summary in outputs/GRPO_2K_RESULTS.txt"
