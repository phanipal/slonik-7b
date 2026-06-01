#!/bin/bash
# Wait for GRPO to finish, then verify and eval against BIRD-PG.
# Writes status to logs/pipeline.log and a final summary to outputs/GRPO_RESULTS.txt.

cd /mnt/d/AI/Projects/slonik-7b
exec > >(tee -a logs/pipeline.log) 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "════════════════════════════════════════"
log "  GRPO → EVAL PIPELINE STARTED"
log "════════════════════════════════════════"

# ─── PHASE 1: WAIT FOR GRPO ───────────────────────────────────────
log ""
log "▸ PHASE 1: Waiting for GRPO to finish..."
GRPO_PID=$(pgrep -f "slonik.training.grpo" | head -1)
if [ -z "$GRPO_PID" ]; then
    log "  ⚠ No GRPO process running. Will proceed if merged checkpoint exists."
else
    log "  Watching PID $GRPO_PID"
    while ps -p $GRPO_PID > /dev/null 2>&1; do
        STEP=$(grep -ao "[0-9]\+/[0-9]\+ \[" logs/grpo.log 2>/dev/null | tail -1 | tr -d ' [')
        log "  ... GRPO running (step: ${STEP:-?})"
        sleep 300  # check every 5 min
    done
    log "  ✓ GRPO process ended"
fi

# Give it a moment to finalize the merge step
sleep 30

# ─── PHASE 2: VERIFY MERGED CHECKPOINT ────────────────────────────
log ""
log "▸ PHASE 2: Verifying GRPO merged checkpoint..."

GRPO_DIR="checkpoints/grpo-merged"
if [ ! -d "$GRPO_DIR" ]; then
    log "  ✗ FATAL: $GRPO_DIR does not exist. GRPO may have crashed before merging."
    log "  Available checkpoints:"
    ls -d checkpoints/grpo/checkpoint-* 2>/dev/null | sed 's/^/    /' | tee -a logs/pipeline.log
    log "  Pipeline aborted."
    exit 1
fi

SHARDS=$(ls $GRPO_DIR/model-*.safetensors 2>/dev/null | wc -l)
SIZE=$(du -sh $GRPO_DIR 2>/dev/null | awk '{print $1}')
log "  ✓ $GRPO_DIR exists"
log "  ✓ $SHARDS safetensors shards, total size $SIZE"

if [ "$SHARDS" -lt 4 ]; then
    log "  ✗ FATAL: expected 4 shards, found $SHARDS. Merge likely failed."
    exit 1
fi

# Quick load test (10 sec — won't actually generate, just opens the model)
log "  Verifying model loads..."
timeout 120 python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('$GRPO_DIR')
print(f'  Config OK: {c.model_type}, vocab_size={c.vocab_size}, hidden_size={c.hidden_size}')
" 2>&1 | sed 's/^/    /' | tee -a logs/pipeline.log

# ─── PHASE 3: START POSTGRES ──────────────────────────────────────
log ""
log "▸ PHASE 3: Starting Postgres for BIRD eval..."

if docker ps --format '{{.Names}}' | grep -q "^slonik-pg$"; then
    log "  ✓ slonik-pg already running"
else
    docker start slonik-pg 2>&1 | sed 's/^/    /' | tee -a logs/pipeline.log
    sleep 5
fi

# Confirm we can actually query
docker exec slonik-pg psql -U postgres -d postgres -c "SELECT count(*) FROM account;" 2>&1 \
    | sed 's/^/    /' | tee -a logs/pipeline.log

# ─── PHASE 4: RUN BIRD-PG EVAL ────────────────────────────────────
log ""
log "▸ PHASE 4: Running BIRD-PG eval against GRPO model..."
log "  This will take ~17 min for 500 examples..."

OUTFILE="outputs/bird_pg_results_grpo.jsonl"
EVAL_LOG="logs/bird_pg_grpo.log"

python -u scripts/eval_bird_pg.py \
    --model $GRPO_DIR \
    --out $OUTFILE \
    > $EVAL_LOG 2>&1

EVAL_EXIT=$?
log "  Eval exit code: $EVAL_EXIT"

# ─── PHASE 5: REPORT ──────────────────────────────────────────────
log ""
log "▸ PHASE 5: Final report"

SUMMARY_FILE="outputs/GRPO_RESULTS.txt"
{
    echo "════════════════════════════════════════════════════════"
    echo "  SLONIK-7B GRPO PIPELINE — RESULTS"
    echo "  Completed: $(date)"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "Model: $GRPO_DIR"
    echo "Eval set: BIRD Mini-Dev PostgreSQL (500 examples)"
    echo ""
    grep -A 25 "BIRD-PG MINI-DEV RESULTS" $EVAL_LOG | head -30
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  COMPARISON vs SFT-only Slonik-7B"
    echo "════════════════════════════════════════════════════════"
    echo "  SFT-only (33.20%) → GRPO post-eval:"
    grep -E "Slonik-7B \(this run\)" $EVAL_LOG | tail -1
    echo ""
    echo "  Per-difficulty SFT baseline:"
    echo "    simple:      48.6%"
    echo "    moderate:    29.6%"
    echo "    challenging: 19.6%"
    echo ""
} > $SUMMARY_FILE

log "  ✓ Final summary saved: $SUMMARY_FILE"
log ""
log "════════════════════════════════════════"
log "  PIPELINE COMPLETE"
log "════════════════════════════════════════"
cat $SUMMARY_FILE
