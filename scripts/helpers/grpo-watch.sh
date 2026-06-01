#!/bin/bash
# Clean GRPO status dashboard

cd /mnt/d/AI/Projects/slonik-7b

clear
echo "═══════════════════════════════════════════════════════════════════"
echo "  SLONIK-7B GRPO STATUS  ·  $(date '+%H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════════"

# Process
echo ""
echo "▸ PROCESS"
PID=$(pgrep -f "slonik.training.grpo" | head -1)
if [ -z "$PID" ]; then
    echo "  ✗ NOT RUNNING"
else
    UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "  ✓ PID $PID  ·  uptime $UPTIME"
fi

# Progress
echo ""
echo "▸ PROGRESS"
LATEST_STEP=$(grep -ao "[0-9]\+/[0-9]\+ \[" logs/grpo_2k.log 2>/dev/null | tail -1 | tr -d ' [')
STEP_TIME=$(grep -ao "[0-9]\+\.[0-9]\+s/it" logs/grpo_2k.log 2>/dev/null | tail -1)
echo "  Step:       ${LATEST_STEP:-—}"
echo "  Step time:  ${STEP_TIME:-—}"

# Rewards (last 10 steps)
echo ""
echo "▸ EXEC REWARD (last 10 steps)"
grep -a "weighted_exec/mean" logs/grpo_2k.log 2>/dev/null \
  | grep -oE "weighted_exec/mean': [0-9.]+" \
  | awk -F"': " '{printf "%5.2f  ", $2}' \
  | tail -c 70
echo ""

# Combined rewards trend
EXEC_AVG=$(grep -a "weighted_exec/mean" logs/grpo_2k.log 2>/dev/null \
  | grep -oE "weighted_exec/mean': [0-9.]+" \
  | awk -F"': " '{sum+=$2; n++} END {if(n>0) printf "%.3f", sum/n; else print "—"}')
echo "  Average so far:  $EXEC_AVG"

# Losses & KL
echo ""
echo "▸ TRAINING METRICS (latest)"
LATEST=$(grep -a "^{'loss'" logs/grpo_2k.log 2>/dev/null | tail -1)
if [ -n "$LATEST" ]; then
    LOSS=$(echo "$LATEST" | grep -oE "'loss': [0-9.e+-]+" | head -1 | awk -F': ' '{print $2}')
    KL=$(echo "$LATEST" | grep -oE "'kl': [0-9.e+-]+" | head -1 | awk -F': ' '{print $2}')
    GRAD=$(echo "$LATEST" | grep -oE "'grad_norm': [0-9.e+-]+" | head -1 | awk -F': ' '{print $2}')
    LR=$(echo "$LATEST" | grep -oE "'learning_rate': [0-9.e+-]+" | head -1 | awk -F': ' '{print $2}')
    echo "  loss:     $LOSS"
    echo "  kl:       $KL"
    echo "  gradnorm: $GRAD"
    echo "  lr:       $LR"
fi

# GPU
echo ""
echo "▸ GPU"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "  VRAM:   %s MiB used  ·  %s MiB free\n  Util:   %s%%  ·  %s°C\n", $1, $2, $3, $4}'

# Checkpoints
echo ""
echo "▸ CHECKPOINTS"
if [ -d checkpoints/grpo ]; then
    ls -1 checkpoints/grpo/ 2>/dev/null | grep "^checkpoint-" | sort -V | tail -3 | sed 's/^/  /'
    LATEST_CKPT=$(ls -1 checkpoints/grpo/ 2>/dev/null | grep "^checkpoint-" | sort -V | tail -1)
    [ -n "$LATEST_CKPT" ] && echo "  ──────────────────"
    [ -n "$LATEST_CKPT" ] && echo "  latest: $LATEST_CKPT"
else
    echo "  (none yet)"
fi

# ETA
echo ""
echo "▸ ETA"
if [ -n "$LATEST_STEP" ] && [ -n "$STEP_TIME" ]; then
    CURRENT=$(echo $LATEST_STEP | awk -F'/' '{print $1}')
    TOTAL=$(echo $LATEST_STEP | awk -F'/' '{print $2}')
    SECS=$(echo $STEP_TIME | grep -oE "[0-9]+\.[0-9]+")
    REMAINING=$(echo "($TOTAL - $CURRENT) * $SECS / 60" | bc -l 2>/dev/null)
    [ -n "$REMAINING" ] && printf "  ~%.0f min remaining  (%.1f hours)\n" "$REMAINING" "$(echo $REMAINING/60 | bc -l)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
