#!/bin/bash
# Show all GRPO processes, expect exactly 1.

COUNT=$(pgrep -f "slonik.training.grpo" | wc -l)
echo "GRPO processes running: $COUNT"
echo ""
if [ "$COUNT" -gt 0 ]; then
    ps -o pid,etime,pcpu,pmem,cmd -p $(pgrep -f "slonik.training.grpo" | tr '\n' ',' | sed 's/,$//')
fi
if [ "$COUNT" -gt 1 ]; then
    echo ""
    echo "⚠ WARNING: $COUNT processes running. Should be 1."
    echo "  Kill extras with: pkill -9 -f slonik.training.grpo"
fi
