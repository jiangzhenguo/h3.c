#!/bin/bash
# Re-measure VPIPE in the same session as the h3.c bf16 A/B, so the
# end-to-end comparison is not polluted by day-to-day thermal drift.
#   cool 10 min -> vpipe (i8_gemm on, same spec as before) -> done
set -u
WORK=/Users/jiangzhenguo/ai/vpipe-work
VPIPE_BIN=/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe
COOL=${COOL:-600}

stamp() { date "+%H:%M:%S"; }

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before VPIPE ====="
sleep "$COOL"

echo "[$(stamp)] ===== VPIPE run (cool start) ====="
cd "$WORK" || exit 1
unset VPIPE_I8_GEMM VPIPE_I8_TIER
( /usr/bin/time -l "$VPIPE_BIN" --launch h3-t2v-compare.vpipeline 2>&1 ) \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/vpipe-recheck.log"
echo "[$(stamp)] VPIPE exit=${PIPESTATUS[0]}"
echo "[$(stamp)] ===== DONE ====="
