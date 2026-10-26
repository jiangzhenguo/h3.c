#!/bin/bash
# Controlled VPIPE run with i8_gemm OFF + per-section DiT profiling.
#   cool 10 min -> vpipe (i8_gemm=false, VPIPE_H3_DIT_PROFILE=1)
set -u
WORK=/Users/jiangzhenguo/ai/vpipe-work
VPIPE_BIN=/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe

echo "[$(date '+%H:%M:%S')] ===== COOLDOWN 600s before VPIPE (i8_gemm OFF) ====="
sleep 600

echo "[$(date '+%H:%M:%S')] ===== VPIPE run4 (i8_gemm OFF, cool start) ====="
cd "$WORK" || exit 1
unset VPIPE_I8_GEMM VPIPE_I8_TIER
export VPIPE_H3_DIT_PROFILE=1
bash "$WORK/sample-thermal.sh" "$WORK/thermal-vpipe4.log" 900 &
SAMPLER=$!
( /usr/bin/time -l "$VPIPE_BIN" --launch h3-t2v-compare.vpipeline \
    --stage-cfg 'generate-video::i8_gemm=false' 2>&1 ) \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/vpipe-run4.log"
echo "[$(date '+%H:%M:%S')] VPIPE run4 done"
kill $SAMPLER 2>/dev/null
echo "[$(date '+%H:%M:%S')] ===== DONE ====="
