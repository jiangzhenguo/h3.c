#!/bin/bash
# Controlled vpipe-vs-h3c benchmark:
#   cool 10 min -> vpipe (with thermal sampling) -> cool 10 min -> h3.c (with thermal sampling)
set -u
WORK=/Users/jiangzhenguo/ai/vpipe-work
VPIPE_BIN=/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe
H3_DIR=/Users/jiangzhenguo/ai/h3/h3.c
MODEL=/Users/jiangzhenguo/ai/h3/MiniMax-H3
PROMPT=$(cat "$WORK/prompt.txt")
COOL=600

stamp() { date "+%H:%M:%S"; }

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before VPIPE ====="
sleep "$COOL"

echo "[$(stamp)] ===== VPIPE run3 (cool start) ====="
cd "$WORK" || exit 1
unset VPIPE_I8_GEMM VPIPE_I8_TIER
bash "$WORK/sample-thermal.sh" "$WORK/thermal-vpipe3.log" 900 &
SAMPLER=$!
( /usr/bin/time -l "$VPIPE_BIN" --launch h3-t2v-compare.vpipeline 2>&1 ) \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/vpipe-run3.log"
echo "[$(stamp)] VPIPE run3 exit=$?"
kill $SAMPLER 2>/dev/null

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before h3.c ====="
sleep "$COOL"

echo "[$(stamp)] ===== h3.c run3 (cool start) ====="
cd "$H3_DIR" || exit 1
bash "$WORK/sample-thermal.sh" "$WORK/thermal-h3c3.log" 900 &
SAMPLER=$!
( /usr/bin/time -l ./h3 -d "$MODEL" -p "$PROMPT" \
    --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
    --layers 50 --reuse 1 --core-reuse 1 --profile \
    -o "$WORK/outputs/h3c-run3-960x544-124f-s6-seed42.mp4" 2>&1 ) \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/h3c-run3.log"
echo "[$(stamp)] h3.c run3 exit=$?"
kill $SAMPLER 2>/dev/null

echo "[$(stamp)] ===== DONE ====="
