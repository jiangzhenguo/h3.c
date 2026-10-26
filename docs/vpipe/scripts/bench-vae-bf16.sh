#!/bin/bash
# A/B benchmark for the h3.c video-VAE bf16 port.
#   cool 10 min -> h3.c (bf16 default) -> cool 10 min -> h3.c (H3_VAE_PRECISION=f32)
# Same prompt / geometry / seed / step count as the earlier VPIPE-vs-h3c benchmark,
# so the "video VAE decoder" line of --profile is directly comparable.
set -u
WORK=/Users/jiangzhenguo/ai/vpipe-work
H3_DIR=/Users/jiangzhenguo/ai/h3/h3.c
H3_BIN=${H3_BIN:-/Users/jiangzhenguo/ai/vpipe-work/h3-bf16-bin}
MODEL=/Users/jiangzhenguo/ai/h3/MiniMax-H3
PROMPT=$(cat "$WORK/prompt.txt")
COOL=${COOL:-600}

stamp() { date "+%H:%M:%S"; }

run_h3() {  # $1 = tag, $2 = output mp4, rest = env assignments
  local tag="$1"; shift
  local out="$1"; shift
  cd "$H3_DIR" || exit 1
  env "$@" /usr/bin/time -l "$H3_BIN" -d "$MODEL" -p "$PROMPT" \
      --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
      --layers 50 --reuse 1 --core-reuse 1 --profile \
      -o "$out" 2>&1 \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/h3c-$tag.log"
  echo "[$(stamp)] h3.c $tag exit=${PIPESTATUS[0]}"
}

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before bf16 run ====="
sleep "$COOL"
echo "[$(stamp)] ===== h3.c bf16 (default) ====="
run_h3 bf16 "$WORK/outputs/h3c-bf16-960x544-124f-s6-seed42.mp4" H3_VAE_PRECISION=bf16

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before f32 run ====="
sleep "$COOL"
echo "[$(stamp)] ===== h3.c f32 (baseline path) ====="
run_h3 f32 "$WORK/outputs/h3c-f32ref-960x544-124f-s6-seed42.mp4" H3_VAE_PRECISION=f32

echo "[$(stamp)] ===== DONE ====="
