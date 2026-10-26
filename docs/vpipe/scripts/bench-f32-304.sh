#!/bin/bash
# Isolation run: the refactored F32 path forced back to the OLD 304 px tile.
# Comparing its output against the pre-change binary's output (old f32 @304)
# proves the refactor is numerically neutral on the F32 path, and isolates
# the tile-size change as the only semantic difference in the new build.
set -u
WORK=/Users/jiangzhenguo/ai/vpipe-work
H3_DIR=/Users/jiangzhenguo/ai/h3/h3.c
H3_BIN=/Users/jiangzhenguo/ai/vpipe-work/h3-bf16-bin
MODEL=/Users/jiangzhenguo/ai/h3/MiniMax-H3
PROMPT=$(cat "$WORK/prompt.txt")
COOL=${COOL:-600}

stamp() { date "+%H:%M:%S"; }

echo "[$(stamp)] ===== COOLDOWN ${COOL}s before f32@304 run ====="
sleep "$COOL"

echo "[$(stamp)] ===== h3.c f32 @ 304 px tile (old tiling) ====="
cd "$H3_DIR" || exit 1
env H3_VAE_PRECISION=f32 H3_VAE_TILE_PIXELS=304 /usr/bin/time -l "$H3_BIN" \
    -d "$MODEL" -p "$PROMPT" \
    --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
    --layers 50 --reuse 1 --core-reuse 1 --profile \
    -o "$WORK/outputs/h3c-f32-304-960x544-124f-s6-seed42.mp4" 2>&1 \
  | python3 -u -c "
import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write(f'[{time.time()-t0:8.2f}] {line}')
" > "$WORK/h3c-f32-304.log"
echo "[$(stamp)] h3.c f32@304 exit=${PIPESTATUS[0]}"
echo "[$(stamp)] ===== DONE ====="
