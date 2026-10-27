#!/bin/bash
set -u
BENCH=/tmp/h3bench
MODEL=/Users/jiangzhenguo/ai/h3/MiniMax-H3
FORK=/Users/jiangzhenguo/ai/h3/h3.c
PROMPT_FILE=$FORK/docs/vpipe/config/prompt.txt
COOL=${COOL:-300}
stamp() { date '+%H:%M:%S'; }
run() {
  local tag=$1; shift
  echo "[$(stamp)] ===== RUN $tag ====="
  cd "$FORK" || return 1
  env "$@" /usr/bin/time -l "$FORK/h3" -d "$MODEL" -p "$(cat $PROMPT_FILE)" \
      --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
      --layers 50 --reuse 1 --core-reuse 1 --profile \
      -o "$BENCH/out-$tag.mp4" > "$BENCH/e2e-$tag.log" 2>&1
  echo "[$(stamp)] $tag exit=$?"
  grep -aE 'video VAE|real|maximum resident' "$BENCH/e2e-$tag.log" | tail -6
}
sleep "$COOL"
run fork-int8 H3_VAE_PRECISION=int8
sleep "$COOL"
run fork-bf16-e2e H3_VAE_PRECISION=bf16
echo "[$(stamp)] ===== E2E INT8 DONE ====="
