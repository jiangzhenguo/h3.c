#!/bin/bash
# Sample VPIPE's GPU thermal/clock telemetry every 10s into a log.
# Usage: ./sample-thermal.sh <out.log> <duration_s>
OUT="$1"
DUR="${2:-600}"
: > "$OUT"
end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  printf '%s %s\n' "$(date +%H:%M:%S)" \
    "$(/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe --gpu-thermal 1500 2>/dev/null)" >> "$OUT"
  sleep 8
done
