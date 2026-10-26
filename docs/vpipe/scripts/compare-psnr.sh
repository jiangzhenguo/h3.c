#!/bin/bash
# Compare two H.264 mp4 files with ffmpeg's built-in PSNR / SSIM filters.
# Usage: compare-psnr.sh A.mp4 B.mp4 ["label"]
set -u
FF=${FF:-/opt/homebrew/bin/ffmpeg}
A="$1"; B="$2"; LABEL="${3:-$(basename "$A") vs $(basename "$B")}"

echo "label    : $LABEL"
echo "A        : $A"
echo "B        : $B"

PSNR=$("$FF" -hide_banner -i "$A" -i "$B" -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
       | grep -o "average:[0-9.]*" | head -1 | cut -d: -f2)
SSIM=$("$FF" -hide_banner -i "$A" -i "$B" -lavfi "[0:v][1:v]ssim" -f null - 2>&1 \
       | grep -o "All:[0-9.]*" | head -1 | cut -d: -f2)
NFR=$("$FF" -hide_banner -i "$A" -i "$B" -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
      | grep -c "^\[Parsed_psnr" || true)

echo "PSNR (Y) : ${PSNR:-n/a} dB"
echo "SSIM(All): ${SSIM:-n/a}"
echo "frames   : ${NFR}"
