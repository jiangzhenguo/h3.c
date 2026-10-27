#!/usr/bin/env python3
"""Isolated video-VAE decode A/B: PR #73 vs the local fork (#80 BF16 + #81 tile).

Every run replays the SAME dumped latent through the same vae_bench tool, so
the numbers are directly comparable; only the binary and the env differ.
"""
import json
import os
import re
import subprocess
import sys
import time

BENCH = "/tmp/h3bench"
MODEL = "/Users/jiangzhenguo/ai/h3/MiniMax-H3"
VAE_DIR = f"{MODEL}/FL2VA/video_vae/source"
FORK = "/Users/jiangzhenguo/ai/h3/h3.c"
PR73 = "/Users/jiangzhenguo/ai/h3/h3-pr73"
BINS = {
    "73": (PR73, f"{PR73}/vae_bench"),
    "fork": (FORK, f"{BENCH}/vae_bench_fork"),
}
LATENT = f"{BENCH}/latent.bin"
OUT = f"{BENCH}/rgb"
CACHE = f"{BENCH}/vaecache"
RESULTS = f"{BENCH}/results.jsonl"
os.makedirs(OUT, exist_ok=True)


def run(build, prec, tile=256, cache=None, ref=None, label=None):
    cwd, binary = BINS[build]
    env = dict(os.environ)
    env["H3_VAE_PRECISION"] = prec
    if tile:
        env["H3_VAE_TILE_PIXELS"] = str(tile)
    else:
        env.pop("H3_VAE_TILE_PIXELS", None)
    if cache:
        env["H3_VAE_CACHE"] = cache
    else:
        env.pop("H3_VAE_CACHE", None)
    name = label or f"{build}-{prec}-t{tile}"
    outp = f"{OUT}/{name}.rgbf32"
    argv = [binary, VAE_DIR, LATENT, outp]
    if ref:
        argv.append(f"{OUT}/{ref}.rgbf32")
    t0 = time.time()
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    rec = {
        "build": build, "prec": prec, "tile": tile, "cached": bool(cache),
        "wall_s": round(wall, 2), "name": name,
    }
    m = re.search(r"load ([\d.]+)s\s+decode ([\d.]+)s", p.stdout)
    if m:
        rec["load_s"] = float(m.group(1))
        rec["decode_s"] = float(m.group(2))
    m2 = re.search(r"vs ref: max_abs ([\d.]+)\s+rmse ([\d.]+)\s+psnr ([\d.]+) dB", p.stdout)
    if m2:
        rec["psnr_db"] = float(m2.group(3))
        rec["max_abs"] = float(m2.group(1))
        rec["rmse"] = float(m2.group(2))
    m3 = re.search(r"latent (\d+)x(\d+)x(\d+) -> (\d+) frames (\d+)x(\d+)", p.stdout)
    if m3:
        rec["frames"] = int(m3.group(4))
        rec["out"] = f"{m3.group(5)}x{m3.group(6)}"
    if p.returncode != 0:
        rec["error"] = p.stderr.strip()[-400:]
    if p.stderr.strip():
        rec["stderr"] = p.stderr.strip()[-200:]
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("all", "sweep"):
        # Round 1: forward order. Round 2: palindrome, so every config is
        # measured once early and once late -> thermal drift cancels out.
        r1 = [("73", "f32"), ("fork", "f32"), ("73", "bf16"),
              ("fork", "bf16"), ("73", "int8")]
        r2 = list(reversed(r1))
        for rnd, plan in (("r1", r1), ("r2", r2)):
            for build, prec in plan:
                ref = f"{build}-f32-t256" if prec != "f32" else None
                label = f"{build}-{prec}-t256"
                if prec == "f32":
                    label = f"{build}-f32-t256"
                run(build, prec, 256, ref=ref, label=label)

    if phase in ("all", "cache"):
        # #73 only: build the on-disk lowered-weight cache, then reuse it.
        run("73", "bf16", 256, cache=CACHE, label="73-bf16-t256-cache1")
        run("73", "bf16", 256, cache=CACHE, label="73-bf16-t256-cache2")

    if phase in ("all", "default"):
        # Each build with its own default tile policy (#73 auto-searches,
        # the fork pins 256 through #81).
        run("73", "bf16", None, label="73-bf16-tdefault")
        run("73", "f32", None, label="73-f32-tdefault")
        run("fork", "bf16", None, label="fork-bf16-tdefault")
        run("fork", "f32", None, label="fork-f32-tdefault")


if __name__ == "__main__":
    main()
