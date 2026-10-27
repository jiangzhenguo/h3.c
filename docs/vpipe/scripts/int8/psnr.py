#!/usr/bin/env python3
"""Pairwise PSNR/RMSE/max-abs between the .rgbf32 dumps of the sweep."""
import glob
import os
import numpy as np

OUT = "/tmp/h3bench/rgb"
files = sorted(glob.glob(f"{OUT}/*.rgbf32"))
if not files:
    raise SystemExit("no dumps yet")

def load(path):
    return np.memmap(path, dtype=np.float32, mode="r")

def compare(a, b):
    ma, mb = load(a), load(b)
    if ma.shape != mb.shape:
        return None
    n = ma.size
    max_abs = 0.0
    sq = 0.0
    step = 20_000_000
    for i in range(0, n, step):
        x = np.asarray(ma[i:i + step], dtype=np.float64)
        y = np.asarray(mb[i:i + step], dtype=np.float64)
        d = x - y
        max_abs = max(max_abs, float(np.abs(d).max()))
        sq += float((d * d).sum())
    mse = sq / n
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    return max_abs, (mse ** 0.5), psnr

base = os.path.basename
print(f"{'file':<28} {'size':>14}")
for f in files:
    print(f"  {base(f):<26} {os.path.getsize(f)/1e6:>10.1f} MB")
print()
print(f"{'A':<26} {'B':<26} {'max_abs':>10} {'rmse':>10} {'psnr_dB':>10}")
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        r = compare(files[i], files[j])
        if r is None:
            continue
        print(f"{base(files[i])[:-6]:<26} {base(files[j])[:-6]:<26} "
              f"{r[0]:>10.5f} {r[1]:>10.6f} {r[2]:>10.2f}")
