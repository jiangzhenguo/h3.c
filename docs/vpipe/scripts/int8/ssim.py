#!/usr/bin/env python3
"""Frame-wise SSIM between two .rgbf32 dumps (frame-major [F,H,W,3])."""
import sys
import numpy as np

OUT = "/tmp/h3bench/rgb"
K1, K2, L = 0.01, 0.03, 1.0


def gauss_kernel(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur_axis(img, k, axis):
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    tmp = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    n = img.shape[axis]
    for i, w in enumerate(k):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        out += w * tmp[tuple(sl)]
    return out


def blur(img, k):
    return blur_axis(blur_axis(img, k, 1), k, 2)


def ssim(a, b):
    k = gauss_kernel()
    mu_x = blur(a, k); mu_y = blur(b, k)
    xx = blur(a * a, k); yy = blur(b * b, k); xy = blur(a * b, k)
    vx = xx - mu_x * mu_x; vy = yy - mu_y * mu_y; vxy = xy - mu_x * mu_y
    c1 = (K1 * L) ** 2; c2 = (K2 * L) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * vxy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def load(name):
    arr = np.memmap(f"{OUT}/{name}.rgbf32", dtype=np.float32, mode="r")
    return np.asarray(arr, dtype=np.float64).reshape(-1, 544, 960, 3)


pairs = [("f32-256", "int8-256"), ("f32-256", "pbf16-0"), ("f32-256", "f32-304")]
for a, b in pairs:
    try:
        x = load(a); y = load(b)
    except FileNotFoundError:
        print(f"skip {a} vs {b}"); continue
    print(f"{a} vs {b}: SSIM {ssim(x, y):.6f}")
