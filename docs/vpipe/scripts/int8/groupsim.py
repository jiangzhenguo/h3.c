#!/usr/bin/env python3
"""Simulate the VAE's INT8 activation quantization: per row (current) vs
per 1024-element group, on data with the real activation's heavy tail."""
import numpy as np

rng = np.random.default_rng(7)
rows, din, dout = 512, 2048, 1024

# The rms_norm output of the VAE: rms 0.53, a few large outliers up to ~10.6.
x = rng.standard_normal((rows, din)) * 0.53
n_out = rows * 4
idx = rng.choice(rows * din, n_out, replace=False)
x.ravel()[idx] = rng.uniform(4.0, 10.6, n_out)
# The projections add a little correlation on top of the white noise.
w = rng.standard_normal((din, dout)) * 0.02
w[:, 0] += 0.05

ref = x @ w


def q_rows(t):
    s = np.abs(t).max(axis=1, keepdims=True) / 127.0
    return np.round(t / s).clip(-127, 127) * s


def q_groups(t, g=1024):
    out = np.empty_like(t)
    for i in range(0, t.shape[1], g):
        blk = t[:, i:i + g]
        s = np.abs(blk).max(axis=1, keepdims=True) / 127.0
        out[:, i:i + g] = np.round(blk / s).clip(-127, 127) * s
    return out


def q_weights(t):
    s = np.abs(t).max(axis=0, keepdims=True) / 127.0
    return np.round(t / s).clip(-127, 127) * s


for name, xq in (("per-row ", q_rows(x)), ("per-group", q_groups(x))):
    for wq, wtag in ((q_weights(w), "int8 weight"), (w, "fp32 weight")):
        got = xq @ wq
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"{name} act + {wtag}: rel-L2 {rel:.5f}")
    # activation-only error share
    print(f"   input quant rel-L2: {np.linalg.norm(xq - x) / np.linalg.norm(x):.5f}")
