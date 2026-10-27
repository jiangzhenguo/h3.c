#!/usr/bin/env python3
"""The final H.264 encode is the real noise floor: compare it against the
F32 VAE decode, and put the INT8 VAE error in the same units."""
import numpy as np
import sys
sys.path.insert(0, "/tmp/h3bench")
from ssim import ssim, blur_axis, gauss_kernel

F, H, W = 124, 544, 960
ref = np.asarray(np.memmap("/tmp/h3bench/rgb/f32-256.rgbf32", dtype=np.float32,
                            mode="r"), dtype=np.float64).reshape(F, H, W, 3)
codec = np.fromfile("/tmp/h3bench/rgb/h264.rgbf24", dtype=np.uint8)
codec = codec.reshape(F, H, W, 3).astype(np.float64) / 255.0
i8 = np.asarray(np.memmap("/tmp/h3bench/rgb/final-int8.rgbf32", dtype=np.float32,
                          mode="r"), dtype=np.float64).reshape(F, H, W, 3)


def stats(a, b, name):
    d = a - b
    mse = float((d * d).mean())
    rmse = mse ** 0.5
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    print(f"{name:<28} max_abs {np.abs(d).max():.5f}  rmse {rmse:.6f}  "
          f"psnr {psnr:6.2f} dB  色阶 {rmse / (1 / 255):.3f}  "
          f"ssim {ssim(a, b):.6f}")


stats(ref, codec, "H.264 编码器自身误差")
stats(ref, i8, "INT8 VAE 误差")
