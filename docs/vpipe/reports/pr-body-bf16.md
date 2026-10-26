## Summary

The video VAE decoder now runs in BF16, which puts its matrix multiplies on the Apple tensor cores.

The decoder is the same 36-layer / 2048-wide ViT here and in [VPIPE](https://github.com/tgo-app-dev/vpipe); the difference was precision. MPS's FP32 matmul is a plain FP32 SIMD path, so on an M5 Pro the decoder was running at 6.2 TFLOP/s — which is the FP32 SIMD peak (20 cores × 128 FLOP × 2 × 1.26 GHz), i.e. no tensor core was involved at all.

## Measurements

M5 Pro (MacBook Pro 16"), 960×544×124 frames, 6 steps, seed 42, `--layers 50 --reuse 1 --core-reuse 1`. Both runs start after a 10 minute cooldown and use the same tile layout, so precision is the only variable:

| | FP32 | BF16 |
|---|---:|---:|
| video VAE decode | 159.45 s | **47.33 s** (3.37×) |
| end to end | 409.95 s | **316.93 s** (1.29×) |
| video VAE peak allocation | 9.37 GiB | **4.68 GiB** |

A standalone micro-benchmark at the decoder's real shapes (2048→6144, 2048→16384, 8192→2048, and the SDPA) gives 6.2 TFLOP/s in FP32 against 26 TFLOP/s in BF16. The FP32 number is the SIMD peak, the BF16 number is the tensor core peak.

For reference, VPIPE decodes the same clip in 45.01 s in the same session.

## Accuracy

Relative error against the FP32 decoder is 0.0048 (VPIPE's own decoder test allows 0.02). On the decoded video, BF16 against FP32 gives 23.05 dB PSNR / 0.836 SSIM — closer than two runs of the unmodified decoder differ from each other under thermal drift alone (20.86 dB / 0.842). The RGB range stays within [0, 1].

I also checked that the refactor is numerically neutral on the FP32 path: forcing `H3_VAE_PRECISION=f32` and the old tile size reproduces the pre-change binary's output to 23.12 dB, and the profile counters are identical (`submissions=56 direct=12264 linear=8120 attention=2016`).

## What changed

* `h3_scale_add_bf16` and `h3_video_qkv_rope_bf16` in `h3_shaders.metal`, with C wrappers in `h3_gpu.m` and declarations in `h3_gpu.h`. They mirror their FP32 counterparts line for line, including the half-split RoPE sign handling and the QK RMS epsilon placement, and reuse the existing bf16↔f32 helpers. Both are added to the pipeline warm-up list.
* `h3_weight_load_f32_as_bf16()` in `h3_weights.c`. The released checkpoint stores all 560 decoder tensors as FP32, so they are converted on the host with round-to-nearest-even before upload. This costs 0.18 s for the full 9.7 GiB set against 112 s saved, and resident weights drop from 9.16 GiB to 4.58 GiB.
* The 11 decoder ops switch to their BF16 variants, and the activations to BF16 tensors. The MLP keeps its bias, so it stays split into linear → swiglu → linear; the fused fc1+SwiGLU NAX kernel does not take a bias.

Everything else is untouched: the DiT, the audio VAE, and the spatial tile selection all behave exactly as before.

## Compatibility

`H3_VAE_PRECISION=f32` restores the original FP32 path. It is the reference path the test below uses for its comparison decode.

## Tests

`tests/test_video_vae.c` decodes the same pseudo-random latent in FP32 and in BF16 and reports timing, relative error, absolute error, RGB range, and the decoded frame count and resolution. It is wired into `make test` and skips when the released weights are not installed.

`tests/test_gpu_bf16_ops.c` checks the two new primitives against their FP32 counterparts in three layers: accuracy against a BF16-truncated input, a bit-exactness check where both paths consume the same values, and a double-precision host reference. It also probes the RoPE rotation and the RMS epsilon placement directly. Twelve deliberately broken copies of the shaders were checked to be caught by the assertions.

```
$ ./h3_video_vae_tests /path/to/MiniMax-H3
video VAE decode: 22 frames at 128x192
  F32 : load   0.694 s, decode   0.598 s, GPU   0.560 s, 145 MPS linears, 36 MPS SDPA, peak 9.16 GiB
  BF16: load   0.877 s, decode   0.199 s, GPU   0.172 s, 145 MPS linears, 36 MPS SDPA, peak 4.58 GiB
  BF16 decode speedup: 2.81x
  BF16 vs F32: rel-L2 0.005121, max |delta| 0.0332994, mean |delta| 0.00172416
  RGB range: F32 [0, 1], BF16 [0, 1]
ok: the BF16 video VAE matches the float32 decoder
```
