## Summary

Pins the video VAE decoder's default spatial tile to the 256 pixels the released checkpoint was trained on, instead of auto-tuning it.

Stacked on #80 — the diff here is just the one function.

The released checkpoint trains the decoder on 256-pixel spatial tiles (`vae_tile_size` in `FL2VA/video_vae/config.json`), and the RoPE table is derived from the tile's own latent size (see `prepare_rope()`), so a different tile size decodes a slightly different function than the one the model saw in training. The auto-tuned default picks 304 pixels for 960×544, which is 19×19 latent cells against the checkpoint's 16×16.

## Measurements

960×544×124, same machine and checkpoint:

| | 304 px (auto-tuned) | 256 px (checkpoint) |
|---|---:|---:|
| decoded frames, PSNR (Y) | — | **41.62 dB** between the two |
| decoded frames, SSIM | — | **0.9856** between the two |
| decoder time, FP32 | 126.09 s | 159.45 s |
| decoder time, BF16 | 35.4 s | 58.1 s |

## This is a fidelity-over-speed trade, not a quality fix

The two tilings are not perceptibly different — 41.6 dB is far above the noise floor between two runs of the same decoder under different thermal states (20.86 dB measured on this machine).

What the change costs is time. 304 pixels uses 8 tiles against 15, so it decodes about 26% less in FP32. Under BF16 the gap widens to 64%, because the fixed per-tile dispatch cost does not shrink along with the matrix multiplies, and after the BF16 port the decoder is compute-light enough for that overhead to matter.

So the argument for this change is strict fidelity to the checkpoint's training configuration, not a visible quality gain. If you would rather keep the auto-tuning, `H3_VAE_TILE_PIXELS` already exposes the same knob and the tile size can be selected explicitly.
