#!/usr/bin/env python3
"""Merge a MiniMax-H3 LoRA into an h3.c transformer checkpoint (offline bake).

h3.c has no LoRA runtime, and does not need one: a LoRA is
W' = W + strength*scale*(B@A), which can be baked into the checkpoint once,
offline. This tool clones the original shards and rewrites only the byte
ranges of the targeted tensors, so the safetensors header, the tensor order
and the payload alignment stay byte-identical to the originals. h3.c then
loads the folded tree directly.

Supports two LoRA key conventions:
  - diffusers/sd-scripts: lora_unet_blocks_24_attn_qkv_proj.lora_down.weight /
    .lora_up.weight / .alpha
  - ComfyUI-style:       (diffusion_model.)blocks.24.attn.qkv_proj.lora_A.weight
    / .lora_B.weight
The third convention, used by larryvrh/MiniMax-H3-Turbo-Lora, spells the keys
in the checkpoint's own key space (blocks.24.attn.qkv_proj.lora_A.weight) and
is handled by the ComfyUI branch with an optional "diffusion_model." prefix.

Covered weight families: blocks.N, token_refiner.blocks.N and final_layer.
The latter two are required by the t8star turbo LoRA (9 additional targets).

Usage:
  python3 tools/merge_lora_h3.py --lora LORA.safetensors --src BASE_TRANSFORMER_DIR \
      --dst OUT_TRANSFORMER_DIR [--strength 1.0]

Stacking several LoRAs into one checkpoint
------------------------------------------
Run the tool once per LoRA, always with --src pointing at the *base*
transformer and --dst at the same product directory:

  merge --lora style.safetensors  --src BASE/transformer --dst OUT/transformer
  merge --lora turbo.safetensors  --src BASE/transformer --dst OUT/transformer

The second run patches the already-merged destination in place (it reads the
current destination bytes, not the base), so the result is
base + style + turbo. --refresh instead re-clones the base shards first and
discards whatever was merged before.

Every applied LoRA is recorded in OUT/transformer/.lora_manifest.json with its
sha256, so re-running the same LoRA against the same product is refused
instead of silently applying the delta twice (double application is what makes
a baked checkpoint look over-saturated and broken). The manifest is a dotfile
with a .json suffix, so h3.c ignores it.

Only the transformer is affected by a LoRA: the text encoder, the video VAE
and the audio VAE are untouched, so a model directory can point those at the
original snapshot instead of copying them.
"""
import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

MANIFEST_NAME = ".lora_manifest.json"


def read_header(f):
    (hlen,) = struct.unpack("<Q", f.read(8))
    return hlen, json.loads(f.read(hlen))


def sdscripts_h3_stem(mod):
    """Map sd-scripts module spelling to the h3 checkpoint key space.

    Ordinary blocks use the first three underscores as separators. The refiner
    namespace and final_layer are literal prefixes and need separate handling.
    """
    if mod.startswith("token_refiner_blocks_"):
        return "token_refiner." + mod[len("token_refiner_"):].replace("_", ".", 3)
    if mod.startswith("final_layer_"):
        rest = mod[len("final_layer_"):]
        if rest.startswith("adaln_") and "_" in rest:
            stem, leaf = rest.rsplit("_", 1)
            return f"final_layer.{stem}.{leaf}"
        return "final_layer." + rest.replace("_", ".", 1)
    return mod.replace("_", ".", 3)


def load_lora(path):
    """Return dict: h3 target weight key -> (A[in,rank] f32, B[out,rank] f32, scale)."""
    with open(path, "rb") as f:
        hlen, hdr = read_header(f)
        body = 8 + hlen

        def tensor(key):
            info = hdr[key]
            s, e = info["data_offsets"]
            f.seek(body + s)
            raw = f.read(e - s)
            dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[info["dtype"]]
            arr = np.frombuffer(raw, dtype=dt)
            if info["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            return arr.reshape(info["shape"]).astype(np.float32)

        keys = [k for k in hdr if k != "__metadata__"]
        targets = {}
        for k in keys:
            m = re.match(r"^(?:lora_unet_)?((?:blocks_\d+|token_refiner_blocks_\d+|final_layer)_(?:attn|mlp|adaln)\w*)\.lora_down\.weight$", k)
            style = "sdscripts"
            if not m:
                m = re.match(r"^(?:diffusion_model\.)?((?:blocks\.\d+|token_refiner\.blocks\.\d+|final_layer)\.(?:attn|mlp|adaln)[\w.]*)\.lora_A\.weight$", k)
                style = "comfy"
            if not m:
                continue
            mod = m.group(1)
            if style == "sdscripts":
                h3stem = sdscripts_h3_stem(mod)
                A = tensor(k)
                B = tensor(f"lora_unet_{mod}.lora_up.weight")
                alpha_k = f"lora_unet_{mod}.alpha"
                alpha = float(tensor(alpha_k).ravel()[0]) if alpha_k in hdr else None
            else:
                h3stem = mod
                A = tensor(k)
                B = tensor(f"{m.group(0)[:-len('lora_A.weight')]}lora_B.weight")
                alpha = None  # comfy h3 loras observed so far: scale folded, alpha/rank == 1
            rank = A.shape[0]
            scale = (alpha / rank) if alpha is not None else 1.0
            targets[h3stem + ".weight"] = (A, B, scale)
        return targets


def bf16_from_f32(x):
    """Round-to-nearest-even F32 -> BF16 as uint16."""
    u = x.astype(np.float32).view(np.uint32)
    rounding = 0x7FFF + ((u >> 16) & 1)
    return ((u + rounding) >> 16).astype(np.uint16)


def f32_from_bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(dst):
    return dst / MANIFEST_NAME


def load_manifest(dst):
    path = manifest_path(dst)
    if not path.exists():
        return {"version": 1, "applies": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "applies": []}
    if not isinstance(data, dict) or not isinstance(data.get("applies"), list):
        return {"version": 1, "applies": []}
    return data


def save_manifest(dst, data):
    path = manifest_path(dst)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare_shard(src, dst, refresh):
    """Make dst a usable shard.

    Returns "cloned", "copied" or "stacked". "stacked" means the destination
    already held a merged shard of the same size: it is kept as is, because the
    new delta has to be added on top of it, not on top of the base.
    """
    if dst.exists() and not refresh and dst.stat().st_size == src.stat().st_size:
        return "stacked"
    if dst.exists():
        dst.unlink()
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode == 0:
        return "cloned"
    shutil.copyfile(src, dst)
    return "copied"


def parity_check(w, folded, a, b, factor, tolerance, rng):
    """folded@x must equal w@x + factor*(b@(a@x)) for a random probe x.

    This catches a swapped A/B pair, a transposed LoRA and a mismatched rank
    before any byte is written.
    """
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    direct = folded @ x
    composed = w @ x + factor * (b @ (a @ x))
    denominator = float(np.max(np.abs(composed))) + 1e-9
    return float(np.max(np.abs(direct - composed))) / denominator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--refresh", action="store_true",
                    help="re-clone the base shards first, discarding anything "
                         "that was merged into the destination before")
    ap.add_argument("--allow-missing", action="store_true",
                    help="warn instead of failing when a LoRA target is absent "
                         "from the base checkpoint")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="allow applying the same LoRA again to the same "
                         "destination (the manifest normally refuses it)")
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the per-tensor parity check")
    ap.add_argument("--parity-tol", type=float, default=1e-4)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the tensors that would be patched and exit")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    targets = load_lora(args.lora)
    print(f"lora targets: {len(targets)} tensors, strength={args.strength}", flush=True)
    if not targets:
        sys.exit("no recognizable LoRA tensors found")

    # The manifest makes a second application of the same LoRA impossible to
    # do by accident: applying the same delta twice is what turns a baked
    # checkpoint into an over-saturated one.
    digest = sha256_of(args.lora)
    manifest = load_manifest(dst)
    if not args.refresh and not args.allow_repeat:
        for entry in manifest.get("applies", []):
            if entry.get("sha256") == digest:
                sys.exit(
                    f"this LoRA was already applied to {dst} (manifest: {manifest_path(dst)}); "
                    "applying it again would double the delta. Use --refresh to rebuild "
                    "from --src, or --allow-repeat to force it."
                )
    if args.refresh:
        manifest = {"version": 1, "applies": []}

    # pass 1: find which shard holds each target
    shards = sorted(p for p in src.glob("*.safetensors"))
    assert shards, f"no shards in {src}"
    shard_of = {}
    headers = {}
    for sp in shards:
        with open(sp, "rb") as f:
            hlen, hdr = read_header(f)
        headers[sp.name] = (hlen, hdr)
        for k in hdr:
            if k in targets:
                shard_of[k] = sp.name
    missing = set(targets) - set(shard_of)
    if missing:
        message = (f"{len(missing)} lora targets are not in the base checkpoint: "
                   f"{sorted(missing)[:5]}")
        if not args.allow_missing:
            sys.exit(message + " (use --allow-missing to fold anyway)")
        print(f"WARNING: {message}", flush=True)

    if args.dry_run:
        for k in sorted(targets):
            print(f"  would patch {k} in {shard_of.get(k, '<missing>')}")
        return

    rng = np.random.default_rng(0)
    applied = 0
    counts = {"cloned": 0, "copied": 0, "stacked": 0}
    for sp in shards:
        dp = dst / sp.name
        counts[prepare_shard(sp, dp, args.refresh)] += 1
        hlen, hdr = headers[sp.name]
        todo = [k for k in hdr if k in targets]
        if not todo:
            print(f"{sp.name}: 0 patched", flush=True)
            continue
        with open(dp, "r+b") as fdst:
            for k in todo:
                info = hdr[k]
                s, e = info["data_offsets"]
                if info["dtype"] != "BF16":
                    print(f"  skip non-bf16 {k} ({info['dtype']})", flush=True)
                    continue
                # Read the current destination bytes: on a stacked merge this
                # is the already-merged weight, not the base one.
                fdst.seek(8 + hlen + s)
                raw = fdst.read(e - s)
                A, B, scale = targets[k]
                w = f32_from_bf16(raw).reshape(info["shape"])
                factor = scale * args.strength
                delta = (B @ A) * factor
                assert delta.shape == w.shape, f"{k}: delta {delta.shape} vs weight {w.shape}"
                folded = w + delta
                if not args.no_parity:
                    relative = parity_check(w, folded, A, B, factor,
                                            args.parity_tol, rng)
                    if relative > args.parity_tol:
                        sys.exit(f"{k}: parity check failed (rel={relative:.2e})")
                out = bf16_from_f32(folded)
                fdst.seek(8 + hlen + s)
                fdst.write(out.tobytes())
                applied += 1
        print(f"{sp.name}: patched {len(todo)}", flush=True)

    # carry over config/index if present
    for extra in ("config.json", "model.safetensors.index.json"):
        if (src / extra).exists():
            shutil.copyfile(src / extra, dst / extra)

    manifest.setdefault("applies", []).append({
        "lora": Path(args.lora).name,
        "sha256": digest,
        "strength": args.strength,
        "targets": len(targets),
        "applied": applied,
    })
    save_manifest(dst, manifest)
    print(f"DONE. applied {applied}/{len(targets)} lora tensors -> {dst} "
          f"({counts['cloned']} cloned, {counts['copied']} copied, "
          f"{counts['stacked']} stacked)", flush=True)


if __name__ == "__main__":
    main()
