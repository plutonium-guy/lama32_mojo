#!/usr/bin/env python3
"""Ground truth for the Mojo Waypoint port (torch CPU, deterministic).

Drives the reference WorldModel (assets/model/transformer/model.py) and
StaticKVCache (assets/model/modular_blocks.py) directly — no diffusers
pipeline machinery — with flex_attention monkeypatched to SDPA + a dense
boolean mask so it runs on CPU.

Scenario: no start image, NF latent frames from fixed-seed noise, scripted
controls (idle for the first half, button+mouse for the second). Dumps to
assets/oracle/:
  noise_f{i}.bin   f32 initial latent noise  [32*32*64]
  v0_f0.bin        f32 pass-0 velocity, frame 0, sigma 1.0 (gate 1)
  lat_f{i}.bin     f32 denoised latent per frame (gate 2)
  rgb_f{i}.bin     u8 decoded frames [4*512*1024*3] per latent (gate 3)
  frame_*.png      first/last decoded frame for eyeballing
  manifest.json    sigmas, control script, shapes, timings
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "model")
OUT_DIR = os.path.join(ROOT, "assets", "oracle")

NF = 8  # latent frames to generate


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def patch_flex_attention():
    """flex_attention -> SDPA with a dense bool mask built from the BlockMask."""
    import torch.nn.attention.flex_attention as fa

    def sdpa_flex(q, k, v, block_mask=None, enable_gqa=False, **kw):
        mask = None
        if block_mask is not None:
            T, L = block_mask.seq_lengths
            bs = block_mask.BLOCK_SIZE
            bs = bs[1] if isinstance(bs, (tuple, list)) else bs
            nb = block_mask.full_kv_num_blocks[0, 0]  # [Qb]
            idx = block_mask.full_kv_indices[0, 0]  # [Qb, KVb]
            qb_n, kv_n = idx.shape
            m = torch.zeros(qb_n, kv_n, dtype=torch.bool)
            for qb in range(qb_n):
                m[qb, idx[qb, : nb[qb]].long()] = True
            mask = (
                m.repeat_interleave(bs, 0).repeat_interleave(bs, 1)[:T, :L]
            )[None, None]
        if enable_gqa:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, 1)
            v = v.repeat_interleave(rep, 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    fa.flex_attention = sdpa_flex


def control_script(f):
    """Deterministic controls per latent frame: idle, then move+look."""
    if f < NF // 2:
        return [], (0.0, 0.0), 0
    return [17], (0.08, -0.02), 0  # one button held + mouse velocity


def ctrl_tensors(f, n_buttons, dtype):
    btns, mouse, scroll = control_script(f)
    button = torch.zeros(1, 1, n_buttons, dtype=dtype)
    for b in btns:
        button[0, 0, b] = 1.0
    mouse_t = torch.tensor([[list(mouse)]], dtype=dtype)
    scroll_t = torch.tensor([[[float(scroll > 0) - float(scroll < 0)]]], dtype=dtype)
    return mouse_t, button, scroll_t


def main():
    torch.manual_seed(0)
    patch_flex_attention()

    wm = load_module("wp_model", os.path.join(MODEL_DIR, "transformer", "model.py"))
    blocks = load_module("wp_blocks", os.path.join(MODEL_DIR, "modular_blocks.py"))

    with open(os.path.join(MODEL_DIR, "transformer", "config.json")) as fh:
        cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_") and k != "auto_map"}

    print("building WorldModel...")
    model = wm.WorldModel(**cfg)

    from safetensors.torch import load_file

    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    # tolerate a uniform prefix on checkpoint keys
    model_keys = set(model.state_dict().keys())
    if not (set(sd.keys()) & model_keys):
        for pref in ("model.", "transformer.", "module."):
            if any(k.startswith(pref) for k in sd):
                sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
                break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print("  missing (first 10):", missing[:10])
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])
    assert not missing, "missing weights — key mapping wrong"

    model = model.to(torch.bfloat16).eval()

    C, H, W = cfg["channels"], cfg["height"] * cfg["patch"][0], cfg["width"] * cfg["patch"][1]
    sigmas = torch.tensor(cfg["scheduler_sigmas"], dtype=torch.bfloat16)
    ts_mult = int(cfg["base_fps"]) // int(cfg["inference_fps"] / cfg["temporal_compression"])
    print(f"latent [{C},{H},{W}], sigmas {cfg['scheduler_sigmas']}, ts_mult {ts_mult}")

    kv_cache = blocks.StaticKVCache(model.config, batch_size=1, dtype=torch.bfloat16)

    os.makedirs(OUT_DIR, exist_ok=True)
    gen = torch.Generator().manual_seed(1234)
    latents = []
    t_frames = []

    with torch.inference_mode():
        for f in range(NF):
            t0 = time.time()
            noise = torch.randn(1, 1, C, H, W, generator=gen)
            x = noise.to(torch.bfloat16)
            # dump bf16-rounded values: Mojo must start from identical latents
            x.float().numpy().astype(np.float32).tofile(
                os.path.join(OUT_DIR, f"noise_f{f}.bin"))
            mouse, button, scroll = ctrl_tensors(f, cfg["n_buttons"], torch.bfloat16)
            fts = torch.tensor([[f * ts_mult]], dtype=torch.long)
            fidx = torch.tensor([[f]], dtype=torch.long)
            sigma = torch.empty(1, 1, dtype=torch.bfloat16)

            kv_cache.set_frozen(True)
            for i, (sig, dsig) in enumerate(zip(sigmas, sigmas.diff())):
                v = model(x=x, sigma=sigma.fill_(sig),
                          frame_timestamp=fts, frame_idx=fidx,
                          mouse=mouse, button=button, scroll=scroll,
                          kv_cache=kv_cache)
                if f == 0 and i == 0:
                    v.float().numpy().astype(np.float32).tofile(
                        os.path.join(OUT_DIR, "v0_f0.bin"))
                x = x + dsig * v

            x.float().numpy().astype(np.float32).tofile(
                os.path.join(OUT_DIR, f"lat_f{f}.bin"))
            latents.append(x.clone())

            kv_cache.set_frozen(False)
            model(x=x, sigma=sigma.fill_(0.0),
                  frame_timestamp=fts, frame_idx=fidx,
                  mouse=mouse, button=button, scroll=scroll,
                  kv_cache=kv_cache)
            dt = time.time() - t0
            t_frames.append(dt)
            print(f"frame {f}: {dt:.1f}s  (5 passes)")

    # ---- VAE decode (f32 for a stable reference) ----
    ae = load_module("wp_ae", os.path.join(MODEL_DIR, "vae", "ae_model.py"))
    with open(os.path.join(MODEL_DIR, "vae", "config.json")) as fh:
        vcfg = {k: v for k, v in json.load(fh).items()
                if not k.startswith("_") and k != "auto_map"}
    vae = ae.ChunkedStreamingTAEHV(**vcfg)
    vsd = load_file(os.path.join(MODEL_DIR, "vae", "diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(vsd)
    vae = vae.float().eval()

    t0 = time.time()
    for f, lat in enumerate(latents):
        frames = vae.decode(lat.squeeze(1).float())  # [T,H,W,3] u8
        frames.numpy().tofile(os.path.join(OUT_DIR, f"rgb_f{f}.bin"))
        if f in (0, NF - 1):
            from PIL import Image
            Image.fromarray(frames[-1].numpy()).save(
                os.path.join(OUT_DIR, f"frame_{f}.png"))
        print(f"decoded latent {f} -> {tuple(frames.shape)}")
    t_vae = time.time() - t0

    manifest = {
        "nf": NF,
        "latent_shape": [C, H, W],
        "sigmas": cfg["scheduler_sigmas"],
        "sigmas_bf16": [float(s) for s in sigmas.float()],
        "dsigmas_bf16": [float(d) for d in sigmas.diff().float()],
        "ts_mult": ts_mult,
        "controls": [
            {"buttons": control_script(f)[0],
             "mouse": list(control_script(f)[1]),
             "scroll": control_script(f)[2]}
            for f in range(NF)
        ],
        "seed": 1234,
        "sec_per_latent_frame_torch_cpu": t_frames,
        "sec_vae_decode_total": t_vae,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("oracle written to", OUT_DIR)


if __name__ == "__main__":
    main()
