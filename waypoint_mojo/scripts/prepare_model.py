#!/usr/bin/env python3
"""Convert Waypoint-1.5-1B to one bf16 safetensors for the Mojo port.

Reads assets/model/model.safetensors (+ vae) and writes
assets/mojo/model.safetensors with a Mojo-friendly layout:

- per layer: qkv stacked [4096,2048] (fused dispatch), o, fc1, fc2,
  MLPFusion fu_x/fu_c/fu_o on ctrl layers (idx % 3 == 0)
- patchify conv folded to a matmul [2048, 256] (patch vec zero-padded
  128 -> 256 for the m%256 kernel requirement); unpatchify to [128, 2048]+bias
- ctrl_emb fc1 zero-padded [8192, 259] -> [8192, 512]
- baked conditioning tables (the whole point — see PLAN.md recipe #2):
  NoiseConditioner + all CondHeads + out AdaLN evaluated at the 5
  scheduler sigmas offline. tab.cond [5,24,6,2048] (s0,b0,g0,s1,b1,g1),
  tab.out [5,2,2048]. CondHead/NoiseConditioner weights are then dropped
  (~0.6 B params never uploaded).
- vlamb [24] value-residual scalars
- vae.* decoder+encoder weights (f16 -> bf16)
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "model")
OUT_DIR = os.path.join(ROOT, "assets", "mojo")

BF = torch.bfloat16


def bake_tables(sd, cfg):
    """Evaluate sigma-conditioned heads exactly as the bf16 reference would."""
    sigmas = torch.tensor(cfg["scheduler_sigmas"]).to(BF)
    S = sigmas.numel()
    L = cfg["n_layers"]
    D = cfg["d_model"]

    # NoiseConditioner in f32 (deployment keeps it fp32), output cast bf16
    fc1 = sd["denoise_step_emb.mlp.fc1.weight"].float()
    fc2 = sd["denoise_step_emb.mlp.fc2.weight"].float()
    freq = torch.logspace(0, -1, steps=256, base=10_000.0, dtype=torch.float32)
    s = sigmas.float() * 1000.0
    phase = s[:, None] * freq[None, :]
    emb = torch.cat((phase.sin(), phase.cos()), dim=-1) * 2 ** 0.5
    cond = (F.silu(emb @ fc1.T) @ fc2.T).to(BF)  # [S, D]

    def cond_head(prefix):
        h = F.silu(cond + sd[prefix + ".bias_in"])
        return [h @ sd[f"{prefix}.cond_proj.{i}.weight"].T for i in range(3)]

    tab = torch.zeros(S, L, 6, D, dtype=BF)
    for layer in range(L):
        p = f"transformer.blocks.{layer}"
        outs = cond_head(p + ".attn_cond_head") + cond_head(p + ".mlp_cond_head")
        for j, t in enumerate(outs):
            tab[:, layer, j] = t

    ab = F.silu(cond) @ sd["out_norm.fc.weight"].T  # [S, 2D]
    tab_out = torch.stack((ab[:, :D], ab[:, D:]), dim=1)  # [S, 2, D]
    return tab, tab_out, [float(x) for x in sigmas.float()]


def main():
    with open(os.path.join(MODEL_DIR, "transformer", "config.json")) as fh:
        cfg = json.load(fh)
    L, D = cfg["n_layers"], cfg["d_model"]

    print("reading transformer checkpoint...")
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    out = {}

    # patchify conv k2s2 -> matmul rows [D, C*2*2], zero-padded to 256 cols
    pw = sd["patchify.weight"].reshape(D, -1)  # [2048, 128]
    pad = torch.zeros(D, 256, dtype=BF)
    pad[:, : pw.shape[1]] = pw
    out["patchify"] = pad
    # unpatchify convT k2s2 -> [C*2*2, D] + bias[C]
    out["unpatch.w"] = sd["unpatchify.weight"].reshape(D, -1).T.contiguous()
    out["unpatch.b"] = sd["unpatchify.bias"]

    vlamb = torch.zeros(L, dtype=BF)
    for i in range(L):
        p = f"transformer.blocks.{i}"
        out[f"L{i}.qkv"] = torch.cat(
            [sd[f"{p}.attn.q_proj.weight"], sd[f"{p}.attn.k_proj.weight"],
             sd[f"{p}.attn.v_proj.weight"]], dim=0)
        out[f"L{i}.o"] = sd[f"{p}.attn.out_proj.weight"]
        out[f"L{i}.fc1"] = sd[f"{p}.dit_mlp.fc1.weight"]
        out[f"L{i}.fc2"] = sd[f"{p}.dit_mlp.fc2.weight"]
        vlamb[i] = sd[f"{p}.attn.v_lamb"]
        if i % cfg["ctrl_conditioning_period"] == 0:
            out[f"L{i}.fu_x"] = sd[f"{p}.ctrl_mlpfusion.fc1_x.weight"]
            out[f"L{i}.fu_c"] = sd[f"{p}.ctrl_mlpfusion.fc1_c.weight"]
            out[f"L{i}.fu_o"] = sd[f"{p}.ctrl_mlpfusion.fc2.weight"]
    out["vlamb"] = vlamb

    cw = sd["ctrl_emb.mlp.fc1.weight"]  # [8192, 259]
    cpad = torch.zeros(cw.shape[0], 512, dtype=BF)
    cpad[:, : cw.shape[1]] = cw
    out["ctrl.fc1"] = cpad
    out["ctrl.fc2"] = sd["ctrl_emb.mlp.fc2.weight"]

    print("baking sigma tables...")
    tab, tab_out, sig_bf16 = bake_tables(sd, cfg)
    out["tab.cond"] = tab
    out["tab.out"] = tab_out

    print("reading vae...")
    vsd = load_file(os.path.join(MODEL_DIR, "vae",
                                 "diffusion_pytorch_model.safetensors"))
    for name, t in vsd.items():
        out["vae." + name] = t.to(BF)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "model.safetensors")
    n = sum(t.numel() for t in out.values())
    print(f"writing {path}: {len(out)} tensors, {n/1e6:.1f}M params, "
          f"{2*n/1e9:.2f} GB bf16")
    save_file(out, path, metadata={"sigmas_bf16": json.dumps(sig_bf16)})
    print("done")


if __name__ == "__main__":
    main()
