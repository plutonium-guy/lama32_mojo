#!/usr/bin/env python3
"""INT8 group-wise weight quantization (W8A32) for the Mojo kernel path.

Packs every 2-D weight [n, m] row-wise as [f16 scales (m/64) | int8 qs (m)],
symmetric per group of 64 along the input dim:

    scale_g = max|w_g| / 127   (rounded to f16 first, so dequant matches)
    q       = clamp(round(w / scale_g), -127, 127)

The packed row is stored as an F16 tensor of shape [n, m/64 + m/2] under the
original tensor name, so safetensors.mojo + resident.mojo load it unchanged
(2 bytes per element, offsets in u16 units). 1-D tensors (norms) stay bf16.

Reads  assets/model/model.safetensors  (bf16)
Writes assets/model-q8/model.safetensors
"""

import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "assets", "model", "model.safetensors")
OUT_DIR = os.path.join(ROOT, "assets", "model-q8")

GROUP = 64


def pack_q8(w: torch.Tensor) -> torch.Tensor:
    n, m = w.shape
    assert m % GROUP == 0 and m % 2 == 0, (n, m)
    x = w.float().numpy().reshape(n, m // GROUP, GROUP)

    scale = np.abs(x).max(axis=2) / 127.0                # (n, m/64)
    scale = np.maximum(scale, 1e-8).astype(np.float16)   # kernel sees f16
    q = np.rint(x / scale[:, :, None].astype(np.float32))
    q = np.clip(q, -127, 127).astype(np.int8).reshape(n, m)

    packed = np.concatenate(
        [scale.view(np.uint16), q.view(np.uint16)], axis=1)
    return torch.from_numpy(packed.view(np.float16))


def main():
    sd = load_file(IN_PATH)
    out = {}
    q_bytes = raw_bytes = 0
    for name, t in sd.items():
        raw_bytes += t.numel() * 2
        if t.dim() == 2:
            out[name] = pack_q8(t)
            q_bytes += out[name].numel() * 2
        else:
            out[name] = t.to(torch.bfloat16)
            q_bytes += t.numel() * 2

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "model.safetensors")
    save_file(out, out_path)
    print(f"wrote {out_path}")
    print(f"bf16 {raw_bytes/1e9:.2f} GB -> q8 {q_bytes/1e9:.2f} GB "
          f"({raw_bytes/q_bytes:.2f}x)")


if __name__ == "__main__":
    main()
