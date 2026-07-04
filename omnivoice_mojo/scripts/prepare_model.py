#!/usr/bin/env python3
"""Convert the F32 OmniVoice checkpoint to one bf16 safetensors for Mojo.

Reads assets/model/{model.safetensors, audio_tokenizer/model.safetensors}
(run `pixi run download` first) and writes assets/mojo/model.safetensors:

- llm.* backbone + audio_embeddings + audio_heads (verbatim names)
- codec decode path under a `codec.` prefix: quantizer codebooks +
  project_out, fc2, acoustic_decoder.*

Everything bf16; Mojo kernels read bf16 weights with f32 activations.
"""

import os

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "model")
OUT_DIR = os.path.join(ROOT, "assets", "mojo")

NUM_QUANTIZERS = 8


def main():
    out = {}

    main_path = os.path.join(MODEL_DIR, "model.safetensors")
    print(f"reading {main_path}")
    main_sd = load_file(main_path)
    for name, t in main_sd.items():
        if name.startswith("llm.") or name in (
            "audio_embeddings.weight",
            "audio_heads.weight",
        ):
            out[name] = t.to(torch.bfloat16)

    codec_path = os.path.join(MODEL_DIR, "audio_tokenizer", "model.safetensors")
    print(f"reading {codec_path}")
    codec_sd = load_file(codec_path)
    for name, t in codec_sd.items():
        keep = name.startswith("acoustic_decoder.") or name.startswith("fc2.")
        if name.startswith("quantizer.quantizers."):
            idx = int(name.split(".")[2])
            keep = idx < NUM_QUANTIZERS and (
                name.endswith(".codebook.embed") or ".project_out." in name
            )
        if keep:
            out["codec." + name] = t.to(torch.bfloat16)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "model.safetensors")
    n = sum(t.numel() for t in out.values())
    print(f"writing {out_path}: {len(out)} tensors, {n/1e6:.1f}M params, "
          f"{2*n/1e9:.2f} GB bf16")
    save_file(out, out_path)
    print("done")


if __name__ == "__main__":
    main()
