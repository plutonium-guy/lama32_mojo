# waypoint_mojo — Waypoint-1.5-1B world model on Apple Metal in Mojo

Port of [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
(real-time interactive video world model: frame-causal rectified-flow DiT +
taehv tiny VAE) to Mojo on Metal, following this repo's llama32/qwen3/omnivoice
pattern. `PLAN.md` documents the general recipe for running any diffusion
model efficiently; this is the worked example.

## Layout

- `src/wp_common.mojo` — DiT kernels: adaLN with baked sigma tables, fused
  q/k rms_norm + ortho 3-axis RoPE, value residual, compact ring-KV
  attention (16 slots + current-frame tail per layer), MLPFusion controller
  conditioning, patchify/unpatchify.
- `src/wp_vae.mojo` — streaming taehv decoder (MemBlock temporal state,
  TGrow time upsampling, pixel-shuffle): 1 latent → 4 RGB frames 512×1024.
- `src/waypoint_gpu.mojo` — model shell + oracle validation (`pixi run test`).
- `scripts/prepare_model.py` — checkpoint converter; bakes NoiseConditioner
  + all CondHeads + out-AdaLN at the 5 scheduler sigmas into lookup tables
  (drops ~0.6 B params of conditioning weights/compute). 3.7 GB → 2.49 GB.
- `scripts/gen_oracle.py` — torch-CPU ground truth (flex_attention
  monkeypatched to SDPA + dense mask). ~19 min/frame on M4 CPU.

## Usage

```
pixi run download   # HF weights -> assets/model (bf16 file only)
pixi run convert    # -> assets/mojo/model.safetensors (2.49 GB)
pixi run oracle     # torch CPU ground truth -> assets/oracle (~2.5 h)
pixi run test       # Mojo vs oracle: 3 gates + timing
```

## Verification gates (M4, 2026-07-05, 8 frames — ALL PASS)

Run `pixi run test --teacher` for the meaningful comparison: cache passes
persist the *oracle's* latents, so per-frame error can't compound.

| gate | teacher-forced | free-running |
|---|---|---|
| 1. pass-0 velocity max-abs-diff | 0.14 (bound 0.25*) | same |
| 2. per-frame latent max-abs-diff | 0.045–0.22 (frames 1–7) | 0.43 → 2.7 by frame 7 |
| 3. decoded RGB PSNR | **53.6–55.7 dB** | 38 dB → 20 dB |

*The torch reference computes every op in bf16; Mojo accumulates in f32.
A torch-vs-torch A/B on a small WorldModel puts pure bf16↔f32 drift at
~0.1 max on this velocity scale, so 0.25 is the bug-vs-precision line.

The free-running divergence is autoregressive chaos, not error: the world
model feeds its own output back through the KV cache, so precision-level
differences separate trajectories after ~5 frames. Teacher forcing shows
each individual frame is computed faithfully (and the 54 dB VAE numbers are
a pure decoder comparison — the taehv port is essentially exact).

## Performance (correctness kernels, before the perf phase)

~20–56 s per latent frame (5 transformer passes; grows with attention
context) ≈ 130–200 GFLOPS effective, + 6–8 s taehv decode per latent
(4 output frames). Oracle torch-CPU reference: 580–1155 s per latent frame,
2.4 s VAE. The perf phase in PLAN.md (simdgroup matmuls, fused QKV,
flash-style attention, 2-step sigmas) is what stands between this and
interactive rates; an RTX 5090 does 56 fps for scale.
