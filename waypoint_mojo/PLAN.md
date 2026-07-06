# Running diffusion models fast — plan, with Waypoint-1.5-1B as the worked example

Goal: a repeatable recipe for running *any* diffusion model efficiently on
Apple Metal in Mojo (following the llama32/qwen3/omnivoice pattern in this
repo), instantiated as a full port of
[Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
— a real-time interactive video world model (frame-causal rectified-flow
diffusion transformer).

## The generic recipe

Cost of a diffusion model = `steps × backbone_flops + VAE_flops`. Every
optimization is one of these levers:

1. **Pick a few-step model / scheduler.** Sampling steps multiply the whole
   backbone cost linearly. Distilled / self-forced models (DMD, LCM,
   turbo) run at 2–8 steps instead of 20–50. Waypoint ships self-forced:
   sigmas `[1.0, 0.9, 0.75, 0.3, 0.0]` = 4 steps, and 2-step works.
2. **Precompute the conditioning.** Timestep/noise embeddings and every
   adaLN projection derived from them are functions of the sigma schedule —
   a *finite set*. Tabulate them offline; at inference they are lookups.
   (Waypoint's own runtime does this on the fly via bf16 LUTs; we bake it
   into the converted checkpoint — this also removes ~0.6 B params of
   resident weights and compute.) Same for text conditioning: encode once
   per prompt, never per step.
3. **Diffusion transformer ≠ LLM decode.** Every pass processes a full
   token batch (here 512 tokens/frame), so matmuls are **compute-bound** —
   the s=1 bandwidth/instruction-bound tricks from LLM decode
   ([QUANTIZATION.md](../QUANTIZATION.md)) don't transfer. What matters is
   tiled matmul kernels with input reuse (commit 4c07235's `k_mm_tile`
   direction), fused QKV / fused epilogues, and f32 accumulation in bf16
   compute.
4. **Cache what is frozen across steps.** For autoregressive video/world
   models: history KV is constant during a frame's denoise steps — compute
   it once per frame (Waypoint: frozen ring-buffer KV cache, one extra
   "cache pass" at sigma 0 persists the clean frame). For image models the
   analog is caching text cross-attn K/V across steps.
5. **Cheap VAE.** A distilled tiny autoencoder (TAESD/TAEHV family) is
   10–50× cheaper than the full KL-VAE and visually close. Waypoint already
   uses one (taehv1_5, pure 3×3/1×1 convs + ReLU).
6. **Quantize for memory, expect little speed.** Compute-bound batched
   matmuls don't get faster from W8; do it to halve residency (int8
   pipeline from QUANTIZATION.md reuses as-is). Measure before believing.
7. **Verify every stage against a framework oracle** before optimizing
   anything (the repo's standing rule): repack weights → single-pass
   max-abs-diff gate → multi-step drift gate → output-quality gate (PSNR) →
   only then perf work.

## Waypoint-1.5-1B analysis (from `transformer/model.py` + `modular_blocks.py`)

Three parts, like every latent diffusion system:
**conditioning → backbone → VAE**.

### Backbone: WorldDiT, 24 layers, d_model 2048

- Latent frame `[C=32, H=32, W=64]` → patchify Conv2d k2 s2 (= linear
  128→2048 per 2×2 patch) → **512 tokens** (grid 16×32) per frame.
- Per block: adaLN (`rms_norm(x)·(1+s)+b`, gate `g` on residual) with
  **two CondHeads** (attn and mlp, 3× d→d linears each) driven by the
  noise embedding; self-attn GQA 32 q-heads / 16 kv-heads, d_head 64;
  q/k rms_norm (no weight); **ortho RoPE** (3-axis: x,y spatial normalized
  to (−1,1) × 8 dims each, t = frame index × 16 dims; pair-rotate via
  even/odd unfold, output layout `[y0…|y1…]` non-interleaved);
  **value residual** (`v = lerp(v, v_layer0, v_lamb)` per layer, scalar);
  plain MLP 2048→8192→2048 with SiLU (not SwiGLU).
- **Ctrl conditioning** at layers 0,3,…,21: MLPFusion
  `x += fc2(silu(fc1_x(rms(x)) + fc1_c(rms(ctrl_emb))))`; ctrl_emb = MLP
  over `[mouse(2) | buttons(256 one-hot) | scroll(1)]` → 2048, per frame.
  `fc1_c(rms(ctrl_emb))` is per-frame constant → compute once, broadcast.
- Prompt conditioning is **null** in the 1B config → no cross-attn, no
  UMT5 text encoder at all. Skip entirely.
- Out: AdaLN(cond) → SiLU → unpatchify ConvT k2 s2 (+bias) → velocity in
  latent space.
- Params: ~1.86 B total in checkpoint; ~1.15 B active per pass once
  CondHeads/NoiseConditioner are tabulated (they only ever see 5 sigmas).

### Attention pattern + KV cache (the interesting part)

- Frame-causal: each forward denoises **one frame (512 tokens)** attending
  to (a) itself via a tail slice, (b) a ring buffer of past frames.
- 18 "local" layers: ring of `local_window=16` frames.
  6 "global" layers (idx%4==3): `global_window=128` with
  `pinned_dilation=8` → only every 8th frame is persisted, into
  `128/8=16` buckets. **Effective span per layer ≤ 17×512 = 8704 tokens.**
- During the 4 denoise steps the cache is *frozen* (current frame writes
  only the tail slice); a 5th pass at sigma=0 with the denoised latent
  persists it into the ring. So: 5 transformer passes per latent frame.
- Mojo layout: per layer, compact `[17×512, kv_dim]` buffer (16 ring
  slots + tail); block-mask semantics reduce to "attend all written
  slots" — no per-position masking inside the kernel, just a slot count.

### Sampling

Rectified flow, Euler: `x += (σ_{i+1} − σ_i) · v(x, σ_i)` over sigmas
`[1.0, 0.9, 0.75, 0.3, 0.0]`. Deterministic given the initial noise.
One-pass CFG is baked in by self-forcing → no uncond pass, no guidance
math at inference.

### VAE: taehv1_5 (streaming tiny autoencoder)

- Decoder: Clamp(tanh(x/3)·3) → conv stack (channels 256/128/64/64),
  3×3 convs + ReLU, MemBlocks (concat previous-frame feature, i.e. per-block
  temporal state), 3× spatial 2× upsample, 2× temporal TGrow (1×1 convs),
  final pixel_shuffle(2) → **one latent → 4 RGB frames 512×1024**, ~30 M
  params, all conv — maps to a small set of Metal kernels (conv3x3+relu,
  conv1x1 as matmul, nearest-upsample, pixel_shuffle).
- Streaming: first latent primes MemBlock state (3 warmup frames trimmed);
  each subsequent latent yields 4 frames. State = one feature map per
  MemBlock; ping-pong buffers in the activation arena.

### Budget (M4, 10-core GPU, ~4.4 TFLOPS f16)

Per latent frame ≈ 5 passes × 512 tok × 1.15 B × 2 ≈ 5.9 TFLOP + VAE
(~0.1 TFLOP). At a realistic 40–60 % of peak → **~2–3 s per latent frame
≈ 1.3–2 rendered fps** at 512×1024. Real-time (like the RTX 5090's 56 fps)
is not on the table on this GPU; the goal is the *efficient ceiling*:
measured tok-passes/s and fps, plus the 2-step mode (3 passes → ~1.7×).

## Mojo mapping

Subproject `waypoint_mojo/`, reusing `../src` (safetensors.mojo,
resident.mojo, llama_common.mojo kernels where shapes allow):

| piece | plan |
|---|---|
| weights | `scripts/prepare_model.py`: transformer + vae safetensors → one packed bf16 file; **bake per-sigma tables**: NoiseConditioner(σ) [5×2048], per-layer CondHead outputs s0,b0,g0,s1,b1,g1 [5×24×6×2048], out-AdaLN a,b [5×2×2048]; fold patchify/unpatchify convs to plain matmul layout; drop CondHead/NoiseConditioner/CFG weights |
| matmul | batched-token tiled `mm` (512×K×N) — the tiled kernels from commit 4c07235, extended to rectangular batches; fused QKV and fused fc1_x+fc1_c where contiguity allows |
| adaLN | one kernel: rms_norm + per-frame scale/bias (table row), and gated residual add |
| rope | host precomputes cos/sin `[512, 32]` per frame index (t axis changes only); kernel applies pair-rotation with the non-interleaved output layout |
| attention | GQA scores/out kernels over compact KV `[17×512]` with valid-slot count; q/k rms_norm fused into rope kernel; value-residual lerp fused into v-projection epilogue |
| ctrl | host builds 259-dim input; `mm` for ctrl_emb MLP once per frame; MLPFusion kernel at 8 layers |
| sampler | host loop: 4 (or 2) Euler steps + cache pass; noise from `randn` oracle dump (verification) or host RNG (demo) |
| KV cache | per-layer ring in the weights/activations arena; local layers write every frame, global layers every 8th (bucket = (idx+7)/8 % 16); frozen flag = skip ring write |
| taehv decode | new kernels: conv3x3(+ReLU, +concat-past for MemBlock), conv1x1-as-mm (TPool/TGrow), nearest 2× upsample, pixel_shuffle, tanh clamp; MemBlock state buffers persisted across latents |
| output | PPM/BMP frame writer in Mojo (compare/animate offline) |

## Verification (oracle, as qwen3/omnivoice)

`scripts/gen_oracle.py` (torch CPU, fp32 reference where cheap, fixed
seed; flex_attention monkeypatched to SDPA + boolean mask so it runs on
CPU) dumps for a scripted run — no start image, fixed noise per frame,
scripted controls (frames 0–3 idle, 4–7 W held + mouse dx):

1. initial noise latents per frame `[N,32,32,64]`
2. pass-0 velocity for frame 0 (σ=1.0) → **gate 1**: max-abs-diff vs Mojo
   (bf16 tolerance ~1e-2 on unit-scale latents)
3. denoised latent per frame → **gate 2**: drift over 8 frames within
   tolerance; report per-frame max-abs-diff
4. decoded RGB frames → **gate 3**: PSNR > 35 dB vs oracle frames
5. timing per stage → perf baseline

## Phases

- **P0**: env + downloads (pixi env with torch/diffusers; model →
  `assets/model/`) ✅; oracle script.
- **P1**: `prepare_model.py` conversion + Mojo DiT single pass (frame 0,
  σ=1.0, empty history) → gate 1.
- **P2**: full per-frame loop (4 steps + cache pass), ring KV, 8 frames →
  gate 2.
- **P3**: taehv decoder in Mojo → gate 3; end-to-end `worldgen` binary
  writing frames.
- **P4**: perf — fused QKV, tiled/fused matmuls, per-frame constants
  hoisted, 2-step mode, W8 via existing quantize pipeline (measure),
  report tok-passes/s + s/frame vs oracle torch-MPS baseline.
- **Later**: interactive loop (keyboard→controls, live window), INT8
  taehv, 360p profile for speed.
