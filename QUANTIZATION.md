# Quantization Plan (idea, not yet implemented)

Status: proposal — 2026-07-03. Current stack runs bf16 weights resident
(Qwen3-0.6B: 47 tok/s decode, Llama 3.2 1B: ~28 tok/s on M4 after
precompiled-handle dispatch optimization).

## Why it fits this codebase

- Kernels read weights only through `bf()` / `bf4()` / `bf8()` helpers in
  `src/llama_common.mojo`. Weight buffer is an opaque u16 arena.
- `k_mm_w` is ~99% of weight traffic → one kernel + one decode helper swap
  covers nearly everything.
- Decode is memory-bandwidth-bound → compression converts directly to tok/s.
- Oracle parity harness (`pixi run oracle` + `pixi run test`) already exists
  to validate quality tolerance.

## Options by bit width

| Format | Compression | Quality | Fit |
|--------|-------------|---------|-----|
| INT8, group-wise scales (group 64) | 2x | negligible loss | best for Qwen3-0.6B (small models quantize worse) |
| INT4 (Q4_0 / Q4_K llama.cpp-style) | 4x | good on ≥1B | game-changer for llama31 8B — fits 16 GB easily |
| 3/2-bit packed | 5-8x | poor on 0.6B-1B | not recommended at these sizes |
| odd widths (5/6-bit) | between | ok | packing/alignment mess, low payoff |

## Implementation path (INT8 first)

1. **Offline converter** (Python script): safetensors bf16 → packed blob —
   u8 weights + f16 per-group scales (group 64 along input dim).
2. **`Weights` arena** becomes u8 buffer (or keep u16 arena, view as bytes).
   Norm weights (1-D, tiny, quality-sensitive) stay bf16.
3. **Kernel**: `k_mm_w` inner loop `bf8()` → `dq8()` — load 8 packed
   weights + 1 scale, dequant in registers. Same TG_MM structure.
4. **Validate**: regenerate oracle, relax logits tolerance (~1e-1 vs
   current 3e-2), greedy-token match must hold.

## Expected numbers (M4, Qwen3-0.6B)

- Weight traffic 1.2 GB → 0.6 GB per token → est. **70-80 tok/s**.
- Ceiling note: host dispatch floor ~13 ms/token (~420 dispatches ×
  ~32 µs even with precompiled handles) caps ~75 tok/s. Beyond that needs
  dispatch fusion (qk-norm merge, rope+kv-copy merge, res_add+rmsnorm merge).
- INT4 on llama31 8B: ~4.5 GB resident, decode est. 3-4x vs bf16 (which
  doesn't fit memory comfortably today).

## Non-goals / notes

- MAX ships a `quantization` package (Q4_0 etc.) but that targets the MAX
  graph API, not this hand-rolled kernel path — not directly reusable.
- Activations stay f32; this is weight-only quantization (W8A32 / W4A32).
