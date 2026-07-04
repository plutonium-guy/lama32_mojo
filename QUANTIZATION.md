# Quantization

Status: **INT8 implemented for Qwen3-0.6B** — 2026-07-04. INT4 for
llama31 8B still planned (see below). Original proposal follows, updated
with measured results.

## What shipped (W8A32, group 64)

- `qwen3_mojo/scripts/quantize.py` — offline converter: every 2-D bf16
  weight row [m] → `[m/64 f16 group scales | m int8]`, symmetric
  (`scale = max|w|/127`, rounded to f16 before requantizing so kernel
  dequant matches). Stored as an F16 tensor of shape [n, m/64 + m/2] under
  the original name, so `safetensors.mojo` + `resident.mojo` load the
  packed file **unchanged** (2 bytes/element, offsets in u16 units).
  Norms (1-D) stay bf16. 1.50 GB → 0.78 GB (1.94x).
- `src/llama_q8.mojo` — shared q8 kernels + layer runner:
  `k_mm_q8`, `k_embed_gather_q8`, `mm_q8_op`, `run_layer_q8` (norms /
  rope / attention / swiglu reuse llama_common). The s==1 fused q|k|v and
  gate|up dispatches carry over (contiguity holds in packed units).
- `qwen3_mojo/src/qwen_q8.mojo` — model shell + oracle validation
  (`pixi run test-q8`), `chat_q8.mojo` REPL (`pixi run chat-q8`).

## Measured (M4, Qwen3-0.6B)

| | bf16 | q8 |
|---|---|---|
| resident weights | 1.24 GB | 0.61 GB |
| decode (same-session A/B) | 21–25 tok/s | 22–29 tok/s |
| prefill (15 tok) | 0.33 s | 0.20 s |
| logits max-abs-diff vs bf16 oracle | 6.5e-5 | 0.75 (span ±25) |
| greedy token match | — | holds |

Chat quality: coherent (thinking mode intact, correct answers) at
temperature 0.

## What the estimate got wrong (important)

The plan predicted 70–80 tok/s from halved weight traffic. Two premises
failed on measurement:

1. **`k_mm_w` is not bandwidth-bound at s=1.** The one-output-per-
   threadgroup design reaches only ~70 GB/s / ~35 GMAC/s on M4; it sits at
   the instruction/memory crossover. Halving bytes with a same-shape q8
   loop (dequant adds convert+mul per weight) made it *slower* (20 tok/s)
   until the loop was restructured group-major: each thread walks whole
   64-weight groups, accumulates unscaled int8·f32 dots, and applies the
   f16 scale once per group. That got q8 to ~39 GMAC/s vs bf16's ~32-36 on
   the head matmul — a ~15% win, not 2x.
   (Also: never materialize an f16 scale via take-address-of-local in a
   GPU kernel — it spills to stack memory; use SIMD bitcast.)
2. The 47 tok/s bf16 baseline recorded earlier did not reproduce on the
   day of this work (21–25 tok/s same code, likely thermal/OS state), so
   relative numbers are what matter.

## Next steps

- **Tiled s=1 matmul** (multiple outputs per threadgroup sharing x loads,
  as `omnivoice_mojo`'s `k_mm_tile` does for s≥4) — that, not
  quantization, is the main decode headroom; q8's traffic halving pays
  off only once kernels are near bandwidth.
- Dispatch fusion (qk-norm merge, rope+kv-copy merge, res_add+rmsnorm
  merge) toward the ~13 ms/token dispatch floor.
- **INT4 (Q4_K-style) for llama31 8B** — the fits-in-16-GB win stands
  regardless of kernel speed: ~4.5 GB resident vs ~16 GB bf16. Same
  packed-row trick works; use group 32 + two nibbles per byte.
- OmniVoice: quantize the Qwen3-0.6B backbone the same way (its seq-level
  forwards use the tiled kernel, where traffic reduction should show more).

## Non-goals / notes

- MAX ships a `quantization` package (Q4_0 etc.) but that targets the MAX
  graph API, not this hand-rolled kernel path — not directly reusable.
- Activations stay f32; this is weight-only quantization (W8A32 / W4A32).
