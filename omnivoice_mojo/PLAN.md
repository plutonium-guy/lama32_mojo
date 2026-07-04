# OmniVoice → Mojo port plan

Port of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (massively
multilingual zero-shot TTS, diffusion-LM style) to Mojo on Apple Metal,
following the llama32/qwen3 pattern in this repo.

## Model analysis

OmniVoice = three parts:

1. **Backbone**: Qwen3-0.6B (`llm.*` in checkpoint) — hidden 1024, 28 layers,
   GQA 16q/8kv, head_dim 128, inter 3072, RMSNorm eps 1e-6, RoPE theta 1e6,
   per-head q/k norm. Identical shapes to `qwen3_mojo`, **but run with fully
   bidirectional attention and no KV cache** (masked-diffusion LM: every
   iteration recomputes the full sequence). vocab resized to 151676.
2. **Audio token adapters**:
   - `audio_embeddings` [8×1025, 1024]: at audio positions, the input
     embedding is the **sum over 8 codebooks** of
     `audio_embeddings[id_c + c*1025]`. Text positions use
     `llm.embed_tokens` row of the text id (ids replicated across the 8
     codebook rows).
   - `audio_heads` [8×1025, 1024] (no bias): hidden → per-codebook logits
     [C=8, V=1025]. Only needed on target-region rows.
3. **Audio codec** (HiggsAudio v2 tokenizer, `audio_tokenizer/` on HF,
   25 Hz frame rate, 24 kHz audio, hop 960). Decode path (all we need for
   TTS output):
   - RVQ: for each of 8 quantizers: codebook embed [1024, 64] lookup →
     `project_out` Linear 64→1024 (+bias); sum over quantizers → (1024, T).
   - `fc2` Linear 1024→256.
   - DAC decoder (weight-norm folded): Conv1d 256→1024 k7 p3 →
     5 blocks (Snake1d → ConvTranspose1d k=2s, p=ceil(s/2), outpad=s%2 →
     3 residual units [Snake, Conv k7 dil 1/3/9 pad 3·dil, Snake, Conv k1])
     with strides [8,5,4,2,3], channels 1024→512→256→128→64→32 →
     Snake → Conv1d 32→1 k7 p3. **No final tanh** (OmniVoice variant).
   - Snake1d: `x + sin(alpha*x)^2 / (alpha + 1e-9)`.
   - Total upsample 960× → 24 kHz.

### Inference algorithm (32-step iterative unmasking, MaskGIT-style)

- Build cond sequence: style tokens (`<|denoise|>` if voice-clone +
  `<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>`)
  + `<|text_start|>{ref_text + text}<|text_end|>` + [ref audio tokens] +
  T target positions all = MASK (1024). Uncond sequence = target region only.
- Target length T from `RuleDurationEstimator` (char phonetic weights;
  fallback ref "Nice to meet you." = 25 tokens; low-threshold power boost).
- Unmask schedule: shifted timesteps `t' = 0.1*t / (1 + (0.1-1)*t)`,
  per step unmask `ceil(total*(t[i+1]-t[i]))` of the 8·T slots.
- Each step: 2 forwards (cond, uncond), CFG in log-prob space
  `log_softmax(lc + 2.0*(lc - lu))`, ban MASK id, greedy pred + confidence,
  confidence − 5.0·codebook_layer, gumbel position noise (temp 5.0),
  top-k over still-masked slots, fill.

## Mojo mapping

New subproject `omnivoice_mojo/` reusing `../src` (llama_common.mojo,
safetensors.mojo, resident.mojo):

| piece | plan |
|---|---|
| weights | `scripts/prepare_model.py`: HF download → single bf16 safetensors (llm + audio adapters + codec decode path, weight-norm folded) |
| forward | reuse `mm_op`/`rmsnorm_op`/rope kernels; new **bidirectional** scores kernel + own run_layer (no cache) |
| embeddings | new kernel: per position, text row or 8-codebook sum |
| logits | `audio_heads` matmul on target rows only; GPU CFG kernel emits (pred id, confidence) per (codebook, frame) — tiny readback |
| sampler | host: schedule, layer penalty, gumbel, top-k, fill |
| codec | new kernels: conv1d (generic k/dil/pad), convtranspose1d, snake; RVQ+fc2 via conv1d k1 |
| tokenizer | HF `tokenizers` via Python interop (as chat.mojo) |
| duration | port of RuleDurationEstimator (unicodedata via Python interop) |
| output | pure-Mojo 16-bit PCM WAV writer |

## Verification (oracle, as qwen3_mojo)

`scripts/gen_oracle.py` (torch CPU, deterministic: position_temperature=0,
class_temperature=0) dumps for a fixed prompt:
1. prepared input ids + audio mask → Mojo must reproduce
2. step-0 cond/uncond logits → max-abs-diff check (bf16 tolerance)
3. final tokens (deterministic loop) → token match rate
4. decoded waveform from oracle tokens → codec check via SNR

## Phases / scope

- **P1**: auto-voice + voice-design modes fully in Mojo (text → wav).
- **P2**: voice cloning — ref audio encoded offline by
  `scripts/encode_ref.py` (torch; HuBERT semantic branch is encode-only),
  Mojo consumes precomputed ref tokens.
- **Later**: port codec encoder + HuBERT for full-Mojo cloning; long-text
  chunking; silence removal post-processing; batching.
