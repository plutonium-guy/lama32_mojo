# omnivoice_mojo

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — multilingual
zero-shot TTS (diffusion-LM over audio tokens) — running natively in Mojo on
Apple Metal. Follows the qwen3_mojo pattern: bf16 weights resident on GPU,
f32 activations, shared kernels from `../src/llama_common.mojo`.

Everything needed to go text → 24 kHz WAV runs in Mojo:

- **Backbone**: Qwen3-0.6B with fully **bidirectional** attention, no KV
  cache (each unmasking step recomputes the sequence) — `ov_common.mojo`,
  `omnivoice_gpu.mojo`.
- **Audio adapters**: 8-codebook summed input embeddings + fused
  audio-heads/CFG-predict kernel.
- **Sampler**: 32-step MaskGIT-style unmasking (shifted schedule, CFG 2.0,
  layer penalty, gumbel position noise) — `sampler.mojo`.
- **Codec**: HiggsAudio v2 decode path (RVQ dequantize + DAC decoder,
  960× upsample) — `codec.mojo`.
- **Extras**: rule duration estimator (`duration.mojo`), WAV writer
  (`wav.mojo`). Text tokenization uses HF `tokenizers` via Python interop.

Python (torch) is used only offline: weight conversion, oracle generation,
and encoding voice-clone references (the codec *encoder* + HuBERT are not
ported yet).

## Setup

```bash
pixi run download    # k2-fsa/OmniVoice from HF (~3.3 GB)
pixi run convert     # -> assets/mojo/model.safetensors (1.27 GB bf16)
pixi run build       # -> ./omnivoice_tts
```

## Use

```bash
# auto voice
pixi run ./omnivoice_tts "Hello there!" --lang en --out out.wav

# voice design
pixi run ./omnivoice_tts "Hello!" --lang en --instruct "female, british accent"

# voice cloning (encode the reference once with torch, then reuse)
pixi run python scripts/encode_ref.py ref.wav "reference transcript" ref.json
pixi run ./omnivoice_tts "New text in the cloned voice." --ref ref.json

# knobs: --seconds 4.0 --speed 1.2 --steps 32 --guidance 2.0
```

## Verification

```bash
pixi run oracle       # torch CPU reference dumps (deterministic variant)
pixi run test         # LM: logits diff, step-0 pred agreement, timings
pixi run test-codec   # codec: SNR vs torch waveform
```

Measured on M4 (bf16 weights vs f32 torch oracle):

- step-0 logits max-abs-diff ≈ 0.28, **step-0 prediction agreement 98%**
  (bf16 drift compounds through the iterative sampler, so exact final-token
  match is not a meaningful gate)
- codec decode **SNR 45 dB** vs torch
- end-to-end outputs transcribed verbatim by Whisper (auto + cloning modes)
- ~0.86 s/step at seq≈165 (≈9× real-time for 3 s audio with 32 steps),
  codec ≈0.6 s per audio-second

## Not ported yet

- codec encoder + HuBERT semantic branch (voice cloning refs need
  `scripts/encode_ref.py`)
- long-text chunking (>30 s), silence-removal post-processing, batching
- speed: k_mm_tile does ~144 GMAC/s; a shared-memory tiled matmul and
  batching the cond/uncond forwards are the next wins
