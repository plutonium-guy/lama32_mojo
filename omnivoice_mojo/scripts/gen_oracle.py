#!/usr/bin/env python3
"""Reference dumps for the Mojo OmniVoice port (torch CPU, deterministic).

Standalone reimplementation of OmniVoice inference (no omnivoice package):
Qwen3 backbone with full bidirectional attention + audio embeddings/heads +
32-step iterative unmasking with CFG. Deterministic variant:
position_temperature=0, class_temperature=0 (greedy everywhere).

Writes to assets/oracle/:
  manifest.json        prompt, cond ids (8 x L), target_len, config
  embeds.bin           f32 (L, 1024) cond input embeddings
  step0_cond.bin       f32 (8, T, 1025) target-region logits, cond, step 0
  step0_uncond.bin     f32 (8, T, 1025) uncond logits, step 0
  final_tokens.bin     i32 (8, T) tokens after full deterministic loop
  wav_oracle.bin       f32 (n,) codec decode of final tokens
  oracle.wav           same, listenable
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel, Qwen3Config, Qwen3Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "model")
OUT_DIR = os.path.join(ROOT, "assets", "oracle")

TEXT = "Hello, this is a test."
LANG = "en"
TARGET_LEN = 75          # 3 s at 25 Hz; fixed so Mojo needs no estimator here
NUM_STEP = 32
GUIDANCE = 2.0
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
NUM_CB = 8
VOCAB = 1025
MASK_ID = 1024

torch.manual_seed(0)


def build_model():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    llm_cfg = Qwen3Config(**{k: v for k, v in cfg["llm_config"].items()
                             if k not in ("architectures", "_name_or_path")})
    llm = Qwen3Model(llm_cfg)
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    llm_sd = {k[len("llm."):]: v for k, v in sd.items() if k.startswith("llm.")}
    missing, unexpected = llm.load_state_dict(llm_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    llm.eval()
    return llm, sd["audio_embeddings.weight"], sd["audio_heads.weight"]


def prepare_inputs(tokenizer):
    style = f"<|lang_start|>{LANG}<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style, return_tensors="pt").input_ids
    text_ids = tokenizer(f"<|text_start|>{TEXT}<|text_end|>",
                         return_tensors="pt").input_ids
    prefix = torch.cat([style_ids, text_ids], dim=1)          # (1, N)
    prefix = prefix.repeat(NUM_CB, 1)                          # (8, N)
    target = torch.full((NUM_CB, TARGET_LEN), MASK_ID, dtype=torch.long)
    input_ids = torch.cat([prefix, target], dim=1).unsqueeze(0)  # (1, 8, L)
    L = input_ids.shape[2]
    audio_mask = torch.zeros(1, L, dtype=torch.bool)
    audio_mask[0, L - TARGET_LEN:] = True
    return input_ids, audio_mask


def forward(llm, audio_emb, audio_heads, input_ids, audio_mask):
    text_embeds = llm.embed_tokens(input_ids[:, 0, :])
    offsets = (torch.arange(NUM_CB) * VOCAB).view(1, -1, 1)
    shifted = input_ids * audio_mask.unsqueeze(1) + offsets
    audio_embeds = F.embedding(shifted, audio_emb).sum(dim=1)
    embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    L = input_ids.shape[2]
    attn = torch.ones(1, 1, L, L, dtype=torch.bool)
    hidden = llm(inputs_embeds=embeds, attention_mask=attn,
                 return_dict=True).last_hidden_state       # (1, L, H)
    logits = hidden @ audio_heads.T                          # (1, L, 8*1025)
    logits = logits.view(1, L, NUM_CB, VOCAB).permute(0, 2, 1, 3)
    return embeds, logits                                    # (1, 8, L, V)


def time_steps():
    t = torch.linspace(0.0, 1.0, NUM_STEP + 1)
    return (T_SHIFT * t / (1 + (T_SHIFT - 1) * t)).tolist()


def cfg_predict(c_logits, u_logits):
    """Returns (pred (8,T), conf (8,T)) — deterministic (greedy) variant."""
    lc = F.log_softmax(c_logits, dim=-1)
    lu = F.log_softmax(u_logits, dim=-1)
    lp = torch.log_softmax(lc + GUIDANCE * (lc - lu), dim=-1)
    lp[..., MASK_ID] = -float("inf")
    return lp.argmax(dim=-1), lp.max(dim=-1)[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    llm, audio_emb, audio_heads = build_model()

    input_ids, audio_mask = prepare_inputs(tokenizer)
    L = input_ids.shape[2]
    T = TARGET_LEN
    print(f"seq len {L}, target {T}")

    # uncond stream: target region only
    u_ids = input_ids[..., L - T:].clone()
    u_mask = torch.ones(1, T, dtype=torch.bool)

    # unmask schedule
    ts = time_steps()
    total, rem, sched = T * NUM_CB, T * NUM_CB, []
    for step in range(NUM_STEP):
        num = rem if step == NUM_STEP - 1 else min(
            math.ceil(total * (ts[step + 1] - ts[step])), rem)
        sched.append(int(num))
        rem -= int(num)

    tokens = torch.full((NUM_CB, T), MASK_ID, dtype=torch.long)
    layer_ids = torch.arange(NUM_CB).view(-1, 1)

    with torch.inference_mode():
        for step in range(NUM_STEP):
            embeds, c_all = forward(llm, audio_emb, audio_heads,
                                    input_ids, audio_mask)
            _, u_all = forward(llm, audio_emb, audio_heads, u_ids, u_mask)
            c_logits = c_all[0, :, L - T:, :].float()
            u_logits = u_all[0, :, :, :].float()

            if step == 0:
                embeds[0].float().numpy().tofile(
                    os.path.join(OUT_DIR, "embeds.bin"))
                c_logits.numpy().tofile(os.path.join(OUT_DIR, "step0_cond.bin"))
                u_logits.numpy().tofile(os.path.join(OUT_DIR, "step0_uncond.bin"))

            pred, scores = cfg_predict(c_logits, u_logits)
            scores = scores - layer_ids * LAYER_PENALTY
            scores = scores.masked_fill(tokens != MASK_ID, -float("inf"))

            k = sched[step]
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[topk_idx] = pred.flatten()[topk_idx]
            tokens = flat.view(NUM_CB, T)

            input_ids[0, :, L - T:] = tokens
            u_ids[0] = tokens
            print(f"step {step:2d}: unmasked {k:3d}, "
                  f"remaining {(tokens == MASK_ID).sum().item()}")

    tokens.numpy().astype(np.int32).tofile(
        os.path.join(OUT_DIR, "final_tokens.bin"))

    print("decoding audio...")
    codec = HiggsAudioV2TokenizerModel.from_pretrained(
        os.path.join(MODEL_DIR, "audio_tokenizer")).eval()
    with torch.inference_mode():
        wav = codec.decode(tokens.unsqueeze(0)).audio_values[0, 0].numpy()
    wav.tofile(os.path.join(OUT_DIR, "wav_oracle.bin"))

    import soundfile as sf
    sf.write(os.path.join(OUT_DIR, "oracle.wav"), wav, 24000)

    manifest = {
        "text": TEXT, "lang": LANG, "seq_len": L, "target_len": T,
        "num_step": NUM_STEP, "guidance": GUIDANCE, "t_shift": T_SHIFT,
        "layer_penalty": LAYER_PENALTY,
        "prefix_ids": input_ids[0, 0, : L - T].tolist(),
        "schedule": sched,
        "wav_len": int(wav.shape[0]),
    }
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"))
    print("oracle written to", OUT_DIR)


if __name__ == "__main__":
    main()
