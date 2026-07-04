#!/usr/bin/env python3
"""Encode a voice-cloning reference for the Mojo TTS CLI.

The HiggsAudio v2 *encoder* (HuBERT semantic branch + DAC acoustic encoder)
is not ported to Mojo; this script runs it in torch and writes the ref
tokens JSON that `omnivoice_tts --ref` consumes:

    {"text": "<transcript>", "tokens": [[...] x 8]}

Usage: python scripts/encode_ref.py ref.wav "transcript of the audio" ref.json
"""

import json
import sys
import os

import soundfile as sf
import torch
import torchaudio
from transformers import HiggsAudioV2TokenizerModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "model")
SR = 24000


def main():
    wav_path, text, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)                      # (channels, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    rms = float(wav.pow(2).mean().sqrt())
    if 0 < rms < 0.1:
        wav = wav * 0.1 / rms

    codec = HiggsAudioV2TokenizerModel.from_pretrained(
        os.path.join(MODEL_DIR, "audio_tokenizer")).eval()
    hop = codec.config.hop_length
    clip = wav.shape[-1] % hop
    if clip:
        wav = wav[:, :-clip]

    with torch.inference_mode():
        codes = codec.encode(wav.unsqueeze(0)).audio_codes[0]   # (8, T)

    text = text.strip()
    if text and text[-1] not in ".!?。！？，,;":
        text += "."

    json.dump({"text": text, "tokens": codes.tolist()}, open(out_path, "w"))
    dur = wav.shape[-1] / SR
    print(f"wrote {out_path}: {codes.shape[1]} frames ({dur:.1f}s), rms {rms:.3f}")


if __name__ == "__main__":
    main()
