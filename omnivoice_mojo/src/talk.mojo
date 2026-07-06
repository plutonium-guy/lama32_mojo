"""Ask a question, hear the answer: Qwen3 latent thinking -> OmniVoice TTS.

One Mojo process, one Metal context, both models resident:
  1. Qwen3-0.6B prefills the question + "<think>\n", runs GPU-resident
     latent (soft-embedding) thinking steps — no host sync, no tokens —
  2. exits to "</think>\n\nThe answer is" and greedy-decodes a short
     discrete answer,
  3. OmniVoice speaks the answer to a 24 kHz WAV.

Run (from omnivoice_mojo):
  pixi run talk "Why is the sky blue?"
  pixi run talk "..." --steps 48 --out answer.wav
"""

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from qwen_gpu import Qwen, load_qwen
from latent_think import (
    prefill_to_soft, latent_steps, mean_row_norm, encode, decode_ids, H,
)
from omnivoice_gpu import load_omnivoice, default_gen_config
from codec import Codec, codec_arena_floats, UPSAMPLE
from llama_common import Acts
from duration import estimate_target_tokens
from wav import write_wav

comptime QWEN_TOK = "/Volumes/T7 Shield/llama32_mojo/qwen3_mojo/assets/model/tokenizer.json"
comptime OV_TOK = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/model/tokenizer.json"
comptime EOS = 151645
comptime MAXLEN = 1024
comptime FRAME_RATE = 25
comptime SAMPLE_RATE = 24000
comptime NCB = 8


def speakable(text: String) raises -> String:
    """Strip markdown noise and keep the first two sentences — spoken
    answers should be concise (and OmniVoice attention is O(T^2))."""
    var re = Python.import_module("re")
    var s = PythonObject(text)
    s = re.sub("[*#`_$\\\\]+", "", s)
    s = re.sub("[\\r\\n]+", " ", s)
    s = re.sub("[ \\t]+", " ", s)
    s = s.strip()
    var parts = re.split("(?<=[.!?]) ", s)
    var keep = PythonObject("")
    var n = 0
    for i in range(len(parts)):
        if n >= 2:
            break
        var p = parts[i]
        if String(p).byte_length() > 0:
            keep = keep + p + PythonObject(" ")
            n += 1
    return String(String(keep).strip())


def main() raises:
    var question = String("Why is the sky blue?")
    var n_latent = 32
    var out_path = String("answer.wav")
    var args = argv()
    if len(args) > 1:
        question = String(args[1])
    var i = 2
    while i + 1 < len(args):
        if args[i] == "--steps":
            n_latent = atol(String(args[i + 1]))
        elif args[i] == "--out":
            out_path = String(args[i + 1])
        i += 2

    var tk_mod = Python.import_module("tokenizers")
    var qtok = tk_mod.Tokenizer.from_file(QWEN_TOK)
    var vtok = tk_mod.Tokenizer.from_file(OV_TOK)

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    print("[loading Qwen3-0.6B]")
    var qwen = load_qwen(ctx, MAXLEN)
    var osoft = qwen.a.alloc(H)
    var oent = qwen.a.alloc(1)
    var tnorm = mean_row_norm(qwen)

    # ---- think (latent) + answer (discrete) ----
    var prompt = (String("<|im_start|>user\n") + question
                  + String("<|im_end|>\n<|im_start|>assistant\n<think>\n"))
    var t0 = perf_counter_ns()
    prefill_to_soft(qwen, encode(qtok, prompt), osoft, oent,
                    Float32(1.0), tnorm, Float32(1e-3))
    var lres = latent_steps(qwen, n_latent, osoft, oent,
                            Float32(1.0), Float32(0.5), Float32(0.8),
                            tnorm, Float32(1e-3))
    var t_think = Float64(perf_counter_ns() - t0) / 1e9
    print("[thought for", lres[0], "latent steps in", t_think, "s ]")

    t0 = perf_counter_ns()
    var nxt = qwen.forward_argmax(
        encode(qtok, String("</think>\n\nThe answer is")))
    var out = List[Int]()
    while qwen.n_cached < MAXLEN and len(out) < 100:
        if nxt == EOS:
            break
        out.append(nxt)
        var step: List[Int] = [nxt]
        nxt = qwen.forward_argmax(step)
    var answer = String("The answer is") + decode_ids(qtok, out)
    var t_ans = Float64(perf_counter_ns() - t0) / 1e9
    print("\n" + answer + "\n")
    print("[", len(out), "answer tokens in", t_ans, "s =",
          Float64(len(out)) / t_ans, "tok/s ]")

    # ---- speak ----
    var speech = speakable(answer)
    var T = estimate_target_tokens(speech, String(""), 0, 1.0)
    var head = encode(vtok,
                      String("<|lang_start|>None<|lang_end|>"
                             "<|instruct_start|>None<|instruct_end|>"
                             "<|text_start|>") + speech
                      + String("<|text_end|>"))
    var P = len(head)
    var prefix8 = List[Int](capacity=NCB * P)
    for _ in range(NCB):
        for t in range(P):
            prefix8.append(head[t])

    print("[loading OmniVoice]")
    var model = load_omnivoice(ctx, P + T + 8)
    var gc = default_gen_config()
    t0 = perf_counter_ns()
    var tokens = model.generate(prefix8, P, T, gc, verbose=False)
    print("[speech tokens:", Float64(perf_counter_ns() - t0) / 1e9, "s for",
          Float64(T) / FRAME_RATE, "s audio ]")

    var ca = Acts(ctx, codec_arena_floats(T))
    var codec = Codec(ctx, T)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](T * UPSAMPLE)
    t0 = perf_counter_ns()
    var wav = codec.decode(ctx, model.w, ca, model.kn, tokens, T, out_buf)
    print("[codec:", Float64(perf_counter_ns() - t0) / 1e9, "s ]")

    var peak = Float32(0)
    for s_i in range(len(wav)):
        var v = wav[s_i]
        if v < 0:
            v = -v
        if v > peak:
            peak = v
    if peak > 1e-6:
        for s_i in range(len(wav)):
            wav[s_i] = wav[s_i] / peak * 0.5
    write_wav(out_path, wav, SAMPLE_RATE)
    print("wrote", out_path, "(", Float64(len(wav)) / SAMPLE_RATE, "s )")
