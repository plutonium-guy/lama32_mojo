"""OmniVoice TTS on Metal, end to end: text -> 24 kHz WAV.

Usage:
  pixi run tts "Hello there, nice to meet you."
  pixi run tts "..." --lang en --out out.wav
  pixi run tts "..." --instruct "female, british accent"     # voice design
  pixi run tts "..." --ref ref.json                          # voice cloning
  pixi run tts "..." --seconds 4.0 --steps 32 --guidance 2.0

--ref takes the JSON written by scripts/encode_ref.py:
  {"tokens": [[...] x8], "text": "reference transcript"}

Tokenizer runs via the HF tokenizers Python package (as qwen3_mojo/chat.mojo);
the LLM, sampler, and codec are the Mojo GPU stack.
"""

from std.python import Python, PythonObject
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from omnivoice_gpu import (
    OmniVoice, load_omnivoice, default_gen_config, NCB,
)
from codec import Codec, codec_arena_floats, UPSAMPLE
from llama_common import Acts
from duration import estimate_target_tokens
from wav import write_wav

comptime TOKENIZER = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/model/tokenizer.json"
comptime FRAME_RATE = 25
comptime SAMPLE_RATE = 24000


def encode_ids(tok: PythonObject, text: String) raises -> List[Int]:
    var pyids = tok.encode(text, add_special_tokens=False).ids
    var ids = List[Int]()
    for i in range(len(pyids)):
        ids.append(atol(String(pyids[i])))
    return ids^


def clean_text(text: String) raises -> String:
    """OmniVoice._combine_text normalization via Python re: strip newlines,
    normalize Chinese parens, collapse spaces, drop spaces around CJK."""
    var re = Python.import_module("re")
    var s = PythonObject(text)
    s = re.sub("[\\r\\n]+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub("[ \\t]+", " ", s)
    s = re.sub("(?<=[一-鿿])\\s+|\\s+(?=[一-鿿])", "", s)
    return String(s.strip())


def main() raises:
    var text = String("")
    var lang = String("None")
    var instruct = String("None")
    var out_path = String("out.wav")
    var ref_path = String("")
    var seconds = Float64(0)
    var speed = Float64(1.0)
    var gc = default_gen_config()

    var args = argv()
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--lang":
            lang = String(args[i + 1])
            i += 2
        elif arg == "--instruct":
            instruct = String(args[i + 1])
            i += 2
        elif arg == "--out":
            out_path = String(args[i + 1])
            i += 2
        elif arg == "--ref":
            ref_path = String(args[i + 1])
            i += 2
        elif arg == "--seconds":
            seconds = atof(String(args[i + 1]))
            i += 2
        elif arg == "--speed":
            speed = atof(String(args[i + 1]))
            i += 2
        elif arg == "--steps":
            gc.num_step = atol(String(args[i + 1]))
            i += 2
        elif arg == "--guidance":
            gc.guidance = Float32(atof(String(args[i + 1])))
            i += 2
        else:
            text = arg
            i += 1
    if text.byte_length() == 0:
        print("usage: tts \"text\" [--lang en] [--instruct \"female\"] "
              "[--ref ref.json] [--seconds 4] [--speed 1.0] [--out out.wav]")
        return

    seed(Int(perf_counter_ns()))
    text = clean_text(text)

    var tk_mod = Python.import_module("tokenizers")
    var tok = tk_mod.Tokenizer.from_file(TOKENIZER)
    var json = Python.import_module("json")
    var bi = Python.import_module("builtins")

    # voice-clone reference (optional)
    var has_ref = ref_path.byte_length() > 0
    var ref_text = String("")
    var ref_tokens = List[Int]()          # (8, R) row-major
    var R = 0
    if has_ref:
        var rj = json.loads(bi.open(ref_path, "r").read())
        ref_text = String(rj["text"])
        var rows = rj["tokens"]
        R = len(rows[0])
        for c in range(NCB):
            for t in range(R):
                ref_tokens.append(atol(String(rows[c][t])))

    # target length
    var T: Int
    if seconds > 0:
        T = Int(seconds * FRAME_RATE)
    else:
        T = estimate_target_tokens(text, ref_text, R, speed)
    print("text:", text)
    print("target:", T, "frames (", Float64(T) / FRAME_RATE, "s )")

    # prompt: style + text (+ ref audio tokens)
    var style = String("")
    if has_ref:
        style += "<|denoise|>"
    style += "<|lang_start|>" + lang + "<|lang_end|>"
    style += "<|instruct_start|>" + instruct + "<|instruct_end|>"
    var full_text = text
    if has_ref:
        full_text = clean_text(ref_text) + " " + text
    var head = encode_ids(tok, style + "<|text_start|>" + full_text
                          + "<|text_end|>")
    var P = len(head) + R
    var prefix8 = List[Int](capacity=NCB * P)
    for c in range(NCB):
        for t in range(len(head)):
            prefix8.append(head[t])
        for t in range(R):
            prefix8.append(ref_tokens[c * R + t])
    var astart = len(head)                # first audio position (ref or target)
    if not has_ref:
        astart = P

    var maxlen = P + T + 8
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_omnivoice(ctx, maxlen)

    var t0 = perf_counter_ns()
    var tokens = model.generate(prefix8, astart, T, gc, verbose=True)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("generate:", dt, "s for", Float64(T) / FRAME_RATE, "s audio")

    # decode to waveform (codec shares the weight buffer; its arena is
    # sized for the decoder's activations)
    var ca = Acts(ctx, codec_arena_floats(T))
    var codec = Codec(ctx, T)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](T * UPSAMPLE)
    t0 = perf_counter_ns()
    var wav = codec.decode(ctx, model.w, ca, model.kn, tokens, T, out_buf)
    print("codec:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    # peak-normalize to 0.5 (matches OmniVoice's no-reference postprocess)
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
    print("wrote", out_path, "(", len(wav), "samples )")
