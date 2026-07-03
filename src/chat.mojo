"""Interactive chat REPL: Llama-3.2-1B-Instruct on Metal, weights load once.

Tokenizer runs at the Python edge (HF tokenizers); everything else is the
from-scratch Mojo GPU stack in llama_gpu.mojo. Multi-turn conversation rides
the persistent KV cache: each turn only appends its own tokens (no re-prefill).
Generation stops on <|eot_id|>/<|end_of_text|>.

Sampling follows the model's generation_config: temperature 0.6, top-p 0.9
(temperature 0 = greedy).

Run (from mojo_ocr for the pixi env):
  pixi run mojo run -I "../llama32_mojo/src" "../llama32_mojo/src/chat.mojo" \
      ["one-shot question"] [temperature] [top_p]
With no question argument, starts the REPL. Commands: /reset, /quit.
"""

from std.python import Python, PythonObject
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns
from llama_gpu import Llama, load_llama, VOCAB
from sample import sample
from std.gpu.host import DeviceContext

comptime TOKENIZER = "/Volumes/T7 Shield/llama32_mojo/assets/model/tokenizer.json"
comptime MAXLEN = 2048
comptime EOT = 128009           # <|eot_id|>
comptime EOS = 128001           # <|end_of_text|>
comptime EOM = 128008           # <|eom_id|>


def encode_turn(tok: PythonObject, question: String, first: Bool) raises -> List[Int]:
    """Llama-3 template tokens for one user turn + assistant header.

    Later turns start with <|eot_id|>: the sampled stop token of the previous
    assistant reply is never forwarded through the model, so it is prepended
    here to keep the KV cache byte-exact with the template.
    """
    var prefix = String("<|begin_of_text|>") if first else String("<|eot_id|>")
    var prompt = (
        prefix
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        + question
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    var pyids = tok.encode(prompt, add_special_tokens=False).ids
    var ids = List[Int]()
    for i in range(len(pyids)):
        ids.append(atol(String(pyids[i])))
    return ids^


def generate(mut model: Llama, tok: PythonObject, ids: List[Int],
             temp: Float32, top_p: Float32) raises:
    """Prefill this turn's tokens, then stream a sampled reply."""
    var greedy = temp <= 0
    var t0 = perf_counter_ns()
    var logits = List[Float32]()
    var nxt: Int
    if greedy:
        nxt = model.forward_argmax(ids)
    else:
        logits = model.forward(ids)
        nxt = sample(logits, temp, top_p)
    var prefill_s = Float64(perf_counter_ns() - t0) / 1e9
    var out = List[Int]()
    var shown = String("")
    t0 = perf_counter_ns()
    while model.n_cached < MAXLEN:
        if nxt == EOT or nxt == EOS or nxt == EOM:
            break
        out.append(nxt)
        # decode-all-and-print-delta keeps multi-byte unicode intact
        var pyout = Python.list()
        for i in range(len(out)):
            pyout.append(out[i])
        var text = String(tok.decode(pyout))
        var nb = text.byte_length()
        var ob = shown.byte_length()
        if nb > ob:
            print(text[byte=ob:nb], end="", flush=True)
        shown = text
        var step: List[Int] = [nxt]
        if greedy:
            nxt = model.forward_argmax(step)
        else:
            logits = model.forward(step)
            nxt = sample(logits, temp, top_p)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print()
    print("[", len(out), "tokens,", Float64(len(out)) / dt, "tok/s, prefill",
          prefill_s, "s, ctx", model.n_cached, "/", MAXLEN, "]")


def main() raises:
    var oneshot = String("")
    if len(argv()) > 1:
        oneshot = String(argv()[1])
    var temp = Float32(0.6)                 # generation_config defaults
    var top_p = Float32(0.9)
    if len(argv()) > 2:
        temp = Float32(atof(String(argv()[2])))
    if len(argv()) > 3:
        top_p = Float32(atof(String(argv()[3])))
    seed(Int(perf_counter_ns()))
    print("temperature:", temp, " top_p:", top_p)

    var tk_mod = Python.import_module("tokenizers")
    var tok = tk_mod.Tokenizer.from_file(TOKENIZER)

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_llama(ctx, MAXLEN)

    if oneshot.byte_length() > 0:
        generate(model, tok, encode_turn(tok, oneshot, True), temp, top_p)
        return

    print("\nREPL ready. /reset clears context, /quit exits.")
    # Python input(): Mojo's own input() drains the whole stdin buffer on its
    # first call, which silently eats every queued line when stdin is a pipe.
    var bi = Python.import_module("builtins")
    while True:
        var line = String("")
        try:
            line = String(bi.input("\nyou> "))
        except:
            break                            # EOF (ctrl-d)
        if line == "":
            continue
        if line == "/quit" or line == "/exit":
            break
        if line == "/reset":
            model.n_cached = 0
            print("[context cleared]")
            continue
        var first = model.n_cached == 0
        var ids = encode_turn(tok, line, first)
        # auto-reset when this turn + a reply headroom won't fit
        if model.n_cached + len(ids) + 64 > MAXLEN:
            print("[context full -> cleared]")
            model.n_cached = 0
            ids = encode_turn(tok, line, True)
        generate(model, tok, ids, temp, top_p)
