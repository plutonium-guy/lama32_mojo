"""Interactive chat REPL: Meta-Llama-3.1-8B-Instruct-abliterated on Metal.

Tokenizer at the Python edge (HF tokenizers); inference is the Mojo GPU stack
in llama_gpu.mojo. One-time warmup loads ~14 GB of layer weights disk -> GPU;
lm_head stays resident. Decode is GPU-bound after warmup.

Run:
  pixi run chat
  pixi run chat "What is the capital of France?" [temperature] [top_p]
Commands: /reset, /quit
"""

from std.math import exp
from std.python import Python, PythonObject
from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns
from llama_gpu import Llama, load_llama, VOCAB
from std.gpu.host import DeviceContext

comptime TOKENIZER = "/Volumes/T7 Shield/llama32_mojo/llama31_mojo/assets/model/tokenizer.json"
comptime MAXLEN = 2048
comptime EOT = 128009
comptime EOS = 128001
comptime EOM = 128008


def argmax(logits: List[Float32]) -> Int:
    var best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def sample(logits: List[Float32], temp: Float32, top_p: Float32) raises -> Int:
    if temp <= 0:
        return argmax(logits)
    var mx = logits[argmax(logits)]
    var probs = List[Float32](capacity=len(logits))
    var total = Float32(0)
    for i in range(len(logits)):
        var p = exp((logits[i] - mx) / temp)
        probs.append(p)
        total += p
    var cand_idx = List[Int]()
    var cand_p = List[Float32]()
    var floor = total * Float32(1e-6)
    for i in range(len(probs)):
        if probs[i] > floor:
            cand_idx.append(i)
            cand_p.append(probs[i])
    var target = total * top_p
    var nuc_idx = List[Int]()
    var nuc_p = List[Float32]()
    var cum = Float32(0)
    while cum < target and len(cand_idx) > 0:
        var b = 0
        for i in range(1, len(cand_p)):
            if cand_p[i] > cand_p[b]:
                b = i
        cum += cand_p[b]
        nuc_idx.append(cand_idx[b])
        nuc_p.append(cand_p[b])
        cand_idx[b] = cand_idx[len(cand_idx) - 1]
        cand_p[b] = cand_p[len(cand_p) - 1]
        _ = cand_idx.pop()
        _ = cand_p.pop()
    var r = Float32(random_float64()) * cum
    var acc = Float32(0)
    for i in range(len(nuc_idx)):
        acc += nuc_p[i]
        if r <= acc:
            return nuc_idx[i]
    return nuc_idx[len(nuc_idx) - 1]


def encode_turn(tok: PythonObject, question: String, first: Bool) raises -> List[Int]:
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
    var greedy = temp <= 0
    var t0 = perf_counter_ns()
    var out = List[Int]()
    var shown = String("")
    var nxt: Int
    var logits: List[Float32]
    if greedy:
        nxt = model.forward_argmax(ids)
    else:
        logits = model.forward(ids)
        nxt = sample(logits, temp, top_p)
    var prefill_s = Float64(perf_counter_ns() - t0) / 1e9
    t0 = perf_counter_ns()
    while model.n_cached < MAXLEN:
        if nxt == EOT or nxt == EOS or nxt == EOM:
            break
        out.append(nxt)
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
    var temp = Float32(0.6)
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
    print("loading model + warming GPU weights (one-time ~15 GB)...")
    var model = load_llama(ctx, MAXLEN, warm=True)

    if oneshot.byte_length() > 0:
        generate(model, tok, encode_turn(tok, oneshot, True), temp, top_p)
        return

    print("\nREPL ready. /reset clears context, /quit exits.")
    var bi = Python.import_module("builtins")
    while True:
        var line = String("")
        try:
            line = String(bi.input("\nyou> "))
        except:
            break
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
        if model.n_cached + len(ids) + 64 > MAXLEN:
            print("[context full -> cleared]")
            model.n_cached = 0
            ids = encode_turn(tok, line, True)
        generate(model, tok, ids, temp, top_p)
