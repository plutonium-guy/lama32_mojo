"""Interactive chat REPL on INT8 group-quantized weights (see chat.mojo).

Same REPL as chat.mojo but on the q8 stack (qwen_q8.mojo): ~2x less
weight traffic per token, quality gated by the oracle greedy match.

Run:
  pixi run chat-q8
  pixi run chat-q8 "What is the capital of France?" [temperature] [top_p]
Commands: /reset, /quit
"""

from std.python import Python, PythonObject
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns
from qwen_q8 import QwenQ8, load_qwen_q8, VOCAB
from sample import sample
from std.gpu.host import DeviceContext

comptime TOKENIZER = "/Volumes/T7 Shield/llama32_mojo/qwen3_mojo/assets/model/tokenizer.json"
comptime MAXLEN = 2048
comptime EOS = 151645


def encode_turn(tok: PythonObject, question: String, first: Bool) raises -> List[Int]:
    """Qwen3 chat template (user turn + assistant header).

    On continuing turns the previous assistant reply is still open in the KV
    cache (generation stops at EOS without feeding it), so close it here.
    """
    var prompt = String("") if first else String("<|im_end|>\n")
    prompt += (
        String("<|im_start|>user\n")
        + question
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )
    var pyids = tok.encode(prompt, add_special_tokens=False).ids
    var ids = List[Int]()
    for i in range(len(pyids)):
        ids.append(atol(String(pyids[i])))
    return ids^


def ends_with_replacement(s: String) -> Bool:
    """True if s ends with U+FFFD (partial UTF-8 from a split multibyte char)."""
    var n = s.byte_length()
    if n < 3:
        return False
    var p = s.unsafe_ptr()
    return p[n - 3] == 0xEF and p[n - 2] == 0xBF and p[n - 1] == 0xBD


def generate(mut model: QwenQ8, tok: PythonObject, ids: List[Int],
             temp: Float32, top_p: Float32) raises:
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
    var n_out = 0
    var pending = List[Int]()  # tokens not yet printed (may end mid-UTF-8 char)
    t0 = perf_counter_ns()
    while model.n_cached < MAXLEN:
        if nxt == EOS:
            break
        n_out += 1
        pending.append(nxt)
        var pyout = Python.list()
        for i in range(len(pending)):
            pyout.append(pending[i])
        var text = String(tok.decode(pyout))
        if len(pending) >= 4 or not ends_with_replacement(text):
            print(text, end="", flush=True)
            pending.clear()
        var step: List[Int] = [nxt]
        if greedy:
            nxt = model.forward_argmax(step)
        else:
            logits = model.forward(step)
            nxt = sample(logits, temp, top_p)
    if len(pending) > 0:
        var pyout = Python.list()
        for i in range(len(pending)):
            pyout.append(pending[i])
        print(String(tok.decode(pyout)), end="", flush=True)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print()
    if nxt != EOS:
        print("[truncated: context full]")
    print("[", n_out, "tokens,", Float64(n_out) / dt, "tok/s, prefill",
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
    var model = load_qwen_q8(ctx, MAXLEN)

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
