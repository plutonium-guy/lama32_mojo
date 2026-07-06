"""Latent ("soft") thinking prototype for Qwen3-0.6B on Metal.

Instead of decode -> argmax -> host readback -> re-embed per thinking token,
feed the *expected embedding* back directly (Coconut / Soft-Thinking style):

    e_next = softmax(logits / T)^T  @  embed_tokens

The mixture stays in the convex hull of real token embeddings (tolerable
zero-shot, unlike raw hidden-state feedback) and the whole thinking phase is
GPU-resident: no per-step synchronize, no argmax readback.

Protocol:
  prefill(prompt + "<think>\n") -> N latent steps -> feed "</think>\n\n"
  -> greedy-decode the answer.
Baseline for comparison: normal discrete greedy thinking.

Run: pixi run think ["question"] [n_latent] [temperature]
"""

from std.math import ceildiv, sqrt, exp, log
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from std.sys import argv
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from llama_common import (
    Acts, BLOCK, PAD, mm_op, rmsnorm_op, run_layer, k_embed_gather,
)
from qwen_gpu import Qwen, load_qwen, VOCAB

comptime TOKENIZER = "/Volumes/T7 Shield/llama32_mojo/qwen3_mojo/assets/model/tokenizer.json"
comptime MAXLEN = 2048
comptime EOS = 151645
comptime H = 1024
comptime TG_SM = 256


# ============================ kernels =========================================

def k_softmax_soft(a: UnsafePointer[Float32, MutAnyOrigin],
                   op: Int, oent: Int, n: Int, invt: Float32, eps: Float32):
    """Tempered softmax with relative-mass truncation + entropy output.

    Truncation: pre-normalization exp values are relative to the max
    (e_max = 1), so dropping e < eps is exactly p_v < eps * p_max —
    threshold top-p with no sort. Entropy H(p) is written to a[oent].
    """
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG_SM, Float32,
                                  address_space = AddressSpace.SHARED]()
    # max
    var mx = Float32(-3.0e38)
    var i = t
    while i < n:
        if a[op + i] > mx:
            mx = a[op + i]
        i += TG_SM
    shared[t] = mx
    barrier()
    var stride = TG_SM // 2
    while stride > 0:
        if t < stride:
            if shared[t + stride] > shared[t]:
                shared[t] = shared[t + stride]
        barrier()
        stride //= 2
    mx = shared[0]
    barrier()
    # exp + truncate + sum
    var sm = Float32(0)
    i = t
    while i < n:
        var e = exp((a[op + i] - mx) * invt)
        if e < eps:
            e = 0
        a[op + i] = e
        sm += e
        i += TG_SM
    shared[t] = sm
    barrier()
    stride = TG_SM // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        barrier()
        stride //= 2
    var inv = Float32(1) / shared[0]
    barrier()
    # normalize + entropy
    var hp = Float32(0)
    i = t
    while i < n:
        var p = a[op + i] * inv
        a[op + i] = p
        if p > 0:
            hp -= p * log(p)
        i += TG_SM
    shared[t] = hp
    barrier()
    stride = TG_SM // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        barrier()
        stride //= 2
    if t == 0:
        a[oent] = shared[0]


def k_row_norm(w: UnsafePointer[UInt16, MutAnyOrigin],
               a: UnsafePointer[Float32, MutAnyOrigin],
               oemb: Int, oy: Int, v: Int, h: Int):
    """Per-row L2 norms of the embedding matrix (startup statistic)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= v:
        return
    var ss = Float32(0)
    for j in range(h):
        var bits = UInt32(w[oemb + r * h + j]) << 16
        var x = UnsafePointer(to=bits).bitcast[Float32]()[]
        ss += x * x
    a[oy + r] = sqrt(ss)


def k_renorm(a: UnsafePointer[Float32, MutAnyOrigin],
             oe: Int, h: Int, target: Float32):
    """Rescale e to the mean embedding-row norm (undo Jensen shrinkage)."""
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG_SM, Float32,
                                  address_space = AddressSpace.SHARED]()
    var ss = Float32(0)
    var i = t
    while i < h:
        ss += a[oe + i] * a[oe + i]
        i += TG_SM
    shared[t] = ss
    barrier()
    var stride = TG_SM // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        barrier()
        stride //= 2
    var nrm = sqrt(shared[0])
    if nrm < 1e-6:
        return
    var s = target / nrm
    i = t
    while i < h:
        a[oe + i] *= s
        i += TG_SM


def k_soft_embed(w: UnsafePointer[UInt16, MutAnyOrigin],
                 a: UnsafePointer[Float32, MutAnyOrigin],
                 op: Int, oemb: Int, oy: Int, v: Int, h: Int):
    """y[j] = sum_v p[v] * E[v, j] — expected embedding under p.

    Adjacent threads read adjacent columns of E (coalesced); p[v] is a
    broadcast load. E is streamed once (same traffic as the lm_head matmul).
    """
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= h:
        return
    var acc = Float32(0)
    for i in range(v):
        var p = a[op + i]
        if p > 1e-8:
            var bits = UInt32(w[oemb + i * h + j]) << 16
            acc += p * UnsafePointer(to=bits).bitcast[Float32]()[]
    a[oy + j] = acc


# ============================ latent driver ===================================

comptime ENT_CHECK = 8                  # entropy readback period (steps)


def forward_to_soft(mut m: Qwen, oh_in: Int, s: Int, osoft: Int, oent: Int,
                    invt: Float32, target_norm: Float32,
                    eps: Float32) raises:
    """Layers -> norm -> logits -> truncated tempered softmax -> expected
    embedding, rescaled to the mean row norm. Fully enqueued; no sync."""
    var oh = oh_in
    for L in range(m.cfg.layers):
        oh = run_layer(m.ctx, m.w.buf, m.a, m.cfg, m.lo[L],
                       oh, s, m.n_cached, m.kc[L], m.vc[L], m.oinv)
        if s > 4 and (L & 3) == 3:
            m.ctx.synchronize()
    m.n_cached += s
    var onrm = rmsnorm_op(m.ctx, m.w.buf, m.a, oh + (s - 1) * H, 1,
                          m.w.o("model.norm.weight"), H, m.cfg.eps)
    var olg = mm_op(m.ctx, m.w.buf, m.a, onrm, 1, H,
                    m.w.o("model.embed_tokens.weight"), VOCAB)
    m.ctx.enqueue_function[k_softmax_soft](
        m.a.buf.unsafe_ptr(), olg, oent, VOCAB, invt, eps,
        grid_dim=1, block_dim=TG_SM)
    m.ctx.enqueue_function[k_soft_embed](
        m.w.buf.unsafe_ptr(), m.a.buf.unsafe_ptr(),
        olg, m.w.o("model.embed_tokens.weight"), osoft, VOCAB, H,
        grid_dim=ceildiv(H, BLOCK), block_dim=BLOCK)
    if target_norm > 0:
        m.ctx.enqueue_function[k_renorm](
            m.a.buf.unsafe_ptr(), osoft, H, target_norm,
            grid_dim=1, block_dim=TG_SM)


def mean_row_norm(mut m: Qwen) raises -> Float32:
    """Mean L2 norm of embedding rows (renorm target), computed once."""
    from llama_common import k_export
    var mark = m.a.mark()
    var onorms = m.a.alloc(VOCAB)
    m.ctx.enqueue_function[k_row_norm](
        m.w.buf.unsafe_ptr(), m.a.buf.unsafe_ptr(),
        m.w.o("model.embed_tokens.weight"), onorms, VOCAB, H,
        grid_dim=ceildiv(VOCAB, BLOCK), block_dim=BLOCK)
    m.ctx.enqueue_function[k_export](
        m.a.buf.unsafe_ptr(), m.lgbuf.unsafe_ptr(), onorms, PAD, VOCAB,
        grid_dim=ceildiv(VOCAB, BLOCK), block_dim=BLOCK)
    m.ctx.synchronize()
    m.a.reset(mark)
    var sm = Float64(0)
    with m.lgbuf.map_to_host() as hst:
        for i in range(VOCAB):
            sm += Float64(hst[PAD + i])
    return Float32(sm / Float64(VOCAB))


def read_entropy(mut m: Qwen, oent: Int) raises -> Float32:
    from llama_common import k_export
    m.ctx.enqueue_function[k_export](
        m.a.buf.unsafe_ptr(), m.lgbuf.unsafe_ptr(), oent, PAD, 1,
        grid_dim=1, block_dim=BLOCK)
    m.ctx.synchronize()
    with m.lgbuf.map_to_host() as hst:
        return hst[PAD]


def prefill_to_soft(mut m: Qwen, ids: List[Int], osoft: Int, oent: Int,
                    invt: Float32, target_norm: Float32,
                    eps: Float32) raises:
    var mark = m.a.mark()
    var oh = m.embed(ids)
    forward_to_soft(m, oh, len(ids), osoft, oent, invt, target_norm, eps)
    m.ctx.synchronize()
    m.a.reset(mark)


def latent_steps(mut m: Qwen, n: Int, osoft: Int, oent: Int,
                 t_hi: Float32, t_lo: Float32, tau: Float32,
                 target_norm: Float32, eps: Float32
                 ) raises -> Tuple[Int, Float32]:
    """Up to n soft steps: temperature annealed t_hi -> t_lo (geometric,
    explore-then-commit), entropy checked every ENT_CHECK steps, early stop
    when H(p) < tau. Returns (steps used, final entropy)."""
    # Entropy gates the stop, but instantaneous H is misleading: most
    # thinking steps are near-deterministic connectives (H ~ 0) with H
    # spiking only at decision points. Require LOW entropy at two
    # consecutive checks, and never stop inside the warmup.
    var ent = Float32(0)
    var done = 0
    var low_streak = 0
    comptime WARMUP = 16
    for i in range(n):
        if m.n_cached + 1 > m.maxlen:
            raise Error("KV cache full")
        var frac = Float64(i) / Float64(max(n - 1, 1))
        var temp = Float32(Float64(t_hi) * ((Float64(t_lo) / Float64(t_hi)) ** frac))
        var mark = m.a.mark()
        # osoft is outside the scratch mark: run_layer reads it, and the
        # k_soft_embed at the end overwrites it strictly afterwards.
        forward_to_soft(m, osoft, 1, osoft, oent, Float32(1.0) / temp,
                        target_norm, eps)
        m.a.reset(mark)
        done = i + 1
        if tau > 0 and done >= WARMUP and done % ENT_CHECK == 0:
            ent = read_entropy(m, oent)
            if ent < tau:
                low_streak += 1
                if low_streak >= 2:
                    break
            else:
                low_streak = 0
    m.ctx.synchronize()
    if tau <= 0:
        ent = read_entropy(m, oent)
    return (done, ent)


# ============================ tokenizer helpers ===============================

def encode(tok: PythonObject, text: String) raises -> List[Int]:
    var pyids = tok.encode(text, add_special_tokens=False).ids
    var ids = List[Int]()
    for i in range(len(pyids)):
        ids.append(atol(String(pyids[i])))
    return ids^


def decode_ids(tok: PythonObject, ids: List[Int]) raises -> String:
    var py = Python.list()
    for i in range(len(ids)):
        py.append(ids[i])
    return String(tok.decode(py))


def greedy_answer(mut m: Qwen, tok: PythonObject, first: Int,
                  max_new: Int) raises -> Tuple[String, Int, Float64]:
    """Greedy decode from token `first` until EOS; returns (text, n, secs)."""
    var out = List[Int]()
    var nxt = first
    var t0 = perf_counter_ns()
    while m.n_cached < MAXLEN and len(out) < max_new:
        if nxt == EOS:
            break
        out.append(nxt)
        var step: List[Int] = [nxt]
        nxt = m.forward_argmax(step)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    return (decode_ids(tok, out), len(out), dt)


# ============================ main ============================================

def main() raises:
    var question = String("If 3 people can paint 3 fences in 3 hours, "
                          "how many hours do 9 people need for 9 fences?")
    var n_latent = 48
    var t_hi = Float32(1.0)
    var t_lo = Float32(0.5)
    var tau = Float32(0.8)              # entropy early-stop (nats)
    if len(argv()) > 1:
        question = String(argv()[1])
    if len(argv()) > 2:
        n_latent = atol(String(argv()[2]))
    if len(argv()) > 3:
        t_hi = Float32(atof(String(argv()[3])))
    if len(argv()) > 4:
        t_lo = Float32(atof(String(argv()[4])))
    if len(argv()) > 5:
        tau = Float32(atof(String(argv()[5])))
    var eps = Float32(1e-3)
    if len(argv()) > 6:
        eps = Float32(atof(String(argv()[6])))
    var use_renorm = True
    if len(argv()) > 7 and String(argv()[7]) == "0":
        use_renorm = False
    var skip_baseline = False
    if len(argv()) > 8 and String(argv()[8]) == "skip":
        skip_baseline = True

    var tk_mod = Python.import_module("tokenizers")
    var tok = tk_mod.Tokenizer.from_file(TOKENIZER)
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_qwen(ctx, MAXLEN)
    var osoft = model.a.alloc(H)        # persistent soft-embedding slot
    var oent = model.a.alloc(1)         # persistent entropy slot
    var tnorm = mean_row_norm(model)
    print("mean embedding row norm:", tnorm)
    if not use_renorm:
        tnorm = Float32(0)
    print("config: eps", eps, " renorm", use_renorm)

    var prompt = (String("<|im_start|>user\n") + question
                  + String("<|im_end|>\n<|im_start|>assistant\n"))

    # ---- baseline: normal discrete thinking (greedy) ----
    var ids = List[Int]()
    if not skip_baseline:
        print("\n=== baseline: discrete thinking (greedy) ===")
        ids = encode(tok, prompt)
        var nxt0 = model.forward_argmax(ids)
        var res = greedy_answer(model, tok, nxt0, 700)
        print(res[0])
        print("[", res[1], "tokens in", res[2], "s =",
              Float64(res[1]) / res[2], "tok/s ]")

    # ---- latent thinking run ----
    print("\n=== latent: <=", n_latent, "soft steps, T", t_hi, "->", t_lo,
          ", stop at H <", tau, "===")
    model.n_cached = 0
    ids = encode(tok, prompt + String("<think>\n"))
    prefill_to_soft(model, ids, osoft, oent, Float32(1.0) / t_hi, tnorm, eps)

    var t0 = perf_counter_ns()
    var lres = latent_steps(model, n_latent, osoft, oent,
                            t_hi, t_lo, tau, tnorm, eps)
    var dt_lat = Float64(perf_counter_ns() - t0) / 1e9
    print("[", lres[0], "latent steps in", dt_lat, "s =",
          Float64(lres[0]) / dt_lat, "steps/s, final H =", lres[1], "nats ]")

    # prime the answer channel so the model can't restart discrete CoT —
    # if the latent steps carried information, it shows up here
    var closing = encode(tok, String("</think>\n\nThe answer is"))
    var nxt = model.forward_argmax(closing)
    var res2 = greedy_answer(model, tok, nxt, 120)
    print("The answer is" + res2[0])
    print("[ answer:", res2[1], "tokens in", res2[2], "s =",
          Float64(res2[1]) / res2[2], "tok/s ]")

    # control: same primer with NO thinking at all (0 latent steps)
    print("\n=== control: no thinking, primed answer ===")
    model.n_cached = 0
    ids = encode(tok, prompt + String("<think>\n</think>\n\nThe answer is"))
    nxt = model.forward_argmax(ids)
    var res3 = greedy_answer(model, tok, nxt, 120)
    print("The answer is" + res3[0])
