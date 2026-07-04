"""OmniVoice-specific GPU kernels + bidirectional layer runner (Metal).

Extends the shared llama_common.mojo machinery with what the masked-diffusion
TTS needs: non-causal attention (no KV cache reuse — every unmasking step
recomputes the full sequence), mixed text/audio input embeddings, a fused
CFG-predict kernel, and 1-D conv kernels for the HiggsAudio DAC decoder.

Same Metal conventions as llama_common: bf16 weight buffer (u16) + f32
activation arena, at most 3 buffers bound per kernel, scalar params.
"""

from std.math import ceildiv, sqrt, exp, sin, log
from std.memory import alloc, stack_allocation
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, PAD, TG_MM, TG,
    bf, bf4, bf8, mm_op, rmsnorm_op, k_rope_qk, k_res_add, k_add, k_swiglu_mul,
    k_softmax_rows, k_att_out, k_rmsnorm2_w, k_res_add_rmsnorm, KPtr,
)


def _box[T: Movable](var f: T) -> KPtr:
    var p = alloc[T](1)
    p.init_pointee_move(f^)
    return p.bitcast[NoneType]()


# ============================ kernels =========================================


def k_scores_bidir(a: UnsafePointer[Float32, MutAnyOrigin],
                   oq: Int, okc: Int, osc: Int, s: Int, nctx: Int,
                   nheads: Int, group: Int, qdim: Int, kvdim: Int, headdim: Int,
                   blen: Int):
    """GQA scores (heads, s, nctx) with full bidirectional attention.
    If blen > 0, s = batch * blen and attention is block-diagonal."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= nheads * s * nctx:
        return
    var h = idx // (s * nctx)
    var r = idx % (s * nctx)
    var i = r // nctx
    var j = r % nctx
    if blen > 0 and (i // blen) != (j // blen):
        a[osc + idx] = Float32(-3.0e38)
        return
    var qb = oq + i * qdim + h * headdim
    var kb = okc + j * kvdim + (h // group) * headdim
    var acc4 = SIMD[DType.float32, 4](0)
    var d = 0
    while d < headdim:
        acc4 += a.load[width=4](qb + d) * a.load[width=4](kb + d)
        d += 4
    a[osc + idx] = acc4.reduce_add() / sqrt(Float32(headdim))


def k_ov_embed(w: UnsafePointer[UInt16, MutAnyOrigin],
               a: UnsafePointer[Float32, MutAnyOrigin],
               oids: UnsafePointer[Int32, MutAnyOrigin],
               otext: Int, oaudio: Int, oy: Int,
               L: Int, H: Int, astart: Int, ncb: Int, vocab: Int):
    """Input embeddings for (8, L) ids: text rows before astart (row 0 of
    ids, llm.embed_tokens), sum of per-codebook audio embeddings after."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= L * H:
        return
    var t = idx // H
    var d = idx % H
    if t < astart:
        a[oy + idx] = bf(w, otext + Int(oids[t]) * H + d)
    else:
        var acc = Float32(0)
        for c in range(ncb):
            var id = Int(oids[c * L + t]) + c * vocab
            acc += bf(w, oaudio + id * H + d)
        a[oy + idx] = acc


def k_ov_embed_pair(w: UnsafePointer[UInt16, MutAnyOrigin],
                    a: UnsafePointer[Float32, MutAnyOrigin],
                    oids0: UnsafePointer[Int32, MutAnyOrigin],
                    oids1: UnsafePointer[Int32, MutAnyOrigin],
                    otext: Int, oaudio: Int, oy: Int,
                    L: Int, H: Int, astart0: Int, astart1: Int,
                    ncb: Int, vocab: Int):
    """Embed two (ncb, L) id sequences into stacked rows [0..L) and [L..2L)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= 2 * L * H:
        return
    var batch = idx // (L * H)
    var rest = idx % (L * H)
    var t = rest // H
    var d = rest % H
    var oids = oids0 if batch == 0 else oids1
    var astart = astart0 if batch == 0 else astart1
    var row = oy + batch * L * H + idx % (L * H)
    if t < astart:
        a[row] = bf(w, otext + Int(oids[t]) * H + d)
    else:
        var acc = Float32(0)
        for c in range(ncb):
            var id = Int(oids[c * L + t]) + c * vocab
            acc += bf(w, oaudio + id * H + d)
        a[row] = acc


def k_cfg_predict(a: UnsafePointer[Float32, MutAnyOrigin],
                  g: UnsafePointer[Float32, MutAnyOrigin],
                  oc: Int, ou: Int, T: Int, ncb: Int, vocab: Int,
                  mask_id: Int, guidance: Float32):
    """Per (codebook, frame): CFG-combined greedy prediction + confidence.

    c/u logits live in the arena as (T, ncb*vocab) rows (audio_heads output).
    Writes pred id (as f32) to g[2*(c*T+t)] and confidence to g[2*(c*T+t)+1].
    conf = max over non-mask v of log_softmax(lc + guidance*(lc - lu)).
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= ncb * T:
        return
    var c = idx // T
    var t = idx % T
    var cb = oc + t * (ncb * vocab) + c * vocab
    var ub = ou + t * (ncb * vocab) + c * vocab

    var cmax = a[cb]
    var umax = a[ub]
    for v in range(1, vocab):
        if a[cb + v] > cmax:
            cmax = a[cb + v]
        if a[ub + v] > umax:
            umax = a[ub + v]
    var cse = Float32(0)
    var use = Float32(0)
    for v in range(vocab):
        cse += exp(a[cb + v] - cmax)
        use += exp(a[ub + v] - umax)
    var clse = cmax + log(cse)
    var ulse = umax + log(use)

    # combined = (1+g)*lc - g*lu; conf needs log_softmax over ALL v
    # (mask id included in the normalizer, banned only for argmax).
    var comb_max = Float32(-3.0e38)
    for v in range(vocab):
        var lc = a[cb + v] - clse
        var lu = a[ub + v] - ulse
        var comb = lc + guidance * (lc - lu)
        if comb > comb_max:
            comb_max = comb
    var comb_se = Float32(0)
    var best = 0
    var best_val = Float32(-3.0e38)
    for v in range(vocab):
        var lc = a[cb + v] - clse
        var lu = a[ub + v] - ulse
        var comb = lc + guidance * (lc - lu)
        comb_se += exp(comb - comb_max)
        if v != mask_id and comb > best_val:
            best_val = comb
            best = v
    g[2 * idx] = Float32(best)
    g[2 * idx + 1] = best_val - (comb_max + log(comb_se))


def k_conv1d(w: UnsafePointer[UInt16, MutAnyOrigin],
             a: UnsafePointer[Float32, MutAnyOrigin],
             ox: Int, owt: Int, obias: Int, oy: Int,
             cin: Int, cout: Int, tin: Int, tout: Int,
             ksize: Int, stride: Int, pad: Int, dil: Int):
    """y[co, t] = bias[co] + sum_{ci,k} x[ci, t*stride - pad + k*dil] * W.
    W layout [cout, cin, k] bf16; x/y channel-major (c, t) f32.
    obias < 0 means no bias."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= cout * tout:
        return
    var co = idx // tout
    var t = idx % tout
    var acc = Float32(0)
    if obias >= 0:
        acc = bf(w, obias + co)
    var t0 = t * stride - pad
    for ci in range(cin):
        var xb = ox + ci * tin
        var wb = owt + (co * cin + ci) * ksize
        for k in range(ksize):
            var j = t0 + k * dil
            if j >= 0 and j < tin:
                acc += a[xb + j] * bf(w, wb + k)
    a[oy + idx] = acc


def k_convtr1d(w: UnsafePointer[UInt16, MutAnyOrigin],
               a: UnsafePointer[Float32, MutAnyOrigin],
               ox: Int, owt: Int, obias: Int, oy: Int,
               cin: Int, cout: Int, tin: Int, tout: Int,
               ksize: Int, stride: Int, pad: Int):
    """ConvTranspose1d: y[co, t] = bias + sum over taps where
    (t + pad - k) divides stride. W layout [cin, cout, k] bf16."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= cout * tout:
        return
    var co = idx // tout
    var t = idx % tout
    var acc = Float32(0)
    if obias >= 0:
        acc = bf(w, obias + co)
    for k in range(ksize):
        var num = t + pad - k
        if num < 0 or num % stride != 0:
            continue
        var j = num // stride
        if j >= tin:
            continue
        for ci in range(cin):
            acc += a[ox + ci * tin + j] * bf(w, owt + (ci * cout + co) * ksize + k)
    a[oy + idx] = acc


def k_snake(w: UnsafePointer[UInt16, MutAnyOrigin],
            a: UnsafePointer[Float32, MutAnyOrigin],
            oalpha: Int, ox: Int, oy: Int, C: Int, T: Int):
    """Snake activation: y = x + sin(alpha*x)^2 / (alpha + 1e-9), alpha per channel."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= C * T:
        return
    var c = idx // T
    var al = bf(w, oalpha + c)
    var x = a[ox + idx]
    var s = sin(al * x)
    a[oy + idx] = x + s * s / (al + Float32(1e-9))


def k_rvq_decode(w: UnsafePointer[UInt16, MutAnyOrigin],
                 a: UnsafePointer[Float32, MutAnyOrigin],
                 oids: UnsafePointer[Int32, MutAnyOrigin],
                 oemb: Int, oproj: Int, obias: Int, oy: Int,
                 T: Int, H: Int, ncb: Int, cdim: Int, cbsize: Int):
    """RVQ dequantize: y[h, t] = sum_c (bias_c[h] + P_c[h,:] @ E_c[id_ct,:]).

    The 8 codebook embed tables ([cbsize, cdim]) are uploaded contiguously at
    oemb, project_out weights ([H, cdim]) at oproj, biases ([H]) at obias.
    ids are (ncb, T) row-major. Output y is channel-major (H, T)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= H * T:
        return
    var h = idx // T
    var t = idx % T
    var acc = Float32(0)
    for c in range(ncb):
        acc += bf(w, obias + c * H + h)
        var eb = oemb + c * cbsize * cdim + Int(oids[c * T + t]) * cdim
        var pb = oproj + c * H * cdim + h * cdim
        var acc4 = SIMD[DType.float32, 4](0)
        var d = 0
        while d < cdim:
            acc4 = bf4(w, eb + d).fma(bf4(w, pb + d), acc4)
            d += 4
        acc += acc4.reduce_add()
    a[oy + idx] = acc


def k_copy(a: UnsafePointer[Float32, MutAnyOrigin], os: Int, od: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[od + i] = a[os + i]


def k_export_slice(a: UnsafePointer[Float32, MutAnyOrigin],
                   g: UnsafePointer[Float32, MutAnyOrigin],
                   osrc: Int, odst: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        g[odst + i] = a[osrc + i]


# ============================ host-side =======================================


struct OvKernels(Copyable, Movable):
    var scores: KPtr
    var embed: KPtr
    var embedpair: KPtr
    var cfg: KPtr
    var conv: KPtr
    var convtr: KPtr
    var snake: KPtr
    var rvq: KPtr
    var cpy: KPtr
    var exp: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.scores = _box(ctx.compile_function[k_scores_bidir]())
        self.embed = _box(ctx.compile_function[k_ov_embed]())
        self.embedpair = _box(ctx.compile_function[k_ov_embed_pair]())
        self.cfg = _box(ctx.compile_function[k_cfg_predict]())
        self.conv = _box(ctx.compile_function[k_conv1d]())
        self.convtr = _box(ctx.compile_function[k_convtr1d]())
        self.snake = _box(ctx.compile_function[k_snake]())
        self.rvq = _box(ctx.compile_function[k_rvq_decode]())
        self.cpy = _box(ctx.compile_function[k_copy]())
        self.exp = _box(ctx.compile_function[k_export_slice]())


def run_layer_bidir(ctx: DeviceContext, w: DeviceBuffer[DType.uint16],
                    mut a: Acts, kn: OvKernels, cfg: Config, lo: LayerOffs,
                    ox: Int, s: Int, oinv: Int, blen: Int = 0) raises -> Int:
    """One pre-norm GQA block with full bidirectional attention over s tokens.

    No KV cache: k/v live in per-call scratch (every diffusion step recomputes
    the whole sequence). Positions are 0..s-1 (RoPE pos0 = 0).
    """
    var H = cfg.hidden
    var QD = cfg.q_dim()
    var KVD = cfg.kv_dim()
    var xn = rmsnorm_op(ctx, w, a, ox, s, lo.in_norm, H, cfg.eps)
    var oq = mm_op(ctx, w, a, xn, s, H, lo.q, QD)
    var okk = mm_op(ctx, w, a, xn, s, H, lo.k, KVD)
    var ov = mm_op(ctx, w, a, xn, s, H, lo.v, KVD)
    if lo.q_norm >= 0:
        var oqn = a.alloc(s * QD)
        var okn = a.alloc(s * KVD)
        ctx.enqueue_function(
            a.kn.rms2.bitcast[type_of(ctx.compile_function[k_rmsnorm2_w]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(),
            oq, okk, lo.q_norm, lo.k_norm, oqn, okn,
            s * cfg.n_heads, s * cfg.n_kv, cfg.head_dim, cfg.eps,
            grid_dim=s * cfg.n_heads + s * cfg.n_kv, block_dim=TG)
        oq = oqn
        okk = okn
    ctx.enqueue_function(
        a.kn.rope.bitcast[type_of(ctx.compile_function[k_rope_qk]())]()[],
        a.buf.unsafe_ptr(), oq, okk, oinv, s, 0,
        cfg.n_heads, cfg.n_kv, QD, KVD, cfg.half(),
        grid_dim=ceildiv(s * (cfg.n_heads + cfg.n_kv) * cfg.half(), BLOCK),
        block_dim=BLOCK)
    var nctx = s
    var osc = a.alloc(cfg.n_heads * s * nctx)
    ctx.enqueue_function(
        kn.scores.bitcast[type_of(ctx.compile_function[k_scores_bidir]())]()[],
        a.buf.unsafe_ptr(), oq, okk, osc, s, nctx,
        cfg.n_heads, cfg.group(), QD, KVD, cfg.head_dim, blen,
        grid_dim=ceildiv(cfg.n_heads * s * nctx, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function(
        a.kn.softmax.bitcast[type_of(ctx.compile_function[k_softmax_rows]())]()[],
        a.buf.unsafe_ptr(), osc, cfg.n_heads * s, nctx,
        grid_dim=ceildiv(cfg.n_heads * s, BLOCK), block_dim=BLOCK)
    var oao = a.alloc(s * QD)
    ctx.enqueue_function(
        a.kn.attout.bitcast[type_of(ctx.compile_function[k_att_out]())]()[],
        a.buf.unsafe_ptr(), osc, ov, oao, s, nctx,
        QD, KVD, cfg.head_dim, cfg.group(),
        grid_dim=ceildiv(s * QD, BLOCK), block_dim=BLOCK)
    var oo = mm_op(ctx, w, a, oao, s, QD, lo.o, H)
    var oh = a.alloc(s * H)
    ctx.enqueue_function(
        a.kn.resadd.bitcast[type_of(ctx.compile_function[k_res_add]())]()[],
        a.buf.unsafe_ptr(), ox, oo, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    var oz = rmsnorm_op(ctx, w, a, oh, s, lo.post_norm, H, cfg.eps)
    var og = mm_op(ctx, w, a, oz, s, H, lo.gate, cfg.inter)
    var ou = mm_op(ctx, w, a, oz, s, H, lo.up, cfg.inter)
    ctx.enqueue_function(
        a.kn.swiglu.bitcast[type_of(ctx.compile_function[k_swiglu_mul]())]()[],
        a.buf.unsafe_ptr(), og, ou, s * cfg.inter,
        grid_dim=ceildiv(s * cfg.inter, BLOCK), block_dim=BLOCK)
    var om = mm_op(ctx, w, a, og, s, cfg.inter, lo.down, H)
    ctx.enqueue_function(
        a.kn.add.bitcast[type_of(ctx.compile_function[k_add]())]()[],
        a.buf.unsafe_ptr(), om, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    return oh
