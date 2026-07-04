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
    Config, LayerOffs, Acts, BLOCK, PAD, TG_MM, bf, bf4, bf8,
    mm_op, rmsnorm_op, k_rope_qk, k_copy2, k_res_add, k_add, k_swiglu_mul,
    k_softmax_rows, k_att_out, KPtr,
)


def _box[T: Movable](var f: T) -> KPtr:
    var p = alloc[T](1)
    p.init_pointee_move(f^)
    return p.bitcast[NoneType]()


# ============================ kernels =========================================


def k_mm_tile(w: UnsafePointer[UInt16, MutAnyOrigin],
              a: UnsafePointer[Float32, MutAnyOrigin],
              ox: Int, owt: Int, oy: Int, s: Int, m: Int, n: Int):
    """Tiled y (s,n) = x (s,m) @ Wbf16 (n,m)^T: 8 rows x 8 cols per 32-thread
    group. The shared k_mm_w does one output per group — optimal for s == 1
    but ~2 KB memory traffic per output; the tile reuses each x/w load 8x,
    which dominates for the seq-level forwards here. Requires m % 256 == 0."""
    var nrb = (s + 7) // 8
    var blk = Int(block_idx.x)
    var rb = (blk % nrb) * 8
    var cb = (blk // nrb) * 8
    var t = Int(thread_idx.x)
    var acc = InlineArray[Float32, 64](fill=0)
    var k = t * 8
    while k < m:
        var x = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        for r in range(8):
            x[r] = a.load[width=8](ox + min(rb + r, s - 1) * m + k)
        for c in range(8):
            var j = cb + c
            if j >= n:
                break
            var w8 = bf8(w, owt + j * m + k)
            for r in range(8):
                acc[r * 8 + c] += (x[r] * w8).reduce_add()
        k += TG_MM * 8
    var shared = stack_allocation[
        64 * TG_MM, Float32, address_space = AddressSpace.SHARED]()
    for e in range(64):
        shared[e * TG_MM + t] = acc[e]
    barrier()
    if t < 32:
        for half in range(2):
            var e = half * TG_MM + t
            var r = rb + e // 8
            var j = cb + e % 8
            if r < s and j < n:
                var tot = SIMD[DType.float32, 4](0)
                var q = 0
                while q < TG_MM:
                    tot += shared.load[width=4](e * TG_MM + q)
                    q += 4
                a[oy + r * n + j] = tot.reduce_add()


def k_scores_bidir(a: UnsafePointer[Float32, MutAnyOrigin],
                   oq: Int, okc: Int, osc: Int, s: Int, nctx: Int,
                   nheads: Int, group: Int, qdim: Int, kvdim: Int, headdim: Int):
    """GQA scores (heads, s, nctx) with full bidirectional attention."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= nheads * s * nctx:
        return
    var h = idx // (s * nctx)
    var r = idx % (s * nctx)
    var i = r // nctx
    var j = r % nctx
    var acc = Float32(0)
    var qb = oq + i * qdim + h * headdim
    var kb = okc + j * kvdim + (h // group) * headdim
    for d in range(headdim):
        acc += a[qb + d] * a[kb + d]
    a[osc + idx] = acc / sqrt(Float32(headdim))


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
    var mmt: KPtr
    var scores: KPtr
    var embed: KPtr
    var cfg: KPtr
    var conv: KPtr
    var convtr: KPtr
    var snake: KPtr
    var rvq: KPtr
    var cpy: KPtr
    var exp: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.mmt = _box(ctx.compile_function[k_mm_tile]())
        self.scores = _box(ctx.compile_function[k_scores_bidir]())
        self.embed = _box(ctx.compile_function[k_ov_embed]())
        self.cfg = _box(ctx.compile_function[k_cfg_predict]())
        self.conv = _box(ctx.compile_function[k_conv1d]())
        self.convtr = _box(ctx.compile_function[k_convtr1d]())
        self.snake = _box(ctx.compile_function[k_snake]())
        self.rvq = _box(ctx.compile_function[k_rvq_decode]())
        self.cpy = _box(ctx.compile_function[k_copy]())
        self.exp = _box(ctx.compile_function[k_export_slice]())


def mm_op_t(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
            kn: OvKernels, ox: Int, s: Int, m: Int, owt: Int,
            n: Int) raises -> Int:
    """Tiled matmul for multi-row inputs; falls back to k_mm_w for s < 4.
    Chunks row blocks so one dispatch stays under the watchdog budget."""
    if s < 4:
        return mm_op(ctx, w, a, ox, s, m, owt, n)
    var oy = a.alloc(s * n)
    var ncb = ceildiv(n, 8)
    var rows_per_chunk = max(8, ((1 << 30) // (n * m)) * 8)
    var r0 = 0
    while r0 < s:
        var rows = min(rows_per_chunk, s - r0)
        ctx.enqueue_function(
            kn.mmt.bitcast[type_of(ctx.compile_function[k_mm_tile]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox + r0 * m, owt, oy + r0 * n, rows, m, n,
            grid_dim=ceildiv(rows, 8) * ncb, block_dim=TG_MM)
        r0 += rows
        if r0 < s:
            ctx.synchronize()
    return oy


def run_layer_bidir(ctx: DeviceContext, w: DeviceBuffer[DType.uint16],
                    mut a: Acts, kn: OvKernels, cfg: Config, lo: LayerOffs,
                    ox: Int, s: Int, oinv: Int) raises -> Int:
    """One pre-norm GQA block with full bidirectional attention over s tokens.

    No KV cache: k/v live in per-call scratch (every diffusion step recomputes
    the whole sequence). Positions are 0..s-1 (RoPE pos0 = 0).
    """
    var H = cfg.hidden
    var QD = cfg.q_dim()
    var KVD = cfg.kv_dim()
    var xn = rmsnorm_op(ctx, w, a, ox, s, lo.in_norm, H, cfg.eps)
    var oq = mm_op_t(ctx, w, a, kn, xn, s, H, lo.q, QD)
    var okk = mm_op_t(ctx, w, a, kn, xn, s, H, lo.k, KVD)
    var ov = mm_op_t(ctx, w, a, kn, xn, s, H, lo.v, KVD)
    if lo.q_norm >= 0:
        oq = rmsnorm_op(ctx, w, a, oq, s * cfg.n_heads, lo.q_norm,
                        cfg.head_dim, cfg.eps)
        okk = rmsnorm_op(ctx, w, a, okk, s * cfg.n_kv, lo.k_norm,
                         cfg.head_dim, cfg.eps)
    ctx.enqueue_function(
        a.kn.rope.bitcast[type_of(ctx.compile_function[k_rope_qk]())]()[],
        a.buf.unsafe_ptr(), oq, okk, oinv, s, 0,
        cfg.n_heads, cfg.n_kv, QD, KVD, cfg.half(),
        grid_dim=ceildiv(s * (cfg.n_heads + cfg.n_kv) * cfg.half(), BLOCK),
        block_dim=BLOCK)
    var osc = a.alloc(cfg.n_heads * s * s)
    ctx.enqueue_function(
        kn.scores.bitcast[type_of(ctx.compile_function[k_scores_bidir]())]()[],
        a.buf.unsafe_ptr(), oq, okk, osc, s, s,
        cfg.n_heads, cfg.group(), QD, KVD, cfg.head_dim,
        grid_dim=ceildiv(cfg.n_heads * s * s, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function(
        a.kn.softmax.bitcast[type_of(ctx.compile_function[k_softmax_rows]())]()[],
        a.buf.unsafe_ptr(), osc, cfg.n_heads * s, s,
        grid_dim=ceildiv(cfg.n_heads * s, BLOCK), block_dim=BLOCK)
    # k_att_out reads a v "cache"; v rows are already contiguous at ov.
    var oao = a.alloc(s * QD)
    ctx.enqueue_function(
        a.kn.attout.bitcast[type_of(ctx.compile_function[k_att_out]())]()[],
        a.buf.unsafe_ptr(), osc, ov, oao, s, s,
        QD, KVD, cfg.head_dim, cfg.group(),
        grid_dim=ceildiv(s * QD, BLOCK), block_dim=BLOCK)
    var oo = mm_op_t(ctx, w, a, kn, oao, s, QD, lo.o, H)
    var oh = a.alloc(s * H)
    ctx.enqueue_function(
        a.kn.resadd.bitcast[type_of(ctx.compile_function[k_res_add]())]()[],
        a.buf.unsafe_ptr(), ox, oo, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    var oz = rmsnorm_op(ctx, w, a, oh, s, lo.post_norm, H, cfg.eps)
    var og = mm_op_t(ctx, w, a, kn, oz, s, H, lo.gate, cfg.inter)
    var ou = mm_op_t(ctx, w, a, kn, oz, s, H, lo.up, cfg.inter)
    ctx.enqueue_function(
        a.kn.swiglu.bitcast[type_of(ctx.compile_function[k_swiglu_mul]())]()[],
        a.buf.unsafe_ptr(), og, ou, s * cfg.inter,
        grid_dim=ceildiv(s * cfg.inter, BLOCK), block_dim=BLOCK)
    var om = mm_op_t(ctx, w, a, kn, og, s, cfg.inter, lo.down, H)
    ctx.enqueue_function(
        a.kn.add.bitcast[type_of(ctx.compile_function[k_add]())]()[],
        a.buf.unsafe_ptr(), om, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    return oh
