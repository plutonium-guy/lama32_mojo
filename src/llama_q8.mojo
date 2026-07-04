"""INT8 weight-only (W8A32) kernels + layer runner (see QUANTIZATION.md).

Packed layout (scripts/quantize.py): each 2-D weight row [m] becomes
[m/64 f16 group scales | m int8 values], i.e. m/64 + m/2 u16 elements.
Offsets stay in u16 units so SafeTensors/Weights load the packed file
unchanged. 1-D tensors (norms) remain bf16 and keep using llama_common's
kernels; only the matmul and embedding-gather weight reads change.

Dequant: w = int8 * f16_scale(group of 64 along the input dim).
"""

from std.math import ceildiv
from std.memory import alloc, stack_allocation, bitcast
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, TG, TG_MM, PAD, MM_CHUNK_MACS, KPtr,
    rmsnorm_op, k_rope_qk, k_copy2, k_res_add, k_add, k_swiglu_mul,
    k_softmax_rows, k_scores, k_att_out,
)

comptime QG = 64                        # quantization group (input dim)


def q8_rowstride(m: Int) -> Int:
    """Packed row length in u16 elements: m/64 scales + m/2 byte-pairs."""
    return m // QG + m // 2


def f16(w: UnsafePointer[UInt16, MutAnyOrigin], i: Int) -> Float32:
    # SIMD bitcast (as bf8 does) — the take-address-of-local variant spills
    # to GPU stack memory and wrecks the matmul hot loop.
    return bitcast[DType.float16, 1](w.load[width=1](i)).cast[
        DType.float32]()[0]


def i8x16(w: UnsafePointer[UInt16, MutAnyOrigin],
          byte_off: Int) -> SIMD[DType.float32, 16]:
    """16 consecutive int8 weights as f32 (unscaled)."""
    var u = w.bitcast[UInt8]().load[width=16](byte_off)
    return bitcast[DType.int8, 16](u).cast[DType.float32]()


def _box[T: Movable](var f: T) -> KPtr:
    var p = alloc[T](1)
    p.init_pointee_move(f^)
    return p.bitcast[NoneType]()


# ============================ kernels =========================================


def k_mm_q8(w: UnsafePointer[UInt16, MutAnyOrigin],
            a: UnsafePointer[Float32, MutAnyOrigin],
            ox: Int, owt: Int, oy: Int, s: Int, m: Int, n: Int, e0: Int):
    """y (s,n) = x (s,m) @ Wq8 (n,m)^T; same one-output-per-group structure
    as k_mm_w, reading packed q8 rows. Each thread walks whole quant groups
    (64 weights): the partial dot accumulates unscaled and the f16 scale is
    applied once per group, keeping the hot loop at ~convert+fma per weight.
    Requires m % (TG_MM*QG) == 0 falling back per-group otherwise; all
    models here have m in {1024, 2048, 3072, ...} = k*64."""
    var idx = e0 + Int(block_idx.x)
    if idx >= s * n:
        return
    var i = idx // n
    var j = idx % n
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG_MM, Float32, address_space = AddressSpace.SHARED]()
    var xb = ox + i * m
    var row = owt + j * q8_rowstride(m)         # u16 base of packed row
    var wb = 2 * (row + m // QG)                # byte base of int8 section
    var acc = Float32(0)
    var g = t
    var ngroups = m // QG
    while g < ngroups:
        var k = g * QG
        var gacc = SIMD[DType.float32, 16](0)
        gacc = a.load[width=16](xb + k).fma(i8x16(w, wb + k), gacc)
        gacc = a.load[width=16](xb + k + 16).fma(i8x16(w, wb + k + 16), gacc)
        gacc = a.load[width=16](xb + k + 32).fma(i8x16(w, wb + k + 32), gacc)
        gacc = a.load[width=16](xb + k + 48).fma(i8x16(w, wb + k + 48), gacc)
        acc += f16(w, row + g) * gacc.reduce_add()
        g += TG_MM
    shared[t] = acc
    barrier()
    if t == 0:
        var tot = SIMD[DType.float32, 4](0)
        var q = 0
        while q < TG_MM:
            tot += shared.load[width=4](q)
            q += 4
        a[oy + idx] = tot.reduce_add()


def k_embed_gather_q8(w: UnsafePointer[UInt16, MutAnyOrigin],
                      a: UnsafePointer[Float32, MutAnyOrigin],
                      oemb: Int, oids: UnsafePointer[Int32, MutAnyOrigin],
                      oy: Int, s: Int, d: Int):
    """Batched embedding gather from packed q8 rows."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= s * d:
        return
    var i = idx // d
    var c = idx % d
    var row = oemb + Int(oids[i]) * q8_rowstride(d)
    var sc = f16(w, row + c // QG)
    var b = w.bitcast[UInt8]()[2 * (row + d // QG) + c]
    a[oy + idx] = Float32(Int8(b)) * sc


# ============================ host-side =======================================


struct Q8Kernels(Copyable, Movable):
    var mm: KPtr
    var embg: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.mm = _box(ctx.compile_function[k_mm_q8]())
        self.embg = _box(ctx.compile_function[k_embed_gather_q8]())


def mm_q8_op(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
             qk: Q8Kernels, ox: Int, s: Int, m: Int, owt: Int,
             n: Int) raises -> Int:
    """Chunked q8 matmul; mirrors llama_common.mm_op."""
    var oy = a.alloc(s * n)
    var total = s * n
    var chunk = max(BLOCK, MM_CHUNK_MACS // m)
    var e0 = 0
    while e0 < total:
        var cnt = min(chunk, total - e0)
        ctx.enqueue_function(
            qk.mm.bitcast[type_of(ctx.compile_function[k_mm_q8]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox, owt, oy, s, m, n, e0, grid_dim=cnt, block_dim=TG_MM)
        e0 += cnt
        if e0 < total:
            ctx.synchronize()
    return oy


def run_layer_q8(ctx: DeviceContext, w: DeviceBuffer[DType.uint16],
                 mut a: Acts, qk: Q8Kernels, cfg: Config, lo: LayerOffs,
                 ox: Int, s: Int, pos0: Int, kc: Int, vc: Int,
                 oinv: Int) raises -> Int:
    """llama_common.run_layer with q8 weight matmuls (norms stay bf16).

    The s==1 fused q|k|v and gate|up dispatches carry over: contiguity in
    the packed arena means k = q + QD*rowstride(H) etc., and k_mm_q8 indexes
    rows uniformly, so a single (QD+2*KVD)-row matmul still works.
    """
    var H = cfg.hidden
    var QD = cfg.q_dim()
    var KVD = cfg.kv_dim()
    var rsH = q8_rowstride(H)
    var nctx = pos0 + s
    var xn = rmsnorm_op(ctx, w, a, ox, s, lo.in_norm, H, cfg.eps)
    var oq: Int
    var okk: Int
    var ov: Int
    if s == 1 and lo.k == lo.q + QD * rsH and lo.v == lo.k + KVD * rsH:
        oq = mm_q8_op(ctx, w, a, qk, xn, 1, H, lo.q, QD + 2 * KVD)
        okk = oq + QD
        ov = okk + KVD
    else:
        oq = mm_q8_op(ctx, w, a, qk, xn, s, H, lo.q, QD)
        okk = mm_q8_op(ctx, w, a, qk, xn, s, H, lo.k, KVD)
        ov = mm_q8_op(ctx, w, a, qk, xn, s, H, lo.v, KVD)
    if lo.q_norm >= 0:
        oq = rmsnorm_op(ctx, w, a, oq, s * cfg.n_heads, lo.q_norm,
                        cfg.head_dim, cfg.eps)
        okk = rmsnorm_op(ctx, w, a, okk, s * cfg.n_kv, lo.k_norm,
                         cfg.head_dim, cfg.eps)
    ctx.enqueue_function(
        a.kn.rope.bitcast[type_of(ctx.compile_function[k_rope_qk]())]()[],
        a.buf.unsafe_ptr(), oq, okk, oinv, s, pos0,
        cfg.n_heads, cfg.n_kv, QD, KVD, cfg.half(),
        grid_dim=ceildiv(s * (cfg.n_heads + cfg.n_kv) * cfg.half(), BLOCK),
        block_dim=BLOCK)
    ctx.enqueue_function(
        a.kn.copy2.bitcast[type_of(ctx.compile_function[k_copy2]())]()[],
        a.buf.unsafe_ptr(), okk, kc + pos0 * KVD, ov, vc + pos0 * KVD, s * KVD,
        grid_dim=ceildiv(2 * s * KVD, BLOCK), block_dim=BLOCK)
    var osc = a.alloc(cfg.n_heads * s * nctx)
    ctx.enqueue_function(
        a.kn.scores.bitcast[type_of(ctx.compile_function[k_scores]())]()[],
        a.buf.unsafe_ptr(), oq, kc, osc, s, pos0, nctx,
        cfg.n_heads, cfg.group(), QD, KVD, cfg.head_dim,
        grid_dim=ceildiv(cfg.n_heads * s * nctx, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function(
        a.kn.softmax.bitcast[type_of(ctx.compile_function[k_softmax_rows]())]()[],
        a.buf.unsafe_ptr(), osc, cfg.n_heads * s, nctx,
        grid_dim=ceildiv(cfg.n_heads * s, BLOCK), block_dim=BLOCK)
    var oao = a.alloc(s * QD)
    ctx.enqueue_function(
        a.kn.attout.bitcast[type_of(ctx.compile_function[k_att_out]())]()[],
        a.buf.unsafe_ptr(), osc, vc, oao, s, nctx,
        QD, KVD, cfg.head_dim, cfg.group(),
        grid_dim=ceildiv(s * QD, BLOCK), block_dim=BLOCK)
    var oo = mm_q8_op(ctx, w, a, qk, oao, s, QD, lo.o, H)
    var oh = a.alloc(s * H)
    ctx.enqueue_function(
        a.kn.resadd.bitcast[type_of(ctx.compile_function[k_res_add]())]()[],
        a.buf.unsafe_ptr(), ox, oo, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    var oz = rmsnorm_op(ctx, w, a, oh, s, lo.post_norm, H, cfg.eps)
    var og: Int
    var ou: Int
    if s == 1 and lo.up == lo.gate + cfg.inter * rsH:
        og = mm_q8_op(ctx, w, a, qk, oz, 1, H, lo.gate, 2 * cfg.inter)
        ou = og + cfg.inter
    else:
        og = mm_q8_op(ctx, w, a, qk, oz, s, H, lo.gate, cfg.inter)
        ou = mm_q8_op(ctx, w, a, qk, oz, s, H, lo.up, cfg.inter)
    ctx.enqueue_function(
        a.kn.swiglu.bitcast[type_of(ctx.compile_function[k_swiglu_mul]())]()[],
        a.buf.unsafe_ptr(), og, ou, s * cfg.inter,
        grid_dim=ceildiv(s * cfg.inter, BLOCK), block_dim=BLOCK)
    var om = mm_q8_op(ctx, w, a, qk, og, s, cfg.inter, lo.down, H)
    ctx.enqueue_function(
        a.kn.add.bitcast[type_of(ctx.compile_function[k_add]())]()[],
        a.buf.unsafe_ptr(), om, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    return oh
