"""Waypoint DiT GPU machinery (Metal): kernels + layer runner.

Follows llama_common conventions: bf16 weights in a u16 buffer, f32
activation arena, at most 2 buffers per kernel, offsets as Int params.

Waypoint-specific pieces (see PLAN.md):
- adaLN rms (no weight, bf16 eps) with baked per-sigma scale/bias tables
- ortho 3-axis RoPE fused with the unweighted q/k rms_norm
- value residual (per-layer lerp toward layer-0 v)
- compact ring KV cache: 16 frame slots + 1 current-frame tail per layer;
  attention iterates a virtual compact index and maps to physical slots
  (nw written slots, one optionally excluded = the slot being overwritten)
- MLPFusion controller conditioning every 3rd layer
- patchify/unpatchify as matmuls with gather/scatter kernels
"""

from std.math import ceildiv, sqrt, exp, cos, sin
from std.memory import alloc, stack_allocation
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from llama_common import (
    Acts, Kernels, KPtr, _box, bf, bf4, bf8, mm_op, BLOCK, TG, TG_MM, PAD,
)

# model constants (Waypoint-1.5-1B)
comptime D = 2048                       # d_model
comptime NH = 32                        # query heads
comptime NKV = 16                       # kv heads
comptime HD = 64                        # head dim
comptime FF = 8192                      # mlp hidden
comptime LAYERS = 24
comptime TPF = 512                      # tokens per frame (grid 16 x 32)
comptime GW = 32                        # token grid width
comptime GH = 16                        # token grid height
comptime C = 32                         # latent channels
comptime LH = 32                        # latent height (GH * patch)
comptime LW = 64                        # latent width  (GW * patch)
comptime LAT = C * LH * LW              # latent elements (65536)
comptime KVD = NKV * HD                 # kv row dim (1024)
comptime QD = NH * HD                   # q row dim (2048)
comptime SLOTS = 17                     # 16 ring frames + current tail
comptime RING = 16
comptime EPS_BF16 = Float32(0.0078125)  # torch F.rms_norm default eps on bf16
comptime NROT = HD // 2                 # rotation pairs per head (32)


# ============================ kernels =========================================

def k_stage_in(g: UnsafePointer[Float32, MutAnyOrigin],
               a: UnsafePointer[Float32, MutAnyOrigin],
               osrc: Int, odst: Int, n: Int):
    """Small staging buffer -> acts arena (host upload without arena map)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[odst + i] = g[osrc + i]


def k_patch_gather(a: UnsafePointer[Float32, MutAnyOrigin],
                   olat: Int, oy: Int):
    """latent [C,LH,LW] -> token vectors [TPF, 256] (patch vec padded 128->256).

    vec[c*4 + ky*2 + kx] = lat[c, ty*2+ky, tx*2+kx], token = ty*GW + tx.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= TPF * 256:
        return
    var tok = idx // 256
    var e = idx % 256
    if e >= C * 4:
        a[oy + idx] = 0
        return
    var c = e // 4
    var ky = (e % 4) // 2
    var kx = e % 2
    var ty = tok // GW
    var tx = tok % GW
    a[oy + idx] = a[olat + c * (LH * LW) + (ty * 2 + ky) * LW + (tx * 2 + kx)]


def k_unpatch_scatter(w: UnsafePointer[UInt16, MutAnyOrigin],
                      a: UnsafePointer[Float32, MutAnyOrigin],
                      omm: Int, ob: Int, olat: Int):
    """[TPF, 128] matmul output (+ channel bias) -> latent [C,LH,LW]."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= TPF * 128:
        return
    var tok = idx // 128
    var e = idx % 128
    var c = e // 4
    var ky = (e % 4) // 2
    var kx = e % 2
    var ty = tok // GW
    var tx = tok % GW
    a[olat + c * (LH * LW) + (ty * 2 + ky) * LW + (tx * 2 + kx)] = (
        a[omm + idx] + bf(w, ob + c))


def k_ada_rms(w: UnsafePointer[UInt16, MutAnyOrigin],
              a: UnsafePointer[Float32, MutAnyOrigin],
              ox: Int, ots: Int, otb: Int, oy: Int, rows: Int, d: Int):
    """y = rms_norm(x) * (1 + s) + b, s/b bf16 rows in the weight buffer."""
    var i = Int(block_idx.x)
    if i >= rows:
        return
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG, Float32, address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, 4](0)
    var k = t * 4
    while k < d:
        var v = a.load[width=4](ox + i * d + k)
        acc = v.fma(v, acc)
        k += TG * 4
    shared[t] = acc[0] + acc[1] + acc[2] + acc[3]
    barrier()
    var stride = TG // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        barrier()
        stride //= 2
    var inv = Float32(1) / sqrt(shared[0] / Float32(d) + EPS_BF16)
    k = t * 4
    while k < d:
        var x4 = a.load[width=4](ox + i * d + k)
        var s4 = bf4(w, ots + k) + 1.0
        a.store(oy + i * d + k, x4 * inv * s4 + bf4(w, otb + k))
        k += TG * 4


def k_rms_plain(a: UnsafePointer[Float32, MutAnyOrigin],
                ox: Int, oy: Int, rows: Int, d: Int):
    """Unweighted rms_norm (torch F.rms_norm, bf16 eps)."""
    var i = Int(block_idx.x)
    if i >= rows:
        return
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG, Float32, address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, 4](0)
    var k = t * 4
    while k < d:
        var v = a.load[width=4](ox + i * d + k)
        acc = v.fma(v, acc)
        k += TG * 4
    shared[t] = acc[0] + acc[1] + acc[2] + acc[3]
    barrier()
    var stride = TG // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        barrier()
        stride //= 2
    var inv = Float32(1) / sqrt(shared[0] / Float32(d) + EPS_BF16)
    k = t * 4
    while k < d:
        a.store(oy + i * d + k, a.load[width=4](ox + i * d + k) * inv)
        k += TG * 4


def k_ada_gate_res(w: UnsafePointer[UInt16, MutAnyOrigin],
                   a: UnsafePointer[Float32, MutAnyOrigin],
                   ox: Int, otg: Int, ores: Int, oy: Int, rows: Int, d: Int):
    """y = res + x * g (g = bf16 gate row per column)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= rows * d:
        return
    var col = idx % d
    a[oy + idx] = a[ores + idx] + a[ox + idx] * bf(w, otg + col)


def k_qk_normrope(a: UnsafePointer[Float32, MutAnyOrigin],
                  oq: Int, ok: Int, okc: Int, orc: Int, tpos: Int):
    """Per head-row: unweighted rms_norm then ortho RoPE.

    q rows ([TPF, QD], NH heads) transform in place; k rows ([TPF, KVD],
    NKV heads) write to cache offset okc (tail slot). orc = rope constants
    in acts: xy[8] then inv_t[16]. Rotation input pairs are (x[2i], x[2i+1]);
    output layout is non-interleaved: y0 -> [i], y1 -> [NROT + i].
    One thread per head-row (TPF * (NH + NKV) rows of HD).
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= TPF * (NH + NKV):
        return
    var tok = row // (NH + NKV)
    var h = row % (NH + NKV)
    var src: Int
    var dst: Int
    if h < NH:
        src = oq + tok * QD + h * HD
        dst = src
    else:
        src = ok + tok * KVD + (h - NH) * HD
        dst = okc + tok * KVD + (h - NH) * HD
    # rms over HD
    var ss = Float32(0)
    var d = 0
    while d < HD:
        var v4 = a.load[width=4](src + d)
        ss += (v4 * v4).reduce_add()
        d += 4
    var inv = Float32(1) / sqrt(ss / Float32(HD) + EPS_BF16)
    # rope freqs for this token
    var xpos = Float32(tok % GW)
    var ypos = Float32(tok // GW)
    var xn = (2.0 * xpos + 1.0) / Float32(GW) - 1.0
    var yn = (2.0 * ypos + 1.0) / Float32(GH) - 1.0
    var out = InlineArray[Float32, HD](fill=0)
    for i in range(NROT):
        var f: Float32
        if i < 8:
            f = xn * a[orc + i]
        elif i < 16:
            f = yn * a[orc + (i - 8)]
        else:
            f = Float32(tpos) * a[orc + 8 + (i - 16)]
        var cf = cos(f)
        var sf = sin(f)
        var x0 = a[src + 2 * i] * inv
        var x1 = a[src + 2 * i + 1] * inv
        out[i] = x0 * cf - x1 * sf
        out[NROT + i] = x1 * cf + x0 * sf
    for i in range(HD):
        a[dst + i] = out[i]


def k_v_store(a: UnsafePointer[Float32, MutAnyOrigin],
              ov: Int, ov1: Int, ovc: Int, lamb: Float32, first: Int):
    """Value residual + cache store for one frame [TPF, KVD].

    first==1 (layer 0): v1 <- v, cache <- v.
    else: cache <- v + lamb * (v1 - v).
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= TPF * KVD:
        return
    var v = a[ov + i]
    if first == 1:
        a[ov1 + i] = v
        a[ovc + i] = v
    else:
        a[ovc + i] = v + lamb * (a[ov1 + i] - v)


def _slot_phys(vi: Int, nring: Int, excl: Int) -> Int:
    """Virtual compact slot index -> physical slot (ring skip + tail)."""
    if vi >= nring:
        return RING                     # tail
    if excl >= 0 and vi >= excl:
        return vi + 1
    return vi


def k_scores_slots(a: UnsafePointer[Float32, MutAnyOrigin],
                   oq: Int, okc: Int, osc: Int, nctx: Int,
                   nring: Int, excl: Int):
    """GQA scores [NH, TPF, nctx] over compact valid slots (no causal mask).

    nctx = (nring + 1) * TPF; kv j -> physical cache position via slot map.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= NH * TPF * nctx:
        return
    var h = idx // (TPF * nctx)
    var r = idx % (TPF * nctx)
    var i = r // nctx
    var j = r % nctx
    var slot = _slot_phys(j // TPF, nring, excl)
    var jj = slot * TPF + (j % TPF)
    var qb = oq + i * QD + h * HD
    var kb = okc + jj * KVD + (h // 2) * HD
    var acc4 = SIMD[DType.float32, 4](0)
    var d = 0
    while d < HD:
        acc4 += a.load[width=4](qb + d) * a.load[width=4](kb + d)
        d += 4
    a[osc + idx] = acc4.reduce_add() / sqrt(Float32(HD))


def k_att_out_slots(a: UnsafePointer[Float32, MutAnyOrigin],
                    osc: Int, ovc: Int, oy: Int, nctx: Int,
                    nring: Int, excl: Int):
    """y [TPF, QD] = probs [NH, TPF, nctx] @ v via the same slot map."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= TPF * QD:
        return
    var i = idx // QD
    var cd = idx % QD
    var h = cd // HD
    var d = cd % HD
    var pb = osc + (h * TPF + i) * nctx
    var acc = Float32(0)
    for vi in range(nctx // TPF):
        var vb = ovc + _slot_phys(vi, nring, excl) * TPF * KVD + (h // 2) * HD + d
        var pbase = pb + vi * TPF
        for e in range(TPF):
            acc += a[pbase + e] * a[vb + e * KVD]
    a[oy + idx] = acc


def k_silu(a: UnsafePointer[Float32, MutAnyOrigin], ox: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var v = a[ox + i]
        a[ox + i] = v / (Float32(1) + exp(-v))


def k_row_add_silu(a: UnsafePointer[Float32, MutAnyOrigin],
                   ox: Int, orow: Int, rows: Int, d: Int):
    """x[r, c] = silu(x[r, c] + row[c]) (MLPFusion cond broadcast)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= rows * d:
        return
    var v = a[ox + idx] + a[orow + idx % d]
    a[ox + idx] = v / (Float32(1) + exp(-v))


def k_axpy(a: UnsafePointer[Float32, MutAnyOrigin],
           oy: Int, ox: Int, s: Float32, n: Int):
    """y += s * x (Euler step: latent += dsigma * velocity)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[oy + i] += s * a[ox + i]


def k_copy(a: UnsafePointer[Float32, MutAnyOrigin], os: Int, od: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[od + i] = a[os + i]


# ============================ host side =======================================

struct WPKernels(Copyable, Movable):
    var stage: KPtr
    var pgather: KPtr
    var upscatter: KPtr
    var adarms: KPtr
    var rmsp: KPtr
    var gateres: KPtr
    var normrope: KPtr
    var vstore: KPtr
    var scores: KPtr
    var attout: KPtr
    var silu: KPtr
    var rowsilu: KPtr
    var axpy: KPtr
    var cpy: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.stage = _box(ctx.compile_function[k_stage_in]())
        self.pgather = _box(ctx.compile_function[k_patch_gather]())
        self.upscatter = _box(ctx.compile_function[k_unpatch_scatter]())
        self.adarms = _box(ctx.compile_function[k_ada_rms]())
        self.rmsp = _box(ctx.compile_function[k_rms_plain]())
        self.gateres = _box(ctx.compile_function[k_ada_gate_res]())
        self.normrope = _box(ctx.compile_function[k_qk_normrope]())
        self.vstore = _box(ctx.compile_function[k_v_store]())
        self.scores = _box(ctx.compile_function[k_scores_slots]())
        self.attout = _box(ctx.compile_function[k_att_out_slots]())
        self.silu = _box(ctx.compile_function[k_silu]())
        self.rowsilu = _box(ctx.compile_function[k_row_add_silu]())
        self.axpy = _box(ctx.compile_function[k_axpy]())
        self.cpy = _box(ctx.compile_function[k_copy]())


@fieldwise_init
struct WPLayer(Copyable, Movable):
    """Weight-buffer offsets (u16 units) for one block."""
    var q: Int                          # qkv stacked: k = q + QD*D, v = k + KVD*D
    var o: Int
    var fc1: Int
    var fc2: Int
    var fu_x: Int                       # -1 when not a ctrl layer
    var fu_c: Int
    var fu_o: Int
    var vlamb: Float32
    var is_global: Bool                 # global attention layer (idx % 4 == 3)
    var kc: Int                         # acts offsets: [SLOTS*TPF, KVD] k cache
    var vc: Int


struct WPCacheState(Copyable, Movable):
    """Host-side ring bookkeeping for one layer."""
    var nw: Int                         # ring slots written (0..RING)
    var next_slot: Int

    def __init__(out self):
        self.nw = 0
        self.next_slot = 0


@fieldwise_init
struct FrameMask(Copyable, Movable):
    """Attention extent for the current frame (per reference upsert mask)."""
    var nring: Int                      # compact valid ring slots
    var excl: Int                       # physical slot being overwritten (-1: none)
    var nctx: Int                       # (nring + 1) * TPF


def ada_rms_op(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
               kn: WPKernels, ox: Int, ots: Int, otb: Int,
               rows: Int, d: Int) raises -> Int:
    var oy = a.alloc(rows * d)
    ctx.enqueue_function(
        kn.adarms.bitcast[type_of(ctx.compile_function[k_ada_rms]())]()[],
        w.unsafe_ptr(), a.buf.unsafe_ptr(), ox, ots, otb, oy, rows, d,
        grid_dim=rows, block_dim=TG)
    return oy
