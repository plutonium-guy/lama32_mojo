"""Shared transformer GPU machinery (Metal, M4): kernels + layer runner.

Used by llama32_mojo, llama31_mojo, and qwen3_mojo. Model dimensions are
runtime values in Config; matmul cost dominates so extra dims are free.

Metal constraints baked in (learned on the OCR port):
- at most 2 buffers bound per kernel (arg-stomp bug); other params are Ints.
- both arenas reserve PAD head elements.
- matmuls are chunked to MM_CHUNK_MACS per dispatch (WindowServer watchdog).
"""

from std.math import ceildiv, sqrt, exp, cos, sin
from std.memory import memcpy, alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation, bitcast

comptime BLOCK = 256
comptime TG = 64                        # threads per reduction group
comptime TG_MM = 32                     # matmul group = one Apple SIMD-group
comptime PAD = 32                       # reserved head elements per arena
comptime MM_CHUNK_MACS = 1 << 30


@fieldwise_init
struct Config(Copyable, Movable):
    var hidden: Int
    var layers: Int
    var n_heads: Int
    var n_kv: Int
    var head_dim: Int
    var inter: Int
    var vocab: Int
    var eps: Float32
    var theta: Float64
    var rope_factor: Float64
    var low_ff: Float64
    var high_ff: Float64
    var orig_ctx: Float64

    def group(self) -> Int:
        return self.n_heads // self.n_kv

    def kv_dim(self) -> Int:
        return self.n_kv * self.head_dim

    def q_dim(self) -> Int:
        return self.n_heads * self.head_dim

    def half(self) -> Int:
        return self.head_dim // 2


@fieldwise_init
struct LayerOffs(Copyable, Movable):
    """u16-element offsets of one layer's weights in the weight buffer."""
    var q: Int
    var k: Int
    var v: Int
    var o: Int
    var gate: Int
    var up: Int
    var down: Int
    var in_norm: Int
    var post_norm: Int
    var q_norm: Int                     # -1 = none (Llama); else head_dim RMS
    var k_norm: Int


# ============================ kernels =========================================
# w = bf16 weight buffer (u16), a = f32 activation arena. 2 pointers max.

def bf(w: UnsafePointer[UInt16, MutAnyOrigin], i: Int) -> Float32:
    var bits = UInt32(w[i]) << 16
    return UnsafePointer(to=bits).bitcast[Float32]()[]


def bf4(w: UnsafePointer[UInt16, MutAnyOrigin], i: Int) -> SIMD[DType.float32, 4]:
    var u = w.load[width=4](i).cast[DType.uint32]() << 16
    return bitcast[DType.float32, 4](u)


def bf8(w: UnsafePointer[UInt16, MutAnyOrigin], i: Int) -> SIMD[DType.float32, 8]:
    var u = w.load[width=8](i).cast[DType.uint32]() << 16
    return bitcast[DType.float32, 8](u)


def k_mm_w(w: UnsafePointer[UInt16, MutAnyOrigin],
           a: UnsafePointer[Float32, MutAnyOrigin],
           ox: Int, owt: Int, oy: Int, s: Int, m: Int, n: Int, e0: Int):
    """y (s,n) = x (s,m) @ Wbf16 (n,m)^T; one TG_MM group per output element.

    Threads stride k with 8-wide (16 B) loads; one barrier, then lane 0 sums
    the TG_MM partials. Requires m % (TG_MM*8) == 0 (m % 256, as before).
    """
    var idx = e0 + Int(block_idx.x)
    if idx >= s * n:
        return
    var i = idx // n
    var j = idx % n
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG_MM, Float32, address_space = AddressSpace.SHARED]()
    var xb = ox + i * m
    var wb = owt + j * m
    var acc = SIMD[DType.float32, 8](0)
    var k = t * 8
    while k < m:
        acc = a.load[width=8](xb + k).fma(bf8(w, wb + k), acc)
        k += TG_MM * 8
    shared[t] = acc.reduce_add()
    barrier()
    if t == 0:
        var tot = SIMD[DType.float32, 4](0)
        var q = 0
        while q < TG_MM:
            tot += shared.load[width=4](q)
            q += 4
        a[oy + idx] = tot.reduce_add()


def k_mm_tile(w: UnsafePointer[UInt16, MutAnyOrigin],
              a: UnsafePointer[Float32, MutAnyOrigin],
              ox: Int, owt: Int, oy: Int, s: Int, m: Int, n: Int):
    """Tiled y (s,n) = x (s,m) @ Wbf16 (n,m)^T: 8 rows x 8 cols per 32-thread
    group. Reuses each x/w load 8x — main decode headroom vs k_mm_w.
    Requires m % 256 == 0."""
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


def k_embed_row(w: UnsafePointer[UInt16, MutAnyOrigin],
                a: UnsafePointer[Float32, MutAnyOrigin],
                oemb: Int, tok_id: Int, oy: Int, d: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < d:
        a[oy + i] = bf(w, oemb + tok_id * d + i)


def k_bf16_to_f32(w: UnsafePointer[UInt16, MutAnyOrigin],
                  a: UnsafePointer[Float32, MutAnyOrigin],
                  ow: Int, oa: Int, n: Int):
    """Decode n contiguous bf16 weights into the f32 activation arena."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[oa + i] = bf(w, ow + i)


def k_embed_gather(w: UnsafePointer[UInt16, MutAnyOrigin],
                   a: UnsafePointer[Float32, MutAnyOrigin],
                   oemb: Int, oids: UnsafePointer[Int32, MutAnyOrigin],
                   oy: Int, s: Int, d: Int):
    """Batched embedding: s token ids -> s rows in one dispatch."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= s * d:
        return
    var row = idx // d
    var col = idx % d
    var tok = Int(oids[row])
    a[oy + idx] = bf(w, oemb + tok * d + col)


def k_rmsnorm_w(w: UnsafePointer[UInt16, MutAnyOrigin],
                a: UnsafePointer[Float32, MutAnyOrigin],
                ox: Int, owt: Int, oy: Int, rows: Int, d: Int, eps: Float32):
    """One TG group per row: parallel sum-of-squares, then scaled write."""
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
    var inv = Float32(1) / sqrt(shared[0] / Float32(d) + eps)
    k = t * 4
    while k < d:
        var x4 = a.load[width=4](ox + i * d + k)
        a.store(oy + i * d + k, bf4(w, owt + k) * x4 * inv)
        k += TG * 4


def k_rmsnorm2_w(w: UnsafePointer[UInt16, MutAnyOrigin],
                 a: UnsafePointer[Float32, MutAnyOrigin],
                 oxq: Int, oxk: Int, owtq: Int, owtk: Int,
                 oyq: Int, oyk: Int, rowsq: Int, rowsk: Int, d: Int, eps: Float32):
    """Fused per-head Q-norm + K-norm: one TG group per row, grid over both."""
    var i = Int(block_idx.x)
    var rows = rowsq + rowsk
    if i >= rows:
        return
    var ox: Int
    var owt: Int
    var oy: Int
    if i < rowsq:
        ox = oxq + i * d
        owt = owtq
        oy = oyq + i * d
    else:
        var ri = i - rowsq
        ox = oxk + ri * d
        owt = owtk
        oy = oyk + ri * d
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG, Float32, address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, 4](0)
    var k = t * 4
    while k < d:
        var v = a.load[width=4](ox + k)
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
    var inv = Float32(1) / sqrt(shared[0] / Float32(d) + eps)
    k = t * 4
    while k < d:
        var x4 = a.load[width=4](ox + k)
        a.store(oy + k, bf4(w, owt + k) * x4 * inv)
        k += TG * 4


def k_rope_qk(a: UnsafePointer[Float32, MutAnyOrigin],
              oq: Int, ok: Int, oinv: Int, s: Int, pos0: Int,
              nheads: Int, nkv: Int, qdim: Int, kvdim: Int, half: Int):
    """HF rotate_half RoPE in place on q and k."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nall = nheads + nkv
    if idx >= s * nall * half:
        return
    var i = idx // (nall * half)
    var r = idx % (nall * half)
    var h = r // half
    var d = r % half
    var freq = Float32(pos0 + i) * a[oinv + d]
    var c = cos(freq)
    var sn = sin(freq)
    var base: Int
    if h < nheads:
        base = oq + i * qdim + h * 2 * half
    else:
        base = ok + i * kvdim + (h - nheads) * 2 * half
    var x0 = a[base + d]
    var x1 = a[base + half + d]
    a[base + d] = x0 * c - x1 * sn
    a[base + half + d] = x1 * c + x0 * sn


def k_rope_qk_copy2(a: UnsafePointer[Float32, MutAnyOrigin],
                    oq: Int, ok: Int, ov: Int, okc: Int, ovc: Int,
                    oinv: Int, s: Int, pos0: Int,
                    nheads: Int, nkv: Int, qdim: Int, kvdim: Int, half: Int):
    """Fused RoPE on q/k + KV cache append (one dispatch).

    Race-free by construction: each rope thread writes ITS rotated k pair
    to both the k activation and the cache (no cross-thread read of a
    value another thread rotates); v is not roped so its copy threads
    are independent. Grid covers s*(nheads+nkv)*half + s*kvdim."""
    var nall = nheads + nkv
    var nrope = s * nall * half
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nrope:
        var i = idx // (nall * half)
        var r = idx % (nall * half)
        var h = r // half
        var d = r % half
        var freq = Float32(pos0 + i) * a[oinv + d]
        var c = cos(freq)
        var sn = sin(freq)
        var y0: Float32
        var y1: Float32
        if h < nheads:
            var base = oq + i * qdim + h * 2 * half
            var x0 = a[base + d]
            var x1 = a[base + half + d]
            a[base + d] = x0 * c - x1 * sn
            a[base + half + d] = x1 * c + x0 * sn
        else:
            var hk = h - nheads
            var base = ok + i * kvdim + hk * 2 * half
            var x0 = a[base + d]
            var x1 = a[base + half + d]
            y0 = x0 * c - x1 * sn
            y1 = x1 * c + x0 * sn
            a[base + d] = y0
            a[base + half + d] = y1
            var cbase = okc + (pos0 + i) * kvdim + hk * 2 * half
            a[cbase + d] = y0
            a[cbase + half + d] = y1
    elif idx < nrope + s * kvdim:
        var vi = idx - nrope
        a[ovc + pos0 * kvdim + vi] = a[ov + vi]


def k_copy2(a: UnsafePointer[Float32, MutAnyOrigin],
            os1: Int, od1: Int, os2: Int, od2: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[od1 + i] = a[os1 + i]
    elif i < 2 * n:
        a[od2 + i - n] = a[os2 + i - n]


def k_res_add(a: UnsafePointer[Float32, MutAnyOrigin],
              ox: Int, oo: Int, oy: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[oy + i] = a[ox + i] + a[oo + i]


def k_res_add_rmsnorm(w: UnsafePointer[UInt16, MutAnyOrigin],
                      a: UnsafePointer[Float32, MutAnyOrigin],
                      ox: Int, oo: Int, owt: Int, oy: Int,
                      rows: Int, d: Int, eps: Float32):
    """Fused residual add + RMSNorm (one TG group per row)."""
    var i = Int(block_idx.x)
    if i >= rows:
        return
    var t = Int(thread_idx.x)
    var shared = stack_allocation[TG, Float32, address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, 4](0)
    var xb = ox + i * d
    var ob = oo + i * d
    var k = t * 4
    while k < d:
        var v = a.load[width=4](xb + k) + a.load[width=4](ob + k)
        a.store(oy + i * d + k, v)
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
    var inv = Float32(1) / sqrt(shared[0] / Float32(d) + eps)
    k = t * 4
    while k < d:
        var x4 = a.load[width=4](oy + i * d + k)
        a.store(oy + i * d + k, bf4(w, owt + k) * x4 * inv)
        k += TG * 4


def k_add(a: UnsafePointer[Float32, MutAnyOrigin], ox: Int, oy: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[oy + i] += a[ox + i]


def k_scores(a: UnsafePointer[Float32, MutAnyOrigin],
             oq: Int, okc: Int, osc: Int, s: Int, pos0: Int, nctx: Int,
             nheads: Int, group: Int, qdim: Int, kvdim: Int, headdim: Int):
    """GQA scores (heads, s, nctx): dot(q_i, k_j)/sqrt(hd), causal -inf."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= nheads * s * nctx:
        return
    var h = idx // (s * nctx)
    var r = idx % (s * nctx)
    var i = r // nctx
    var j = r % nctx
    var o = osc + idx
    if j > pos0 + i:
        a[o] = Float32(-3.0e38)
        return
    var qb = oq + i * qdim + h * headdim
    var kb = okc + j * kvdim + (h // group) * headdim
    var acc4 = SIMD[DType.float32, 4](0)
    var d = 0
    while d + 4 <= headdim:
        acc4 += a.load[width=4](qb + d) * a.load[width=4](kb + d)
        d += 4
    while d < headdim:
        acc4[0] += a[qb + d] * a[kb + d]
        d += 1
    a[o] = acc4.reduce_add() / sqrt(Float32(headdim))


def k_softmax_rows(a: UnsafePointer[Float32, MutAnyOrigin],
                   op: Int, rows: Int, n: Int):
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= rows:
        return
    var off = op + r * n
    var mx = a[off]
    for i in range(1, n):
        if a[off + i] > mx:
            mx = a[off + i]
    var sm = Float32(0)
    for i in range(n):
        a[off + i] = exp(a[off + i] - mx)
        sm += a[off + i]
    for i in range(n):
        a[off + i] /= sm


def k_att_out(a: UnsafePointer[Float32, MutAnyOrigin],
              osc: Int, ovc: Int, oy: Int, s: Int, nctx: Int,
              hidden: Int, kvdim: Int, headdim: Int, group: Int):
    """y (s, hidden) from probs (heads, s, nctx) @ v cache (nctx, kvdim)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= s * hidden:
        return
    var i = idx // hidden
    var c = idx % hidden
    var h = c // headdim
    var d = c % headdim
    var pb = osc + (h * s + i) * nctx
    var vb = ovc + (h // group) * headdim + d
    var acc4 = SIMD[DType.float32, 4](0)
    var j = 0
    while j + 4 <= nctx:
        var p4 = a.load[width=4](pb + j)
        var v4 = SIMD[DType.float32, 4](
            a[vb + j * kvdim], a[vb + (j + 1) * kvdim],
            a[vb + (j + 2) * kvdim], a[vb + (j + 3) * kvdim])
        acc4 += p4 * v4
        j += 4
    while j < nctx:
        acc4[0] += a[pb + j] * a[vb + j * kvdim]
        j += 1
    a[oy + idx] = acc4.reduce_add()


def k_swiglu_mul(a: UnsafePointer[Float32, MutAnyOrigin], og: Int, ou: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var v = a[og + i]
    a[og + i] = (v / (Float32(1) + exp(-v))) * a[ou + i]


def k_export(a: UnsafePointer[Float32, MutAnyOrigin],
             g: UnsafePointer[Float32, MutAnyOrigin], osrc: Int, odst: Int, n: Int):
    """Copy an acts slice into a small dedicated readback buffer — mapping the
    whole acts arena to host costs ~8 ms/token; mapping the small one doesn't."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        g[odst + i] = a[osrc + i]


def k_argmax(a: UnsafePointer[Float32, MutAnyOrigin],
             oid: UnsafePointer[Int32, MutAnyOrigin], op: Int, n: Int):
    """Parallel argmax over n logits; writes winning index to oid[0]."""
    var t = Int(thread_idx.x)
    var best_i = Int32(0)
    var best_v = a[op]
    var i = t
    while i < n:
        if a[op + i] > best_v:
            best_v = a[op + i]
            best_i = Int32(i)
        i += TG_MM
    var sval = stack_allocation[TG_MM, Float32, address_space = AddressSpace.SHARED]()
    var sidx = stack_allocation[TG_MM, Int32, address_space = AddressSpace.SHARED]()
    sval[t] = best_v
    sidx[t] = best_i
    barrier()
    var stride = TG_MM // 2
    while stride > 0:
        if t < stride:
            if sval[t + stride] > sval[t]:
                sval[t] = sval[t + stride]
                sidx[t] = sidx[t + stride]
        barrier()
        stride //= 2
    if t == 0:
        oid[0] = sidx[0]


# ============================ host-side shared ================================

comptime KPtr = UnsafePointer[NoneType, UntrackedOrigin[mut=True]]


def _box[T: Movable](var f: T) -> KPtr:
    """Move a compiled kernel handle to the heap, return a type-erased pointer.

    The box lives for the process lifetime (never freed); call sites bitcast
    back with type_of(ctx.compile_function[k_x]()).
    """
    var p = alloc[T](1)
    p.init_pointee_move(f^)
    return p.bitcast[NoneType]()


struct Kernels(Copyable, Movable):
    """Precompiled kernel handles: enqueue_function(handle, ...) skips the
    per-call kernel resolution of the enqueue_function[k_x](...) template
    path (~3x cheaper dispatch; decode is dispatch-bound)."""
    var mm: KPtr
    var mmt: KPtr
    var rms: KPtr
    var rms2: KPtr
    var rope: KPtr
    var ropec: KPtr                     # fused rope + KV append
    var copy2: KPtr
    var scores: KPtr
    var softmax: KPtr
    var attout: KPtr
    var swiglu: KPtr
    var resadd: KPtr
    var add: KPtr
    var embg: KPtr
    var exp: KPtr
    var argmax: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.mm = _box(ctx.compile_function[k_mm_w]())
        self.mmt = _box(ctx.compile_function[k_mm_tile]())
        self.rms = _box(ctx.compile_function[k_rmsnorm_w]())
        self.rms2 = _box(ctx.compile_function[k_rmsnorm2_w]())
        self.rope = _box(ctx.compile_function[k_rope_qk]())
        self.ropec = _box(ctx.compile_function[k_rope_qk_copy2]())
        self.copy2 = _box(ctx.compile_function[k_copy2]())
        self.scores = _box(ctx.compile_function[k_scores]())
        self.softmax = _box(ctx.compile_function[k_softmax_rows]())
        self.attout = _box(ctx.compile_function[k_att_out]())
        self.swiglu = _box(ctx.compile_function[k_swiglu_mul]())
        self.resadd = _box(ctx.compile_function[k_res_add]())
        self.add = _box(ctx.compile_function[k_add]())
        self.embg = _box(ctx.compile_function[k_embed_gather]())
        self.exp = _box(ctx.compile_function[k_export]())
        self.argmax = _box(ctx.compile_function[k_argmax]())


struct Acts:
    """f32 activation arena: bump allocation + resettable scratch mark.

    Also carries the precompiled kernel handles (kn) so every dispatch site
    that already receives Acts gets cheap enqueues without signature churn.
    """
    var buf: DeviceBuffer[DType.float32]
    var top: Int
    var cap: Int
    var kn: Kernels

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        self.buf = ctx.enqueue_create_buffer[DType.float32](cap)
        self.top = PAD
        self.cap = cap
        self.kn = Kernels(ctx)

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.top
        self.top += n
        if self.top > self.cap:
            raise Error("acts arena overflow")
        return off

    def mark(self) -> Int:
        return self.top

    def reset(mut self, m: Int):
        self.top = m


def rope_inv_freq(cfg: Config) -> List[Float32]:
    """Standard RoPE inverse frequencies (Qwen, classic Llama)."""
    var half = cfg.half()
    var out = List[Float32](capacity=half)
    for d in range(half):
        out.append(Float32(1.0 / (cfg.theta ** (Float64(2 * d) / Float64(cfg.head_dim)))))
    return out^


def llama3_inv_freq(cfg: Config) -> List[Float32]:
    """Per-dim RoPE inverse frequencies with llama3 wavelength scaling."""
    var half = cfg.half()
    var out = List[Float32](capacity=half)
    for d in range(half):
        var inv = 1.0 / (cfg.theta ** (Float64(2 * d) / Float64(cfg.head_dim)))
        var wavelen = 2.0 * 3.14159265358979323846 / inv
        var scaled = inv
        if wavelen >= cfg.orig_ctx / cfg.high_ff:
            if wavelen > cfg.orig_ctx / cfg.low_ff:
                scaled = inv / cfg.rope_factor
            else:
                var smooth = (cfg.orig_ctx / wavelen - cfg.low_ff) / (cfg.high_ff - cfg.low_ff)
                scaled = (1.0 - smooth) * inv / cfg.rope_factor + smooth * inv
        out.append(Float32(scaled))
    return out^


def mm_op_w(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
            ox: Int, s: Int, m: Int, owt: Int, n: Int) raises -> Int:
    """Chunked k_mm_w matmul; used for s in {2,3}."""
    var oy = a.alloc(s * n)
    var total = s * n
    var chunk = max(BLOCK, MM_CHUNK_MACS // m)
    var e0 = 0
    while e0 < total:
        var cnt = min(chunk, total - e0)
        ctx.enqueue_function(
            a.kn.mm.bitcast[type_of(ctx.compile_function[k_mm_w]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox, owt, oy, s, m, n, e0, grid_dim=cnt, block_dim=TG_MM)
        e0 += cnt
        if e0 < total:
            ctx.synchronize()
    return oy


def mm_op_t(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
            ox: Int, s: Int, m: Int, owt: Int, n: Int) raises -> Int:
    """Tiled matmul for s==1 decode and s>=4 prefill."""
    var oy = a.alloc(s * n)
    var ncb = ceildiv(n, 8)
    var rows_per_chunk = max(8, ((1 << 30) // (n * m)) * 8)
    var r0 = 0
    while r0 < s:
        var rows = min(rows_per_chunk, s - r0)
        ctx.enqueue_function(
            a.kn.mmt.bitcast[type_of(ctx.compile_function[k_mm_tile]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox + r0 * m, owt, oy + r0 * n, rows, m, n,
            grid_dim=ceildiv(rows, 8) * ncb, block_dim=TG_MM)
        r0 += rows
        if r0 < s:
            ctx.synchronize()
    return oy


def mm_op(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
          ox: Int, s: Int, m: Int, owt: Int, n: Int) raises -> Int:
    """Matmul dispatch: tiled for s>=4; k_mm_w for s<=3.

    s==1 is latency-bound: k_mm_w's one-output-per-group shape gives 8x
    the threadgroups of the tile kernel and measures ~6x faster (30 vs
    ~5 GMAC/s microbench) — parallelism beats load reuse at s==1."""
    if s >= 4:
        return mm_op_t(ctx, w, a, ox, s, m, owt, n)
    return mm_op_w(ctx, w, a, ox, s, m, owt, n)


def rmsnorm_op(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
               ox: Int, s: Int, owt: Int, d: Int, eps: Float32) raises -> Int:
    var oy = a.alloc(s * d)
    ctx.enqueue_function(
        a.kn.rms.bitcast[type_of(ctx.compile_function[k_rmsnorm_w]())]()[],
        w.unsafe_ptr(), a.buf.unsafe_ptr(),
        ox, owt, oy, s, d, eps, grid_dim=s, block_dim=TG)
    return oy


def run_layer(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
              cfg: Config, lo: LayerOffs, ox: Int, s: Int, pos0: Int,
              kc: Int, vc: Int, oinv: Int) raises -> Int:
    """One pre-norm GQA block with KV append; returns offset of (s, hidden).

    When s==1 and q|k|v (resp. gate|up) are contiguous in the weight buffer,
    they run as single fused dispatches (single row: no interleaving issue).
    """
    var H = cfg.hidden
    var QD = cfg.q_dim()
    var KVD = cfg.kv_dim()
    var nctx = pos0 + s
    var xn = rmsnorm_op(ctx, w, a, ox, s, lo.in_norm, H, cfg.eps)
    var oq: Int
    var okk: Int
    var ov: Int
    if s == 1 and lo.k == lo.q + QD * H and lo.v == lo.k + KVD * H:
        oq = mm_op(ctx, w, a, xn, 1, H, lo.q, QD + 2 * KVD)
        okk = oq + QD
        ov = okk + KVD
    else:
        oq = mm_op(ctx, w, a, xn, s, H, lo.q, QD)
        okk = mm_op(ctx, w, a, xn, s, H, lo.k, KVD)
        ov = mm_op(ctx, w, a, xn, s, H, lo.v, KVD)
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
        a.kn.ropec.bitcast[
            type_of(ctx.compile_function[k_rope_qk_copy2]())]()[],
        a.buf.unsafe_ptr(), oq, okk, ov, kc, vc, oinv, s, pos0,
        cfg.n_heads, cfg.n_kv, QD, KVD, cfg.half(),
        grid_dim=ceildiv(
            s * (cfg.n_heads + cfg.n_kv) * cfg.half() + s * KVD, BLOCK),
        block_dim=BLOCK)
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
    var oo = mm_op(ctx, w, a, oao, s, QD, lo.o, H)
    var oh = a.alloc(s * H)
    ctx.enqueue_function(
        a.kn.resadd.bitcast[type_of(ctx.compile_function[k_res_add]())]()[],
        a.buf.unsafe_ptr(), ox, oo, oh, s * H,
        grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
    var oz = rmsnorm_op(ctx, w, a, oh, s, lo.post_norm, H, cfg.eps)
    var og: Int
    var ou: Int
    if s == 1 and lo.up == lo.gate + H * cfg.inter:
        og = mm_op(ctx, w, a, oz, 1, H, lo.gate, 2 * cfg.inter)
        ou = og + cfg.inter
    else:
        og = mm_op(ctx, w, a, oz, s, H, lo.gate, cfg.inter)
        ou = mm_op(ctx, w, a, oz, s, H, lo.up, cfg.inter)
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


def argmax_op(ctx: DeviceContext, mut a: Acts, lgbuf: DeviceBuffer[DType.float32],
              argbuf: DeviceBuffer[DType.int32], op: Int, vocab: Int) raises:
    """GPU argmax over logits in lgbuf; result in argbuf[0]."""
    ctx.enqueue_function(
        a.kn.argmax.bitcast[type_of(ctx.compile_function[k_argmax]())]()[],
        lgbuf.unsafe_ptr(), argbuf.unsafe_ptr(), op, vocab,
        grid_dim=1, block_dim=TG_MM)


def read_argmax_gpu(argbuf: DeviceBuffer[DType.int32]) raises -> Int:
    """Read GPU argmax result (4-byte host map)."""
    with argbuf.map_to_host() as h:
        return Int(h[0])


def read_argmax_buf(lgbuf: DeviceBuffer[DType.float32], vocab: Int) raises -> Int:
    """Argmax on logits already in lgbuf — no List copy."""
    var best = 0
    with lgbuf.map_to_host() as h:
        var p = h.unsafe_ptr() + PAD
        for i in range(1, vocab):
            if p[i] > p[best]:
                best = i
    return best


def read_logits_buf(lgbuf: DeviceBuffer[DType.float32], vocab: Int) raises -> List[Float32]:
    var logits = List[Float32](length=vocab, fill=0)
    with lgbuf.map_to_host() as h:
        memcpy(dest=logits.unsafe_ptr(), src=h.unsafe_ptr() + PAD, count=vocab)
    return logits^


def read_f32_bin(path: String, count: Int) raises -> List[Float32]:
    var f = open(path, "r")
    var raw = f.read_bytes(count * 4)
    f.close()
    var out = List[Float32](capacity=count)
    for i in range(count):
        var bits = (UInt32(raw[4 * i]) | (UInt32(raw[4 * i + 1]) << 8)
                    | (UInt32(raw[4 * i + 2]) << 16) | (UInt32(raw[4 * i + 3]) << 24))
        out.append(UnsafePointer(to=bits).bitcast[Float32]()[])
    return out^
