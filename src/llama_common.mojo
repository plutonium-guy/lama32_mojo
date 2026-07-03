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
    var acc = Float32(0)
    var qb = oq + i * qdim + h * headdim
    var kb = okc + j * kvdim + (h // group) * headdim
    for d in range(headdim):
        acc += a[qb + d] * a[kb + d]
    a[o] = acc / sqrt(Float32(headdim))


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
    var acc = Float32(0)
    var pb = osc + (h * s + i) * nctx
    var vb = ovc + (h // group) * headdim + d
    for j in range(nctx):
        acc += a[pb + j] * a[vb + j * kvdim]
    a[oy + idx] = acc


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
    var rms: KPtr
    var rope: KPtr
    var copy2: KPtr
    var scores: KPtr
    var softmax: KPtr
    var attout: KPtr
    var swiglu: KPtr
    var resadd: KPtr
    var add: KPtr
    var embg: KPtr
    var exp: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.mm = _box(ctx.compile_function[k_mm_w]())
        self.rms = _box(ctx.compile_function[k_rmsnorm_w]())
        self.rope = _box(ctx.compile_function[k_rope_qk]())
        self.copy2 = _box(ctx.compile_function[k_copy2]())
        self.scores = _box(ctx.compile_function[k_scores]())
        self.softmax = _box(ctx.compile_function[k_softmax_rows]())
        self.attout = _box(ctx.compile_function[k_att_out]())
        self.swiglu = _box(ctx.compile_function[k_swiglu_mul]())
        self.resadd = _box(ctx.compile_function[k_res_add]())
        self.add = _box(ctx.compile_function[k_add]())
        self.embg = _box(ctx.compile_function[k_embed_gather]())
        self.exp = _box(ctx.compile_function[k_export]())


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


def mm_op(ctx: DeviceContext, w: DeviceBuffer[DType.uint16], mut a: Acts,
          ox: Int, s: Int, m: Int, owt: Int, n: Int) raises -> Int:
    """Chunked matmul; no dispatch exceeds MM_CHUNK_MACS (watchdog safety)."""
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
