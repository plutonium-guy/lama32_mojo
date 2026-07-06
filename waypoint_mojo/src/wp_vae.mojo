"""taehv1_5 streaming decoder on Metal (latent [32,32,64] -> 4 RGB 512x1024).

Faithful port of vae/ae_model.py ChunkedStreamingTAEHV.decode:
- MemBlocks carry one feature map of temporal state each (previous step's
  block input); TGrow 1x1 convs double the time axis twice (1 latent ->
  4 frames); nearest 2x upsamples between stages; pixel_shuffle(2) at the
  end.
- Streaming warmup: the first latent is fed 4x (3 warmup passes seed the
  MemBlock memories, outputs discarded) exactly like frames_to_trim in the
  reference.

Weights live in the shared bf16 arena under vae.decoder.N.*.
"""

from std.math import ceildiv, exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx
from resident import Weights
from llama_common import Acts, BLOCK, PAD, bf, KPtr, _box
from wp_common import WPKernels, k_copy
from wp_common import LH, LW, C as LATC

comptime FH = 512                       # output frame height
comptime FW = 1024                      # output frame width


# ============================ kernels =========================================

def k_clamp3(a: UnsafePointer[Float32, MutAnyOrigin], ox: Int, oy: Int, n: Int):
    """tanh(x/3)*3 (taehv latent Clamp)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var v = a[ox + i] / 3.0
        var e = exp(2.0 * v)
        a[oy + i] = (e - 1.0) / (e + 1.0) * 3.0


def k_conv3x3(w: UnsafePointer[UInt16, MutAnyOrigin],
              a: UnsafePointer[Float32, MutAnyOrigin],
              ox1: Int, c1: Int, ox2: Int, c2: Int,
              ow: Int, ob: Int, ores: Int, oy: Int,
              cout: Int, h: Int, wd: Int, relu: Int):
    """3x3 pad-1 conv; optional 2nd input (channel concat), bias, residual,
    relu. Weight layout [cout, c1+c2, 3, 3] bf16."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= cout * h * wd:
        return
    var co = idx // (h * wd)
    var p = idx % (h * wd)
    var y = p // wd
    var x = p % wd
    var cin = c1 + c2
    var acc = Float32(0)
    if ob >= 0:
        acc = bf(w, ob + co)
    var wb = ow + co * cin * 9
    for ci in range(cin):
        var src = (ox1 + ci * h * wd) if ci < c1 else (ox2 + (ci - c1) * h * wd)
        var wc = wb + ci * 9
        for ky in range(3):
            var iy = y + ky - 1
            if iy < 0 or iy >= h:
                continue
            for kx in range(3):
                var ix = x + kx - 1
                if ix < 0 or ix >= wd:
                    continue
                acc += a[src + iy * wd + ix] * bf(w, wc + ky * 3 + kx)
    if ores >= 0:
        acc += a[ores + idx]
    if relu == 1 and acc < 0:
        acc = 0
    a[oy + idx] = acc


def k_conv1x1(w: UnsafePointer[UInt16, MutAnyOrigin],
              a: UnsafePointer[Float32, MutAnyOrigin],
              ox: Int, ow: Int, oy: Int, cin: Int, cout: Int, hw: Int):
    """1x1 conv (TGrow/TPool), no bias."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= cout * hw:
        return
    var co = idx // hw
    var p = idx % hw
    var acc = Float32(0)
    for ci in range(cin):
        acc += a[ox + ci * hw + p] * bf(w, ow + co * cin + ci)
    a[oy + idx] = acc


def k_upsample2(a: UnsafePointer[Float32, MutAnyOrigin],
                ox: Int, oy: Int, c: Int, h: Int, wd: Int):
    """Nearest 2x spatial upsample [c,h,w] -> [c,2h,2w]."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var h2 = h * 2
    var w2 = wd * 2
    if idx >= c * h2 * w2:
        return
    var ci = idx // (h2 * w2)
    var p = idx % (h2 * w2)
    var y = p // w2
    var x = p % w2
    a[oy + idx] = a[ox + ci * h * wd + (y // 2) * wd + (x // 2)]


def k_pshuffle(a: UnsafePointer[Float32, MutAnyOrigin],
               ox: Int, oy: Int):
    """pixel_shuffle(2): [12, FH/2, FW/2] -> [3, FH, FW], clamped to [0,1]."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= 3 * FH * FW:
        return
    var c = idx // (FH * FW)
    var p = idx % (FH * FW)
    var y = p // FW
    var x = p % FW
    var src = (c * 4 + (y % 2) * 2 + (x % 2)) * (FH // 2) * (FW // 2)
    var v = a[ox + src + (y // 2) * (FW // 2) + (x // 2)]
    if v < 0:
        v = 0
    if v > 1:
        v = 1
    a[oy + idx] = v


def k_fill0(a: UnsafePointer[Float32, MutAnyOrigin], ox: Int, n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        a[ox + i] = 0


# ============================ host side =======================================

def vae_weight_names() -> List[String]:
    var names = List[String]()
    names.append("vae.decoder.1.weight")
    names.append("vae.decoder.1.bias")
    for b in [3, 4, 5, 9, 10, 11, 15, 16, 17]:
        for c in [0, 2, 4]:
            var p = "vae.decoder." + String(b) + ".conv." + String(c) + "."
            names.append(p + "weight")
            names.append(p + "bias")
    names.append("vae.decoder.7.conv.weight")
    names.append("vae.decoder.8.weight")
    names.append("vae.decoder.13.conv.weight")
    names.append("vae.decoder.14.weight")
    names.append("vae.decoder.19.conv.weight")
    names.append("vae.decoder.20.weight")
    names.append("vae.decoder.22.weight")
    names.append("vae.decoder.22.bias")
    return names^


struct WPVaeKernels(Copyable, Movable):
    var clamp3: KPtr
    var conv3: KPtr
    var conv1: KPtr
    var up2: KPtr
    var pshuf: KPtr
    var fill0: KPtr

    def __init__(out self, ctx: DeviceContext) raises:
        self.clamp3 = _box(ctx.compile_function[k_clamp3]())
        self.conv3 = _box(ctx.compile_function[k_conv3x3]())
        self.conv1 = _box(ctx.compile_function[k_conv1x1]())
        self.up2 = _box(ctx.compile_function[k_upsample2]())
        self.pshuf = _box(ctx.compile_function[k_pshuffle]())
        self.fill0 = _box(ctx.compile_function[k_fill0]())


@fieldwise_init
struct MemBlockState(Copyable, Movable):
    """One MemBlock: weight offsets + temporal memory buffer."""
    var w0: Int                         # conv.0 [n, 2n, 3, 3]
    var b0: Int
    var w2: Int                         # conv.2 [n, n, 3, 3]
    var b2: Int
    var w4: Int                         # conv.4 [n, n, 3, 3]
    var b4: Int
    var mem: Int                        # [n, h, w] previous-step input
    var ch: Int
    var h: Int
    var wd: Int


struct WPVae:
    var vk: WPVaeKernels
    var blocks: List[MemBlockState]     # 9 MemBlocks (3 per stage)
    var oframes: List[Int]              # 4 decoded frames [3, FH, FW] in acts
    var primed: Bool
    var nmem: Int                       # total memory elements (for reset)
    var mem0: Int

    def __init__(out self, ctx: DeviceContext, w: Weights, mut a: Acts) raises:
        self.vk = WPVaeKernels(ctx)
        self.blocks = List[MemBlockState]()
        self.oframes = List[Int]()
        self.primed = False
        self.mem0 = a.mark()
        # stage A: 256 @ 32x64, stage B: 128 @ 64x128, stage C: 64 @ 128x256
        var ids_a = [3, 4, 5]
        var ids_b = [9, 10, 11]
        var ids_c = [15, 16, 17]
        for i in range(9):
            var bid: Int
            var ch: Int
            var h: Int
            var wd: Int
            if i < 3:
                bid = ids_a[i]
                ch = 256
                h = LH
                wd = LW
            elif i < 6:
                bid = ids_b[i - 3]
                ch = 128
                h = LH * 2
                wd = LW * 2
            else:
                bid = ids_c[i - 6]
                ch = 64
                h = LH * 4
                wd = LW * 4
            var p = "vae.decoder." + String(bid) + ".conv."
            self.blocks.append(MemBlockState(
                w.o(p + "0.weight"), w.o(p + "0.bias"),
                w.o(p + "2.weight"), w.o(p + "2.bias"),
                w.o(p + "4.weight"), w.o(p + "4.bias"),
                a.alloc(ch * h * wd), ch, h, wd))
        self.nmem = a.mark() - self.mem0
        for _ in range(4):
            self.oframes.append(a.alloc(3 * FH * FW))
        self.reset(ctx, a)

    def reset(mut self, ctx: DeviceContext, mut a: Acts) raises:
        ctx.enqueue_function(
            self.vk.fill0.bitcast[
                type_of(ctx.compile_function[k_fill0]())]()[],
            a.buf.unsafe_ptr(), self.mem0, self.nmem,
            grid_dim=ceildiv(self.nmem, BLOCK), block_dim=BLOCK)
        self.primed = False

    def _conv3(self, ctx: DeviceContext, w: Weights, mut a: Acts,
               ox1: Int, c1: Int, ox2: Int, c2: Int, ow: Int, ob: Int,
               ores: Int, cout: Int, h: Int, wd: Int, relu: Int) raises -> Int:
        var oy = a.alloc(cout * h * wd)
        ctx.enqueue_function(
            self.vk.conv3.bitcast[
                type_of(ctx.compile_function[k_conv3x3]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox1, c1, ox2, c2, ow, ob, ores, oy, cout, h, wd, relu,
            grid_dim=ceildiv(cout * h * wd, BLOCK), block_dim=BLOCK)
        return oy

    def _memblock(self, ctx: DeviceContext, w: Weights, mut a: Acts,
                  kn: WPKernels, bi: Int, ox: Int) raises -> Int:
        var b = self.blocks[bi].copy()
        var n = b.ch * b.h * b.wd
        var c = self._conv3(ctx, w, a, ox, b.ch, b.mem, b.ch,
                            b.w0, b.b0, -1, b.ch, b.h, b.wd, 1)
        c = self._conv3(ctx, w, a, c, b.ch, -1, 0, b.w2, b.b2, -1,
                        b.ch, b.h, b.wd, 1)
        c = self._conv3(ctx, w, a, c, b.ch, -1, 0, b.w4, b.b4, ox,
                        b.ch, b.h, b.wd, 1)
        # memory <- this step's input (after use)
        ctx.enqueue_function(
            kn.cpy.bitcast[type_of(ctx.compile_function[k_copy]())]()[],
            a.buf.unsafe_ptr(), ox, b.mem, n,
            grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        return c

    def _conv1(self, ctx: DeviceContext, w: Weights, mut a: Acts,
               ox: Int, ow: Int, cin: Int, cout: Int, hw: Int) raises -> Int:
        var oy = a.alloc(cout * hw)
        ctx.enqueue_function(
            self.vk.conv1.bitcast[
                type_of(ctx.compile_function[k_conv1x1]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox, ow, oy, cin, cout, hw,
            grid_dim=ceildiv(cout * hw, BLOCK), block_dim=BLOCK)
        return oy

    def _up2(self, ctx: DeviceContext, mut a: Acts,
             ox: Int, c: Int, h: Int, wd: Int) raises -> Int:
        var oy = a.alloc(c * h * 2 * wd * 2)
        ctx.enqueue_function(
            self.vk.up2.bitcast[
                type_of(ctx.compile_function[k_upsample2]())]()[],
            a.buf.unsafe_ptr(), ox, oy, c, h, wd,
            grid_dim=ceildiv(c * h * 2 * wd * 2, BLOCK), block_dim=BLOCK)
        return oy

    def _feed(mut self, ctx: DeviceContext, w: Weights, mut a: Acts,
              kn: WPKernels, olat: Int, keep: Bool) raises:
        """One latent through the decoder; frames land in self.oframes.

        When keep is False (streaming warmup) outputs are discarded but the
        MemBlock state still updates, matching the reference."""
        var mark = a.mark()
        # stage A @ 32x64
        var oc = a.alloc(LATC * LH * LW)
        ctx.enqueue_function(
            self.vk.clamp3.bitcast[
                type_of(ctx.compile_function[k_clamp3]())]()[],
            a.buf.unsafe_ptr(), olat, oc, LATC * LH * LW,
            grid_dim=ceildiv(LATC * LH * LW, BLOCK), block_dim=BLOCK)
        var x = self._conv3(ctx, w, a, oc, LATC, -1, 0,
                            w.o("vae.decoder.1.weight"),
                            w.o("vae.decoder.1.bias"), -1, 256, LH, LW, 1)
        for bi in range(3):
            x = self._memblock(ctx, w, a, kn, bi, x)
        x = self._up2(ctx, a, x, 256, LH, LW)
        x = self._conv1(ctx, w, a, x, w.o("vae.decoder.7.conv.weight"),
                        256, 256, LH * 2 * LW * 2)
        # stage B @ 64x128
        x = self._conv3(ctx, w, a, x, 256, -1, 0,
                        w.o("vae.decoder.8.weight"), -1, -1,
                        128, LH * 2, LW * 2, 0)
        for bi in range(3, 6):
            x = self._memblock(ctx, w, a, kn, bi, x)
        x = self._up2(ctx, a, x, 128, LH * 2, LW * 2)
        var hb = LH * 4
        var wb = LW * 4
        var g = self._conv1(ctx, w, a, x, w.o("vae.decoder.13.conv.weight"),
                            128, 256, hb * wb)
        ctx.synchronize()
        # two time steps: channels [0:128], [128:256]
        for t in range(2):
            var mt = a.mark()
            var xt = self._conv3(ctx, w, a, g + t * 128 * hb * wb, 128, -1, 0,
                                 w.o("vae.decoder.14.weight"), -1, -1,
                                 64, hb, wb, 0)
            for bi in range(6, 9):
                xt = self._memblock(ctx, w, a, kn, bi, xt)
            xt = self._up2(ctx, a, xt, 64, hb, wb)
            var hc = hb * 2
            var wc = wb * 2
            var g2 = self._conv1(ctx, w, a, xt,
                                 w.o("vae.decoder.19.conv.weight"),
                                 64, 128, hc * wc)
            ctx.synchronize()
            for u in range(2):
                var mu = a.mark()
                var xd = self._conv3(ctx, w, a, g2 + u * 64 * hc * wc, 64,
                                     -1, 0, w.o("vae.decoder.20.weight"),
                                     -1, -1, 64, hc, wc, 1)
                var xo = self._conv3(ctx, w, a, xd, 64, -1, 0,
                                     w.o("vae.decoder.22.weight"),
                                     w.o("vae.decoder.22.bias"), -1,
                                     12, hc, wc, 0)
                if keep:
                    ctx.enqueue_function(
                        self.vk.pshuf.bitcast[
                            type_of(ctx.compile_function[k_pshuffle]())]()[],
                        a.buf.unsafe_ptr(), xo, self.oframes[t * 2 + u],
                        grid_dim=ceildiv(3 * FH * FW, BLOCK), block_dim=BLOCK)
                ctx.synchronize()
                a.reset(mu)
            a.reset(mt)
        a.reset(mark)

    def decode(mut self, ctx: DeviceContext, w: Weights, mut a: Acts,
               kn: WPKernels, olat: Int) raises:
        """Decode one latent; 4 RGB frames land in self.oframes[0..3]."""
        if not self.primed:
            for _ in range(3):
                self._feed(ctx, w, a, kn, olat, False)
            self.primed = True
        self._feed(ctx, w, a, kn, olat, True)
