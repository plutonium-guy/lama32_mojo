"""Waypoint-1.5-1B world model on Metal — DiT backbone + rectified-flow loop.

Validates against the torch oracle (scripts/gen_oracle.py):
  gate 1: pass-0 velocity (frame 0, sigma 1.0) max-abs-diff
  gate 2: denoised latent drift per frame
Run: pixi run test   (needs `pixi run convert` and `pixi run oracle` first)
"""

from std.math import ceildiv, sqrt, cos, sin, log
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from safetensors import SafeTensors
from resident import Weights
from llama_common import (
    Acts, BLOCK, TG, PAD, mm_op, k_bf16_to_f32, k_export, k_res_add,
    k_copy2, k_softmax_rows, read_f32_bin,
)
from wp_common import (
    WPKernels, WPLayer, WPCacheState, FrameMask, ada_rms_op,
    k_stage_in, k_patch_gather, k_unpatch_scatter, k_ada_rms, k_rms_plain,
    k_ada_gate_res, k_qk_normrope, k_v_store, k_scores_slots, k_att_out_slots,
    k_silu, k_row_add_silu, k_axpy, k_copy,
    D, NH, NKV, HD, FF, LAYERS, TPF, GW, GH, C, LH, LW, LAT, KVD, QD,
    SLOTS, RING,
)
from wp_vae import WPVae, vae_weight_names, FH, FW

comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/waypoint_mojo/assets/mojo"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/waypoint_mojo/assets/oracle"
comptime NSIG = 5                       # sigma table rows (incl. 0.0 cache row)
comptime PI = 3.14159265358979323846


def weight_names() -> List[String]:
    var names = List[String]()
    names.append("patchify")
    names.append("unpatch.w")
    names.append("unpatch.b")
    names.append("ctrl.fc1")
    names.append("ctrl.fc2")
    names.append("vlamb")
    names.append("tab.cond")
    names.append("tab.out")
    for L in range(LAYERS):
        var lp = "L" + String(L) + "."
        names.append(lp + "qkv")
        names.append(lp + "o")
        names.append(lp + "fc1")
        names.append(lp + "fc2")
        if L % 3 == 0:
            names.append(lp + "fu_x")
            names.append(lp + "fu_c")
            names.append(lp + "fu_o")
    names.extend(vae_weight_names())
    return names^


struct Waypoint:
    var ctx: DeviceContext
    var w: Weights
    var a: Acts
    var kn: WPKernels
    var stg: DeviceBuffer[DType.float32]    # host -> acts staging
    var outb: DeviceBuffer[DType.float32]   # acts -> host readback
    var layers: List[WPLayer]
    var cache: List[WPCacheState]
    var oc1: List[Int]                      # per-ctrl-layer fused cond [D]
    var orc: Int                            # rope constants xy[8] | inv_t[16]
    var octrl: Int                          # ctrl input vec [512]
    var olat: Int                           # current latent x [LAT]
    var ovel: Int                           # velocity output [LAT]
    var ov1: Int                            # layer-0 v for value residual
    var oxa: Int                            # ping/pong hidden [TPF*D] each
    var oxb: Int
    var otabc: Int
    var otabo: Int
    var frame: Int

    def __init__(out self, ctx: DeviceContext, var w: Weights) raises:
        self.ctx = ctx
        self.w = w^
        var kvelems = LAYERS * 2 * SLOTS * TPF * KVD
        # scratch: scores worst case NH*TPF*SLOTS*TPF + mlp/qkv intermediates
        # (also covers the VAE's ~75M peak); + VAE memories and frame buffers
        var scratch = NH * TPF * SLOTS * TPF + 24 * 1024 * 1024
        var vaeper = 32 * 1024 * 1024
        self.a = Acts(ctx, PAD + kvelems + 4 * LAT + 3 * TPF * D + 64 * 1024
                      + vaeper + scratch)
        self.kn = WPKernels(ctx)
        self.stg = ctx.enqueue_create_buffer[DType.float32](PAD + LAT)
        self.outb = ctx.enqueue_create_buffer[DType.float32](PAD + 3 * FH * FW)
        self.layers = List[WPLayer]()
        self.cache = List[WPCacheState]()
        self.oc1 = List[Int]()
        self.frame = 0

        self.orc = self.a.alloc(32)
        self.octrl = self.a.alloc(512)
        self.olat = self.a.alloc(LAT)
        self.ovel = self.a.alloc(LAT)
        self.ov1 = self.a.alloc(TPF * KVD)
        self.oxa = self.a.alloc(TPF * D)
        self.oxb = self.a.alloc(TPF * D)
        for _ in range(8):
            self.oc1.append(self.a.alloc(D))
        self.otabc = self.w.o("tab.cond")
        self.otabo = self.w.o("tab.out")

        # rope constants (host-exact replicas of OrthoRoPEAngles)
        with self.stg.map_to_host() as h:
            var n = 4                       # (d_xy + 1) // 2 with d_xy = 8
            var max_freq = Float64(GH) * 0.8    # min(GH, GW) * nyquist
            for i in range(8):
                var step = Float64(i // 2) * (max_freq / 2.0 - 1.0) / Float64(n - 1)
                h[PAD + i] = Float32((1.0 + step) * PI)
            for i in range(16):
                var ex = Float64(2 * (i // 2)) / 16.0
                h[PAD + 8 + i] = Float32(1.0 / (10000.0 ** ex))
        self._stage(PAD, self.orc, 32)

        # value-residual scalars via bf16 decode + readback
        var ovl = self.a.alloc(LAYERS)
        self.ctx.enqueue_function[k_bf16_to_f32](
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            self.w.o("vlamb"), ovl, LAYERS,
            grid_dim=1, block_dim=BLOCK)
        self.ctx.enqueue_function[k_export](
            self.a.buf.unsafe_ptr(), self.outb.unsafe_ptr(), ovl, PAD, LAYERS,
            grid_dim=1, block_dim=BLOCK)
        self.ctx.synchronize()
        var vlambs = List[Float32]()
        with self.outb.map_to_host() as h:
            for L in range(LAYERS):
                vlambs.append(h[PAD + L])

        for L in range(LAYERS):
            var lp = "L" + String(L) + "."
            var fux = -1
            var fuc = -1
            var fuo = -1
            if L % 3 == 0:
                fux = self.w.o(lp + "fu_x")
                fuc = self.w.o(lp + "fu_c")
                fuo = self.w.o(lp + "fu_o")
            self.layers.append(WPLayer(
                self.w.o(lp + "qkv"), self.w.o(lp + "o"),
                self.w.o(lp + "fc1"), self.w.o(lp + "fc2"),
                fux, fuc, fuo, vlambs[L], (L % 4) == 3,
                self.a.alloc(SLOTS * TPF * KVD),
                self.a.alloc(SLOTS * TPF * KVD)))
            self.cache.append(WPCacheState())

    def _stage(mut self, osrc: Int, odst: Int, n: Int) raises:
        self.ctx.enqueue_function(
            self.kn.stage.bitcast[
                type_of(self.ctx.compile_function[k_stage_in]())]()[],
            self.stg.unsafe_ptr(), self.a.buf.unsafe_ptr(), osrc, odst, n,
            grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

    def upload_latent(mut self, vals: List[Float32]) raises:
        with self.stg.map_to_host() as h:
            for i in range(LAT):
                h[PAD + i] = vals[i]
        self._stage(PAD, self.olat, LAT)

    def read_acts(mut self, osrc: Int, n: Int) raises -> List[Float32]:
        self.ctx.enqueue_function(
            self.a.kn.exp.bitcast[
                type_of(self.ctx.compile_function[k_export]())]()[],
            self.a.buf.unsafe_ptr(), self.outb.unsafe_ptr(), osrc, PAD, n,
            grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        var out = List[Float32](capacity=n)
        with self.outb.map_to_host() as h:
            for i in range(n):
                out.append(h[PAD + i])
        return out^

    def set_controls(mut self, mouse_x: Float32, mouse_y: Float32,
                     buttons: List[Int], scroll: Int) raises:
        """Build the 512-padded ctrl vec and per-layer fused conditioning."""
        with self.stg.map_to_host() as h:
            for i in range(512):
                h[PAD + i] = 0
            h[PAD + 0] = mouse_x
            h[PAD + 1] = mouse_y
            for i in range(len(buttons)):
                h[PAD + 2 + buttons[i]] = 1.0
            h[PAD + 258] = Float32(1 if scroll > 0 else (-1 if scroll < 0 else 0))
        self._stage(PAD, self.octrl, 512)
        var mark = self.a.mark()
        var oh = mm_op(self.ctx, self.w.buf, self.a, self.octrl, 1, 512,
                       self.w.o("ctrl.fc1"), FF)
        self.ctx.enqueue_function(
            self.kn.silu.bitcast[
                type_of(self.ctx.compile_function[k_silu]())]()[],
            self.a.buf.unsafe_ptr(), oh, FF,
            grid_dim=ceildiv(FF, BLOCK), block_dim=BLOCK)
        var oe = mm_op(self.ctx, self.w.buf, self.a, oh, 1, FF,
                       self.w.o("ctrl.fc2"), D)
        var on = self.a.alloc(D)
        self.ctx.enqueue_function(
            self.kn.rmsp.bitcast[
                type_of(self.ctx.compile_function[k_rms_plain]())]()[],
            self.a.buf.unsafe_ptr(), oe, on, 1, D,
            grid_dim=1, block_dim=TG)
        var ci = 0
        for L in range(LAYERS):
            if L % 3 != 0:
                continue
            var oc = mm_op(self.ctx, self.w.buf, self.a, on, 1, D,
                           self.layers[L].fu_c, D)
            self.ctx.enqueue_function(
                self.kn.cpy.bitcast[
                    type_of(self.ctx.compile_function[k_copy]())]()[],
                self.a.buf.unsafe_ptr(), oc, self.oc1[ci], D,
                grid_dim=ceildiv(D, BLOCK), block_dim=BLOCK)
            ci += 1
        self.ctx.synchronize()
        self.a.reset(mark)

    def _frame_mask(self, L: Int) raises -> FrameMask:
        """Attention extent for this frame, per the reference upsert mask."""
        var st = self.cache[L].copy()
        var write_step = True
        if self.layers[L].is_global:
            write_step = (self.frame % 8) == 0
        var excl = -1
        if write_step and st.next_slot < st.nw:
            excl = st.next_slot
        var nring = st.nw
        if excl >= 0:
            nring -= 1
        return FrameMask(nring, excl, (nring + 1) * TPF)

    def _persist_slot(mut self, L: Int) raises:
        """Cache pass bookkeeping: tail -> ring slot when this frame persists."""
        var write_step = True
        if self.layers[L].is_global:
            write_step = (self.frame % 8) == 0
        if not write_step:
            return
        var slot = self.cache[L].next_slot
        var kc = self.layers[L].kc
        var vc = self.layers[L].vc
        var n = TPF * KVD
        self.ctx.enqueue_function(
            self.a.kn.copy2.bitcast[
                type_of(self.ctx.compile_function[k_copy2]())]()[],
            self.a.buf.unsafe_ptr(),
            kc + RING * n, kc + slot * n, vc + RING * n, vc + slot * n, n,
            grid_dim=ceildiv(2 * n, BLOCK), block_dim=BLOCK)
        if self.cache[L].nw < RING:
            self.cache[L].nw += 1
        self.cache[L].next_slot = (slot + 1) % RING

    def forward_pass(mut self, sig_idx: Int, want_out: Bool,
                     persist: Bool) raises:
        """One transformer pass over the current latent (olat)."""
        var mark = self.a.mark()
        var otok = self.a.alloc(TPF * 256)
        self.ctx.enqueue_function(
            self.kn.pgather.bitcast[
                type_of(self.ctx.compile_function[k_patch_gather]())]()[],
            self.a.buf.unsafe_ptr(), self.olat, otok,
            grid_dim=ceildiv(TPF * 256, BLOCK), block_dim=BLOCK)
        var ox0 = mm_op(self.ctx, self.w.buf, self.a, otok, TPF, 256,
                        self.w.o("patchify"), D)
        self.ctx.enqueue_function(
            self.kn.cpy.bitcast[
                type_of(self.ctx.compile_function[k_copy]())]()[],
            self.a.buf.unsafe_ptr(), ox0, self.oxa, TPF * D,
            grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)
        self.a.reset(mark)

        var xin = self.oxa
        var ci = 0
        for L in range(LAYERS):
            var xout = self.oxb if xin == self.oxa else self.oxa
            self._layer(L, sig_idx, xin, xout, ci)
            if L % 3 == 0:
                ci += 1
            xin = xout
            self.ctx.synchronize()

        if want_out:
            var m2 = self.a.mark()
            var ot = self.otabo + sig_idx * 2 * D
            var on = ada_rms_op(self.ctx, self.w.buf, self.a, self.kn,
                                xin, ot, ot + D, TPF, D)
            self.ctx.enqueue_function(
                self.kn.silu.bitcast[
                    type_of(self.ctx.compile_function[k_silu]())]()[],
                self.a.buf.unsafe_ptr(), on, TPF * D,
                grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)
            var ou = mm_op(self.ctx, self.w.buf, self.a, on, TPF, D,
                           self.w.o("unpatch.w"), 128)
            self.ctx.enqueue_function(
                self.kn.upscatter.bitcast[
                    type_of(self.ctx.compile_function[k_unpatch_scatter]())]()[],
                self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
                ou, self.w.o("unpatch.b"), self.ovel,
                grid_dim=ceildiv(TPF * 128, BLOCK), block_dim=BLOCK)
            self.a.reset(m2)

        if persist:
            for L in range(LAYERS):
                self._persist_slot(L)
        self.ctx.synchronize()

    def _layer(mut self, L: Int, sig_idx: Int, xin: Int, xout: Int,
               ci: Int) raises:
        var lo = self.layers[L].copy()
        var mark = self.a.mark()
        var otab = self.otabc + ((sig_idx * LAYERS + L) * 6) * D
        var fm = self._frame_mask(L)
        var nring = fm.nring
        var excl = fm.excl
        var nctx = fm.nctx
        var tail = RING * TPF * KVD

        var xn = ada_rms_op(self.ctx, self.w.buf, self.a, self.kn,
                            xin, otab, otab + D, TPF, D)
        var oq = mm_op(self.ctx, self.w.buf, self.a, xn, TPF, D, lo.q, QD)
        var ok = mm_op(self.ctx, self.w.buf, self.a, xn, TPF, D,
                       lo.q + QD * D, KVD)
        var ov = mm_op(self.ctx, self.w.buf, self.a, xn, TPF, D,
                       lo.q + (QD + KVD) * D, KVD)
        self.ctx.enqueue_function(
            self.kn.normrope.bitcast[
                type_of(self.ctx.compile_function[k_qk_normrope]())]()[],
            self.a.buf.unsafe_ptr(), oq, ok, lo.kc + tail, self.orc, self.frame,
            grid_dim=ceildiv(TPF * (NH + NKV), BLOCK), block_dim=BLOCK)
        self.ctx.enqueue_function(
            self.kn.vstore.bitcast[
                type_of(self.ctx.compile_function[k_v_store]())]()[],
            self.a.buf.unsafe_ptr(), ov, self.ov1, lo.vc + tail, lo.vlamb,
            1 if L == 0 else 0,
            grid_dim=ceildiv(TPF * KVD, BLOCK), block_dim=BLOCK)

        var osc = self.a.alloc(NH * TPF * nctx)
        self.ctx.enqueue_function(
            self.kn.scores.bitcast[
                type_of(self.ctx.compile_function[k_scores_slots]())]()[],
            self.a.buf.unsafe_ptr(), oq, lo.kc, osc, nctx, nring, excl,
            grid_dim=ceildiv(NH * TPF * nctx, BLOCK), block_dim=BLOCK)
        self.ctx.enqueue_function(
            self.a.kn.softmax.bitcast[
                type_of(self.ctx.compile_function[k_softmax_rows]())]()[],
            self.a.buf.unsafe_ptr(), osc, NH * TPF, nctx,
            grid_dim=ceildiv(NH * TPF, BLOCK), block_dim=BLOCK)
        var oao = self.a.alloc(TPF * QD)
        self.ctx.enqueue_function(
            self.kn.attout.bitcast[
                type_of(self.ctx.compile_function[k_att_out_slots]())]()[],
            self.a.buf.unsafe_ptr(), osc, lo.vc, oao, nctx, nring, excl,
            grid_dim=ceildiv(TPF * QD, BLOCK), block_dim=BLOCK)
        var oo = mm_op(self.ctx, self.w.buf, self.a, oao, TPF, QD, lo.o, D)
        var ox1 = self.a.alloc(TPF * D)
        self.ctx.enqueue_function(
            self.kn.gateres.bitcast[
                type_of(self.ctx.compile_function[k_ada_gate_res]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            oo, otab + 2 * D, xin, ox1, TPF, D,
            grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)

        var ox2 = ox1
        if lo.fu_x >= 0:
            var oxr = self.a.alloc(TPF * D)
            self.ctx.enqueue_function(
                self.kn.rmsp.bitcast[
                    type_of(self.ctx.compile_function[k_rms_plain]())]()[],
                self.a.buf.unsafe_ptr(), ox1, oxr, TPF, D,
                grid_dim=TPF, block_dim=TG)
            var oa1 = mm_op(self.ctx, self.w.buf, self.a, oxr, TPF, D,
                            lo.fu_x, D)
            self.ctx.enqueue_function(
                self.kn.rowsilu.bitcast[
                    type_of(self.ctx.compile_function[k_row_add_silu]())]()[],
                self.a.buf.unsafe_ptr(), oa1, self.oc1[ci], TPF, D,
                grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)
            var oy = mm_op(self.ctx, self.w.buf, self.a, oa1, TPF, D,
                           lo.fu_o, D)
            ox2 = self.a.alloc(TPF * D)
            self.ctx.enqueue_function[k_res_add](
                self.a.buf.unsafe_ptr(), ox1, oy, ox2, TPF * D,
                grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)

        var xn2 = ada_rms_op(self.ctx, self.w.buf, self.a, self.kn,
                             ox2, otab + 3 * D, otab + 4 * D, TPF, D)
        var oh = mm_op(self.ctx, self.w.buf, self.a, xn2, TPF, D, lo.fc1, FF)
        self.ctx.enqueue_function(
            self.kn.silu.bitcast[
                type_of(self.ctx.compile_function[k_silu]())]()[],
            self.a.buf.unsafe_ptr(), oh, TPF * FF,
            grid_dim=ceildiv(TPF * FF, BLOCK), block_dim=BLOCK)
        var om = mm_op(self.ctx, self.w.buf, self.a, oh, TPF, FF, lo.fc2, D)
        self.ctx.enqueue_function(
            self.kn.gateres.bitcast[
                type_of(self.ctx.compile_function[k_ada_gate_res]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            om, otab + 5 * D, ox2, xout, TPF, D,
            grid_dim=ceildiv(TPF * D, BLOCK), block_dim=BLOCK)
        self.a.reset(mark)

    def euler_step(mut self, dsig: Float32) raises:
        self.ctx.enqueue_function(
            self.kn.axpy.bitcast[
                type_of(self.ctx.compile_function[k_axpy]())]()[],
            self.a.buf.unsafe_ptr(), self.olat, self.ovel, dsig, LAT,
            grid_dim=ceildiv(LAT, BLOCK), block_dim=BLOCK)


def max_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    var mx = Float32(0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


def main() raises:
    from std.python import Python
    from std.sys import argv

    # --teacher: persist the oracle's denoised latent in each cache pass so
    # per-frame error can't compound through the KV cache. Distinguishes
    # autoregressive drift (inherent, chaotic) from porting bugs.
    var teacher = False
    for i in range(len(argv())):
        if argv()[i] == "--teacher":
            teacher = True

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    if teacher:
        print("teacher-forcing mode: cache passes use oracle latents")

    var st = SafeTensors(MODEL_DIR)
    var names = weight_names()
    var total = PAD
    for i in range(len(names)):
        total += st.get(names[i]).numel()
    print("weights:", Float64(total * 2) / 1e9, "GB bf16")
    var w = Weights(ctx, total)
    var t0 = perf_counter_ns()
    w.upload_all(st, names)
    print("upload done in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var model = Waypoint(ctx, w^)

    var json = Python.import_module("json")
    var bi = Python.import_module("builtins")
    var man = json.loads(bi.open(ORACLE_DIR + "/manifest.json", "r").read())
    var nf = atol(String(man["nf"]))
    var dsig = List[Float32]()
    for i in range(len(man["dsigmas_bf16"])):
        dsig.append(Float32(atof(String(man["dsigmas_bf16"][i]))))

    var vae = WPVae(ctx, model.w, model.a)
    var ln10 = Float64(2.302585092994046)

    var pass_ok = True
    for f in range(nf):
        var tf0 = perf_counter_ns()
        var noise = read_f32_bin(ORACLE_DIR + "/noise_f" + String(f) + ".bin",
                                 LAT)
        model.upload_latent(noise)
        var ctl = man["controls"][f]
        var btns = List[Int]()
        for i in range(len(ctl["buttons"])):
            btns.append(atol(String(ctl["buttons"][i])))
        model.set_controls(
            Float32(atof(String(ctl["mouse"][0]))),
            Float32(atof(String(ctl["mouse"][1]))),
            btns, atol(String(ctl["scroll"])))

        for i in range(len(dsig)):
            model.forward_pass(i, True, False)
            if f == 0 and i == 0:
                var got = model.read_acts(model.ovel, LAT)
                var want = read_f32_bin(ORACLE_DIR + "/v0_f0.bin", LAT)
                # torch runs every op in bf16, Mojo accumulates f32: measured
                # bf16-vs-f32 drift on a 8-layer torch A/B is ~0.1 max, so
                # 0.25 is the bug-vs-precision line for 24 layers.
                var mx = max_abs_diff(got, want)
                print("gate 1 (pass-0 velocity) max_abs_diff =", mx)
                if mx > Float32(0.25):
                    pass_ok = False
            model.euler_step(dsig[i])

        var got_lat = model.read_acts(model.olat, LAT)
        var want_lat = read_f32_bin(ORACLE_DIR + "/lat_f" + String(f) + ".bin",
                                    LAT)
        print("frame", f, "latent max_abs_diff =",
              max_abs_diff(got_lat, want_lat))

        if teacher:
            model.upload_latent(want_lat)
        model.forward_pass(NSIG - 1, False, True)   # cache pass, sigma 0
        model.frame += 1
        print("frame", f, "done in",
              Float64(perf_counter_ns() - tf0) / 1e9, "s (5 passes)")

        # gate 3: taehv decode vs oracle RGB (skipped if oracle hasn't
        # reached its VAE phase yet — rgb bins are written last)
        var want_rgb: List[UInt8]
        try:
            var fr = open(ORACLE_DIR + "/rgb_f" + String(f) + ".bin", "r")
            want_rgb = fr.read_bytes(4 * FH * FW * 3)
            fr.close()
        except:
            print("frame", f, "vae gate skipped (no oracle rgb yet)")
            continue
        var tv0 = perf_counter_ns()
        vae.decode(ctx, model.w, model.a, model.kn, model.olat)
        var se = Float64(0)
        for k in range(4):
            var got = model.read_acts(vae.oframes[k], 3 * FH * FW)
            for c in range(3):
                for p in range(FH * FW):
                    var g = Float64(got[c * FH * FW + p]) * 255.0
                    var gi = Float64(Int(g + 0.5))
                    var wv = Float64(
                        Int(want_rgb[(k * FH * FW + p) * 3 + c]))
                    se += (gi - wv) * (gi - wv)
        var mse = se / Float64(4 * 3 * FH * FW)
        var psnr = Float64(99)
        if mse > 0:
            psnr = 10.0 * log(255.0 * 255.0 / mse) / ln10
        print("frame", f, "vae psnr =", psnr, "dB  (decode",
              Float64(perf_counter_ns() - tv0) / 1e9, "s)")
        if psnr < 35:
            pass_ok = False

    print("ALL PASS" if pass_ok else "CHECK FAILED")
