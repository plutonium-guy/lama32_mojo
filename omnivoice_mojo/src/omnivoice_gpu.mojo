"""OmniVoice diffusion-LM on Metal: Qwen3-0.6B backbone, bidirectional
attention, audio-token adapters, 32-step iterative unmasking with CFG.

Weights: assets/mojo/model.safetensors (bf16, from scripts/prepare_model.py).
The backbone matches qwen3_mojo shapes exactly; the difference is no KV cache
(masked-diffusion recomputes the full sequence every step, non-causal).

Run `pixi run test` to verify against the Python oracle
(scripts/gen_oracle.py must have been run first).
"""

from std.math import ceildiv
from std.memory import memcpy
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from safetensors import SafeTensors
from resident import Weights
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, PAD, TG,
    rope_inv_freq, rmsnorm_op, read_f32_bin, mm_op,
    k_rmsnorm2_w, k_res_add_rmsnorm,
)
from ov_common import (
    OvKernels, run_layer_bidir,
    k_ov_embed, k_ov_embed_pair, k_cfg_predict, k_copy, k_export_slice,
)
from sampler import MASK_ID, unmask_schedule, select_and_fill

comptime LAYERS = 28
comptime NCB = 8                        # audio codebooks
comptime AVOCAB = 1025                  # audio vocab incl. MASK (1024)
comptime HEADS_N = NCB * AVOCAB         # 8200
comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/mojo"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/oracle"


def make_config() -> Config:
    return Config(1024, LAYERS, 16, 8, 128, 3072, 151676,
                  Float32(1e-6), 1000000.0, 1.0, 1.0, 4.0, 8192.0)


def weight_names() -> List[String]:
    """Upload order matters for the codec RVQ tensors: the 8 codebook embeds,
    project_out weights, and biases must each be contiguous (see k_rvq_decode).
    """
    var names = List[String]()
    names.append("llm.embed_tokens.weight")
    names.append("llm.norm.weight")
    for L in range(LAYERS):
        var lp = "llm.layers." + String(L) + "."
        names.append(lp + "input_layernorm.weight")
        names.append(lp + "post_attention_layernorm.weight")
        names.append(lp + "self_attn.q_norm.weight")
        names.append(lp + "self_attn.k_norm.weight")
        names.append(lp + "self_attn.q_proj.weight")
        names.append(lp + "self_attn.k_proj.weight")
        names.append(lp + "self_attn.v_proj.weight")
        names.append(lp + "self_attn.o_proj.weight")
        names.append(lp + "mlp.gate_proj.weight")
        names.append(lp + "mlp.up_proj.weight")
        names.append(lp + "mlp.down_proj.weight")
    names.append("audio_embeddings.weight")
    names.append("audio_heads.weight")
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".codebook.embed")
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".project_out.weight")
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".project_out.bias")
    names.append("codec.fc2.weight")
    names.append("codec.fc2.bias")
    # DAC decoder: conv1, 5 upsampling blocks, final snake + conv2
    names.append("codec.acoustic_decoder.conv1.weight")
    names.append("codec.acoustic_decoder.conv1.bias")
    for b in range(5):
        var bp = "codec.acoustic_decoder.block." + String(b) + "."
        names.append(bp + "snake1.alpha")
        names.append(bp + "conv_t1.weight")
        names.append(bp + "conv_t1.bias")
        for r in range(3):
            var rp = bp + "res_unit" + String(r + 1) + "."
            names.append(rp + "snake1.alpha")
            names.append(rp + "conv1.weight")
            names.append(rp + "conv1.bias")
            names.append(rp + "snake2.alpha")
            names.append(rp + "conv2.weight")
            names.append(rp + "conv2.bias")
    names.append("codec.acoustic_decoder.snake1.alpha")
    names.append("codec.acoustic_decoder.conv2.weight")
    names.append("codec.acoustic_decoder.conv2.bias")
    return names^


@fieldwise_init
struct GenConfig(Copyable, Movable):
    var num_step: Int
    var guidance: Float32
    var t_shift: Float64
    var layer_penalty: Float32
    var position_temp: Float32


def default_gen_config() -> GenConfig:
    return GenConfig(32, 2.0, 0.1, 5.0, 5.0)


struct OmniVoice:
    var ctx: DeviceContext
    var cfg: Config
    var w: Weights
    var a: Acts
    var kn: OvKernels
    var lo: List[LayerOffs]
    var oinv: Int
    var idc: DeviceBuffer[DType.int32]   # cond ids (8, L)
    var idu: DeviceBuffer[DType.int32]   # uncond ids (8, T)
    var idu_pad: DeviceBuffer[DType.int32]  # uncond padded to (8, L)
    var pc: DeviceBuffer[DType.float32]  # (pred, conf) per (codebook, frame)
    var dbg: DeviceBuffer[DType.float32]  # verification readback
    var maxlen: Int

    def __init__(out self, ctx: DeviceContext, var w: Weights, maxlen: Int) raises:
        self.ctx = ctx
        self.cfg = make_config()
        self.w = w^
        # per-layer scratch: cond scores 16*(P+T)^2 + uncond 16*T^2 — worst
        # case ~32*maxlen^2 when the text prefix is short next to T
        # (overflowed the old 16x sizing on long spoken answers); plus
        # ~14e3*L floats; hidden slots and two (T, 8200) logit blocks
        # persist across the per-layer resets.
        var cap = PAD + 36 * maxlen * maxlen + 20000 * maxlen + 8 * 1024 * 1024
        self.a = Acts(ctx, cap)
        self.kn = OvKernels(ctx)
        self.lo = List[LayerOffs]()
        for L in range(LAYERS):
            var lp = "llm.layers." + String(L) + "."
            self.lo.append(LayerOffs(
                self.w.o(lp + "self_attn.q_proj.weight"),
                self.w.o(lp + "self_attn.k_proj.weight"),
                self.w.o(lp + "self_attn.v_proj.weight"),
                self.w.o(lp + "self_attn.o_proj.weight"),
                self.w.o(lp + "mlp.gate_proj.weight"),
                self.w.o(lp + "mlp.up_proj.weight"),
                self.w.o(lp + "mlp.down_proj.weight"),
                self.w.o(lp + "input_layernorm.weight"),
                self.w.o(lp + "post_attention_layernorm.weight"),
                self.w.o(lp + "self_attn.q_norm.weight"),
                self.w.o(lp + "self_attn.k_norm.weight")))
        self.oinv = self.a.alloc(self.cfg.half())
        var inv = rope_inv_freq(self.cfg)
        with self.a.buf.map_to_host() as h:
            for d in range(self.cfg.half()):
                h[self.oinv + d] = inv[d]
        self.idc = ctx.enqueue_create_buffer[DType.int32](NCB * maxlen)
        self.idu = ctx.enqueue_create_buffer[DType.int32](NCB * maxlen)
        self.idu_pad = ctx.enqueue_create_buffer[DType.int32](NCB * maxlen)
        self.pc = ctx.enqueue_create_buffer[DType.float32](2 * NCB * maxlen)
        self.dbg = ctx.enqueue_create_buffer[DType.float32](maxlen * HEADS_N)
        self.maxlen = maxlen

    def forward_ids(mut self, cond: Bool, L: Int,
                    astart: Int) raises -> Int:
        """Full bidirectional forward over the (8, L) ids in the cond
        (or uncond) id buffer. Returns arena offset of the final-norm
        hidden states (L, hidden)."""
        var H = self.cfg.hidden
        var ids = self.idc.unsafe_ptr() if cond else self.idu.unsafe_ptr()
        var ox = self.a.alloc(L * H)
        self.ctx.enqueue_function(
            self.kn.embed.bitcast[
                type_of(self.ctx.compile_function[k_ov_embed]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            ids,
            self.w.o("llm.embed_tokens.weight"),
            self.w.o("audio_embeddings.weight"),
            ox, L, H, astart, NCB, AVOCAB,
            grid_dim=ceildiv(L * H, BLOCK), block_dim=BLOCK)
        for L_i in range(self.cfg.layers):
            var mark = self.a.mark()
            var oh = run_layer_bidir(self.ctx, self.w.buf, self.a, self.kn,
                                     self.cfg, self.lo[L_i], ox, L, self.oinv)
            self.ctx.enqueue_function(
                self.kn.cpy.bitcast[
                    type_of(self.ctx.compile_function[k_copy]())]()[],
                self.a.buf.unsafe_ptr(), oh, ox, L * H,
                grid_dim=ceildiv(L * H, BLOCK), block_dim=BLOCK)
            self.a.reset(mark)
            if (L_i & 3) == 3:
                self.ctx.synchronize()
        return rmsnorm_op(self.ctx, self.w.buf, self.a, ox, L,
                          self.w.o("llm.norm.weight"), H, self.cfg.eps)

    def upload_ids(mut self, prefix8: List[Int], tokens: List[Int],
                   T: Int) raises -> Int:
        """Fill cond ids (per-codebook prefix rows + current tokens) and
        uncond ids (tokens only). prefix8 is (8, P) row-major — rows are
        identical for text but differ when a voice-clone reference's audio
        tokens are part of the prefix. Returns cond length L."""
        var P = len(prefix8) // NCB
        var L = P + T
        with self.idc.map_to_host() as h:
            for c in range(NCB):
                for t in range(P):
                    h[c * L + t] = Int32(prefix8[c * P + t])
                for t in range(T):
                    h[c * L + P + t] = Int32(tokens[c * T + t])
        with self.idu.map_to_host() as h:
            for i in range(NCB * T):
                h[i] = Int32(tokens[i])
        with self.idu_pad.map_to_host() as h:
            for c in range(NCB):
                for t in range(P):
                    h[c * L + t] = Int32(MASK_ID)
                for t in range(T):
                    h[c * L + P + t] = Int32(tokens[c * T + t])
        return L

    def forward_ids_batched(mut self, L: Int, astart: Int) raises -> Int:
        """Batched cond+uncond backbone: one 2L forward with block-diagonal attn."""
        var H = self.cfg.hidden
        var s = 2 * L
        var ox = self.a.alloc(s * H)
        self.ctx.enqueue_function(
            self.kn.embedpair.bitcast[
                type_of(self.ctx.compile_function[k_ov_embed_pair]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            self.idc.unsafe_ptr(), self.idu_pad.unsafe_ptr(),
            self.w.o("llm.embed_tokens.weight"),
            self.w.o("audio_embeddings.weight"),
            ox, L, H, astart, astart, NCB, AVOCAB,
            grid_dim=ceildiv(2 * L * H, BLOCK), block_dim=BLOCK)
        for L_i in range(self.cfg.layers):
            var mark = self.a.mark()
            var oh = run_layer_bidir(self.ctx, self.w.buf, self.a, self.kn,
                                     self.cfg, self.lo[L_i], ox, s, self.oinv, L)
            self.ctx.enqueue_function(
                self.kn.cpy.bitcast[
                    type_of(self.ctx.compile_function[k_copy]())]()[],
                self.a.buf.unsafe_ptr(), oh, ox, s * H,
                grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
            self.a.reset(mark)
            if (L_i & 3) == 3:
                self.ctx.synchronize()
        return rmsnorm_op(self.ctx, self.w.buf, self.a, ox, s,
                          self.w.o("llm.norm.weight"), H, self.cfg.eps)

    def heads(mut self, oh: Int, rows: Int) raises -> Int:
        return mm_op(self.ctx, self.w.buf, self.a, oh, rows,
                     self.cfg.hidden, self.w.o("audio_heads.weight"), HEADS_N)

    def step(mut self, L: Int, T: Int, astart: Int, gc: GenConfig,
             mut preds: List[Float32], mut confs: List[Float32]) raises:
        """One diffusion step: batched cond+uncond forward, CFG predict."""
        var mark = self.a.mark()
        var H = self.cfg.hidden
        var onrm = self.forward_ids_batched(L, astart)
        var ocl = self.heads(onrm + (L - T) * H, T)
        var oul = self.heads(onrm + L * H + (L - T) * H, T)
        self.ctx.enqueue_function(
            self.kn.cfg.bitcast[
                type_of(self.ctx.compile_function[k_cfg_predict]())]()[],
            self.a.buf.unsafe_ptr(), self.pc.unsafe_ptr(),
            ocl, oul, T, NCB, AVOCAB, MASK_ID, gc.guidance,
            grid_dim=ceildiv(NCB * T, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        with self.pc.map_to_host() as h:
            for i in range(NCB * T):
                preds[i] = h[2 * i]
                confs[i] = h[2 * i + 1]
        self.a.reset(mark)

    def generate(mut self, prefix8: List[Int], astart: Int, T: Int,
                 gc: GenConfig, verbose: Bool = False) raises -> List[Int]:
        """Iterative unmasking; returns (8, T) audio tokens row-major.
        astart = index of the first audio position in the cond sequence
        (start of ref-audio tokens for cloning, else start of the target)."""
        var tokens = List[Int](length=NCB * T, fill=MASK_ID)
        var sched = unmask_schedule(T, NCB, gc.num_step, gc.t_shift)
        var preds = List[Float32](length=NCB * T, fill=0)
        var confs = List[Float32](length=NCB * T, fill=0)
        for step in range(gc.num_step):
            var L = self.upload_ids(prefix8, tokens, T)
            self.step(L, T, astart, gc, preds, confs)
            select_and_fill(tokens, preds, confs, sched[step], NCB, T,
                            gc.layer_penalty, gc.position_temp)
            if verbose:
                print("  step", step + 1, "/", gc.num_step, ": revealed",
                      sched[step])
        return tokens^

    def export_dbg(mut self, osrc: Int, n: Int) raises -> List[Float32]:
        self.ctx.enqueue_function(
            self.kn.exp.bitcast[
                type_of(self.ctx.compile_function[k_export_slice]())]()[],
            self.a.buf.unsafe_ptr(), self.dbg.unsafe_ptr(), osrc, 0, n,
            grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        var out = List[Float32](length=n, fill=0)
        with self.dbg.map_to_host() as h:
            memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
        return out^


def load_omnivoice(ctx: DeviceContext, maxlen: Int) raises -> OmniVoice:
    var st = SafeTensors(MODEL_DIR)
    var names = weight_names()
    var total = PAD
    for i in range(len(names)):
        total += st.get(names[i]).numel()
    print("params:", total, "(", Float64(total * 2) / 1e9, "GB bf16 )")
    var w = Weights(ctx, total)
    print("uploading weights...")
    var t0 = perf_counter_ns()
    w.upload_all(st, names)
    print("upload done in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    return OmniVoice(ctx, w^, maxlen)


def read_i32_bin(path: String, count: Int) raises -> List[Int]:
    var f = open(path, "r")
    var raw = f.read_bytes(count * 4)
    f.close()
    var out = List[Int](capacity=count)
    for i in range(count):
        var bits = (UInt32(raw[4 * i]) | (UInt32(raw[4 * i + 1]) << 8)
                    | (UInt32(raw[4 * i + 2]) << 16)
                    | (UInt32(raw[4 * i + 3]) << 24))
        out.append(Int(Int32(bits)))
    return out^


def _cfg_argmax_row(cl: List[Float32], ul: List[Float32], base: Int,
                    guidance: Float32) -> Int:
    """Greedy CFG argmax over one (codebook, frame) vocab row on the host.
    argmax of lc + g*(lc-lu) is invariant to the softmax normalizers only
    through lc/lu, so compute them properly."""
    from std.math import exp, log
    var cmax = cl[base]
    var umax = ul[base]
    for v in range(1, AVOCAB):
        if cl[base + v] > cmax:
            cmax = cl[base + v]
        if ul[base + v] > umax:
            umax = ul[base + v]
    var cse = Float32(0)
    var use = Float32(0)
    for v in range(AVOCAB):
        cse += exp(cl[base + v] - cmax)
        use += exp(ul[base + v] - umax)
    var clse = cmax + log(cse)
    var ulse = umax + log(use)
    var best = 0
    var bv = Float32(-3.0e38)
    for v in range(AVOCAB):
        if v == MASK_ID:
            continue
        var lc = cl[base + v] - clse
        var lu = ul[base + v] - ulse
        var comb = lc + guidance * (lc - lu)
        if comb > bv:
            bv = comb
            best = v
    return best


def maxdiff(got: List[Float32], want: List[Float32]) -> Float32:
    var mx = Float32(0)
    for i in range(len(want)):
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


def main() raises:
    from std.python import Python

    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var json = Python.import_module("json")
    var bi = Python.import_module("builtins")
    var manifest = json.loads(bi.open(ORACLE_DIR + "/manifest.json", "r").read())
    var T = atol(String(manifest["target_len"]))
    var L = atol(String(manifest["seq_len"]))
    var P = L - T
    var prefix8 = List[Int](capacity=NCB * P)
    var pyids = manifest["prefix_ids"]
    for _ in range(NCB):
        for i in range(len(pyids)):
            prefix8.append(atol(String(pyids[i])))
    print("seq len", L, "target", T, "prefix", P)

    var model = load_omnivoice(ctx, L + 32)
    var gc = default_gen_config()

    # --- step-0 checks: embeddings + cond/uncond logits vs oracle ---------
    var tokens = List[Int](length=NCB * T, fill=MASK_ID)
    _ = model.upload_ids(prefix8, tokens, T)
    var mark = model.a.mark()

    var H = model.cfg.hidden
    var oxc = model.forward_ids(True, L, L - T)
    var ocl = model.heads(oxc + (L - T) * H, T)
    var got_cl = model.export_dbg(ocl, T * HEADS_N)
    var want_cl = read_f32_bin(ORACLE_DIR + "/step0_cond.bin", NCB * T * AVOCAB)
    # oracle layout (8, T, 1025); ours (T, 8, 1025)
    var mx = Float32(0)
    for c in range(NCB):
        for t in range(T):
            for v in range(AVOCAB):
                var d = got_cl[t * HEADS_N + c * AVOCAB + v] - want_cl[
                    (c * T + t) * AVOCAB + v]
                if d < 0:
                    d = -d
                if d > mx:
                    mx = d
    print("step0 cond logits max_abs_diff =", mx)

    var oxu = model.forward_ids(False, T, 0)
    var oul = model.heads(oxu, T)
    var got_ul = model.export_dbg(oul, T * HEADS_N)
    var want_ul = read_f32_bin(ORACLE_DIR + "/step0_uncond.bin", NCB * T * AVOCAB)
    var mxu = Float32(0)
    for c in range(NCB):
        for t in range(T):
            for v in range(AVOCAB):
                var d = got_ul[t * HEADS_N + c * AVOCAB + v] - want_ul[
                    (c * T + t) * AVOCAB + v]
                if d < 0:
                    d = -d
                if d > mxu:
                    mxu = d
    print("step0 uncond logits max_abs_diff =", mxu)

    # step-0 greedy CFG prediction agreement (mojo logits vs oracle logits):
    # separates bf16 forward noise from sampling-loop divergence.
    var agree = 0
    for c in range(NCB):
        for t in range(T):
            var bg = _cfg_argmax_row(got_cl, got_ul, t * HEADS_N + c * AVOCAB,
                                     gc.guidance)
            var bw = _cfg_argmax_row(want_cl, want_ul,
                                     (c * T + t) * AVOCAB, gc.guidance)
            if bg == bw:
                agree += 1
    print("step0 pred agreement:", agree, "/", NCB * T)
    model.a.reset(mark)

    # --- phase timings ----------------------------------------------------
    mark = model.a.mark()
    var tb = perf_counter_ns()
    var oxc2 = model.forward_ids(True, L, L - T)
    ctx.synchronize()
    print("  cond fwd:", Float64(perf_counter_ns() - tb) / 1e9, "s")
    tb = perf_counter_ns()
    var ocl2 = model.heads(oxc2 + (L - T) * H, T)
    ctx.synchronize()
    print("  heads:   ", Float64(perf_counter_ns() - tb) / 1e9, "s")
    tb = perf_counter_ns()
    var oxu2 = model.forward_ids(False, T, 0)
    ctx.synchronize()
    print("  uncond:  ", Float64(perf_counter_ns() - tb) / 1e9, "s")
    _ = oxu2
    _ = ocl2
    model.a.reset(mark)

    # --- full deterministic generation vs oracle tokens -------------------
    gc.position_temp = 0.0                 # oracle is the deterministic variant
    var t0 = perf_counter_ns()
    var got_tokens = model.generate(prefix8, L - T, T, gc)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("generate:", dt, "s (", dt / Float64(gc.num_step), "s/step )")

    var want_tokens = read_i32_bin(ORACLE_DIR + "/final_tokens.bin", NCB * T)
    var nmatch = 0
    for i in range(NCB * T):
        if got_tokens[i] == want_tokens[i]:
            nmatch += 1
    print("token match:", nmatch, "/", NCB * T,
          "(", Float64(nmatch) / Float64(NCB * T) * 100.0, "% )")
    # bf16-vs-f32 drift compounds through the iterative sampler, so exact
    # token match degrades without indicating a bug; step-0 agreement is the
    # meaningful correctness gate.
    print("PASS" if (mx < Float32(0.5) and agree * 100 >= NCB * T * 95)
          else "CHECK")
