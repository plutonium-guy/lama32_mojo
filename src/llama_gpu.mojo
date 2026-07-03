"""Llama-3.2-1B-Instruct on Metal (M4): bf16 weights resident, f32 activations.

Model-specific shell over llama_common.mojo (kernels + layer runner shared
with the 8B streaming port in llama31_mojo). 1B fits resident: weights upload
once into a ~2.5 GB u16 arena. Tied lm_head = embed_tokens.

Self-test: oracle prompt -> logits vs assets/oracle/last_logits.bin, greedy 791.
Run: pixi run test  (or: pixi run mojo run -I src src/llama_gpu.mojo)
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from safetensors import SafeTensors
from resident import Weights
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, TG, PAD,
    k_embed_gather, k_export,
    llama3_inv_freq, mm_op, rmsnorm_op, run_layer,
    read_f32_bin, read_argmax_buf, read_logits_buf,
)

comptime VOCAB = 128256
comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/assets/model"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/assets/oracle"


def make_config() -> Config:
    return Config(2048, 16, 32, 8, 64, 8192, VOCAB,
                  Float32(1e-5), 500000.0, 32.0, 1.0, 4.0, 8192.0)


def weight_names() -> List[String]:
    var names = List[String]()
    names.append("model.embed_tokens.weight")
    names.append("model.norm.weight")
    for L in range(16):
        var lp = "model.layers." + String(L) + "."
        names.append(lp + "input_layernorm.weight")
        names.append(lp + "post_attention_layernorm.weight")
        names.append(lp + "self_attn.q_proj.weight")
        names.append(lp + "self_attn.k_proj.weight")
        names.append(lp + "self_attn.v_proj.weight")
        names.append(lp + "self_attn.o_proj.weight")
        names.append(lp + "mlp.gate_proj.weight")
        names.append(lp + "mlp.up_proj.weight")
        names.append(lp + "mlp.down_proj.weight")
    return names^


struct Llama:
    var ctx: DeviceContext
    var cfg: Config
    var w: Weights
    var a: Acts
    var lgbuf: DeviceBuffer[DType.float32]
    var idbuf: DeviceBuffer[DType.int32]
    var lo: List[LayerOffs]
    var kc: List[Int]
    var vc: List[Int]
    var oinv: Int
    var maxlen: Int
    var n_cached: Int

    def __init__(out self, ctx: DeviceContext, var w: Weights, maxlen: Int) raises:
        self.ctx = ctx
        self.cfg = make_config()
        self.w = w^
        var kvd = self.cfg.kv_dim()
        var cache = self.cfg.layers * 2 * maxlen * kvd
        self.a = Acts(ctx, PAD + self.cfg.half() + cache + 32 * 1024 * 1024)
        self.lgbuf = ctx.enqueue_create_buffer[DType.float32](PAD + VOCAB)
        self.idbuf = ctx.enqueue_create_buffer[DType.int32](maxlen)
        self.lo = List[LayerOffs]()
        self.kc = List[Int]()
        self.vc = List[Int]()
        self.maxlen = maxlen
        self.n_cached = 0
        self.oinv = self.a.alloc(self.cfg.half())
        var inv = llama3_inv_freq(self.cfg)
        with self.a.buf.map_to_host() as h:
            for d in range(self.cfg.half()):
                h[self.oinv + d] = inv[d]
        for L in range(self.cfg.layers):
            self.kc.append(self.a.alloc(maxlen * kvd))
            self.vc.append(self.a.alloc(maxlen * kvd))
            var lp = "model.layers." + String(L) + "."
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
                -1, -1))

    def embed(mut self, ids: List[Int]) raises -> Int:
        var s = len(ids)
        var H = self.cfg.hidden
        var ox = self.a.alloc(s * H)
        var oemb = self.w.o("model.embed_tokens.weight")
        with self.idbuf.map_to_host() as h:
            for i in range(s):
                h[i] = Int32(ids[i])
        self.ctx.enqueue_function(
            self.a.kn.embg.bitcast[
                type_of(self.ctx.compile_function[k_embed_gather]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            oemb, self.idbuf.unsafe_ptr(), ox, s, H,
            grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
        return ox

    def _run_forward(mut self, ids: List[Int]) raises:
        """Embed + layers + norm + lm_head -> logits in lgbuf."""
        var s = len(ids)
        if self.n_cached + s > self.maxlen:
            raise Error("KV cache full")
        var scratch = self.a.mark()
        var oh = self.embed(ids)
        for L in range(self.cfg.layers):
            oh = run_layer(self.ctx, self.w.buf, self.a, self.cfg, self.lo[L],
                           oh, s, self.n_cached, self.kc[L], self.vc[L], self.oinv)
            if s > 4:
                self.ctx.synchronize()
        self.n_cached += s
        var onrm = rmsnorm_op(self.ctx, self.w.buf, self.a,
                              oh + (s - 1) * self.cfg.hidden, 1,
                              self.w.o("model.norm.weight"),
                              self.cfg.hidden, self.cfg.eps)
        var olg = mm_op(self.ctx, self.w.buf, self.a, onrm, 1, self.cfg.hidden,
                        self.w.o("model.embed_tokens.weight"), VOCAB)
        self.ctx.enqueue_function(
            self.a.kn.exp.bitcast[
                type_of(self.ctx.compile_function[k_export]())]()[],
            self.a.buf.unsafe_ptr(), self.lgbuf.unsafe_ptr(), olg, PAD, VOCAB,
            grid_dim=ceildiv(VOCAB, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        self.a.reset(scratch)

    def forward(mut self, ids: List[Int]) raises -> List[Float32]:
        self._run_forward(ids)
        return read_logits_buf(self.lgbuf, VOCAB)

    def forward_argmax(mut self, ids: List[Int]) raises -> Int:
        """Greedy decode path — skips full-vocab host List allocation."""
        self._run_forward(ids)
        return read_argmax_buf(self.lgbuf, VOCAB)


def load_llama(ctx: DeviceContext, maxlen: Int) raises -> Llama:
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
    return Llama(ctx, w^, maxlen)


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_llama(ctx, 64)

    var ids: List[Int] = [128000, 128006, 882, 128007, 271, 3923, 374, 279,
                          6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271]
    var t0 = perf_counter_ns()
    var logits = model.forward(ids)
    print("prefill+logits:", Float64(perf_counter_ns() - t0) / 1e9, "s")
    t0 = perf_counter_ns()
    for _ in range(16):
        var step: List[Int] = [791]
        _ = model.forward_argmax(step)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("decode:", 16.0 / dt, "tok/s (", dt / 16.0 * 1000.0, "ms/tok )")

    var want = read_f32_bin(ORACLE_DIR + "/last_logits.bin", VOCAB)
    var mx = Float32(0)
    var gi = 0
    var wi = 0
    for i in range(VOCAB):
        var d = logits[i] - want[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
        if logits[i] > logits[gi]:
            gi = i
        if want[i] > want[wi]:
            wi = i
    print("logits max_abs_diff =", mx)
    print("greedy id: got", gi, "want", wi)
    print("ALL PASS" if (gi == wi and mx < Float32(2e-2)) else "FAILED")
