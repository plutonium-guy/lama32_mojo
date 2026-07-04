"""Qwen3-0.6B on Metal with INT8 group-quantized weights (W8A32).

Same model shell as qwen_gpu.mojo but reading the packed q8 arena
(assets/model-q8, from scripts/quantize.py): int8 weights + f16 group
scales, ~2x less weight traffic per token. Norms stay bf16.

Run: pixi run test-q8   (validates vs the bf16 oracle: relaxed logits
tolerance, greedy-token match, decode tok/s)
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from safetensors import SafeTensors
from resident import Weights
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, PAD,
    k_export, rope_inv_freq, rmsnorm_op, read_f32_bin,
    read_argmax_gpu, read_logits_buf, argmax_op,
)
from llama_q8 import Q8Kernels, run_layer_q8, mm_q8_op, k_embed_gather_q8
from sample import argmax

comptime VOCAB = 151936
comptime LAYERS = 28
comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/qwen3_mojo/assets/model-q8"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/qwen3_mojo/assets/oracle"


def make_config() -> Config:
    return Config(1024, LAYERS, 16, 8, 128, 3072, VOCAB,
                  Float32(1e-6), 1000000.0, 1.0, 1.0, 4.0, 8192.0)


def weight_names() -> List[String]:
    var names = List[String]()
    names.append("model.embed_tokens.weight")
    names.append("model.norm.weight")
    for L in range(LAYERS):
        var lp = "model.layers." + String(L) + "."
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
    return names^


struct QwenQ8:
    var ctx: DeviceContext
    var cfg: Config
    var w: Weights
    var a: Acts
    var qk: Q8Kernels
    var lgbuf: DeviceBuffer[DType.float32]
    var argbuf: DeviceBuffer[DType.int32]
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
        self.qk = Q8Kernels(ctx)
        self.lgbuf = ctx.enqueue_create_buffer[DType.float32](PAD + VOCAB)
        self.argbuf = ctx.enqueue_create_buffer[DType.int32](1)
        self.idbuf = ctx.enqueue_create_buffer[DType.int32](maxlen)
        self.lo = List[LayerOffs]()
        self.kc = List[Int]()
        self.vc = List[Int]()
        self.maxlen = maxlen
        self.n_cached = 0
        self.oinv = self.a.alloc(self.cfg.half())
        var inv = rope_inv_freq(self.cfg)
        with self.a.buf.map_to_host() as h:
            for d in range(self.cfg.half()):
                h[self.oinv + d] = inv[d]
        for _ in range(self.cfg.layers):
            self.kc.append(self.a.alloc(maxlen * kvd))
            self.vc.append(self.a.alloc(maxlen * kvd))
        for L in range(self.cfg.layers):
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
                self.w.o(lp + "self_attn.q_norm.weight"),
                self.w.o(lp + "self_attn.k_norm.weight")))

    def embed(mut self, ids: List[Int]) raises -> Int:
        var s = len(ids)
        var H = self.cfg.hidden
        var ox = self.a.alloc(s * H)
        var oemb = self.w.o("model.embed_tokens.weight")
        with self.idbuf.map_to_host() as h:
            for i in range(s):
                h[i] = Int32(ids[i])
        self.ctx.enqueue_function(
            self.qk.embg.bitcast[
                type_of(self.ctx.compile_function[k_embed_gather_q8]())]()[],
            self.w.buf.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            oemb, self.idbuf.unsafe_ptr(), ox, s, H,
            grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
        return ox

    def _run_forward(mut self, ids: List[Int]) raises:
        var s = len(ids)
        if self.n_cached + s > self.maxlen:
            raise Error("KV cache full")
        var mark = self.a.mark()
        var oh = self.embed(ids)
        for L in range(self.cfg.layers):
            oh = run_layer_q8(self.ctx, self.w.buf, self.a, self.qk,
                              self.cfg, self.lo[L], oh, s, self.n_cached,
                              self.kc[L], self.vc[L], self.oinv)
            if s > 4 and (L & 3) == 3:
                self.ctx.synchronize()
        self.n_cached += s
        var onrm = rmsnorm_op(self.ctx, self.w.buf, self.a,
                              oh + (s - 1) * self.cfg.hidden, 1,
                              self.w.o("model.norm.weight"),
                              self.cfg.hidden, self.cfg.eps)
        var olg = mm_q8_op(self.ctx, self.w.buf, self.a, self.qk, onrm, 1,
                           self.cfg.hidden,
                           self.w.o("model.embed_tokens.weight"), VOCAB)
        self.ctx.enqueue_function(
            self.a.kn.exp.bitcast[
                type_of(self.ctx.compile_function[k_export]())]()[],
            self.a.buf.unsafe_ptr(), self.lgbuf.unsafe_ptr(), olg, PAD, VOCAB,
            grid_dim=ceildiv(VOCAB, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        self.a.reset(mark)

    def forward(mut self, ids: List[Int]) raises -> List[Float32]:
        self._run_forward(ids)
        return read_logits_buf(self.lgbuf, VOCAB)

    def forward_argmax(mut self, ids: List[Int]) raises -> Int:
        self._run_forward(ids)
        argmax_op(self.ctx, self.a, self.lgbuf, self.argbuf, PAD, VOCAB)
        self.ctx.synchronize()
        return read_argmax_gpu(self.argbuf)


def load_qwen_q8(ctx: DeviceContext, maxlen: Int) raises -> QwenQ8:
    var st = SafeTensors(MODEL_DIR)
    var names = weight_names()
    var total = PAD
    for i in range(len(names)):
        total += st.get(names[i]).numel()
    print("packed elems:", total, "(", Float64(total * 2) / 1e9, "GB q8 )")
    var w = Weights(ctx, total)
    print("uploading weights...")
    var t0 = perf_counter_ns()
    w.upload_all(st, names)
    print("upload done in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    return QwenQ8(ctx, w^, maxlen)


def main() raises:
    from std.python import Python

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_qwen_q8(ctx, 128)

    var json = Python.import_module("json")
    var bi = Python.import_module("builtins")
    var manifest = json.loads(
        bi.open(ORACLE_DIR + "/manifest.json", "r").read())
    var ids = List[Int]()
    var pyids = manifest["ids"]
    for i in range(len(pyids)):
        ids.append(atol(String(pyids[i])))

    var t0 = perf_counter_ns()
    var logits = model.forward(ids)
    print("prefill+logits:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    var want = read_f32_bin(ORACLE_DIR + "/last_logits.bin", VOCAB)
    var exp = argmax(want)
    t0 = perf_counter_ns()
    comptime STEPS = 64
    var nxt = exp
    for _ in range(STEPS):
        var step: List[Int] = [nxt]
        nxt = model.forward_argmax(step)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("decode:", Float64(STEPS) / dt, "tok/s (", dt / STEPS * 1000.0,
          "ms/tok )")

    var mx = Float32(0)
    var got = argmax(logits)
    for i in range(VOCAB):
        var d = logits[i] - want[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    # int8 group-64 drift vs the bf16 oracle is ~0.7 on logits spanning ±25;
    # the meaningful gate (QUANTIZATION.md) is the greedy-token match.
    print("logits max_abs_diff =", mx, "(bf16 oracle; q8 tolerance 1.0)")
    print("greedy id: got", got, "want", exp)
    print("ALL PASS" if (got == exp and mx < Float32(1.0)) else "FAILED")
