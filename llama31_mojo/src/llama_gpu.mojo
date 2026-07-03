"""Meta-Llama-3.1-8B-Instruct-abliterated on Metal (M4), optimized streaming.

Shared kernels from llama32_mojo/llama_common.mojo. Weight path:
  disk (once per layer) -> GPU layer buffers (permanent) -> inference.

No host-side layer RAM cache (avoids holding 2x ~14 GB). lm_head + final norm
stay GPU-resident. Embeddings: one file handle, batched row reads, GPU bf16
decode into the acts arena (no full-arena host map). Untied lm_head; RoPE
factor 8. Needs ~16 GB unified memory after warmup.

Run: pixi run test   (imports shared code via -I ../src)
"""

from std.math import ceildiv
from std.time import sleep, perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy
from safetensors import SafeTensors
from llama_common import (
    Config, LayerOffs, Acts, BLOCK, TG, PAD,
    k_export, k_bf16_to_f32,
    llama3_inv_freq, mm_op, rmsnorm_op, run_layer, read_f32_bin,
)

comptime VOCAB = 128256
comptime HIDDEN = 4096
comptime INTER = 14336
comptime KVDIM = 8 * 128
comptime LAYERS = 32
comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/llama31_mojo/assets/model"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/llama31_mojo/assets/oracle"
comptime LAYER_U16 = (HIDDEN * HIDDEN + 2 * HIDDEN * KVDIM + HIDDEN * HIDDEN
                      + 2 * HIDDEN * INTER + INTER * HIDDEN + 2 * HIDDEN)
comptime RESIDENT_U16 = 4096 + VOCAB * HIDDEN
comptime EMBED_CAP = 2048


def make_config() -> Config:
    return Config(HIDDEN, LAYERS, 32, 8, 128, INTER, VOCAB,
                  Float32(1e-5), 500000.0, 8.0, 1.0, 4.0, 8192.0)


def fixed_layer_offs() -> LayerOffs:
    """Offsets for weights packed adjacently (matches run_layer s==1 fusion)."""
    var t = PAD
    var q = t
    t += HIDDEN * HIDDEN
    var k = t
    t += HIDDEN * KVDIM
    var v = t
    t += HIDDEN * KVDIM
    var o = t
    t += HIDDEN * HIDDEN
    var g = t
    t += HIDDEN * INTER
    var u = t
    t += HIDDEN * INTER
    var d = t
    t += INTER * HIDDEN
    var inn = t
    t += HIDDEN
    var pn = t
    return LayerOffs(q, k, v, o, g, u, d, inn, pn, -1, -1)


def layer_tensor_names(L: Int) -> List[String]:
    var lp = "model.layers." + String(L) + "."
    var names = List[String]()
    names.append(lp + "self_attn.q_proj.weight")
    names.append(lp + "self_attn.k_proj.weight")
    names.append(lp + "self_attn.v_proj.weight")
    names.append(lp + "self_attn.o_proj.weight")
    names.append(lp + "mlp.gate_proj.weight")
    names.append(lp + "mlp.up_proj.weight")
    names.append(lp + "mlp.down_proj.weight")
    names.append(lp + "input_layernorm.weight")
    names.append(lp + "post_attention_layernorm.weight")
    return names^


def read_tensor_bytes(st: SafeTensors, name: String) raises -> List[UInt8]:
    """One tensor's raw bf16 bytes; retries exFAT dropouts."""
    var info = st.get(name)
    var attempt = 0
    while True:
        try:
            var f = open(info.path, "r")
            _ = f.seek(UInt64(info.begin))
            var raw = f.read_bytes(info.nbytes())
            f.close()
            return raw^
        except e:
            attempt += 1
            if attempt >= 15:
                raise e.copy()
            print("  read failed (", e, "), retry", attempt)
            sleep(2.0)


def load_layer_to_gpu(st: SafeTensors, L: Int, buf: DeviceBuffer[DType.uint16]) raises:
    """Stream one packed layer shard-by-shard directly into a GPU buffer."""
    var names = layer_tensor_names(L)
    with buf.map_to_host() as h:
        var u16_off = PAD
        var i = 0
        while i < len(names):
            var path = st.get(names[i]).path
            var f = open(path, "r")
            while i < len(names) and st.get(names[i]).path == path:
                var info = st.get(names[i])
                _ = f.seek(UInt64(info.begin))
                var raw = f.read_bytes(info.nbytes())
                memcpy(dest=(h.unsafe_ptr() + u16_off).bitcast[UInt8](),
                       src=raw.unsafe_ptr(), count=len(raw))
                u16_off += info.numel()
                i += 1
            f.close()


struct GpuLayerCache:
    """Permanent GPU buffers — one packed layer each, loaded once from disk."""
    var bufs: List[DeviceBuffer[DType.uint16]]

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        self.bufs = List[DeviceBuffer[DType.uint16]]()
        for _ in range(n):
            self.bufs.append(
                ctx.enqueue_create_buffer[DType.uint16](PAD + LAYER_U16))

    def warm(mut self, st: SafeTensors) raises:
        print("loading layers disk -> GPU (~",
              Float64(LAYER_U16 * 2 * LAYERS) / 1e9, "GB )...")
        var t0 = perf_counter_ns()
        for L in range(LAYERS):
            load_layer_to_gpu(st, L, self.bufs[L])
            if L % 8 == 7:
                print("  layer", L + 1, "/", LAYERS)
        print("gpu warm in", Float64(perf_counter_ns() - t0) / 1e9, "s")


struct ResidentWeights:
    """Lm_head + final norm — uploaded once, never re-read from disk."""
    var buf: DeviceBuffer[DType.uint16]
    var norm: Int
    var lm: Int

    def __init__(out self, ctx: DeviceContext, st: SafeTensors) raises:
        var norm_bytes = read_tensor_bytes(st, "model.norm.weight")
        var lm_bytes = read_tensor_bytes(st, "lm_head.weight")
        self.buf = ctx.enqueue_create_buffer[DType.uint16](PAD + RESIDENT_U16)
        self.norm = PAD
        self.lm = PAD + 4096
        with self.buf.map_to_host() as h:
            memcpy(dest=(h.unsafe_ptr() + self.norm).bitcast[UInt8](),
                   src=norm_bytes.unsafe_ptr(), count=len(norm_bytes))
            memcpy(dest=(h.unsafe_ptr() + self.lm).bitcast[UInt8](),
                   src=lm_bytes.unsafe_ptr(), count=len(lm_bytes))
        print("resident hot weights:", Float64(RESIDENT_U16 * 2) / 1e9, "GB bf16")


struct Llama:
    var ctx: DeviceContext
    var cfg: Config
    var st: SafeTensors
    var gpu: GpuLayerCache
    var hot: ResidentWeights
    var embed_stage: DeviceBuffer[DType.uint16]
    var a: Acts
    var lgbuf: DeviceBuffer[DType.float32]
    var lo: LayerOffs
    var kc: List[Int]
    var vc: List[Int]
    var oinv: Int
    var maxlen: Int
    var n_cached: Int

    def __init__(out self, ctx: DeviceContext, var st: SafeTensors,
                 maxlen: Int, warm: Bool) raises:
        self.ctx = ctx
        self.cfg = make_config()
        self.st = st^
        self.gpu = GpuLayerCache(ctx, LAYERS)
        self.hot = ResidentWeights(ctx, self.st)
        self.embed_stage = ctx.enqueue_create_buffer[DType.uint16](
            PAD + EMBED_CAP * HIDDEN)
        self.lo = fixed_layer_offs()
        var kvd = self.cfg.kv_dim()
        var cache_elems = self.cfg.layers * 2 * maxlen * kvd
        self.a = Acts(ctx, PAD + self.cfg.half() + cache_elems + 64 * 1024 * 1024)
        self.lgbuf = ctx.enqueue_create_buffer[DType.float32](PAD + VOCAB)
        self.kc = List[Int]()
        self.vc = List[Int]()
        self.maxlen = maxlen
        self.n_cached = 0
        self.oinv = self.a.alloc(self.cfg.half())
        var inv = llama3_inv_freq(self.cfg)
        with self.a.buf.map_to_host() as h:
            for d in range(self.cfg.half()):
                h[self.oinv + d] = inv[d]
        for _ in range(self.cfg.layers):
            self.kc.append(self.a.alloc(maxlen * kvd))
            self.vc.append(self.a.alloc(maxlen * kvd))
        if warm:
            self.gpu.warm(self.st)

    def embed(mut self, ids: List[Int]) raises -> Int:
        """Batched disk row reads + GPU bf16 decode (no acts arena host map)."""
        var s = len(ids)
        if s > EMBED_CAP:
            raise Error("embed batch exceeds EMBED_CAP")
        var H = self.cfg.hidden
        var ox = self.a.alloc(s * H)
        var info = self.st.get("model.embed_tokens.weight")
        var row_b = H * 2
        var f = open(info.path, "r")
        with self.embed_stage.map_to_host() as stage:
            var base = stage.unsafe_ptr() + PAD
            for i in range(s):
                _ = f.seek(UInt64(info.begin + ids[i] * row_b))
                var raw = f.read_bytes(row_b)
                memcpy(dest=(base + i * H).bitcast[UInt8](),
                       src=raw.unsafe_ptr(), count=row_b)
        f.close()
        self.ctx.enqueue_function[k_bf16_to_f32](
            self.embed_stage.unsafe_ptr(), self.a.buf.unsafe_ptr(),
            PAD, ox, s * H,
            grid_dim=ceildiv(s * H, BLOCK), block_dim=BLOCK)
        return ox

    def forward(mut self, ids: List[Int]) raises -> List[Float32]:
        var s = len(ids)
        if self.n_cached + s > self.maxlen:
            raise Error("KV cache full")
        var mark = self.a.mark()
        var oh = self.embed(ids)
        for L in range(self.cfg.layers):
            oh = run_layer(self.ctx, self.gpu.bufs[L], self.a, self.cfg, self.lo,
                           oh, s, self.n_cached, self.kc[L], self.vc[L],
                           self.oinv)
            if s > 4:
                self.ctx.synchronize()
        if s <= 4:
            self.ctx.synchronize()
        self.n_cached += s
        var onrm = rmsnorm_op(self.ctx, self.hot.buf, self.a,
                              oh + (s - 1) * self.cfg.hidden, 1, self.hot.norm,
                              self.cfg.hidden, self.cfg.eps)
        var olg = mm_op(self.ctx, self.hot.buf, self.a, onrm, 1,
                        self.cfg.hidden, self.hot.lm, VOCAB)
        self.ctx.enqueue_function[k_export](
            self.a.buf.unsafe_ptr(), self.lgbuf.unsafe_ptr(), olg, PAD, VOCAB,
            grid_dim=ceildiv(VOCAB, BLOCK), block_dim=BLOCK)
        self.ctx.synchronize()
        var logits = List[Float32](length=VOCAB, fill=0)
        with self.lgbuf.map_to_host() as h:
            memcpy(dest=logits.unsafe_ptr(), src=h.unsafe_ptr() + PAD, count=VOCAB)
        self.a.reset(mark)
        return logits^


def load_llama(ctx: DeviceContext, maxlen: Int, warm: Bool = False) raises -> Llama:
    var st = SafeTensors(MODEL_DIR)
    print("tensors:", st.num_tensors(),
          "| gpu layers:", Float64(LAYER_U16 * 2 * LAYERS) / 1e9, "GB after warm")
    return Llama(ctx, st^, maxlen, warm)


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var model = load_llama(ctx, 64, warm=True)

    var ids: List[Int] = [128000, 128006, 882, 128007, 271, 3923, 374, 279,
                          6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271]
    var t0 = perf_counter_ns()
    var logits = model.forward(ids)
    print("prefill+logits:", Float64(perf_counter_ns() - t0) / 1e9, "s")
    t0 = perf_counter_ns()
    for _ in range(4):
        var step: List[Int] = [791]
        _ = model.forward(step)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("decode:", 4.0 / dt, "tok/s (", dt / 4.0 * 1000.0, "ms/tok )")

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
    print("ALL PASS" if (gi == wi and mx < Float32(3e-2)) else "FAILED")
