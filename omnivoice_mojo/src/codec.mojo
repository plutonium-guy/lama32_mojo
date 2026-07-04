"""HiggsAudio v2 codec decoder on Metal: RVQ dequantize + DAC conv decoder.

Turns (8, T) audio tokens at 25 Hz into a 24 kHz waveform (960x upsample):
RVQ (8 codebooks, embed 64 -> project 1024, summed) -> fc2 1024->256 ->
Conv1d 256->1024 k7 -> 5 blocks [Snake, ConvTranspose1d k=2s (outpad s%2),
3 residual units (Snake, Conv k7 dil 1/3/9, Snake, Conv k1)] with strides
[8,5,4,2,3], channels halving 1024->32 -> Snake -> Conv1d 32->1 k7.
No final tanh (OmniVoice's DAC variant).

Run `pixi run test-codec` to verify against the oracle waveform.
"""

from std.math import ceildiv, log
from std.memory import memcpy
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from safetensors import SafeTensors
from resident import Weights
from llama_common import Acts, BLOCK, PAD, read_f32_bin, k_add
from ov_common import (
    OvKernels, k_conv1d, k_convtr1d, k_snake, k_rvq_decode, k_copy,
    k_export_slice,
)

comptime NCB = 8
comptime CBSIZE = 1024
comptime CDIM = 64
comptime RVQ_H = 1024
comptime LATENT = 256
comptime DEC_CH = 1024
comptime UPSAMPLE = 960


def codec_strides() -> List[Int]:
    return [8, 5, 4, 2, 3]


struct Codec:
    """Weight offsets + dispatch helpers; shares the Weights/Acts arenas."""
    var idbuf: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, max_frames: Int) raises:
        self.idbuf = ctx.enqueue_create_buffer[DType.int32](NCB * max_frames)

    def _conv(self, ctx: DeviceContext, w: Weights, mut a: Acts,
              kn: OvKernels, ox: Int, name: String, cin: Int, cout: Int,
              tin: Int, tout: Int, k: Int, stride: Int, pad: Int,
              dil: Int) raises -> Int:
        var oy = a.alloc(cout * tout)
        ctx.enqueue_function(
            kn.conv.bitcast[type_of(ctx.compile_function[k_conv1d]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox, w.o(name + ".weight"), w.o(name + ".bias"), oy,
            cin, cout, tin, tout, k, stride, pad, dil,
            grid_dim=ceildiv(cout * tout, BLOCK), block_dim=BLOCK)
        return oy

    def _conv_at(self, ctx: DeviceContext, w: Weights, mut a: Acts,
                 kn: OvKernels, ox: Int, oy: Int, name: String,
                 cin: Int, cout: Int, tin: Int, tout: Int, k: Int,
                 stride: Int, pad: Int, dil: Int) raises:
        """Conv into a caller-provided arena slot (no alloc)."""
        ctx.enqueue_function(
            kn.conv.bitcast[type_of(ctx.compile_function[k_conv1d]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
            ox, w.o(name + ".weight"), w.o(name + ".bias"), oy,
            cin, cout, tin, tout, k, stride, pad, dil,
            grid_dim=ceildiv(cout * tout, BLOCK), block_dim=BLOCK)

    def _snake_at(self, ctx: DeviceContext, w: Weights, mut a: Acts,
                  kn: OvKernels, ox: Int, oy: Int, name: String, C: Int,
                  T: Int) raises:
        ctx.enqueue_function(
            kn.snake.bitcast[type_of(ctx.compile_function[k_snake]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
            w.o(name + ".alpha"), ox, oy, C, T,
            grid_dim=ceildiv(C * T, BLOCK), block_dim=BLOCK)

    def _res_unit(self, ctx: DeviceContext, w: Weights, mut a: Acts,
                  kn: OvKernels, cur: Int, u: Int, v: Int, name: String,
                  C: Int, T: Int, dil: Int) raises:
        """Residual unit over 3 rotating slots; result lands in u."""
        self._snake_at(ctx, w, a, kn, cur, u, name + ".snake1", C, T)
        self._conv_at(ctx, w, a, kn, u, v, name + ".conv1",
                      C, C, T, T, 7, 1, 3 * dil, dil)
        self._snake_at(ctx, w, a, kn, v, v, name + ".snake2", C, T)
        self._conv_at(ctx, w, a, kn, v, u, name + ".conv2",
                      C, C, T, T, 1, 1, 0, 1)
        ctx.enqueue_function(
            a.kn.add.bitcast[type_of(ctx.compile_function[k_add]())]()[],
            a.buf.unsafe_ptr(), cur, u, C * T,
            grid_dim=ceildiv(C * T, BLOCK), block_dim=BLOCK)

    def decode(mut self, ctx: DeviceContext, w: Weights, mut a: Acts,
               kn: OvKernels, tokens: List[Int], T: Int,
               out_buf: DeviceBuffer[DType.float32]) raises -> List[Float32]:
        """tokens: (8, T) row-major audio codes. Returns waveform (T*960,)."""
        with self.idbuf.map_to_host() as h:
            for i in range(NCB * T):
                h[i] = Int32(tokens[i])
        var mark = a.mark()

        # RVQ dequantize -> (1024, T)
        var oq = a.alloc(RVQ_H * T)
        ctx.enqueue_function(
            kn.rvq.bitcast[type_of(ctx.compile_function[k_rvq_decode]())]()[],
            w.buf.unsafe_ptr(), a.buf.unsafe_ptr(), self.idbuf.unsafe_ptr(),
            w.o("codec.quantizer.quantizers.0.codebook.embed"),
            w.o("codec.quantizer.quantizers.0.project_out.weight"),
            w.o("codec.quantizer.quantizers.0.project_out.bias"),
            oq, T, RVQ_H, NCB, CDIM, CBSIZE,
            grid_dim=ceildiv(RVQ_H * T, BLOCK), block_dim=BLOCK)

        # fc2 (linear as k=1 conv) -> (256, T), then decoder stem -> (1024, T)
        var ox = self._conv(ctx, w, a, kn, oq, "codec.fc2",
                            RVQ_H, LATENT, T, T, 1, 1, 0, 1)
        ox = self._conv(ctx, w, a, kn, ox, "codec.acoustic_decoder.conv1",
                        LATENT, DEC_CH, T, T, 7, 1, 3, 1)

        # Persistent slot carrying each block's output across per-block
        # arena resets; sized for the largest activation (32 ch @ 24 kHz).
        var ocarry = a.alloc(32 * UPSAMPLE * T)
        ctx.enqueue_function(
            kn.cpy.bitcast[type_of(ctx.compile_function[k_copy]())]()[],
            a.buf.unsafe_ptr(), ox, ocarry, DEC_CH * T,
            grid_dim=ceildiv(DEC_CH * T, BLOCK), block_dim=BLOCK)

        var strides = codec_strides()
        var C = DEC_CH
        var t = T
        for b in range(5):
            var bp = "codec.acoustic_decoder.block." + String(b)
            var s = strides[b]
            var cout = C // 2
            var tout = t * s
            var bmark = a.mark()
            var osn = a.alloc(C * t)
            self._snake_at(ctx, w, a, kn, ocarry, osn, bp + ".snake1", C, t)
            var cur = a.alloc(cout * tout)
            var u = a.alloc(cout * tout)
            var v = a.alloc(cout * tout)
            ctx.enqueue_function(
                kn.convtr.bitcast[
                    type_of(ctx.compile_function[k_convtr1d]())]()[],
                w.buf.unsafe_ptr(), a.buf.unsafe_ptr(),
                osn, w.o(bp + ".conv_t1.weight"), w.o(bp + ".conv_t1.bias"),
                cur, C, cout, t, tout, 2 * s, s, (s + 1) // 2,
                grid_dim=ceildiv(cout * tout, BLOCK), block_dim=BLOCK)
            C = cout
            t = tout
            var dil = 1
            for r in range(3):
                self._res_unit(ctx, w, a, kn, cur, u, v,
                               bp + ".res_unit" + String(r + 1), C, t, dil)
                # result in u; rotate so next unit reads it
                var tmp = cur
                cur = u
                u = tmp
                dil *= 3
            ctx.enqueue_function(
                kn.cpy.bitcast[type_of(ctx.compile_function[k_copy]())]()[],
                a.buf.unsafe_ptr(), cur, ocarry, C * t,
                grid_dim=ceildiv(C * t, BLOCK), block_dim=BLOCK)
            a.reset(bmark)
            ctx.synchronize()

        var ofin = a.alloc(C * t)
        self._snake_at(ctx, w, a, kn, ocarry, ofin,
                       "codec.acoustic_decoder.snake1", C, t)
        ox = self._conv(ctx, w, a, kn, ofin, "codec.acoustic_decoder.conv2",
                        C, 1, t, t, 7, 1, 3, 1)

        ctx.enqueue_function(
            kn.exp.bitcast[type_of(ctx.compile_function[k_export_slice]())]()[],
            a.buf.unsafe_ptr(), out_buf.unsafe_ptr(), ox, 0, t,
            grid_dim=ceildiv(t, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
        var wav = List[Float32](length=t, fill=0)
        with out_buf.map_to_host() as h:
            memcpy(dest=wav.unsafe_ptr(), src=h.unsafe_ptr(), count=t)
        a.reset(mark)
        return wav^


def codec_weight_names() -> List[String]:
    var names = List[String]()
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".codebook.embed")
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".project_out.weight")
    for c in range(NCB):
        names.append("codec.quantizer.quantizers." + String(c) + ".project_out.bias")
    names.append("codec.fc2.weight")
    names.append("codec.fc2.bias")
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


def codec_arena_floats(T: Int) -> Int:
    """Peak decoder activation footprint for T frames (with slack):
    carry slot (32*960*T) + block-4 temps (~113*960*T) + stem."""
    return PAD + 160 * UPSAMPLE * T + 4 * 1024 * 1024


comptime MODEL_DIR = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/mojo"
comptime ORACLE_DIR = "/Volumes/T7 Shield/llama32_mojo/omnivoice_mojo/assets/oracle"


def read_i32_bin_(path: String, count: Int) raises -> List[Int]:
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


def main() raises:
    from std.python import Python

    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var json = Python.import_module("json")
    var bi = Python.import_module("builtins")
    var manifest = json.loads(bi.open(ORACLE_DIR + "/manifest.json", "r").read())
    var T = atol(String(manifest["target_len"]))
    var wav_len = atol(String(manifest["wav_len"]))

    var st = SafeTensors(MODEL_DIR)
    var names = codec_weight_names()
    var total = PAD
    for i in range(len(names)):
        total += st.get(names[i]).numel()
    print("codec params:", total)
    var w = Weights(ctx, total)
    w.upload_all(st, names)

    var a = Acts(ctx, codec_arena_floats(T))
    var kn = OvKernels(ctx)
    var codec = Codec(ctx, T)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](T * UPSAMPLE)

    var tokens = read_i32_bin_(ORACLE_DIR + "/final_tokens.bin", NCB * T)
    var t0 = perf_counter_ns()
    var wav = codec.decode(ctx, w, a, kn, tokens, T, out_buf)
    print("decode:", Float64(perf_counter_ns() - t0) / 1e9, "s for",
          len(wav), "samples")

    var want = read_f32_bin(ORACLE_DIR + "/wav_oracle.bin", wav_len)
    var n = min(len(wav), wav_len)
    var err = Float64(0)
    var sig = Float64(0)
    for i in range(n):
        var d = Float64(wav[i] - want[i])
        err += d * d
        sig += Float64(want[i]) * Float64(want[i])
    var snr = 10.0 * (log(sig / (err + 1e-12)) / log(10.0))
    print("len got", len(wav), "want", wav_len, " SNR =", snr, "dB")
    print("PASS" if snr > 20.0 else "CHECK")
