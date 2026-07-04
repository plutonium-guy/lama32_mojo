"""Matmul kernel microbench: k_mm_w (1 output/group) vs k_mm_tile (4x8)."""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from llama_common import Acts, PAD, TG_MM, k_mm_w, k_mm_tile
from std.gpu.host import DeviceBuffer


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    comptime S = 96
    comptime M = 1024
    comptime N = 3072
    var w = ctx.enqueue_create_buffer[DType.uint16](N * M + PAD)
    with w.map_to_host() as h:
        for i in range(N * M + PAD):
            h[i] = UInt16(0x3F80 >> 3)          # small bf16 pattern
    var a = Acts(ctx, PAD + S * M + S * N + 1024)
    var ox = a.alloc(S * M)
    var oy = a.alloc(S * N)
    with a.buf.map_to_host() as h:
        for i in range(S * M):
            h[ox + i] = Float32(0.001)

    comptime REPS = 50
    var macs = Float64(S) * M * N * REPS

    var t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function(
            a.kn.mmt.bitcast[type_of(ctx.compile_function[k_mm_tile]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(), ox, PAD, oy, S, M, N,
            grid_dim=ceildiv(S, 8) * ceildiv(N, 8), block_dim=TG_MM)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("k_mm_tile:", macs / dt / 1e9, "GMAC/s")

    t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function(
            a.kn.mm.bitcast[type_of(ctx.compile_function[k_mm_w]())]()[],
            w.unsafe_ptr(), a.buf.unsafe_ptr(), ox, PAD, oy, S, M, N, 0,
            grid_dim=S * N, block_dim=TG_MM)
    ctx.synchronize()
    dt = Float64(perf_counter_ns() - t0) / 1e9
    print("k_mm_w:   ", macs / dt / 1e9, "GMAC/s")
