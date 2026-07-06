"""Matmul kernel microbench: k_mm_w vs k_mm_tile at various s."""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from llama_common import Acts, PAD, TG_MM, k_mm_w, k_mm_tile


def bench_batch(ctx: DeviceContext, mut a: Acts,
                w_ptr: UnsafePointer[UInt16, MutAnyOrigin],
                ox: Int, oy: Int, s: Int, m: Int, n: Int) raises:
    comptime REPS = 50
    var macs = Float64(s) * Float64(m) * Float64(n) * Float64(REPS)

    var t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function[k_mm_tile](
            w_ptr, a.buf.unsafe_ptr(), ox, PAD, oy, s, m, n,
            grid_dim=ceildiv(s, 8) * ceildiv(n, 8), block_dim=TG_MM)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("  k_mm_tile :", macs / dt / 1e9, "GMAC/s")

    if s == 1:
        t0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.enqueue_function[k_mm_w](
                w_ptr, a.buf.unsafe_ptr(), ox, PAD, oy, s, m, n, 0,
                grid_dim=s * n, block_dim=TG_MM)
        ctx.synchronize()
        dt = Float64(perf_counter_ns() - t0) / 1e9
        print("  k_mm_w    :", macs / dt / 1e9, "GMAC/s")


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    comptime M = 1024
    comptime N = 3072
    comptime SMAX = 512
    var w = ctx.enqueue_create_buffer[DType.uint16](N * M + PAD)
    with w.map_to_host() as h:
        for i in range(N * M + PAD):
            h[i] = UInt16(0x3F80 >> 3)          # small bf16 pattern
    var a = Acts(ctx, PAD + SMAX * M + SMAX * N + 1024)
    var ox = a.alloc(SMAX * M)
    var oy = a.alloc(SMAX * N)
    with a.buf.map_to_host() as h:
        for i in range(SMAX * M):
            h[ox + i] = Float32(0.001)

    for si in range(3):
        var s = 1
        if si == 1:
            s = 96
        elif si == 2:
            s = 512
        print("s =", s, " (m", M, "n", N, ")")
        bench_batch(ctx, a, w.unsafe_ptr(), ox, oy, s, M, N)
