"""Host-side iterative-unmasking sampler for OmniVoice.

The GPU produces, per (codebook, frame), a greedy CFG prediction and its
confidence (see k_cfg_predict). This module owns the unmask schedule and the
per-step position selection: layer penalty, optional gumbel position noise,
top-k over still-masked slots.
"""

from std.math import ceildiv, log
from std.random import random_float64

comptime MASK_ID = 1024


def time_steps(num_step: Int, t_shift: Float64) -> List[Float64]:
    var out = List[Float64](capacity=num_step + 1)
    for i in range(num_step + 1):
        var t = Float64(i) / Float64(num_step)
        out.append(t_shift * t / (1.0 + (t_shift - 1.0) * t))
    return out^


def unmask_schedule(target_len: Int, ncb: Int, num_step: Int,
                    t_shift: Float64) -> List[Int]:
    """How many of the ncb*target_len masked slots to reveal at each step."""
    var ts = time_steps(num_step, t_shift)
    var total = target_len * ncb
    var rem = total
    var sched = List[Int](capacity=num_step)
    for step in range(num_step):
        var num: Int
        if step == num_step - 1:
            num = rem
        else:
            var frac = ts[step + 1] - ts[step]
            num = Int(Float64(total) * frac)
            if Float64(num) < Float64(total) * frac:
                num += 1                      # ceil
            if num > rem:
                num = rem
        sched.append(num)
        rem -= num
    return sched^


def gumbel() -> Float32:
    var u = random_float64()
    return Float32(-log(-log(u + 1e-10) + 1e-10))


def select_and_fill(mut tokens: List[Int], preds: List[Float32],
                    confs: List[Float32], k: Int, ncb: Int, T: Int,
                    layer_penalty: Float32, position_temp: Float32):
    """Pick the k highest-scoring still-masked (codebook, frame) slots and
    commit their predictions. tokens/preds/confs are (ncb, T) row-major."""
    var n = ncb * T
    var scores = List[Float32](capacity=n)
    for i in range(n):
        var c = i // T
        var s = confs[i] - Float32(c) * layer_penalty
        if position_temp > 0:
            s = s / position_temp + gumbel()
        if tokens[i] != MASK_ID:
            s = Float32(-3.0e38)
        scores.append(s)
    for _ in range(k):
        var best = -1
        var bv = Float32(-3.0e38)
        for i in range(n):
            if scores[i] > bv:
                bv = scores[i]
                best = i
        if best < 0:
            break
        tokens[best] = Int(preds[best])
        scores[best] = Float32(-3.0e38)
