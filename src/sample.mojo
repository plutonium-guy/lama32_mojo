"""Fast CPU sampling helpers for chat REPLs."""

from std.math import exp
from std.random import random_float64


def argmax(logits: List[Float32]) -> Int:
    var best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def sample(logits: List[Float32], temp: Float32, top_p: Float32) raises -> Int:
    """Temperature + nucleus sampling; temp <= 0 is greedy.

  Only scores logits within (max - 16) of the peak — skips exp() over the
  full 150k vocab when the tail is negligible.
    """
    if temp <= 0:
        return argmax(logits)
    var mx = logits[argmax(logits)]
    var cutoff = mx - Float32(16.0)
    var cand_idx = List[Int]()
    var cand_p = List[Float32]()
    var total = Float32(0)
    for i in range(len(logits)):
        if logits[i] < cutoff:
            continue
        var p = exp((logits[i] - mx) / temp)
        cand_idx.append(i)
        cand_p.append(p)
        total += p
    if len(cand_idx) == 0:
        return argmax(logits)
    var target = total * top_p
    var nuc_idx = List[Int]()
    var nuc_p = List[Float32]()
    var cum = Float32(0)
    while cum < target and len(cand_idx) > 0:
        var b = 0
        for i in range(1, len(cand_p)):
            if cand_p[i] > cand_p[b]:
                b = i
        cum += cand_p[b]
        nuc_idx.append(cand_idx[b])
        nuc_p.append(cand_p[b])
        cand_idx[b] = cand_idx[len(cand_idx) - 1]
        cand_p[b] = cand_p[len(cand_p) - 1]
        _ = cand_idx.pop()
        _ = cand_p.pop()
    var r = Float32(random_float64()) * cum
    var acc = Float32(0)
    for i in range(len(nuc_idx)):
        acc += nuc_p[i]
        if r <= acc:
            return nuc_idx[i]
    return nuc_idx[len(nuc_idx) - 1]
