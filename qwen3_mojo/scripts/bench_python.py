"""Python benchmark for Qwen3-0.6B on Metal (PyTorch MPS) — speed + oracle check.

Same from-scratch forward as gen_oracle.py / bench, but tensors live on MPS.
Mirrors qwen_gpu.mojo self-test: prefill oracle prompt, compare last_logits,
then 16 greedy decode steps with KV cache.
"""
import json
import os
import struct
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "assets/model/model.safetensors")
ORACLE = os.path.join(ROOT, "assets/oracle")

H, LAYERS, NH, NKV, HD, VOCAB = 1024, 28, 16, 8, 128, 151936
EPS, THETA = 1e-6, 1_000_000.0
QD = NH * HD

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open(MODEL, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
DATA_START = 8 + n

_WEIGHTS: dict[str, torch.Tensor] = {}
INV_FREQ = torch.tensor(
  1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD)),
  dtype=torch.float32,
  device=DEVICE,
)


def sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()


def load_w(name: str) -> torch.Tensor:
    if name not in _WEIGHTS:
        m = header[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        off, end = m["data_offsets"]
        count = (end - off) // 2
        u16 = np.fromfile(MODEL, dtype="<u2", count=count, offset=DATA_START + off)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        _WEIGHTS[name] = torch.from_numpy(f32.reshape(m["shape"]).copy()).to(DEVICE)
    return _WEIGHTS[name]


def rmsnorm(x, w):
    v = x.float()
    return w * (v / torch.sqrt((v * v).mean(-1, keepdim=True) + EPS))


def rope(x, pos0: int):
    s = x.shape[0]
    t = torch.arange(pos0, pos0 + s, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    freqs = t * INV_FREQ.unsqueeze(0)
    cos = torch.cos(freqs).unsqueeze(1)
    sin = torch.sin(freqs).unsqueeze(1)
    x0, x1 = x[..., : HD // 2], x[..., HD // 2 :]
    return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)


class KVCache:
    def __init__(self):
        self.k = [None] * LAYERS
        self.v = [None] * LAYERS
        self.pos = 0


def attn(q, k_all, v_all):
    s, t = q.shape[0], k_all.shape[0]
    group = NH // NKV
    out = torch.empty((s, NH, HD), dtype=torch.float32, device=DEVICE)
    for h in range(NH):
        kv = h // group
        sc = q[:, h] @ k_all[:, kv].T / (HD ** 0.5)
        if s > 1:
            mask = torch.triu(
                torch.full((s, t), float("-inf"), device=DEVICE), diagonal=t - s + 1
            )
            sc = sc + mask
        sc = sc - sc.max(dim=-1, keepdim=True).values
        pr = torch.exp(sc)
        pr = pr / pr.sum(dim=-1, keepdim=True)
        out[:, h] = pr @ v_all[:, kv]
    return out


def layer(x, L: int, cache: KVCache, pos: int):
    p = f"model.layers.{L}."
    s = x.shape[0]
    xn = rmsnorm(x, load_w(p + "input_layernorm.weight"))
    q = (xn @ load_w(p + "self_attn.q_proj.weight").T).reshape(s, NH, HD)
    k_new = (xn @ load_w(p + "self_attn.k_proj.weight").T).reshape(s, NKV, HD)
    v_new = (xn @ load_w(p + "self_attn.v_proj.weight").T).reshape(s, NKV, HD)
    q = rmsnorm(q, load_w(p + "self_attn.q_norm.weight"))
    k_new = rmsnorm(k_new, load_w(p + "self_attn.k_norm.weight"))
    q, k_new = rope(q, pos), rope(k_new, pos)

    if cache.k[L] is None:
        cache.k[L], cache.v[L] = k_new, v_new
    else:
        cache.k[L] = torch.cat([cache.k[L], k_new], dim=0)
        cache.v[L] = torch.cat([cache.v[L], v_new], dim=0)

    out = attn(q, cache.k[L], cache.v[L])
    x = x + (out.reshape(s, QD) @ load_w(p + "self_attn.o_proj.weight").T)
    xn = rmsnorm(x, load_w(p + "post_attention_layernorm.weight"))
    g = xn @ load_w(p + "mlp.gate_proj.weight").T
    u = xn @ load_w(p + "mlp.up_proj.weight").T
    act = g / (1 + torch.exp(-g)) * u
    return x + act @ load_w(p + "mlp.down_proj.weight").T


def forward(cache: KVCache, ids: list[int]) -> torch.Tensor:
    emb = load_w("model.embed_tokens.weight")
    x = emb[torch.tensor(ids, device=DEVICE, dtype=torch.long)].float()
    pos = cache.pos
    for L in range(LAYERS):
        x = layer(x, L, cache, pos)
    cache.pos += len(ids)
    h = rmsnorm(x, load_w("model.norm.weight"))
    return h[-1] @ emb.T


def warm_weights():
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for L in range(LAYERS):
        p = f"model.layers.{L}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            names.append(p + suffix)
    for name in names:
        load_w(name)
    sync()


def read_oracle_logits() -> np.ndarray:
    return np.fromfile(os.path.join(ORACLE, "last_logits.bin"), dtype="<f4", count=VOCAB)


def main():
    manifest = json.load(open(os.path.join(ORACLE, "manifest.json")))
    ids = manifest["ids"]
    backend = "Metal MPS" if DEVICE.type == "mps" else "CPU"
    print(f"python torch {backend} | Qwen3-0.6B")
    print("device:", DEVICE)
    print("prompt tokens:", len(ids))

    t0 = time.perf_counter()
    warm_weights()
    load_s = time.perf_counter() - t0
    print(f"weight upload: {load_s:.3f}s")

    cache = KVCache()
    sync()
    t0 = time.perf_counter()
    logits = forward(cache, ids)
    sync()
    prefill_s = time.perf_counter() - t0

    logits_np = logits.detach().cpu().numpy()
    want = read_oracle_logits()
    mx = float(np.max(np.abs(logits_np - want)))
    got = int(np.argmax(logits_np))
    exp = int(np.argmax(want))
    print(f"prefill+logits: {prefill_s:.3f}s")
    print(f"logits max_abs_diff vs oracle = {mx:.6e}")
    print(f"greedy id: got {got} want {exp}")

    sync()
    t0 = time.perf_counter()
    tok = got
    for _ in range(16):
        logits = forward(cache, [tok])
        sync()
        tok = int(torch.argmax(logits).item())
    decode_s = time.perf_counter() - t0
    print(f"decode: {16.0 / decode_s:.2f} tok/s ({decode_s / 16.0 * 1000.0:.1f} ms/tok)")

    ok = got == exp and mx < 3e-2
    print("ALL PASS" if ok else "FAILED")


if __name__ == "__main__":
    main()
