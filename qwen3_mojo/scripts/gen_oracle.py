"""Numpy ground truth for Qwen3-0.6B (f32, straight from safetensors).

Mirrors transformers Qwen3Model: RMSNorm eps 1e-6, GQA 16q/8kv hd128,
standard RoPE (theta 1e6), per-head Q/K RMSNorm before RoPE, SwiGLU,
tied lm_head.
"""
import json
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "assets/model/model.safetensors")
OUT = os.path.join(ROOT, "assets/oracle")

H, LAYERS, NH, NKV, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
EPS, THETA = 1e-6, 1_000_000.0
QD = NH * HD
KVD = NKV * HD

with open(MODEL, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
DATA_START = 8 + n


def W(name):
    m = header[name]
    assert m["dtype"] == "BF16", (name, m["dtype"])
    off, end = m["data_offsets"]
    count = (end - off) // 2
    u16 = np.fromfile(MODEL, dtype="<u2", count=count, offset=DATA_START + off)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(m["shape"]).copy()


def rmsnorm(x, w):
    v = x.astype(np.float32)
    return w * (v / np.sqrt((v * v).mean(-1, keepdims=True) + EPS))


def inv_freq():
    return (1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))).astype(
        np.float32
    )


INV_FREQ = inv_freq()


def rope(x, pos0):
    s = x.shape[0]
    t = np.arange(pos0, pos0 + s, dtype=np.float32)[:, None]
    freqs = t * INV_FREQ[None, :]
    cos = np.cos(freqs)[:, None, :]
    sin = np.sin(freqs)[:, None, :]
    x0, x1 = x[..., : HD // 2], x[..., HD // 2 :]
    return np.concatenate([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1)


def layer(x, L):
    p = f"model.layers.{L}."
    s = x.shape[0]
    xn = rmsnorm(x, W(p + "input_layernorm.weight"))
    q = (xn @ W(p + "self_attn.q_proj.weight").T).reshape(s, NH, HD)
    k = (xn @ W(p + "self_attn.k_proj.weight").T).reshape(s, NKV, HD)
    v = (xn @ W(p + "self_attn.v_proj.weight").T).reshape(s, NKV, HD)
    q = rmsnorm(q, W(p + "self_attn.q_norm.weight"))
    k = rmsnorm(k, W(p + "self_attn.k_norm.weight"))
    q, k = rope(q, 0), rope(k, 0)
    group = NH // NKV
    mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), 1)
    out = np.empty((s, NH, HD), dtype=np.float32)
    for h in range(NH):
        kv = h // group
        sc = q[:, h] @ k[:, kv].T / np.sqrt(HD) + mask
        sc = sc - sc.max(-1, keepdims=True)
        pr = np.exp(sc)
        pr /= pr.sum(-1, keepdims=True)
        out[:, h] = pr @ v[:, kv]
    x = x + out.reshape(s, QD) @ W(p + "self_attn.o_proj.weight").T
    xn = rmsnorm(x, W(p + "post_attention_layernorm.weight"))
    g = xn @ W(p + "mlp.gate_proj.weight").T
    u = xn @ W(p + "mlp.up_proj.weight").T
    act = g / (1 + np.exp(-g)) * u
    return x + act @ W(p + "mlp.down_proj.weight").T


def main():
    os.makedirs(OUT, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(ROOT, "assets/model/tokenizer.json"))
    prompt = (
        "<|im_start|>user\n"
        "What is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ids = tok.encode(prompt, add_special_tokens=False).ids
    print("prompt ids:", ids)

    emb = W("model.embed_tokens.weight")
    x = emb[ids].astype(np.float32)
    manifest = {"ids": ids}

    def dump(name, a):
        a = np.ascontiguousarray(a, dtype="<f4")
        a.tofile(os.path.join(OUT, name + ".bin"))
        manifest[name] = {"file": name + ".bin", "shape": list(a.shape)}

    for L in range(LAYERS):
        x = layer(x, L)
        if L in (0, 1):
            dump(f"after_layer{L}", x)
        print(f"layer {L}  mean={x.mean():.6f} std={x.std():.6f}")

    h = rmsnorm(x, W("model.norm.weight"))
    logits = h[-1] @ emb.T
    dump("last_logits", logits)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

    top = np.argsort(-logits)[:5]
    print("greedy next id:", int(top[0]), repr(tok.decode([int(top[0])])))
    print("top5:", [(int(t), round(float(logits[t]), 3)) for t in top])


if __name__ == "__main__":
    main()
