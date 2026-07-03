"""Numpy ground truth for Meta-Llama-3.1-8B-Instruct-abliterated (f32).

Reads tensors on demand across the 4 safetensors shards (one f32 tensor
materialized at a time, so this runs fine in 16 GB). Mirrors transformers
LlamaModel: RMSNorm eps 1e-5, GQA 32q/8kv hd128, llama3-scaled RoPE
(theta 5e5, factor 8, low 1, high 4, orig 8192), SwiGLU, UNTIED lm_head.
"""
import glob
import json
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, "assets/model")
OUT = os.path.join(ROOT, "assets/oracle")

H, LAYERS, NH, NKV, HD, INTER, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
EPS, THETA = 1e-5, 500000.0
FACTOR, LOW_FF, HIGH_FF, ORIG_CTX = 8.0, 1.0, 4.0, 8192

# build tensor -> (shard_path, data_start, header) map
_headers = {}
_where = {}
for path in sorted(glob.glob(os.path.join(MDIR, "*.safetensors"))):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    _headers[path] = (hdr, 8 + n)
    for k in hdr:
        if k != "__metadata__":
            _where[k] = path


def W(name):
    path = _where[name]
    hdr, data_start = _headers[path]
    m = hdr[name]
    assert m["dtype"] == "BF16", (name, m["dtype"])
    off, end = m["data_offsets"]
    count = (end - off) // 2
    u16 = np.fromfile(path, dtype="<u2", count=count, offset=data_start + off)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(m["shape"]).copy()


def rmsnorm(x, w):
    v = x.astype(np.float32)
    return w * (v / np.sqrt((v * v).mean(-1, keepdims=True) + EPS))


def llama3_inv_freq():
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    wavelen = 2 * np.pi / inv
    smooth = (ORIG_CTX / wavelen - LOW_FF) / (HIGH_FF - LOW_FF)
    scaled = np.where(
        wavelen < ORIG_CTX / HIGH_FF,
        inv,
        np.where(wavelen > ORIG_CTX / LOW_FF, inv / FACTOR,
                 (1 - smooth) * inv / FACTOR + smooth * inv),
    )
    return scaled.astype(np.float32)


INV_FREQ = llama3_inv_freq()


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
    x = x + out.reshape(s, H) @ W(p + "self_attn.o_proj.weight").T
    xn = rmsnorm(x, W(p + "post_attention_layernorm.weight"))
    g = xn @ W(p + "mlp.gate_proj.weight").T
    u = xn @ W(p + "mlp.up_proj.weight").T
    act = g / (1 + np.exp(-g)) * u
    return x + act @ W(p + "mlp.down_proj.weight").T


def main():
    os.makedirs(OUT, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MDIR, "tokenizer.json"))
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    ids = tok.encode(prompt, add_special_tokens=False).ids
    print("prompt ids:", ids)

    x = W("model.embed_tokens.weight")[ids].astype(np.float32)
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
    logits = h[-1] @ W("lm_head.weight").T          # untied
    dump("last_logits", logits)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

    top = np.argsort(-logits)[:5]
    print("greedy next id:", int(top[0]), repr(tok.decode([int(top[0])])))
    print("top5:", [(int(t), round(float(logits[t]), 3)) for t in top])


if __name__ == "__main__":
    main()
