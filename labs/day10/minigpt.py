"""Day 10: a from-scratch Llama-style GPT (forward + KV cache), checked against Hugging Face.

    uv run python labs/day10/minigpt.py                        # 下载 SmolLM2-135M（约 270 MB）并生成
    uv run python labs/day10/minigpt.py --prompt "The capital of France is"
    uv run python labs/day10/minigpt.py --check --offline      # 不联网：随机小模型与 transformers 对账
    uv run python labs/day10/minigpt.py --check                # 再加上真实权重对账

The model itself (Config → MiniGPT, plus generate) uses only torch tensors, nn.Linear and
nn.Embedding. transformers is imported only inside the checks, as the reference implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


# ---------------------------------------------------------------- 模型：约 150 行

@dataclass
class Config:
    vocab_size: int = 49152
    hidden_size: int = 576           # d
    num_hidden_layers: int = 30      # L
    num_attention_heads: int = 9     # h
    num_key_value_heads: int = 3     # h_kv（GQA）
    intermediate_size: int = 1536    # m
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        raw = json.loads(Path(path).read_text())
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):                                   # (B, T, d)
        x32 = x.float()                                     # 归约在 FP32 里做
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rope_cos_sin(positions, head_dim: int, theta: float, dtype):
    """RoPE 的旋转角，原理见 Day 11。今天只需要知道：它按绝对位置旋转 q 和 k。"""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    angles = positions.float()[:, None] * inv_freq[None, :]      # (T, d_h/2)
    angles = torch.cat([angles, angles], dim=-1)                 # (T, d_h)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x, cos, sin):                                     # x: (B, heads, T, d_h)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class KVCache:
    """一层的历史 K/V。只存 K 和 V：旧的 q 再也用不到（Day 09 §6.1）。"""

    def __init__(self):
        self.k = self.v = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, dh = cfg.hidden_size, cfg.head_dim
        self.h, self.h_kv, self.dh = cfg.num_attention_heads, cfg.num_key_value_heads, dh
        self.q_proj = nn.Linear(d, self.h * dh, bias=False)      # 我在找什么
        self.k_proj = nn.Linear(d, self.h_kv * dh, bias=False)   # 我能匹配什么
        self.v_proj = nn.Linear(d, self.h_kv * dh, bias=False)   # 我交出什么
        self.o_proj = nn.Linear(self.h * dh, d, bias=False)      # 混合各头，写回残差流

    def forward(self, x, cos, sin, positions, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)       # (B, h,    T, d_h)
        k = self.k_proj(x).view(B, T, self.h_kv, self.dh).transpose(1, 2)    # (B, h_kv, T, d_h)
        v = self.v_proj(x).view(B, T, self.h_kv, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)                                        # (B, h_kv, T_k, d_h)
        group = self.h // self.h_kv                                          # GQA：每组共享一套 K/V
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)               # (B, h, T, T_k)
        key_pos = torch.arange(k.shape[2], device=x.device)
        future = key_pos[None, :] > positions[:, None]                      # 按绝对位置判断
        scores = scores.masked_fill(future, float("-inf"))
        probs = scores.float().softmax(dim=-1).to(q.dtype)                  # 沿 Key 维
        out = (probs @ v).transpose(1, 2).reshape(B, T, self.h * self.dh)   # 合头
        return self.o_proj(out)


class MLP(nn.Module):                                                       # SwiGLU
    def __init__(self, cfg: Config):
        super().__init__()
        d, m = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, positions, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions, cache)  # 取信息
        x = x + self.mlp(self.post_attention_layernorm(x))                            # 加工
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, ids, caches: list[KVCache] | None = None):            # ids: (B, T)
        start = caches[0].length if caches else 0
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        x = self.embed_tokens(ids)                                          # (B, T, d)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta, x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, positions, caches[i] if caches else None)
        return self.lm_head(self.norm(x))                                   # (B, T, V)


@torch.no_grad()
def generate(model: MiniGPT, ids, steps: int, use_cache: bool = True):
    """贪心生成。带缓存与否，只差最后一行：下一步喂进去的是新 token，还是整段序列。"""
    caches = [KVCache() for _ in model.layers] if use_cache else None
    seq = feed = ids
    for _ in range(steps):
        logits = model(feed, caches)
        next_id = logits[:, -1:].argmax(dim=-1)                             # (B, 1)
        seq = torch.cat([seq, next_id], dim=1)
        feed = next_id if use_cache else seq
    return seq[:, ids.shape[1]:]


# ---------------------------------------------------------------- 加载真实权重

def download(name: str, model_id: str = MODEL_ID) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_id, name)


def load_pretrained(model_id: str = MODEL_ID, device: str = "cpu") -> MiniGPT:
    """权重只是一组有名字的张量：去掉 'model.' 前缀后，名字与上面的模块一一对应。"""
    from safetensors.torch import load_file

    cfg = Config.from_json(download("config.json", model_id))
    state = {name.removeprefix("model."): tensor.float()
             for name, tensor in load_file(download("model.safetensors", model_id)).items()}
    if cfg.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["embed_tokens.weight"])
    model = MiniGPT(cfg)
    model.load_state_dict(state, strict=True)      # 少一个、多一个或形状不符都会报错
    return model.to(device).eval()


def unique_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())          # 共享的张量只计一次


# ---------------------------------------------------------------- 对账

def hf_reference(cfg: Config, model_id: str | None = None):
    """transformers 的 LlamaForCausalLM，作为参考实现。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    if model_id:
        return LlamaForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, attn_implementation="eager").eval()
    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings, attention_bias=False, mlp_bias=False,
        attn_implementation="eager")
    return LlamaForCausalLM(hf_cfg).eval()


def copy_from_hf(cfg: Config, hf_model) -> MiniGPT:
    state = {name.removeprefix("model."): tensor
             for name, tensor in hf_model.state_dict().items()}
    model = MiniGPT(cfg)
    model.load_state_dict(state, strict=True)
    return model.eval()


def max_diff(a, b) -> float:
    return (a - b).abs().max().item()


def report(label: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} {detail}")
    return ok


@torch.no_grad()
def check_against(model: MiniGPT, hf_model, ids, tol: float) -> bool:
    ok = True
    ours, ref = model(ids), hf_model(ids).logits
    ok &= report("logits 与 transformers 一致", max_diff(ours, ref) < tol,
                 f"最大误差 {max_diff(ours, ref):.2e}")
    ok &= report("每个位置的 top-1 token 一致",
                 bool((ours.argmax(-1) == ref.argmax(-1)).all()), "")

    split = ids.shape[1] // 2                      # 先 prefill 前半段，再逐 token decode
    caches = [KVCache() for _ in model.layers]
    pieces = [model(ids[:, :split], caches)]
    pieces += [model(ids[:, t:t + 1], caches) for t in range(split, ids.shape[1])]
    cached = torch.cat(pieces, dim=1)
    ok &= report("prefill + 逐 token decode = 一次完整前向", max_diff(cached, ours) < tol,
                 f"最大误差 {max_diff(cached, ours):.2e}")
    ok &= report("KV Cache 只存 K/V，长度等于已处理 token 数",
                 caches[0].length == ids.shape[1] and caches[0].k.shape[1] == model.cfg.num_key_value_heads,
                 f"K 形状 {tuple(caches[0].k.shape)}")

    steps = 8
    with_cache = generate(model, ids[:, :split], steps, use_cache=True)
    without_cache = generate(model, ids[:, :split], steps, use_cache=False)
    prompt = ids[:, :split]
    hf_tokens = hf_model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                  pad_token_id=0, max_new_tokens=steps, min_new_tokens=steps,
                                  do_sample=False)[:, split:]
    ok &= report("生成：带缓存 = 不带缓存 = transformers",
                 torch.equal(with_cache, without_cache) and torch.equal(with_cache, hf_tokens),
                 f"{steps} 个 token")
    return ok


def check_offline() -> bool:
    torch.manual_seed(0)
    ok = True
    for h_kv, tied in ((4, False), (2, True), (1, False)):         # MHA、GQA、MQA
        cfg = Config(vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=h_kv, intermediate_size=56, tie_word_embeddings=tied)
        print(f"随机小模型：h=4, h_kv={h_kv}, tied={tied}")
        hf_model = hf_reference(cfg)
        model = copy_from_hf(cfg, hf_model)
        ok &= report("参数量与 transformers 相同",
                     unique_parameters(model) == unique_parameters(hf_model),
                     f"{unique_parameters(model):,}")
        ok &= check_against(model, hf_model, torch.randint(0, cfg.vocab_size, (2, 12)), 1e-5)
    return ok


def check_pretrained(device: str) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day08"))
    from param_count import Config as CountConfig, parameter_counts
    from tokenizers import Tokenizer

    print(f"真实权重：{MODEL_ID}（FP32，{device}）")
    model = load_pretrained(MODEL_ID, device)
    cfg = model.cfg
    expected = sum(parameter_counts(CountConfig(
        width=cfg.hidden_size, layers=cfg.num_hidden_layers, heads=cfg.num_attention_heads,
        kv_heads=cfg.num_key_value_heads, intermediate=cfg.intermediate_size,
        vocab=cfg.vocab_size, tied=cfg.tie_word_embeddings)).values())
    ok = report("参数量 = Day 08 计算器", unique_parameters(model) == expected,
                f"{unique_parameters(model):,}")
    tokenizer = Tokenizer.from_file(download("tokenizer.json"))
    text = "The capital of France is Paris. The capital of Germany is"
    ids = torch.tensor([tokenizer.encode(text).ids], device=device)
    hf_model = hf_reference(cfg, MODEL_ID).to(device)
    return ok & check_against(model, hf_model, ids, 1e-3)


# ---------------------------------------------------------------- 演示

def demo(prompt: str, steps: int, device: str) -> None:
    from tokenizers import Tokenizer

    model = load_pretrained(MODEL_ID, device)
    cfg = model.cfg
    print(f"{MODEL_ID}: d={cfg.hidden_size}, L={cfg.num_hidden_layers}, "
          f"h={cfg.num_attention_heads}, h_kv={cfg.num_key_value_heads}, "
          f"参数 {unique_parameters(model) / 1e6:.1f} M")
    tokenizer = Tokenizer.from_file(download("tokenizer.json"))
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    results = {}
    for use_cache in (True, False):
        generate(model, ids, 2, use_cache)                                # 预热
        start = time.perf_counter()
        out = generate(model, ids, steps, use_cache)
        results[use_cache] = (out, time.perf_counter() - start)
    print(f"\n{prompt}{tokenizer.decode(results[True][0][0].tolist())}\n")
    for use_cache, (out, seconds) in results.items():
        print(f"  {'带 KV Cache  ' if use_cache else '不带 KV Cache'}  "
              f"{steps} token / {seconds:.2f} s = {steps / seconds:6.1f} token/s")
    same = torch.equal(results[True][0], results[False][0])
    print(f"  两种方式生成的 token {'完全相同' if same else '不同！'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="Once upon a time, a little cat")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--check", action="store_true", help="与 transformers 对账")
    parser.add_argument("--offline", action="store_true", help="只做不联网的随机小模型对账")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.check:
        ok = check_offline()
        if not args.offline:
            ok &= check_pretrained(args.device)
        print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
        sys.exit(0 if ok else 1)
    demo(args.prompt, args.max_new_tokens, args.device)


if __name__ == "__main__":
    main()
