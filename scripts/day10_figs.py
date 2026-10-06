"""Day 10 配图：代码地图（这 150 行以后在哪里被优化）、有/无 KV Cache 的逐步耗时实测。"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "day10"))
from minigpt import KVCache, load_pretrained

OUT = ROOT / "assets" / "day10"


def box(ax, x, y, width, height, text, color, fontsize=10.5, text_color="white", ha="center"):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="white", lw=1.5))
    tx = x + width / 2 if ha == "center" else x + 0.15
    ax.text(tx, y + height / 2, text, ha=ha, va="center", color=text_color,
            fontsize=fontsize, linespacing=1.4)


def fig1_code_map():
    rows = [
        ("Embedding", "embed_tokens", "查表：ID → d 维向量", C["neutral"], "—"),
        ("RMSNorm", "RMSNorm", "x / rms(x) · g，FP32 归约", C["neutral"], "Day 12 原理 · Day 27 Triton 融合"),
        ("RoPE", "rope_cos_sin / apply_rope", "按绝对位置旋转 q、k", C["accent"], "Day 11 原理与外推"),
        ("Q/K/V/O 投影", "Attention.q_proj …", "4 个 nn.Linear", C["memory"], "Day 24 矩阵乘 · Day 32–33 权重量化"),
        ("KV Cache", "KVCache.update", "torch.cat 追加 K/V", C["capacity"], "Day 15–16 · Day 31 PagedAttention · Day 34 KV 量化"),
        ("GQA", "repeat_interleave", "展开 K/V 给每组 Query 头", C["capacity"], "Day 17：高效实现不展开"),
        ("打分 + softmax + 加权", "scores / probs / probs @ v", "3 行，写出完整 T×T 表", C["compute"], "Day 25 online softmax · Day 29 FlashAttention"),
        ("FFN", "MLP", "SwiGLU：gate、up、down", C["memory"], "Day 13 原理 · Day 32–33 量化"),
        ("残差", "Block.forward", "x = x + …，两行", C["ok"], "—"),
        ("生成循环", "generate", "argmax；feed 新 token", C["ok"], "Day 20 采样 · Day 51 连续批 · Day 55 投机解码"),
        ("整体", "MiniGPT.forward", "每个 decode 步约 2800 次 aten 调用", C["neutral"], "Day 65 torch.compile · Day 66 CUDA Graph"),
    ]
    fig, ax = plt.subplots(figsize=(14, 8.6))
    ax.set(xlim=(0, 14), ylim=(0, len(rows) + 1.2))
    ax.axis("off")
    headers = [(0.1, "Day 08 部件"), (2.6, "minigpt.py 中的代码"), (6.2, "今天的写法"), (9.6, "以后在哪里优化它")]
    for x, title in headers:
        ax.text(x, len(rows) + 0.6, title, fontsize=12, fontweight="bold", va="center")
    for index, (part, code, how, color, later) in enumerate(rows):
        y = len(rows) - 1 - index + 0.1
        box(ax, 0.1, y, 2.3, 0.8, part, color)
        ax.text(2.6, y + 0.4, code, va="center", fontsize=10.5, family="monospace")
        ax.text(6.2, y + 0.4, how, va="center", fontsize=10.5)
        ax.text(9.6, y + 0.4, later, va="center", fontsize=10.5,
                color=C["compute"] if later != "—" else C["neutral"])
        ax.plot([0.1, 13.9], [y - 0.05, y - 0.05], color=C["grid"], lw=0.8)
    fig.suptitle("图 1 · 代码地图：后面 70 多天的优化，几乎都是在改这 150 行中的某几行", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig1_code_map.png")
    plt.close(fig)


@torch.no_grad()
def median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


@torch.no_grad()
def step_cost(model, length: int, repeats: int = 7) -> tuple[float, float]:
    """序列已有 length 个 token 时，生成下一个 token 的单步耗时（中位数）：(带缓存, 不带缓存)。"""
    ids = torch.randint(0, model.cfg.vocab_size, (1, length), generator=torch.Generator().manual_seed(length))
    caches = [KVCache() for _ in model.layers]
    model(ids[:, :-1], caches)
    snapshot = [(c.k, c.v) for c in caches]

    def cached_step():
        fresh = []
        for k, v in snapshot:                      # update 用 torch.cat 生成新张量，快照不会被改动
            cache = KVCache()
            cache.k, cache.v = k, v
            fresh.append(cache)
        model(ids[:, -1:], fresh)

    return median_ms(cached_step, repeats), median_ms(lambda: model(ids), repeats)


def fig2_cache_timing(prompt: int = 8, steps: int = 400):
    torch.set_grad_enabled(False)
    model = load_pretrained()
    for n in (9, 64, 200):                         # 预热：让分配器和线程池进入稳态
        step_cost(model, n, 3)
    lengths = [prompt + 1] + list(range(32, prompt + steps + 1, 32)) + [prompt + steps]
    costs = [step_cost(model, n) for n in lengths]
    cached = [c for c, _ in costs]
    uncached = [u for _, u in costs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    ax.plot(lengths, uncached, "o-", color=C["compute"], ms=4, label="不带 KV Cache：每步重算整段序列")
    ax.plot(lengths, cached, "o-", color=C["memory"], ms=4, label="带 KV Cache：每步只算新 token")
    ax.set(xlabel="当前序列长度（token）", ylabel="生成下一个 token 的耗时 / ms",
           title="每一步的耗时：线性增长 vs 几乎不变", ylim=(0, max(uncached) * 1.15))
    ax.legend(loc="upper left", fontsize=10)
    ax.annotate(f"短序列时两者接近：\n约 {cached[0]:.0f} ms 的固定开销", xy=(lengths[0], cached[0]),
                xytext=(lengths[len(lengths) // 3], max(uncached) * 0.12), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C["neutral"]))

    ax = axes[1]
    generated = np.arange(1, steps + 1)
    per_step_cached = np.interp(prompt + generated, lengths, cached)
    per_step_uncached = np.interp(prompt + generated, lengths, uncached)
    total_cached = np.cumsum(per_step_cached) / 1e3
    total_uncached = np.cumsum(per_step_uncached) / 1e3
    ax.plot(generated, total_uncached, color=C["compute"], lw=2, label="不带 KV Cache")
    ax.plot(generated, total_cached, color=C["memory"], lw=2, label="带 KV Cache")
    top = total_uncached[-1]
    for n, dx, dy in ((48, 0.04, 0.4), (steps, -0.05, 0.15)):
        ratio = total_uncached[n - 1] / total_cached[n - 1]
        ax.annotate(f"生成 {n} 个：快 {ratio:.1f} 倍", xy=(n, total_uncached[n - 1]),
                    xytext=(n + steps * dx, total_uncached[n - 1] + top * dy),
                    ha="right" if dx < 0 else "left", fontsize=10.5,
                    arrowprops=dict(arrowstyle="->", color=C["neutral"]))
    ax.set(xlabel="已生成的 token 数", ylabel="累计耗时 / s", title="累计耗时：平方增长 vs 线性增长",
           ylim=(0, top * 1.3))
    ax.legend(loc="upper left", fontsize=10)
    fig.suptitle("图 2 · KV Cache 的收益随长度增长：短序列时固定开销占主导", fontsize=15)
    fig.text(0.5, 0.01, f"SmolLM2-135M，FP32，Mac mini M4 CPU；每个长度测 7 次取中位数，累计耗时由单步耗时插值累加；"
             "数值因机器而异",
             ha="center", fontsize=10, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig2_cache_timing.png")
    plt.close(fig)
    for n in (48, steps):
        print(f"  生成 {n} 个 token：带缓存 {total_cached[n-1]:.2f} s，不带 {total_uncached[n-1]:.2f} s，"
              f"{total_uncached[n-1] / total_cached[n-1]:.1f} 倍")
    print(f"  单步：长度 {lengths[0]} 时 {cached[0]:.1f} / {uncached[0]:.1f} ms；"
          f"长度 {lengths[-1]} 时 {cached[-1]:.1f} / {uncached[-1]:.1f} ms")


if __name__ == "__main__":
    print("CJK font:", setup())
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_code_map()
    fig2_cache_timing()
    print("Figures:", OUT)
