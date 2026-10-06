"""Day 09 配图：两类计算、KV Cache 成立的原因、随长度的增长、decode 的 Roofline。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "day09"))
from attention_cost import Config, PROJECTIONS, Workload, flop_counts, payload_bytes

OUT = ROOT / "assets" / "day09"

# Day 02 中两台机器的峰值算力（FLOP/s）与带宽（Byte/s）
MACHINES = {
    "RTX 4050：24 TFLOPS / 192 GB/s": (24e12, 192e9, C["compute"]),
    "M4：3.53 TFLOPS / 85 GB/s": (3.53e12, 85e9, C["memory"]),
}


def box(ax, x, y, width, height, text, color, fontsize=11):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=color,
                           edgecolor="white", linewidth=1.5))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color="white", fontsize=fontsize, linespacing=1.5)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.7, color=C["neutral"]))


def fig1_attention_graph():
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.set(xlim=(0, 12), ylim=(-0.6, 12))
    ax.axis("off")
    ax.set_title("图 1 · 两类计算：有参数的投影随 T 线性，无参数的配对随 T 平方", pad=18)
    box(ax, 3.7, 10.7, 4.6, 0.8, "输入隐藏状态 X：[B, T, d]", C["neutral"])
    for x, label in ((0.4, "Q"), (4.4, "K"), (8.4, "V")):
        box(ax, x, 8.6, 3.2, 1.3, f"{label} 投影 + 拆头\n[B, h, T, d_h]\n2BTd² FLOPs",
            C["memory"], 10.5)
        arrow(ax, (6, 10.7), (x + 1.6, 9.9))
    box(ax, 1.7, 6.6, 5.3, 1.3, "Q @ K.T → scores\n[B, h, T, T]\n2BT²d FLOPs", C["compute"], 10.5)
    arrow(ax, (2, 8.6), (3, 7.9))
    arrow(ax, (6, 8.6), (5.7, 7.9))
    box(ax, 1.7, 4.6, 5.3, 1.25, r"$\div\sqrt{d_h}$" + " → causal mask → softmax\nP：[B, h, T, T]",
        C["accent"])
    arrow(ax, (4.35, 6.6), (4.35, 5.85))
    ax.text(7.5, 6.45, "每个 Query 都要和\n每个可见 Key 打分：\nT × T 个配对",
            ha="left", va="center", fontsize=11, linespacing=1.6, color=C["compute"])
    box(ax, 3.7, 2.4, 4.6, 1.3, "P @ V → Value 加权和\n[B, h, T, d_h]\n2BT²d FLOPs", C["compute"], 10.5)
    arrow(ax, (4.35, 4.6), (5.3, 3.7))
    ax.plot([10, 10], [8.6, 3.05], color=C["neutral"], lw=1.7)
    arrow(ax, (10, 3.05), (8.3, 3.05))
    box(ax, 3.7, 0.3, 4.6, 1.3, "合头 → O 投影\n[B, T, d]\n2BTd² FLOPs", C["memory"], 10.5)
    arrow(ax, (6, 2.4), (6, 1.6))
    ax.text(0.1, 1.6, "蓝：有参数，每个 token 独立\n    合计 8BTd²，随 T 线性",
            va="center", fontsize=10.5, color=C["memory"], linespacing=1.5)
    ax.text(0.1, 0.4, "红：无参数，两两配对\n    合计 4BT²d，随 T 平方",
            va="center", fontsize=10.5, color=C["compute"], linespacing=1.5)
    ax.text(8.6, 0.95, "h × d_h = d：\n头数不改变 FLOPs", va="center",
            fontsize=10.5, color=C["neutral"], linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_attention_graph.png")
    plt.close(fig)


def cells(ax, y, labels, colors, text_colors=None):
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.add_patch(Rectangle((index, y), 1, 1, facecolor=color, edgecolor="white", lw=2))
        ax.text(index + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10,
                color=(text_colors[index] if text_colors else "white"))


def fig3_kv_cache():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1, 1.2]})
    prefill, decode = axes
    n = 6

    prefill.set(xlim=(-1.3, n), ylim=(-1.6, n + 0.6), aspect="equal")
    prefill.axis("off")
    prefill.set_title("Prefill：因果 mask，第 i 行只看 j ≤ i")
    for row in range(n):
        labels = ["可见" if col <= row else "" for col in range(n)]
        colors = [C["memory"] if col <= row else C["grid"] for col in range(n)]
        cells(prefill, n - 1 - row, labels, colors)
        prefill.text(-0.25, n - 0.5 - row, f"q{row + 1}", ha="right", va="center", fontsize=10.5)
    for col in range(n):
        prefill.text(col + 0.5, n + 0.25, f"k{col + 1}", ha="center", va="center", fontsize=10.5)
    prefill.text(n / 2, -0.8, "上三角恒为 0：前面的位置永远看不到后面的 token\n"
                 "→ 新 token 到来时，旧位置的隐藏状态和 K/V 都不变",
                 ha="center", va="center", fontsize=10.5, linespacing=1.6)

    m = n + 1
    decode.set(xlim=(-2.6, m + 0.2), ylim=(-1.6, 6.2), aspect="equal")
    decode.axis("off")
    decode.set_title(f"Decode 第 {m} 个 token：只算新的一行")
    decode.text(-0.25, 4.9, "K/V 来源", ha="right", va="center", fontsize=10.5)
    cells(decode, 4.4, ["缓存"] * n + ["新算"], [C["memory"]] * n + [C["compute"]])
    for col in range(m):
        decode.text(col + 0.5, 5.75, f"k{col + 1}, v{col + 1}", ha="center", va="center", fontsize=9)
    decode.text(-0.25, 2.5, f"q{m} 的分数", ha="right", va="center", fontsize=10.5)
    cells(decode, 2.0, ["可见"] * m, [C["accent"]] * m)
    decode.text(m / 2, 1.2, f"q{m} 与全部 {m} 个 Key 打分，再对 {m} 个 Value 加权求和",
                ha="center", va="center", fontsize=10.5)
    decode.text(m / 2, -0.55, "只为新 token 计算 q/k/v；旧的 q 和 P 用不到，不必保存\n"
                "代价：每一步都要把全部历史 K/V 从显存读一遍",
                ha="center", va="center", fontsize=10.5, linespacing=1.6, color=C["neutral"])
    fig.suptitle("图 3 · 为什么能缓存 K/V：k_j 只依赖 x_j，因果 mask 让 x_j 不随未来改变", fontsize=15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "fig3_kv_cache.png")
    plt.close(fig)


def fig2_cost_scaling():
    config = Config()
    lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    projections, dense_core, causal_core, scores, cache = [], [], [], [], []
    for length in lengths:
        work = Workload(seq_len=length)
        dense = flop_counts(config, work)
        causal = flop_counts(config, work, causal=True)
        projections.append(sum(dense[name] for name in PROJECTIONS) / 1e9)
        dense_core.append((dense["Q @ K.T"] + dense["P @ V"]) / 1e9)
        causal_core.append((causal["Q @ K.T"] + causal["P @ V"]) / 1e9)
        payload = payload_bytes(config, work)
        scores.append(payload["One dense scores OR P tensor (one layer)"] / 1024 ** 3)
        cache.append(payload["KV cache (all layers, after this step)"] / 1024 ** 3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    ax = axes[0]
    ax.plot(lengths, projections, "o-", color=C["memory"], label="Q/K/V/O 投影：有参数，线性")
    ax.plot(lengths, dense_core, "s-", color=C["compute"], label="两次配对：完整矩阵，平方")
    ax.plot(lengths, causal_core, "^--", color=C["accent"], label="两次配对：因果有效（约减半）")
    ax.axvline(8192, color=C["neutral"], ls=":", lw=1)
    ax.axvspan(256, 8192, color=C["memory"], alpha=0.06)
    ax.set(ylabel="GFLOPs / 层（仅 Attention）", title="计算：T < 2d 时投影占主导")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(8192, 2 ** 3.2, "交叉点\nT = 2d = 8192", ha="center", fontsize=9.5,
            bbox=dict(fc="white", ec="none", pad=2))
    ax.text(0.97, 0.04, "2048 → 4096：\n投影 ×2，配对 ×4\n合计只 ×2.4", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, linespacing=1.5)

    ax = axes[1]
    ax.plot(lengths, scores, "s-", color=C["capacity"], label="一层分数表：平方、临时、可被融合消除")
    ax.plot(lengths, cache, "o-", color=C["memory"], label="32 层 KV Cache：线性、常驻")
    ax.set(ylabel="张量大小 / GiB", title="内存：分数表增长更快，KV Cache 必须常驻")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.annotate("T=2048：0.25 vs 1 GiB", xy=(2048, 1), xytext=(450, 7),
                arrowprops=dict(arrowstyle="->", color=C["neutral"]), fontsize=10)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(lengths, ["256", "512", "1k", "2k", "4k", "8k", "16k", "32k"])
        ax.set_xlabel("上下文长度 T（k = 1024）")
    fig.suptitle("图 2 · 参数不变，计算与内存随上下文长度增长", fontsize=15)
    fig.text(0.5, 0.025, "Llama-2-7B 结构：B=1, d=4096, h=32, L=32, FP16；长序列仅为算术外推；右图是张量大小，不是峰值显存",
             ha="center", fontsize=10, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT / "fig2_cost_scaling.png")
    plt.close(fig)


def pairing_intensity(config: Config, seq_len: int, element_bytes: int = 2) -> float:
    """单 token decode 的 QK^T + PV：FLOPs / (K、V、q、输出的字节数)。"""
    work = Workload(seq_len=seq_len, mode="decode", element_bytes=element_bytes)
    flops = flop_counts(config, work)
    payload = payload_bytes(config, work)
    moved = payload["K + V (one layer, all visible tokens)"] + 2 * config.width * element_bytes
    return (flops["Q @ K.T"] + flops["P @ V"]) / moved


def projection_intensity(config: Config, tokens: int, element_bytes: int = 2) -> float:
    """MHA 的 Q/K/V/O 投影：FLOPs / (权重 + 输入输出激活的字节数)。"""
    d = config.width
    flops = 8 * tokens * d * d
    moved = (4 * d * d + 8 * tokens * d) * element_bytes
    return flops / moved


def fig4_decode_roofline():
    mha, gqa = Config(), Config(kv_heads=8)
    mha_pairing, b1 = pairing_intensity(mha, 2048), projection_intensity(mha, 1)
    points = [
        (f"decode 配对（MHA）≈ {mha_pairing:.1f}，decode 投影（B=1）≈ {b1:.1f}\n"
         "配对部分：batch 再大也不变，每个请求有自己的 K/V",
         mha_pairing, C["compute"], (0.55, 2 ** 2.6), "bottom"),
        (f"decode 配对（GQA，8 个 KV 头）≈ {pairing_intensity(gqa, 2048):.1f}\n"
         "每个 K/V 被 4 个 Query 头复用",
         pairing_intensity(gqa, 2048), C["accent"], (5.5, 2 ** -2.3), "top"),
        (f"decode 投影（B=32）≈ {projection_intensity(mha, 32):.1f}\n权重被 32 个请求复用",
         projection_intensity(mha, 32), C["memory"], (40, 2 ** 0.9), "top"),
        (f"prefill 投影（T=2048）≈ {projection_intensity(mha, 2048):.0f}",
         projection_intensity(mha, 2048), C["ok"], (230, 2 ** 3.4), "top"),
    ]
    intensity = np.logspace(-1, 12, 400, base=2)
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, (peak, bandwidth, color) in MACHINES.items():
        roof = np.minimum(peak, bandwidth * intensity) / 1e12
        ax.plot(intensity, roof, color=color, lw=2.2, label=label)
        ridge = peak / bandwidth
        ax.plot([ridge], [peak / 1e12], "o", color=color, ms=6)
        ax.text(ridge * 1.4, peak / 1e12 * 1.3, f"拐点 ≈ {int(ridge)} FLOP/Byte",
                fontsize=9.5, color=color, va="bottom")
    peak, bandwidth, _ = next(iter(MACHINES.values()))
    for label, value, color, xytext, va in points:
        y = min(peak, bandwidth * value) / 1e12
        ax.plot([value], [y], "D", color=color, ms=9, zorder=5)
        ax.annotate(label, xy=(value, y), xytext=xytext, fontsize=9.5, va=va,
                    color=color, linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set(xlim=(0.5, 4096), ylim=(2 ** -5, 64), xlabel="算术强度（FLOP / Byte，FP16）",
           ylabel="可达性能上限 / TFLOPS",
           title="图 4 · decode 为什么慢：每读 1 Byte 的 K/V 只做约 1 次浮点运算")
    ax.legend(loc="lower right", fontsize=10)
    fig.text(0.5, 0.01, "Llama-2-7B 一层 Attention，上下文 2048；菱形标在 RTX 4050 的屋顶上；"
             "字节数只计必须读写的张量，忽略缓存命中",
             ha="center", fontsize=10, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig4_decode_roofline.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_attention_graph()
    fig2_cost_scaling()
    fig3_kv_cache()
    fig4_decode_roofline()
    print("Figures:", OUT)
