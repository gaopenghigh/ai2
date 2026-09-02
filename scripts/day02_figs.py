"""Day 02 教学配图：
  fig1_roofline_anatomy.png       Roofline 的三个部件与"点的三种位置"
  fig2_optimization_directions.png 优化只有三个方向
  fig3_llm_on_roofline.png        把 LLM 的 prefill/decode 打到两台机器的屋顶上

运行：uv run python scripts/day02_figs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, C

OUT = Path(__file__).resolve().parents[1] / "assets" / "day02"
OUT.mkdir(parents=True, exist_ok=True)

# Day 01 实测（Mac mini M4）与标称（RTX 4050）
M4 = dict(name="Mac mini M4", peak=3.5e12, bw=86e9, color=C["memory"])
RTX = dict(name="RTX 4050 Laptop", peak=24e12, bw=192e9, color=C["compute"])

AI = np.logspace(-2, 4, 500)


def roof(ai, peak, bw):
    return np.minimum(peak, ai * bw)


def fig1_anatomy() -> None:
    peak, bw = M4["peak"], M4["bw"]
    ridge = peak / bw

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(AI, roof(AI, peak, bw) / 1e12, lw=3.5, color=C["neutral"], zorder=4)
    ax.fill_between(AI, 1e-4, roof(AI, peak, bw) / 1e12,
                    color=C["ok"], alpha=0.10, zorder=0)
    ax.fill_between(AI, roof(AI, peak, bw) / 1e12, 1e3,
                    color=C["compute"], alpha=0.07, zorder=0)

    ax.axvline(ridge, color=C["accent"], ls="--", lw=2, zorder=2)
    ax.annotate(f"① 平衡点 = 峰值算力 / 峰值带宽 = {ridge:.0f}",
                xy=(ridge, 0.12), xytext=(75, 0.020),
                fontsize=10, color=C["accent"], fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.6))

    ax.annotate("② 斜坡段：性能 = AI × 带宽\n   （每读 1 字节能算几次，就有多少性能）",
                xy=(1.5, 1.5 * bw / 1e12), xytext=(0.035, 8),
                fontsize=10, color=C["memory"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["memory"], lw=1.6))

    ax.annotate("③ 平顶段：性能 = 峰值算力\n   （再提高 AI 也没用了）",
                xy=(400, peak / 1e12), xytext=(75, 0.10),
                fontsize=10, color=C["compute"], fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C["compute"], lw=1.6))

    ax.text(0.035, 130, "物理上不可能到达的区域", fontsize=11,
            color=C["compute"], fontweight="bold")

    # 三种典型的点
    pts = [
        (0.5, 0.5 * bw, "A 贴在屋顶上：已达物理极限\n别再优化了，换思路",
         C["ok"], (-13, 10), "right"),
        (0.5, 0.5 * bw * 0.35, "B 掉在屋顶下：实现有问题\n还有 3 倍空间",
         C["capacity"], (-13, -22), "right"),
        (900, peak, "C 平顶上：GPU 真的在满负荷算",
         C["ok"], (0, 16), "center"),
    ]
    for ai, p, label, col, off, ha in pts:
        ax.scatter([ai], [p / 1e12], s=130, color=col, zorder=6,
                   edgecolors="black", linewidths=1.2)
        ax.annotate(label, (ai, p / 1e12), textcoords="offset points",
                    xytext=off, fontsize=9, color=col, fontweight="bold", ha=ha)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.02, 5000); ax.set_ylim(0.004, 400)
    ax.set_xlabel("算术强度 AI (FLOP / Byte)")
    ax.set_ylabel("可达性能 (TFLOPS)")
    ax.set_title("图 1  Roofline 只有三个部件，却能预测任何算子的性能上限",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_roofline_anatomy.png")
    plt.close(fig)


def fig2_directions() -> None:
    peak, bw = M4["peak"], M4["bw"]
    ridge = peak / bw

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(AI, roof(AI, peak, bw) / 1e12, lw=3, color=C["neutral"], zorder=4,
            label="当前硬件的屋顶")
    ax.plot(AI, roof(AI, peak * 4, bw * 2) / 1e12, lw=2, ls=":", color=C["accent"],
            zorder=3, label="换更强硬件 / 用 Tensor Core")

    start_ai, start_p = 1.0, 1.0 * bw * 0.35
    ax.scatter([start_ai], [start_p / 1e12], s=160, color=C["compute"], zorder=8,
               edgecolors="black", linewidths=1.2)
    ax.annotate("你现在在这", (start_ai, start_p / 1e12),
                textcoords="offset points", xytext=(-14, -6), ha="right",
                fontsize=10, fontweight="bold", color=C["compute"])

    arrows = [
        ((start_ai, start_p), (start_ai, start_ai * bw), C["ok"],
         "① 向上：贴近屋顶\n算子融合 · 访存合并 · 提高占用率", (12, 6)),
        ((start_ai, start_p), (60, start_p), C["memory"],
         "② 向右：提高算术强度\n增大 batch · 量化 · 复用缓存", (0, 10)),
        ((3.0, 3.0 * bw), (3.0, 3.0 * bw * 2), C["accent"],
         "③ 抬高屋顶\n换硬件 · 换更低精度的计算单元", (14, 2)),
    ]
    for (x0, y0), (x1, y1), col, label, off in arrows:
        ax.annotate("", xy=(x1, y1 / 1e12), xytext=(x0, y0 / 1e12),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=3,
                                    mutation_scale=22))
        ax.annotate(label, ((x0 * x1) ** 0.5, (y0 * y1) ** 0.5 / 1e12),
                    textcoords="offset points", xytext=off,
                    fontsize=9.5, color=col, fontweight="bold")

    ax.axvline(ridge, color=C["neutral"], ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.05, 5000); ax.set_ylim(0.015, 60)
    ax.set_xlabel("算术强度 AI (FLOP / Byte)")
    ax.set_ylabel("可达性能 (TFLOPS)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("图 2  性能优化只有三个方向，没有第四个",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_optimization_directions.png")
    plt.close(fig)


def fig3_llm() -> None:
    """同一份 LLM 负载，在两台机器上落在完全不同的区域。"""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for mach in (M4, RTX):
        ax.plot(AI, roof(AI, mach["peak"], mach["bw"]) / 1e12, lw=3,
                color=mach["color"], zorder=3,
                label=f"{mach['name']}  (平衡点 {mach['peak']/mach['bw']:.0f})")
        ax.axvline(mach["peak"] / mach["bw"], color=mach["color"],
                   ls="--", lw=1.2, alpha=0.6, zorder=1)

    # decode 的算术强度 ≈ batch size；prefill ≈ 序列长度
    loads = [("decode\nB=1", 1), ("decode\nB=8", 8), ("decode\nB=32", 32),
             ("decode\nB=128", 128), ("prefill\n2048 tok", 2048)]
    for i, (label, ai) in enumerate(loads):
        ax.axvline(ai, color=C["neutral"], lw=0.8, alpha=0.35, zorder=0)
        ax.text(ai, 0.011, label, ha="center", va="bottom", fontsize=8.5,
                color=C["neutral"], fontweight="bold")
        for mach in (M4, RTX):
            p = min(mach["peak"], ai * mach["bw"])
            ax.scatter([ai], [p / 1e12], s=85, color=mach["color"], zorder=6,
                       edgecolors="black", linewidths=0.9)

    ax.annotate("同一个 batch=32 的负载：\nM4 已经撞上算力墙（4050 还在带宽墙）",
                xy=(32, min(M4["peak"], 32 * M4["bw"]) / 1e12),
                xytext=(80, 0.20), fontsize=9.5, fontweight="bold",
                color=C["accent"], ha="left",
                arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.6))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 8000); ax.set_ylim(0.008, 90)
    ax.set_xlabel("算术强度 AI (FLOP / Byte)  ≈  decode 的 batch size")
    ax.set_ylabel("可达性能 (TFLOPS)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("图 3  同一份 LLM 负载，在不同硬件上撞的墙不一样",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_llm_on_roofline.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"[style] CJK font = {setup()}")
    fig1_anatomy()
    fig2_directions()
    fig3_llm()
    for p in sorted(OUT.glob("*.png")):
        print(f"[ok] {p.relative_to(OUT.parents[1])}  ({p.stat().st_size/1024:.0f} KB)")
