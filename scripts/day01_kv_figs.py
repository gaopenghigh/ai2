"""Day 01 手算练习配图：KV Cache 到底存的是什么

  fig6_kv_cache_origin.png   一个 token 在一层里产生 Q/K/V，只有 K/V 被缓存
                             MHA 与 GQA 对比

运行：uv run python scripts/day01_kv_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

from _style import setup, C

OUT = Path(__file__).resolve().parents[1] / "assets" / "day01"
OUT.mkdir(parents=True, exist_ok=True)

UNIT = 3.2 / 4096          # 4096 维画 3.2 个绘图单位


def draw_panel(ax, title, q_dim, kv_dim, n_q_heads, n_kv_heads, head_dim, note):
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    # 输入
    ax.add_patch(Rectangle((0.2, 2.5), 1.0, 1.0, fc=C["neutral"], ec="black", lw=1.4))
    ax.text(0.7, 3.0, "输入 x", ha="center", va="center", color="white",
            fontsize=9, fontweight="bold")
    ax.text(0.7, 2.3, "4096 维", ha="center", va="top", fontsize=8.5)

    rows = [
        ("W_q", "Q", q_dim, n_q_heads, C["neutral"], "用完即弃", 4.7),
        ("W_k", "K", kv_dim, n_kv_heads, C["memory"], "存进 KV Cache", 3.0),
        ("W_v", "V", kv_dim, n_kv_heads, C["capacity"], "存进 KV Cache", 1.3),
    ]

    for wname, oname, dim, n_head, col, fate, cy in rows:
        ax.annotate("", xy=(1.95, cy), xytext=(1.25, 3.0),
                    arrowprops=dict(arrowstyle="->", color=C["neutral"], lw=1.3))
        ax.add_patch(Rectangle((1.95, cy - 0.35), 0.85, 0.7,
                               fc="white", ec=C["neutral"], lw=1.4))
        ax.text(2.38, cy, wname, ha="center", va="center", fontsize=9,
                fontweight="bold", family="monospace")

        w = dim * UNIT
        ax.add_patch(Rectangle((3.15, cy - 0.3), w, 0.6, fc=col, ec="black",
                               lw=1.3, alpha=0.9))
        ax.text(3.15 + w / 2, cy, f"{oname}  {dim} 维",
                ha="center", va="center", color="white", fontsize=9,
                fontweight="bold")
        ax.text(3.15, cy - 0.42, f"{n_head} 头 × {head_dim}", ha="left", va="top",
                fontsize=8, color=C["neutral"])

        keep = oname != "Q"
        ax.text(6.85, cy, ("→  " if keep else "→  ") + fate,
                ha="left", va="center", fontsize=9.5,
                fontweight="bold" if keep else "normal",
                color=col if keep else C["neutral"])

    ax.add_patch(Rectangle((6.7, 0.75), 3.5, 2.8, fc="none",
                           ec=C["memory"], lw=2, ls="--"))
    ax.text(8.45, 0.5, note, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=C["memory"])


def fig6_kv_cache_origin() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2))

    draw_panel(
        axes[0], "MHA · Llama-2-7B：Q 有几个头，K/V 就有几个头",
        q_dim=4096, kv_dim=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
        note="每层每 token 缓存 4096 + 4096 = 8192 个数",
    )
    draw_panel(
        axes[1], "GQA · Llama-3-8B：32 个 Q 头共享 8 个 K/V 头 → K/V 只有 1/4 宽",
        q_dim=4096, kv_dim=1024, n_q_heads=32, n_kv_heads=8, head_dim=128,
        note="每层每 token 缓存 1024 + 1024 = 2048 个数（省 4 倍）",
    )

    fig.suptitle("图 6  KV Cache 存的是什么：一个 token 经过一层，只有 K 和 V 留了下来",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_kv_cache_origin.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"[style] CJK font = {setup()}")
    fig6_kv_cache_origin()
    p = OUT / "fig6_kv_cache_origin.png"
    print(f"[ok] {p.name}  ({p.stat().st_size/1024:.0f} KB)")
