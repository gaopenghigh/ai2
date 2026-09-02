"""Day 01 加餐配图：真实模型的 KV Cache 演进

  fig7_kv_evolution.png   每 token KV 开销 与 1M 上下文总量（6 个真实模型）

数据来自 labs/day01/mem_calc.py（配置取自 HuggingFace 真实 config.json）。
运行：uv run python scripts/day01_extra_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, C

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from mem_calc import MODELS, GB  # noqa: E402

OUT = ROOT / "assets" / "day01"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "Llama-2-7B": "Llama-2-7B\nMHA · 2023",
    "Qwen3-32B": "Qwen3-32B\nGQA-8 · 2025",
    "Qwen3-30B-A3B": "Qwen3-30B-A3B\nMoE+GQA-4 · 2025",
    "Qwen3.8-27B": "Qwen3.8-27B\n混合线性注意力 · 2026",
    "DeepSeek-V3": "DeepSeek-V3\nMLA · 2024",
    "DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash\nMLA+CSA/HCA · 2026",
}


def fig7_kv_evolution() -> None:
    models = list(MODELS)
    labels = [SHORT[m.name] for m in models]
    per_kb = np.array([m.kv_bytes_per_token() / 1024 for m in models])
    at1m_gb = np.array([m.kv_bytes(1_048_576) / GB for m in models])
    y = np.arange(len(models))

    # 颜色：按 KV 大小排序，越小越"绿"（每个模型一个独立颜色）
    palette = ["#8B1A2B", C["compute"], C["capacity"], C["accent"], C["memory"], C["ok"]]
    cols = [""] * len(models)
    for rank, idx in enumerate(per_kb.argsort()[::-1]):
        cols[idx] = palette[rank]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- 左：每 token KV ---
    ax1.barh(y, per_kb, color=cols, height=0.6)
    ax1.set_xscale("log")
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_ylim(len(models) - 0.4, -1.1)
    ax1.set_xlabel("每个 token 的 KV Cache (KB，对数轴)")
    ax1.set_xlim(3, 2500)
    for i, v in enumerate(per_kb):
        ax1.text(v * 1.15, i, f"{v:.1f} KB", va="center", fontsize=9,
                 fontweight="bold")
    ax1.set_title("每多生成 1 个 token，多占多少显存", fontsize=12, fontweight="bold")

    # --- 右：1M 上下文 ---
    ax2.barh(y, at1m_gb, color=cols, height=0.6)
    ax2.set_xscale("log")
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_ylim(len(models) - 0.4, -1.1)
    ax2.set_xlabel("1M token 上下文的 KV Cache 总量 (GB，对数轴)")
    ax2.set_xlim(3, 6000)
    for i, v in enumerate(at1m_gb):
        ax2.text(v * 1.15, i, f"{v:.0f} GB" if v >= 10 else f"{v:.1f} GB",
                 va="center", fontsize=9, fontweight="bold", zorder=5,
                 bbox=dict(fc="white", ec="none", pad=1.0))

    for gb, name, col in [(6, "RTX 4050\n6 GB", C["compute"]),
                          (80, "单张 H100\n80 GB", C["accent"]),
                          (640, "8×H100\n640 GB", "#3F4A56")]:
        ax2.axvline(gb, color=col, ls="--", lw=1.8, zorder=0)
        ax2.text(gb, -0.62, name, ha="center", va="center", fontsize=8.5,
                 color=col, fontweight="bold",
                 bbox=dict(fc="white", ec="none", pad=1.5))
    ax2.set_title("撑住 1M 上下文，光 KV 就要这么多", fontsize=12, fontweight="bold")

    span = per_kb.max() / per_kb.min()
    fig.suptitle(f"图 7  KV Cache 的架构演进：三年间同样上下文的显存开销降了 {span:.0f} 倍",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_kv_evolution.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"[style] CJK font = {setup()}")
    fig7_kv_evolution()
    p = OUT / "fig7_kv_evolution.png"
    print(f"[ok] {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")
