"""X03 配图：分数到概率、温度、稳定平移、因果 mask。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "extra"))
from x03_softmax import stable_log_softmax, stable_softmax

OUT = ROOT / "assets" / "extra" / "x03"
LOGITS = torch.tensor([0.0, math.log(2), math.log(4)], dtype=torch.float64)
COLORS = [C["memory"], C["capacity"], C["compute"]]


def fig1_scores_to_probs() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8))
    panels = [
        (LOGITS.numpy(), "分数 z：任意实数", "分数", ["0", "ln 2", "ln 4"]),
        (LOGITS.exp().numpy(), "exp(z)：正的相对权重", "未归一化权重", ["1", "2", "4"]),
        (stable_softmax(LOGITS).numpy(), "除以总和 7：得到概率", "概率", ["1/7", "2/7", "4/7"]),
    ]
    for ax, (values, title, ylabel, labels) in zip(axes, panels):
        ax.bar(range(3), values, color=COLORS, width=0.55)
        for index, (value, label) in enumerate(zip(values, labels)):
            ax.text(index, value + max(values) * 0.05, label, ha="center", va="bottom", fontsize=12)
        ax.set(xticks=range(3), xticklabels=["A", "B", "C"], ylim=(0, max(values) * 1.35),
               title=title, ylabel=ylabel)
        ax.grid(axis="x", visible=False)
    fig.suptitle("图 1 · 分数不是概率：先产生正权重，再在同一组候选之间分配总量 1", fontsize=14)
    fig.text(0.5, 0.025, "分数差 ln 2 变成概率比 2；共同加一个常数，不改变任何概率比。",
             ha="center", fontsize=11, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(OUT / "fig1_scores_to_probs.png")
    plt.close(fig)


def fig2_temperature() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True)
    for ax, temperature in zip(axes, (0.5, 1.0, 2.0)):
        probs = stable_softmax(LOGITS, temperature)
        entropy = -(probs * stable_log_softmax(LOGITS, temperature)).sum().item()
        ax.bar(range(3), probs.numpy(), color=COLORS, width=0.55)
        for index, value in enumerate(probs.tolist()):
            ax.text(index, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set(xticks=range(3), xticklabels=["A", "B", "C"], ylim=(0, 1.04),
               title=f"温度 {temperature:g}：" + ("更集中" if temperature < 1 else
                                              "原分布" if temperature == 1 else "更平缓"))
        ax.text(0.5, 0.96, f"p(C)/p(A) = {(probs[2] / probs[0]).item():g}；熵 = {entropy:.3f} nat",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("概率")
    fig.suptitle("图 2 · 温度缩放分数差：最高分仍是 C，但抽到其他候选的机会变了", fontsize=14)
    fig.text(0.5, 0.025, r"$\log(p_i/p_j)=(z_i-z_j)/\tau$"
             + "    正温度不改变排名；softmax 本身不抽样。", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(OUT / "fig2_temperature.png")
    plt.close(fig)


def fig3_stable_shift() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.3))
    ax.axis("off")
    ax.set(xlim=(0, 12), ylim=(0, 5))
    for center, text in ((2, "分数"), (6, "指数权重"), (10, "归一化结果")):
        ax.text(center, 4.55, text, ha="center", va="center", fontsize=12, fontweight="bold")
    rows = [
        (3.05, ["[1000, 1001, 1002]\n直接 exp", "[inf, inf, inf]\nFP32 也会溢出",
                "inf / inf\nNaN"], C["compute"]),
        (1.05, ["[-2, -1, 0]\n先减最大值 1002", "[0.1353, 0.3679, 1]\n每项不超过 1",
                "[0.0900, 0.2447, 0.6652]\n总和为 1"], C["memory"]),
    ]
    for bottom, texts, color in rows:
        for column, text in enumerate(texts):
            left = column * 4 + 0.35
            ax.add_patch(Rectangle((left, bottom), 3.3, 1.05, fc=color, ec="white", lw=1.5))
            ax.text(left + 1.65, bottom + 0.525, text, ha="center", va="center", color="white",
                    fontsize=11, linespacing=1.6)
            if column < 2:
                ax.annotate("", xy=(left + 4, bottom + 0.525), xytext=(left + 3.3, bottom + 0.525),
                            arrowprops=dict(arrowstyle="->", color=C["neutral"], lw=1.7))
    ax.text(6, 0.42, "数学上是同一个分布；浮点运算的中间值不同，决定了代码能否算出来。",
            ha="center", va="center", fontsize=11)
    fig.suptitle("图 3 · 减最大值不是近似，也不是裁剪：把最大的指数固定为 exp(0)=1", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "fig3_stable_shift.png")
    plt.close(fig)


def fig4_causal_rows() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]])
    allowed = torch.ones(3, 3, dtype=torch.bool).tril()
    probs = stable_softmax(scores.masked_fill(~allowed, -torch.inf))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    for ax in axes:
        ax.imshow((~allowed).numpy(), cmap=ListedColormap(["#E6F2F3", "#D8DEE4"]), vmin=0, vmax=1)
        ax.set(xticks=range(3), xticklabels=["Key 0", "Key 1", "Key 2"], yticks=range(3),
               yticklabels=["Query 0", "Query 1", "Query 2"], xlabel="沿这一维分配概率")
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(visible=False)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(3):
        for column in range(3):
            text = f"{scores[row, column].item():.0f}" if allowed[row, column] else r"$-\infty$"
            axes[0].text(column, row, text, ha="center", va="center", fontsize=15)
            axes[1].text(column, row, f"{probs[row, column].item():.3f}",
                         ha="center", va="center", fontsize=14)
    axes[0].set_title("先将未来位置置为负无穷")
    axes[1].set_title("再沿 Key 维做 softmax：每行和为 1")
    fig.suptitle("图 4 · 不同 Query 各分各的：禁止的位置不进入分母，也不占走概率", fontsize=14)
    fig.text(0.5, 0.025, "灰色为被 mask 的位置。整行都被 mask 时，没有可定义的归一化分布；教学实现明确报错。",
             ha="center", fontsize=10.5, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(OUT / "fig4_causal_rows.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_scores_to_probs()
    fig2_temperature()
    fig3_stable_shift()
    fig4_causal_rows()
    print("Figures:", OUT)