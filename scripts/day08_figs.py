"""Day 08 配图：残差流、Q/K/V 查询、Attention 与 FFN 分工、参数预算。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "day08"))
from param_count import Config, layer_counts, parameter_counts

OUT = ROOT / "assets" / "day08"

# §2.4 的小例子：特征为 [名词, 有生命, 代词]，Q/K 空间为 [是名词, 有生命]
TOKENS = ["小猫", "纸箱", "它"]
X = {"小猫": [1, 1, 0], "纸箱": [1, 0, 0], "它": [0, 0, 1]}
Q_IT = [1, 2]
K = {"小猫": [1, 1], "纸箱": [1, 0], "它": [0, 0]}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


RAW_SCORES = [dot(X["它"], X[t]) for t in TOKENS]
QK_SCORES = [dot(Q_IT, K[t]) for t in TOKENS]
RAW_WEIGHTS = softmax(RAW_SCORES)
QK_WEIGHTS = softmax(QK_SCORES)


def box(ax, left, bottom, width, height, text, color, fontsize=11, text_color="white"):
    ax.add_patch(Rectangle((left, bottom), width, height,
                           facecolor=color, edgecolor="white", linewidth=1.5))
    ax.text(left + width / 2, bottom + height / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, linespacing=1.5)


def arrow(ax, start, end, color=None, lw=1.7):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=lw, color=color or C["neutral"]))


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.32, facecolor="white", edgecolor=C["ok"], lw=2, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=15, color=C["ok"],
            fontweight="bold", zorder=4)


def fig1_decoder():
    fig, axes = plt.subplots(1, 2, figsize=(13, 8),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 12))
        ax.axis("off")
    overview, block = axes

    overview.set_title("整模型：从 token ID 到下一个 token")
    stages = [
        ("token ID\nT 个整数，不能做计算", C["neutral"]),
        ("Embedding 查表\n每个位置一个 d 维向量 (T, d)", C["memory"]),
        ("Decoder block × L\n取信息 → 加工，形状保持 (T, d)", C["compute"]),
        ("Final RMSNorm", C["neutral"]),
        ("LM head：d → 词表大小 V\n下一个 token 的分数", C["memory"]),
    ]
    for index, (text, color) in enumerate(stages):
        bottom = 9.9 - 2.1 * index
        box(overview, 0.6, bottom, 8.8, 1.3, text, color, 10.5)
        if index < len(stages) - 1:
            arrow(overview, (5, bottom), (5, bottom - 0.8))

    block.set_title("一个 block：子层从残差流读取，再把增量加回去")
    stream_x = 1.6
    block.plot([stream_x, stream_x], [11.2, 0.9], color=C["ok"], lw=6, solid_capstyle="butt")
    arrow(block, (stream_x, 1.1), (stream_x, 0.4), C["ok"], 2.5)
    block.text(stream_x, 11.55, "残差流 x：(T, d)", ha="center", va="center",
               color=C["ok"], fontsize=11.5, fontweight="bold")
    block.text(stream_x, 0.05, "输出 x：(T, d)", ha="center", va="center",
               color=C["ok"], fontsize=11)

    branches = [
        (10.2, "Attention\n从其他位置取信息", C["memory"], 7.0),
        (5.6, "FFN\n在每个位置上加工", C["compute"], 2.4),
    ]
    for read_y, label, color, write_y in branches:
        arrow(block, (stream_x + 0.1, read_y), (3.4, read_y))
        block.text(2.5, read_y + 0.3, "读", ha="center", fontsize=10, color=C["neutral"])
        box(block, 3.4, read_y - 0.4, 3.0, 0.8, "RMSNorm", C["neutral"], 10.5)
        arrow(block, (4.9, read_y - 0.4), (4.9, read_y - 1.0))
        box(block, 3.4, read_y - 2.3, 5.4, 1.3, label, color, 11.5)
        block.plot([4.9, 4.9], [read_y - 2.3, write_y], color=C["neutral"], lw=1.7)
        arrow(block, (4.9, write_y), (stream_x + 0.35, write_y))
        block.text(3.2, write_y + 0.3, "加回增量", ha="center", fontsize=10, color=C["neutral"])
        plus(block, stream_x, write_y)
    block.text(9.9, 6.4, "输出必须是 d 维，\n才能加回残差流", ha="right", va="center",
               fontsize=10, color=C["neutral"], linespacing=1.5)
    fig.suptitle("图 1 · 每层两步：Attention 取信息，FFN 加工；残差流贯穿所有层", fontsize=15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(OUT / "fig1_decoder.png")
    plt.close(fig)


def fig2_qkv_lookup():
    fig = plt.figure(figsize=(13, 9.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)
    top = fig.add_subplot(grid[0, :])
    top.set(xlim=(0, 13), ylim=(0, 6))
    top.axis("off")
    top.set_title("用 Q/K/V 查询：Query 表达需求，Key 用来被匹配，Value 是交出的内容")

    box(top, 0.1, 2.3, 2.7, 1.5, "“它”的 Query\nq = [1, 2]\n找名词，更要有生命",
        C["accent"], 10.5)
    contents = {"小猫": "猫的内容", "纸箱": "纸箱的内容", "它": "代词的内容"}
    for index, (token, score, weight) in enumerate(zip(TOKENS, QK_SCORES, QK_WEIGHTS)):
        y = 4.6 - 1.75 * index
        box(top, 4.0, y, 2.6, 0.95, f"{token} 的 Key\nk = {K[token]}", C["memory"], 10)
        arrow(top, (2.8, 3.05), (4.0, y + 0.47))
        top.text(7.0, y + 0.47, f"q·k = {score}", ha="left", va="center", fontsize=10.5)
        top.text(8.45, y + 0.47, f"→ {weight:.3f}", ha="left", va="center",
                 fontsize=11, color=C["compute"], fontweight="bold")
        box(top, 9.6, y, 1.6, 0.95, f"v_{token}\n{contents[token]}", C["ok"], 9.5)
        arrow(top, (11.2, y + 0.47), (11.75, 3.05), C["compute"], 0.6 + 3.5 * weight)
    box(top, 11.75, 2.3, 1.2, 1.5, "“它”\n的输出", C["compute"], 10.5)
    top.text(7.6, 5.75, "softmax 权重", ha="center", fontsize=10, color=C["compute"])
    top.text(6.5, 0.0, "输出 ≈ 0.844·v_小猫 + 0.114·v_纸箱 + 0.042·v_它：主要取到了小猫的内容",
             ha="center", va="bottom", fontsize=11)

    panels = [
        (fig.add_subplot(grid[1, 0]), RAW_SCORES, RAW_WEIGHTS, C["neutral"],
         "不用投影：分数 = x_它 · x_j", "只看自己：取不到先行词"),
        (fig.add_subplot(grid[1, 1]), QK_SCORES, QK_WEIGHTS, C["compute"],
         "用 Q/K 投影：分数 = q_它 · k_j", "找到小猫"),
    ]
    for ax, scores, weights, color, title, verdict in panels:
        bars = ax.bar(TOKENS, weights, color=color, width=0.55)
        ax.set(ylim=(0, 1.05), ylabel="“它”分给各位置的注意力权重", title=title)
        for bar, score, weight in zip(bars, scores, weights):
            ax.text(bar.get_x() + bar.get_width() / 2, weight + 0.03,
                    f"{weight:.3f}\n(分数 {score})", ha="center", va="bottom", fontsize=9.5)
        ax.text(0.97, 0.93, verdict, transform=ax.transAxes, ha="right", va="top",
                fontsize=11.5, color=color, fontweight="bold")
        ax.grid(axis="x", visible=False)
    fig.suptitle("图 2 · 为什么需要 Q、K、V：直接用隐藏状态打分，每个位置只会看自己", fontsize=15)
    fig.text(0.5, 0.01, "特征 [名词, 有生命, 代词]：x_小猫=[1,1,0]，x_纸箱=[1,0,0]，x_它=[0,0,1]；"
             "为直观省略 1/√d_h；真实模型的特征由训练得到",
             ha="center", fontsize=10, color=C["neutral"])
    fig.savefig(OUT / "fig2_qkv_lookup.png")
    plt.close(fig)


def fig3_two_mixers():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 7))
        ax.axis("off")
    attention, ffn = axes

    attention.set_title("Attention：收集。跨位置，权重随内容现场计算")
    for index, (token, weight) in enumerate(zip(TOKENS, QK_WEIGHTS)):
        bottom = 5.2 - 1.45 * index
        box(attention, 0.3, bottom, 2.6, 0.8, f"v_{token}", C["ok"])
        arrow(attention, (2.95, bottom + 0.4), (6, 3.35), C["neutral"], 0.6 + 5 * weight)
        attention.text(3.45, bottom + 0.75, f"{weight:.3f}", ha="left", va="center",
                       color=C["compute"], fontsize=12)
    box(attention, 6, 2.85, 3.8, 1, "“它”的输出\n主要是小猫的内容", C["memory"])
    attention.text(5, 0.95, "权重 = softmax(q·k)：换一句话就会变\n权重确定后，输出只是 Value 的加权平均",
                   ha="center", va="center", fontsize=11, linespacing=1.6)

    ffn.set_title("FFN：加工。逐位置，所有位置共用同一套权重")
    for index, token in enumerate(TOKENS):
        bottom = 5.2 - 1.45 * index
        box(ffn, 0.2, bottom, 1.8, 0.8, token, C["memory"])
        arrow(ffn, (2, bottom + 0.4), (2.8, bottom + 0.4))
        box(ffn, 2.8, bottom - 0.05, 4.1, 0.9, "d → 4d → d\n非线性", C["compute"], 10)
        arrow(ffn, (6.9, bottom + 0.4), (7.7, bottom + 0.4))
        box(ffn, 7.7, bottom, 2.1, 0.8, "增量", C["ok"])
    ffn.text(5, 0.95, "每行进来时已带有 Attention 取到的上下文\nFFN 不再跨行，只专心“消化”",
             ha="center", va="center", fontsize=11, linespacing=1.6)
    fig.suptitle("图 3 · 分工：Attention 负责收集信息，FFN 负责加工信息", fontsize=15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "fig3_two_mixers.png")
    plt.close(fig)


def fig4_parameter_budget():
    config = Config()
    counts = layer_counts(config)
    attention = sum(counts[name] for name in ("Q projection", "K projection",
                                             "V projection", "O projection"))
    ffn = sum(counts[name] for name in ("FFN up", "FFN down", "FFN gate"))
    share = ffn / (attention + ffn)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    ax = axes[0]
    values = np.array([attention, ffn]) / 1e6
    bars = ax.barh([1, 0], values, color=[C["memory"], C["compute"]], height=0.5)
    ax.set(yticks=[1, 0], yticklabels=["Attention\nQ/K/V/O", "SwiGLU FFN\ngate/up/down"],
           xlim=(0, 175), ylim=(-0.7, 1.8), xlabel="百万参数 / 层",
           title="每层：FFN 是 Attention 的约 2 倍")
    for bar, value in zip(bars, values):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, f"{value:.2f} M",
                va="center", fontsize=11)
    ax.text(0.5, 0.95, f"FFN 占每层大矩阵参数的 {share:.1%}", transform=ax.transAxes,
            ha="center", va="top", color=C["compute"], fontsize=11)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.axis("off")
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.set_title("从 12d²L 到精确计数")
    total = sum(parameter_counts(config).values())
    approx = 12 * config.width ** 2 * config.layers
    entries = [
        ("12d²L：经典结构的主干估算", f"{approx:,}"),
        ("+ SwiGLU 实际中间宽度带来的差额", "+33,554,432"),
        ("+ 全部 RMSNorm", "+266,240"),
        ("+ Embedding 与独立 LM head", "+262,144,000"),
    ]
    for index, (label, value) in enumerate(entries):
        baseline = 5.8 - index * 1.05
        ax.text(0.1, baseline, label, va="center", fontsize=10.5)
        ax.text(9.9, baseline, value, ha="right", va="center", fontsize=10.5)
    ax.plot([0.1, 9.9], [1.95, 1.95], color=C["grid"])
    ax.text(5, 1.25, f"{total:,} ≈ 6.74 B", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C["memory"])
    ax.text(5, 0.45, f"估算误差约 {1 - approx / total:.0%}", ha="center", va="center",
            fontsize=10.5, color=C["neutral"])
    fig.suptitle("图 4 · Llama-2-7B 的参数在哪里：2/3 在 FFN，12d²L 已足够接近", fontsize=15)
    fig.text(0.5, 0.025, "配置：d=4096，L=32，m=11008，V=32000；MHA、无 bias、RMSNorm、输入输出不共享",
             ha="center", fontsize=10.5, color=C["neutral"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(OUT / "fig4_parameter_budget.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_decoder()
    fig2_qkv_lookup()
    fig3_two_mixers()
    fig4_parameter_budget()
    print("Figures:", OUT)
