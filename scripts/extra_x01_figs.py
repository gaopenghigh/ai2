"""附加课 X01 配图 · 向量与点积

    .venv/bin/python scripts/extra_x01_figs.py

图号 = 正文阅读顺序：
  fig1  二维向量与点积：三种夹角
  fig2  点积的几何含义：投影
  fig3  one-hot vs embedding
  fig4  实验 C 训出来的词向量空间
  fig5  高维的反直觉：随机向量几乎两两垂直
  fig6  一个 token 的一生（接回 Day 01）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "extra"))
OUT = ROOT / "assets" / "extra" / "x01"
OUT.mkdir(parents=True, exist_ok=True)

RB = LinearSegmentedColormap.from_list("rb", ["#FFFFFF", C["memory"]])


def _arrow(ax, v, color, label, lw=2.6):
    ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                mutation_scale=20))
    ax.text(v[0] * 1.12, v[1] * 1.12, label, color=color, fontsize=12,
            fontweight="bold", ha="center", va="center")


# ------------------------------------------------------------------- 图 1
def fig1_dot_geometry() -> None:
    cases = [
        ("方向一致", np.array([3.0, 1.0]), np.array([2.6, 1.6]), C["ok"]),
        ("互相垂直", np.array([3.0, 1.0]), np.array([-1.0, 3.0]), C["capacity"]),
        ("方向相反", np.array([3.0, 1.0]), np.array([-2.6, -1.6]), C["compute"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    for ax, (name, a, b, col) in zip(axes, cases):
        _arrow(ax, a, C["memory"], "a")
        _arrow(ax, b, col, "b")
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        ang = math.degrees(math.acos(max(-1, min(1, cos))))
        ax.add_patch(Arc((0, 0), 1.6, 1.6, theta1=math.degrees(math.atan2(*a[::-1])),
                         theta2=math.degrees(math.atan2(*b[::-1])),
                         color=C["neutral"], lw=1.6))
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-2.6, 3.8)
        ax.set_aspect("equal")
        ax.axhline(0, color=C["grid"], lw=1)
        ax.axvline(0, color=C["grid"], lw=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}   θ ≈ {ang:.0f}°")
        ax.text(0, -2.25,
                f"a·b = {float(a @ b):+.1f}\ncos θ = {cos:+.2f}",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=col, linespacing=1.7)

    fig.text(0.5, 0.035,
             "点积是一个数，却同时回答了两件事：方向合不合（正负）、有多合（大小）",
             ha="center", fontsize=12, color=C["accent"], fontweight="bold")
    fig.suptitle("图 1 · 点积的符号就是「方向合不合」", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig1_dot_geometry.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_projection() -> None:
    a = np.array([4.0, 0.0])
    b = np.array([2.6, 2.2])
    proj = (a @ b) / (a @ a) * a

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    _arrow(ax, a, C["memory"], "a")
    _arrow(ax, b, C["compute"], "b")
    ax.plot([b[0], proj[0]], [b[1], proj[1]], ls="--", lw=1.8, color=C["neutral"])
    ax.plot([0, proj[0]], [0.001, 0.001], lw=7, color=C["capacity"], alpha=0.9,
            solid_capstyle="butt")
    ax.text(proj[0] / 2, -0.42, "b 在 a 上的投影长度 = |b|cos θ",
            ha="center", va="center", fontsize=11.5, color="#9A7000",
            fontweight="bold")

    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ang = math.degrees(math.acos(cos))
    ax.add_patch(Arc((0, 0), 1.5, 1.5, theta1=0, theta2=ang,
                     color=C["neutral"], lw=1.6))
    ax.text(0.95, 0.28, "θ", fontsize=13, color=C["neutral"], fontweight="bold")

    ax.text(2.0, 3.45, r"$a \cdot b \;=\; |a| \times |b|\cos\theta$",
            fontsize=18, ha="center", va="center", color=C["accent"])
    ax.text(2.0, 2.62,
            "「a 有多长」  ×  「b 沿着 a 的方向有多少」",
            fontsize=12, ha="center", va="center", color=C["neutral"],
            fontweight="bold")

    ax.set_xlim(-1.0, 5.4)
    ax.set_ylim(-1.1, 4.0)
    ax.set_aspect("equal")
    ax.axhline(0, color=C["grid"], lw=1)
    ax.axvline(0, color=C["grid"], lw=1)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.5, 0.035,
             "左边是「一堆乘加」（GPU 擅长），右边是「两个箭头的夹角」（人能想象）—— 同一件事",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")
    fig.suptitle("图 2 · 为什么 a·b = |a||b|cos θ", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig2_projection.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig3_onehot_vs_embedding() -> None:
    words = ["cat", "kitten", "dog", "car", "truck"]
    n = len(words)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # --- 上排：one-hot ---
    ax = axes[0][0]
    oh = np.eye(n)
    ax.imshow(np.hstack([oh, np.zeros((n, 7))]), cmap=RB, vmin=0, vmax=1,
              aspect="auto")
    ax.set_yticks(range(n), words)
    ax.set_xticks([])
    ax.set_title("① one-hot：词表多大，向量就多长")
    ax.text(8.5, n / 2 - 0.5, "…\n50000 维\n只有 1 个 1",
            ha="center", va="center", fontsize=11, color=C["neutral"],
            fontweight="bold", linespacing=1.8)

    ax = axes[0][1]
    sim = oh @ oh.T
    im = ax.imshow(sim, cmap=RB, vmin=0, vmax=1)
    ax.set_xticks(range(n), words, rotation=30, ha="right")
    ax.set_yticks(range(n), words)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.0f}", ha="center", va="center",
                    fontsize=10, color="white" if sim[i, j] > 0.5 else C["neutral"])
    ax.set_title("任意两个词的相似度都是 0")

    # --- 下排：embedding ---
    from x01_vectors import train_embeddings                     # noqa: E402
    vocab, idx, emb, _co = train_embeddings(dim=8)
    e = np.stack([emb[idx[w]] for w in words])
    e = e / np.linalg.norm(e, axis=1, keepdims=True)

    ax = axes[1][0]
    ax.imshow(e, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_yticks(range(n), words)
    ax.set_xticks(range(8), [f"d{i}" for i in range(8)])
    ax.set_title("② embedding：8 个数就够（真实模型 4096 个）")
    for i in range(n):
        for j in range(8):
            ax.text(j, i, f"{e[i,j]:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(e[i, j]) > 0.45 else "black")

    ax = axes[1][1]
    sim = e @ e.T
    ax.imshow(sim, cmap=RB, vmin=0, vmax=1)
    ax.set_xticks(range(n), words, rotation=30, ha="right")
    ax.set_yticks(range(n), words)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if sim[i, j] > 0.55 else C["neutral"])
    ax.set_title("相似度出现了结构：左上角一块、右下角一块")

    fig.text(0.5, 0.035,
             "one-hot 里每个词都和其余所有词等距 —— 没有任何「意思」的信息。"
             "embedding 把「意思」变成了「方向」",
             ha="center", fontsize=12, color=C["accent"], fontweight="bold")
    fig.suptitle("图 3 · 从「编号」到「意思」：为什么要用稠密向量",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "fig3_onehot_vs_embedding.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig4_word_space() -> None:
    from x01_vectors import ANIMALS, VEHICLES, train_embeddings  # noqa: E402

    vocab, idx, emb, _co = train_embeddings(dim=8)
    words = ANIMALS + VEHICLES
    e = np.stack([emb[idx[w]] for w in words])
    e = e / np.linalg.norm(e, axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # --- 左：相似度矩阵 ---
    ax = axes[0]
    sim = e @ e.T
    ax.imshow(sim, cmap=RB, vmin=0, vmax=1)
    ax.set_xticks(range(len(words)), words, rotation=40, ha="right")
    ax.set_yticks(range(len(words)), words)
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if sim[i, j] > 0.55 else C["neutral"])
    ax.add_patch(Rectangle((-0.5, -0.5), 5, 5, fc="none", ec=C["ok"], lw=3))
    ax.add_patch(Rectangle((4.5, 4.5), 5, 5, fc="none", ec=C["memory"], lw=3))
    ax.set_title("① 余弦相似度矩阵：两块自己冒出来的")

    # --- 右：投影到 2D ---
    # --- 右：投影到第一主方向（一条轴就分开了） ---
    ax = axes[1]
    _u, _s, vt = np.linalg.svd(e - e.mean(0))
    proj = (e - e.mean(0)) @ vt[0]
    order = np.argsort(proj)
    ws = [words[i] for i in order]
    vs = proj[order]
    cols = [C["ok"] if w in ANIMALS else C["memory"] for w in ws]
    ax.barh(np.arange(len(ws)), vs, color=cols, height=0.62, alpha=0.92)
    ax.set_yticks(np.arange(len(ws)), ws)
    ax.axvline(0, color=C["neutral"], lw=1.4)
    ax.set_xlabel("在第一主方向上的坐标")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.text(-0.55, len(ws) - 0.6, "动物", ha="center", fontsize=13,
            fontweight="bold", color=C["ok"])
    ax.text(0.55, len(ws) - 0.6, "交通工具", ha="center", fontsize=13,
            fontweight="bold", color=C["memory"])
    ax.set_ylim(-0.7, len(ws) + 0.2)
    ax.set_title("② 8 维压到 1 维：一条轴就把两类分开了")

    within = np.mean([sim[i, j] for i in range(10) for j in range(10)
                      if i != j and (i < 5) == (j < 5)])
    across = np.mean([sim[i, j] for i in range(10) for j in range(10)
                      if (i < 5) != (j < 5)])
    fig.text(0.5, 0.035,
             f"类内平均相似度 {within:.2f}　vs　类间平均相似度 {across:.2f} —— "
             "而我们从头到尾没告诉过模型谁是动物谁是车",
             ha="center", fontsize=12, color=C["accent"], fontweight="bold")
    fig.suptitle("图 4 · 只数「谁和谁一起出现」，就长出了语义结构",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUT / "fig4_word_space.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
def fig5_high_dim() -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    for d, col in ((2, C["compute"]), (8, C["capacity"]),
                   (64, C["ok"]), (4096, C["memory"])):
        a = rng.normal(size=(40000, d))
        b = rng.normal(size=(40000, d))
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ax.hist(c, bins=120, density=True, histtype="step", lw=2.4,
                color=col, label=f"d = {d}")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("两个随机向量的 cos θ")
    ax.set_ylabel("概率密度")
    ax.legend()
    ax.set_title("① 维度越高，cos θ 越挤在 0 附近（= 越接近垂直）")

    ax = axes[1]
    ds = np.array([2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    stds = []
    for d in ds:
        a = rng.normal(size=(20000, d))
        b = rng.normal(size=(20000, d))
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        stds.append(c.std())
    ax.loglog(ds, stds, "o", ms=9, color=C["memory"], label="实测标准差")
    ax.loglog(ds, 1 / np.sqrt(ds), "-", lw=2.4, color=C["compute"],
              label=r"理论值 $1/\sqrt{d}$")
    ax.set_xlabel("维度 d")
    ax.set_ylabel("cos θ 的标准差")
    ax.legend()
    ax.set_title("② 标准差正好是 1/√d（实测点全落在线上）")
    ax.annotate("d = 4096 时只有 0.016\n随便两个向量几乎必然垂直",
                xy=(4096, 1 / math.sqrt(4096)), xytext=(22, 0.026),
                fontsize=10.5, color=C["memory"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.6, color=C["memory"]))

    fig.text(0.5, 0.035,
             "好处：高维空间「很空」，能塞下十几万个互不干扰的词向量　|　"
             r"代价：点积的量级随 $\sqrt{d}$ 涨 —— 这就是注意力里 $1/\sqrt{d}$ 的来历",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")
    fig.suptitle("图 5 · 高维空间的反直觉：随机两个向量几乎一定垂直",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig5_high_dim.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 6
def fig6_token_life() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    steps = [
        (2, 16, "「猫」", "一个字", C["accent"], "人看得懂的东西"),
        (22, 16, "12345", "token id", "#8D6A9F", "查词表得到的编号\n（还只是个编号）"),
        (42, 16, "[0.2, -1.3,\n 0.7, …]", "embedding", C["ok"],
         "查嵌入表拿到的\n4096 个数 = 一个方向"),
        (62, 16, "[1.1, 0.4,\n -0.2, …]", "过完 32 层", C["capacity"],
         "还是 4096 个数\n但已经「读过上下文」"),
        (82, 16, "「咪」", "下一个字", C["compute"], "和词表里每个词做点积\n取最像的那个"),
    ]
    for x, w, big, small, col, note in steps:
        ax.add_patch(Rectangle((x, 5.4), w, 2.9, fc=col, ec="white", lw=2,
                               alpha=0.92))
        ax.text(x + w / 2, 6.85, big, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", linespacing=1.5)
        ax.text(x + w / 2, 8.75, small, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=col)
        ax.text(x + w / 2, 4.0, note, ha="center", va="center", fontsize=9.5,
                color=C["neutral"], linespacing=1.7)
        if x > 2:
            ax.annotate("", xy=(x - 0.6, 6.85), xytext=(x - 3.4, 6.85),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                        color=C["neutral"], mutation_scale=18))

    ax.text(50, 1.6,
            "★ 今天讲的「向量 + 点积」，在这条链上出现了两次：\n"
            "第 3 步「一个 token = 一行 4096 个数」（Day 01 §3）　和　"
            "第 5 步「和词表做点积挑最像的」",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C["accent"], linespacing=1.9)

    fig.suptitle("图 6 · 一个 token 的一生（接回 Day 01）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig6_token_life.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    fig1_dot_geometry()
    fig2_projection()
    fig3_onehot_vs_embedding()
    fig4_word_space()
    fig5_high_dim()
    fig6_token_life()
    for p in sorted(OUT.glob("*.png")):
        print("  ->", p.relative_to(ROOT))
