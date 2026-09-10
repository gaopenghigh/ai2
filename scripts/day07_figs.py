"""Day 07 配图 · Week 1 复盘：四堵墙合成一个公式

    .venv/bin/python scripts/day07_figs.py

图号 = 正文阅读顺序：
  fig1  六天 → 一个公式：每一天贡献了哪一项
  fig2  四道关卡：容量先过，算力/带宽取 max，软件开销再加上去
  fig3  五台机器的画像卡：三根柱子涨了几百倍，第四根纹丝不动
  fig4  主线案例：同一个请求在四台机器上的 TPOT 分解与端到端
  fig5  Week 1 的四堵墙 → 后面 11 周每个技术在拆哪一堵
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from _style import C, setup

OUT = Path(__file__).resolve().parents[1] / "assets" / "day07"
OUT.mkdir(parents=True, exist_ok=True)

COMPUTE, MEM, CAP, SOFT = C["compute"], C["memory"], C["capacity"], C["accent"]


def _box(ax, x, y, w, h, fc, text, fs=10, tc="white", weight="bold", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec="white", lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, linespacing=1.6, zorder=10)


# ------------------------------------------------------------------- 图 1
DAYS = [
    ("Day 01", "一个 token 的一生", "decode 上限 = BW / 权重字节\nKV Cache 从哪来",
     ["S", "M"]),
    ("Day 02", "Roofline", "max(F/P, S/B) 这个 max\n以及平衡点 = P/B", ["P", "B"]),
    ("Day 03", "GPU 架构", "P 是怎么来的：核数×ALU×2×频率\nB 是怎么来的：位宽×每线速率",
     ["P", "B"]),
    ("Day 04", "存储层次", "t = t0 + S/B 里的 t0\n融合能改小的是 S", ["S", "T"]),
    ("Day 05", "数值格式", "每参数字节 b\n同时决定了 S 和 M", ["S", "M"]),
    ("Day 06", "软件栈剖面", "N · t_launch\n分子不随硬件变快", ["T"]),
]
TERM_COLOR = {"P": COMPUTE, "B": MEM, "M": CAP, "S": MEM, "T": SOFT}
TERM_NAME = {"P": "算力 P", "B": "带宽 B", "M": "容量 M", "S": "字节 S", "T": "软件 N·t"}


def fig1_assembly() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.6, 7.35)

    # ---- 顶部：主公式 ----
    ax.text(0.5, 7.05, "Week 1 六天，其实只在拼这一个公式",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C["neutral"])
    _box(ax, 0.055, 6.30, 0.20, 0.52, CAP, "先过容量墙\nW + KV  ≤  M", fs=11, tc="#3B2F0B")
    ax.text(0.285, 6.56, "然后", ha="center", va="center", fontsize=11,
            color=C["neutral"])
    _box(ax, 0.325, 6.30, 0.155, 0.52, COMPUTE, "F / P\n算力要这么久", fs=11)
    ax.text(0.497, 6.56, "max", ha="center", va="center", fontsize=12,
            fontweight="bold", color=C["neutral"])
    _box(ax, 0.515, 6.30, 0.155, 0.52, MEM, "S / B\n带宽要这么久", fs=11)
    ax.text(0.688, 6.56, "+", ha="center", va="center", fontsize=15,
            fontweight="bold", color=C["neutral"])
    _box(ax, 0.705, 6.30, 0.185, 0.52, SOFT, "N · t_launch\n软件要这么久", fs=11)
    ax.text(0.945, 6.56, "= t", ha="center", va="center", fontsize=14,
            fontweight="bold", color=C["neutral"])

    # ---- 六天 ----
    for i, (day, topic, gave, terms) in enumerate(DAYS):
        y = 5.20 - i * 0.88
        _box(ax, 0.055, y, 0.115, 0.68, C["neutral"], f"{day}\n{topic}", fs=10.5)
        ax.text(0.195, y + 0.34, gave, ha="left", va="center", fontsize=10.2,
                color="#333", linespacing=1.7)
        for j, t in enumerate(terms):
            _box(ax, 0.735 + j * 0.115, y + 0.13, 0.105, 0.42, TERM_COLOR[t],
                 TERM_NAME[t], fs=9.5,
                 tc="#3B2F0B" if TERM_COLOR[t] == CAP else "white")

    ax.text(0.195, 5.98, "这一天量出了什么", ha="left", va="center", fontsize=10.5,
            color=C["neutral"], fontweight="bold")
    ax.text(0.735, 5.98, "进了公式的哪一项", ha="left", va="center", fontsize=10.5,
            color=C["neutral"], fontweight="bold")

    fig.text(0.5, 0.035,
             "六天看起来在讲六件事，其实是在给同一个公式凑齐参数 —— "
             "Day 07 要做的，就是把这几个参数一次性量出来",
             ha="center", fontsize=11.5, color=C["neutral"], fontweight="bold")
    fig.suptitle("图 1 · Week 1 六天，各贡献了统一性能模型的哪一项",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "fig1_assembly.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_four_gates() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.9))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.85, 4.15)

    def arrow(x0, y0, x1, y1, col=C["neutral"]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), lw=2.0, color=col,
                                     arrowstyle="-|>", mutation_scale=16,
                                     shrinkA=0, shrinkB=0))

    ax.text(0.15, 2.1, "一层\n的活", ha="center", va="center", fontsize=10.5,
            color=C["neutral"], fontweight="bold", linespacing=1.6)
    arrow(0.55, 2.1, 1.15, 2.1)

    # ① 容量墙
    _box(ax, 1.2, 1.28, 1.5, 1.64, CAP,
         "① 容量墙\nW + KV ≤ M\n\n装不下就没有\n「快慢」可言", fs=9.5, tc="#3B2F0B")
    ax.add_patch(FancyArrowPatch((1.95, 1.22), (1.95, 0.42), lw=2.0, color=COMPUTE,
                                 arrowstyle="-|>", mutation_scale=16))
    ax.text(1.95, 0.05, "装不下 → 量化 / 切分 / offload\n（Week 5 / 6 / 7）",
            ha="center", va="center", fontsize=9.5, color=COMPUTE,
            fontweight="bold", linespacing=1.6)
    arrow(2.75, 2.1, 3.35, 2.1)

    # ②③ 并联
    ax.add_patch(Rectangle((3.35, 0.75), 3.05, 2.75, fc="#F4F6F8", ec=C["grid"],
                           lw=1.4, zorder=0))
    ax.text(4.87, 3.72, "② 和 ③ 是并联的：同时在跑，谁慢听谁的",
            ha="center", va="center", fontsize=10.5, color=C["neutral"],
            fontweight="bold")
    _box(ax, 3.5, 2.45, 1.75, 0.85, COMPUTE, "② 算力墙\nF / P", fs=11)
    _box(ax, 3.5, 1.0, 1.75, 0.85, MEM, "③ 带宽墙\nS / B", fs=11)
    ax.add_patch(Polygon([(5.65, 2.15), (6.05, 2.55), (6.45, 2.15), (6.05, 1.75)],
                         fc="white", ec=C["neutral"], lw=1.8, zorder=5))
    ax.text(6.05, 2.15, "max", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C["neutral"], zorder=10)
    arrow(5.25, 2.88, 5.85, 2.35, COMPUTE)
    arrow(5.25, 1.42, 5.85, 1.95, MEM)
    ax.text(4.87, 0.42,
            "算术强度 AI = F/S 决定站哪边：AI < 平衡点 P/B 就是带宽受限（Day 02）",
            ha="center", va="center", fontsize=9.8, color=C["neutral"])

    arrow(6.5, 2.15, 7.05, 2.15)
    ax.text(6.78, 2.5, "+", ha="center", va="center", fontsize=16,
            fontweight="bold", color=C["neutral"])

    # ④ 软件
    _box(ax, 7.1, 1.28, 1.55, 1.64, SOFT,
         "④ 软件开销\nN · t_launch\n\n串联：一分\n躲不掉", fs=9.5)
    arrow(8.7, 2.15, 9.25, 2.15)
    ax.text(9.6, 2.15, "t", ha="center", va="center", fontsize=20,
            fontweight="bold", color=C["neutral"])

    fig.text(0.5, 0.035,
             "容量墙是「能不能」，算力/带宽墙是「多快」，软件开销是「白交多少过路费」 —— "
             "任何优化手段，都只能作用在这四项中的某一项上",
             ha="center", fontsize=11.5, color=C["neutral"], fontweight="bold")
    fig.suptitle("图 2 · 一个请求要过的四道关卡", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "fig2_four_gates.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
MACHINES = ["Mac mini M4\n(实测)", "RTX 4050", "RTX 4090", "A100 80G", "H100 SXM"]
TFLOPS = [3.57, 24.0, 165.0, 312.0, 989.0]
BW = [89.4, 192.0, 1008.0, 2039.0, 3350.0]
RIDGE = [40, 125, 164, 153, 295]
TLAUNCH = [4.64, 5.0, 5.0, 5.0, 5.0]


def fig3_profile_cards() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.4))
    panels = [
        (TFLOPS, "① 峰值算力 P (TFLOPS)", COMPUTE, True, f"{TFLOPS[-1]/TFLOPS[0]:.0f}× "),
        (BW, "② 显存带宽 B (GB/s)", MEM, True, f"{BW[-1]/BW[0]:.0f}× "),
        (RIDGE, "平衡点 P/B (FLOP/Byte)", CAP, False, f"{RIDGE[-1]/RIDGE[0]:.1f}× "),
        (TLAUNCH, "④ 发射开销 t_launch (µs)", SOFT, False, "1.1× "),
    ]
    for ax, (vals, title, col, log, gain) in zip(axes, panels):
        x = np.arange(len(MACHINES))
        ax.bar(x, vals, color=col, alpha=0.92, width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v * (1.12 if log else 1.0) + (0 if log else max(vals) * 0.03),
                    f"{v:g}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color="#333", zorder=10)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) * 0.4, max(vals) * 4)
        else:
            ax.set_ylim(0, max(vals) * 1.28)
        ax.set_xticks(x, ["M4", "4050", "4090", "A100", "H100"], fontsize=9.5)
        ax.set_title(title, fontsize=11.5)
        ax.text(0.5, 0.94, f"M4 → H100 涨 {gain}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=col,
                bbox=dict(fc="white", ec="none", pad=2.0), zorder=10)

    axes[3].axhline(5.0, color=C["neutral"], ls="--", lw=1.4)
    fig.text(0.5, 0.035,
             "硬件在疯狂进步的只有前两根柱子；第四根是软件栈的固定成本，"
             "十年没怎么变 —— 所以卡越快，它占的比例越高（Day 06）",
             ha="center", fontsize=11.5, color=C["neutral"], fontweight="bold")
    fig.suptitle("图 3 · 五台机器的画像卡：三样涨了几百倍，一样纹丝不动",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig3_profile_cards.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
# bench_hw.py --predict 的输出：7B INT4 / 2000 in → 500 out / batch 1
CASE = [
    # 名字, 带宽要的 ms, 软件 ms, 端到端 s, tok/s
    ("Mac mini M4\n(实测画像卡)", 50.87, 7.60, 36.72, 13.6),
    ("RTX 4050", 23.70, 8.16, 17.03, 29.4),
    ("RTX 4090", 4.51, 8.16, 6.50, 77.0),
    ("H100 SXM", 1.36, 8.16, 4.78, 104.5),
]


def fig4_mainline() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    names = [c[0] for c in CASE]
    hw = np.array([c[1] for c in CASE])
    sw = np.array([c[2] for c in CASE])
    x = np.arange(len(CASE))

    ax = axes[0]
    ax.bar(x, hw, color=MEM, alpha=0.92, width=0.6, label="搬权重 + 搬 KV（S/B）")
    ax.bar(x, sw, bottom=hw, color=SOFT, alpha=0.92, width=0.6,
           label="软件开销（1632 × t_launch）")
    for xi, (h, s) in enumerate(zip(hw, sw)):
        ax.text(xi, h + s + 1.6, f"{h + s:.1f} ms", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#333", zorder=10)
        ax.text(xi, h + s / 2, f"{s / (h + s) * 100:.0f}%", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=10)
    ax.set_xticks(x, names, fontsize=9.5)
    ax.set_ylim(0, max(hw + sw) * 1.22)
    ax.set_ylabel("TPOT（每个 token 要多久，ms）")
    ax.set_title("decode 的 TPOT 分解（紫色数字 = 软件占比）")
    ax.legend(loc="upper right", fontsize=9.5)

    ax = axes[1]
    e2e = [c[3] for c in CASE]
    ax.bar(x, e2e, color=C["neutral"], alpha=0.85, width=0.6)
    for xi, (v, tps) in enumerate(zip(e2e, [c[4] for c in CASE])):
        ax.text(xi, v + max(e2e) * 0.02, f"{v:.1f} s", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#333", zorder=10)
        ax.text(xi, v / 2, f"{tps:.0f}\ntok/s", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=10)
    ax.set_xticks(x, names, fontsize=9.5)
    ax.set_ylim(0, max(e2e) * 1.18)
    ax.set_ylabel("端到端耗时（秒）")
    ax.set_title("2000 token 输入 → 500 token 输出，端到端")

    ax.annotate("", xy=(3.05, 7.4), xytext=(2.62, 18.6),
                arrowprops=dict(arrowstyle="->", lw=2.0, color=COMPUTE))
    ax.text(2.55, 22.0, "带宽涨 3.3 倍\n端到端只快 1.36 倍", ha="center", va="center",
            fontsize=10, fontweight="bold", color=COMPUTE, linespacing=1.6,
            bbox=dict(fc="white", ec=COMPUTE, lw=1.2, pad=3.0), zorder=10)

    fig.text(0.5, 0.035,
             "7B INT4 · 2000 token 输入 → 500 token 输出 · batch=1 · eager PyTorch —— "
             "全部由 bench_hw.py 的画像卡算出，没有跑模型",
             ha="center", fontsize=11.5, color=C["neutral"], fontweight="bold")
    fig.suptitle("图 4 · 主线案例：同一个请求，四台机器", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUT / "fig4_mainline.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
ROADMAP = [
    ("① 容量墙 M", CAP, "#3B2F0B", [
        "W5  量化 GPTQ / AWQ / FP8",
        "W3  GQA / MLA 压 KV Cache",
        "W5  PagedAttention 消碎片",
        "W6  ZeRO / 梯度检查点",
        "W7  TP / PP：切到多张卡上",
    ]),
    ("② 算力墙 P", COMPUTE, "white", [
        "W4  Tiling / 寄存器分块",
        "W4  Tensor Core / 对齐到 tile",
        "W2  搞清 FLOPs 到底花在哪",
        "W9  Chunked Prefill 填满算力",
    ]),
    ("③ 带宽墙 B", MEM, "white", [
        "W5  FlashAttention：少搬一个量级",
        "W4  算子融合：中间结果不落地",
        "W8  连续批处理：权重摊给更多请求",
        "W8  投机解码：搬一次权重吐多个",
        "W3  Prefix Cache：算过的不再算",
    ]),
    ("④ 软件开销 N·t", SOFT, "white", [
        "W10  torch.compile：融合 + 少发射",
        "W10  CUDA Graph：N 次发射变 1 次",
        "W8  vLLM / TRT-LLM 的主要价值",
        "W4  持久化 kernel",
    ]),
]


def fig5_roadmap() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.set_axis_off()
    ax.set_xlim(0, 4)
    ax.set_ylim(1.05, 6.6)

    for i, (title, col, tc, items) in enumerate(ROADMAP):
        _box(ax, i + 0.06, 5.55, 0.88, 0.62, col, title, fs=12, tc=tc)
        ax.add_patch(FancyArrowPatch((i + 0.5, 5.5), (i + 0.5, 5.15), lw=2.0,
                                     color=col, arrowstyle="-|>", mutation_scale=14))
        for j, it in enumerate(items):
            y = 4.72 - j * 0.72
            ax.add_patch(FancyBboxPatch((i + 0.06, y), 0.88, 0.56,
                                        boxstyle="round,pad=0,rounding_size=0.03",
                                        fc="white", ec=col, lw=1.5))
            wk, _, rest = it.partition("  ")
            ax.text(i + 0.13, y + 0.28, wk, ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=col)
            ax.text(i + 0.32, y + 0.28, rest, ha="left", va="center", fontsize=9.3,
                    color="#333")

    ax.text(2.0, 6.32, "后面 11 周要学的每一个技术，都是在拆这四堵墙中的某一堵",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=C["neutral"])
    ax.text(2.0, 1.35,
            "学新技术时先问一句：它拆的是哪一堵墙？拆不动的那几堵，它一点忙都帮不上。",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=C["ok"])

    fig.suptitle("图 5 · 从 Week 1 的四堵墙，看清后面 11 周的地图",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(OUT / "fig5_roadmap.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    fig1_assembly()
    fig2_four_gates()
    fig3_profile_cards()
    fig4_mainline()
    fig5_roadmap()
    print("figures →", OUT)
