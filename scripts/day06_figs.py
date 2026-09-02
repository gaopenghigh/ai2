"""Day 06 配图 · 软件栈剖面

    .venv/bin/python scripts/day06_figs.py

图号 = 正文阅读顺序：
  fig1  八层抽象：从 y = model(x) 到 SASS
  fig2  CPU 和 GPU 是两条时间线（喂得饱 vs 喂不饱）
  fig3  开销阶梯实测
  fig4  队列深度：连发越多，平均开销越低
  fig5  GPU 越快，软件开销占比越高
  fig6  三条对策：融合 / CUDA Graph / 持久化 kernel
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import C, setup

OUT = Path(__file__).resolve().parents[1] / "assets" / "day06"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- 图 1
LAYERS = [
    ("你的代码", "y = model(x)", "Python 解释器", 0.03, C["accent"]),
    ("nn.Module / autograd", "记录计算图、准备反向", "Python + C++", 0.4, "#8D6A9F"),
    ("Dispatcher", "按 dtype × device × layout 查表", "C++ 虚表跳转", 0.5, C["capacity"]),
    ("ATen 算子", "aten::mm(a, b)", "后端无关的算子定义", 0.3, "#C79A2E"),
    ("后端库", "cuBLAS / cuDNN / MPSGraph", "挑一个具体 kernel", 0.6, C["ok"]),
    ("运行时", "CUDA Runtime / Metal", "打包命令、管流、管显存", 1.0, "#2E7D6E"),
    ("驱动 + 命令队列", "写命令缓冲区并提交", "用户态 + 内核态驱动", 1.5, C["memory"]),
    ("GPU 硬件", "派发 block → SM 执行 SASS", "Day 03 讲的那一层", 0.5, C["compute"]),
]


def fig1_stack() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_axis_off()
    n = len(LAYERS)
    ax.set_xlim(-0.02, 1.34)
    ax.set_ylim(-0.9, n + 0.3)

    total = sum(x[3] for x in LAYERS)
    for i, (name, what, how, us, col) in enumerate(LAYERS):
        y = n - 1 - i
        ax.add_patch(Rectangle((0.02, y), 0.60, 0.78, fc=col, ec="white",
                               lw=1.6, alpha=0.92))
        ax.text(0.06, y + 0.52, name, ha="left", va="center", color="white",
                fontweight="bold", fontsize=11.5)
        ax.text(0.06, y + 0.23, what, ha="left", va="center", color="white",
                fontsize=9.2)
        ax.text(0.65, y + 0.39, how, ha="left", va="center",
                fontsize=9.5, color=C["neutral"])
        # 右侧耗时条
        ax.add_patch(Rectangle((1.03, y + 0.18), us / total * 0.28, 0.42,
                               fc=col, ec="none", alpha=0.75))
        ax.text(1.02, y + 0.39, f"~{us:.1f} µs", ha="right", va="center",
                fontsize=9.5, color=C["neutral"], fontweight="bold")

    ax.annotate("", xy=(-0.005, 0.2), xytext=(-0.005, n - 0.2),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C["neutral"]))
    ax.text(1.03, n + 0.02, "这一层大概花多久", fontsize=10,
            color=C["neutral"], fontweight="bold")
    ax.text(0.65, n + 0.02, "具体在做什么", fontsize=10,
            color=C["neutral"], fontweight="bold")

    ax.text(0.32, -0.55,
            f"八层加起来 ≈ {total:.0f} µs —— 这就是每个 kernel 的「过路费」\n"
            "和这个 kernel 要算多少完全无关",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C["compute"], linespacing=1.8)

    fig.suptitle("图 1 · 从 y = model(x) 到 GPU 上的一条指令，中间隔着八层",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    fig.savefig(OUT / "fig1_stack.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_async_timeline() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0))

    def draw(ax, launch_us, exec_us, title, ok):
        n = 8
        for i in range(n):
            t0 = i * launch_us
            ax.add_patch(Rectangle((t0, 1.05), launch_us * 0.88, 0.5,
                                   fc=C["accent"], ec="white", lw=1.0))
            if i < 3:
                ax.text(t0 + launch_us * 0.44, 1.30, f"发{i}", ha="center",
                        va="center", fontsize=8, color="white", fontweight="bold")
        prev_end = 0.0
        for i in range(n):
            ready = (i + 1) * launch_us          # 命令投递完成的时刻
            start = max(ready, prev_end)
            if start > prev_end + 1e-9 and i > 0:
                ax.add_patch(Rectangle((prev_end, 0.25), start - prev_end, 0.5,
                                       fc="none", ec=C["compute"], lw=1.3,
                                       ls="--", hatch="///"))
            ax.add_patch(Rectangle((start, 0.25), exec_us * 0.94, 0.5,
                                   fc=C["compute"] if ok else C["memory"],
                                   ec="white", lw=1.0))
            if i < 3:
                ax.text(start + exec_us * 0.47, 0.50, f"算{i}", ha="center",
                        va="center", fontsize=8, color="white", fontweight="bold")
            prev_end = start + exec_us

        ax.set_xlim(0, 102)
        ax.set_ylim(0, 2.0)
        ax.set_yticks([0.5, 1.3], ["GPU 时间线", "CPU 时间线"])
        ax.set_xlabel("时间 (µs)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)

    draw(axes[0], 3, 12,
         "① CPU 发得快、GPU 算得慢 → 队列一直是满的，GPU 满载（正常情况）", True)
    axes[0].text(60, 1.75,
                 "CPU 早早发完就闲着 → 「不 sync 计时」量到的就是这一小段",
                 ha="center", fontsize=10, color=C["accent"], fontweight="bold")

    draw(axes[1], 12, 3,
         "② CPU 发得慢、GPU 算得快 → GPU 每算完一个就得干等（CPU-bound）", False)
    axes[1].text(60, 1.75,
                 "斜线区 = GPU 的气泡：硬件在，但没活干",
                 ha="center", fontsize=10, color=C["compute"], fontweight="bold")

    fig.text(0.5, 0.028,
             "决定你在哪种情况的，是「每个 kernel 的发射开销」和「它的实际执行时间」谁大 —— "
             "这就是 CPU-bound / GPU-bound 的定义",
             ha="center", fontsize=11, color=C["neutral"], fontweight="bold")

    fig.suptitle("图 2 · CPU 和 GPU 是两条独立的时间线",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "fig2_async_timeline.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig3_overhead_ladder() -> None:
    rows = [
        ("纯 Python 空函数", 0.025, C["accent"], "解释器"),
        ("读元数据 x.shape", 0.059, "#8D6A9F", "不进 C++"),
        ("元数据操作 x.view(1)", 0.404, C["capacity"], "进 C++，不发 kernel"),
        ("1 个元素相加", 2.77, C["compute"], "★ 第一次真正发 kernel"),
        ("1024² 相加 (12 MB)", 61.6, "#2E7D6E", "开始有工作量"),
        ("4096² 相加 (200 MB)", 2298.0, C["memory"], "数据搬运主导"),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    n = len(rows)
    ypos = np.arange(n)
    vals = [r[1] for r in rows]
    ax.barh(ypos, vals, color=[r[2] for r in rows], height=0.6, alpha=0.92)
    ax.set_xscale("log")
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlim(0.01, 3e5)
    ax.set_xlabel("端到端耗时 µs（对数轴）")

    for i, (_name, v, _c, note) in enumerate(rows):
        lab = f"{v:,.0f} µs" if v >= 100 else f"{v:.3g} µs"
        ax.text(v * 1.7, i, f"{lab}    ·  {note}", va="center",
                fontsize=10, color=C["neutral"], fontweight="bold", zorder=10,
                bbox=dict(fc="white", ec="none", pad=1.5))

    ax.axvline(2.77, color=C["compute"], ls="--", lw=1.8)
    ax.axhspan(2.5, 3.5, color=C["compute"], alpha=0.08)

    fig.text(0.5, 0.035,
             "从「元数据操作」到「发一个最小 kernel」，耗时跳了 7 倍 —— "
             "这一跳就是整个软件栈的过路费（M4 实测 ≈ 2.8 µs）",
             ha="center", fontsize=11.5, color=C["compute"], fontweight="bold")

    fig.suptitle("图 3 · 开销阶梯：时间到底花在哪一层（Mac mini M4 实测）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig3_overhead_ladder.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig4_queue_depth() -> None:
    n = np.array([1, 10, 100, 1000])
    per = np.array([204.2, 24.58, 5.16, 3.18])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    ax.plot(n, per, "o-", lw=2.6, ms=10, color=C["memory"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("连续发射了多少个小 kernel（中间不 sync）")
    ax.set_ylabel("平均每个 kernel 的耗时 (µs)")
    for xi, yi in zip(n, per):
        ax.annotate(f"{yi:.1f} µs", xy=(xi, yi), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=C["memory"], fontweight="bold")
    ax.axhline(3.18, color=C["compute"], ls="--", lw=1.5)
    ax.text(1.2, 3.5, "稳态：流水线打满后的真实单价", fontsize=10,
            color=C["compute"], fontweight="bold")
    ax.set_ylim(2, 400)
    ax.set_title("① 连发越多，平均开销越低")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 8.9, "为什么第一行那么贵？", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C["compute"])
    ax.text(5, 6.6,
            "发 1 个 kernel 再 sync = 204 µs\n"
            "但稳态单价只有 3.2 µs\n\n"
            "差的那 200 µs，全是【同步本身】的代价：\n"
            "提交命令缓冲区 → 等 GPU 跑完 → 等完成通知回到 CPU",
            ha="center", va="center", fontsize=11, color=C["neutral"],
            linespacing=2.0)
    ax.add_patch(Rectangle((0.6, 1.0), 8.8, 3.4, fc="#FDF1F3",
                           ec=C["compute"], lw=2))
    ax.text(5, 3.75, "所以这些写法在循环里是性能杀手", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C["compute"])
    ax.text(5, 2.1,
            ".item()   .cpu()   .numpy()   .tolist()\n"
            "print(tensor)   float(loss)   if tensor > 0",
            ha="center", va="center", fontsize=11, color=C["neutral"],
            family="monospace", linespacing=2.0)
    ax.set_title("② 一次同步 ≈ 60 个 kernel 的发射开销")

    fig.text(0.5, 0.035,
             "实测：循环里每步加一个 .item()，512³ 矩阵乘从 92 µs 变成 337 µs —— 慢了 3.7 倍",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 4 · 队列深度与同步的代价（Mac mini M4 实测）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig4_queue_depth.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
def fig5_bandwidth_vs_overhead() -> None:
    machines = [
        ("Mac mini\nM4", 91e9),
        ("RTX 4050", 192e9),
        ("RTX 4090", 1008e9),
        ("A100 80G", 2039e9),
        ("H100 SXM", 3350e9),
    ]
    wbytes = 403e6          # 一层 Transformer 的权重（d=4096, fp16）
    n_ops, launch_us = 51, 6.0
    over = n_ops * launch_us

    names = [m[0] for m in machines]
    move = [wbytes / m[1] * 1e6 for m in machines]
    share = [over / (mv + over) * 100 for mv in move]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    xs = np.arange(len(names))
    ax.bar(xs, move, color=C["memory"], width=0.6, label="权重搬运（受带宽限制）")
    ax.bar(xs, [over] * len(names), bottom=move, color=C["compute"],
           width=0.6, label=f"{n_ops} 个算子的发射开销（固定 {over:.0f} µs）")
    ax.set_xticks(xs, names)
    ax.set_yscale("log")
    ax.set_ylabel("一层 Transformer 的耗时 (µs，对数轴)")
    ax.set_ylim(50, 20000)
    ax.legend(loc="upper right", fontsize=9.5)
    for i, (mv, sh) in enumerate(zip(move, share)):
        ax.text(i, (mv + over) * 1.25, f"{sh:.0f}%", ha="center",
                fontsize=11, fontweight="bold", color=C["compute"], zorder=10)
    ax.set_title("① 搬运时间随带宽变快，发射开销纹丝不动")

    ax = axes[1]
    ax.plot([m[1] / 1e9 for m in machines], share, "o-", lw=2.8, ms=11,
            color=C["compute"])
    for (name, bw), sh in zip(machines, share):
        ax.annotate(f"{name.replace(chr(10), ' ')}\n{sh:.0f}%",
                    xy=(bw / 1e9, sh), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color=C["neutral"], fontweight="bold", linespacing=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("显存带宽 (GB/s，对数轴)")
    ax.set_ylabel("软件栈开销占总耗时的比例 (%)")
    ax.set_ylim(0, 95)
    ax.axhline(50, color=C["neutral"], ls="--", lw=1.4)
    ax.text(120, 53, "过了这条线，软件栈就是主要瓶颈", fontsize=10,
            color=C["neutral"], fontweight="bold")
    ax.set_title("② 卡越快，软件栈占比越高")

    fig.text(0.5, 0.035,
             "这解释了一个常见困惑：为什么同一份代码换到 H100 上，"
             "加速远达不到带宽比 —— 因为发射开销那一项一点没变小",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 5 · 同一份代码换到不同的卡上（一层 d=4096 的 Transformer）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig5_bandwidth_vs_overhead.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 6
def fig6_three_cures() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 7.6))

    def timeline(ax, launches, title, note, col):
        t = 0.0
        for i, (lw_, ex) in enumerate(launches):
            ax.add_patch(Rectangle((t, 0.55), lw_, 0.34, fc=C["accent"],
                                   ec="white", lw=0.8))
            ax.add_patch(Rectangle((t + lw_, 0.12), ex, 0.34, fc=col,
                                   ec="white", lw=0.8))
            t += lw_ + ex
        ax.set_xlim(0, 96)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.29, 0.72], ["GPU 执行", "CPU 发射"], fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=11.5)
        ax.text(95, 0.98, note, ha="right", va="top", fontsize=10.5,
                color=col, fontweight="bold")
        ax.set_xlabel("时间 (µs)" if title.startswith("③") else "")

    timeline(axes[0], [(6, 2)] * 9,
             "① eager：9 个小算子，每个都交一次过路费",
             "总时间 72 µs，其中 54 µs 是发射", C["compute"])
    timeline(axes[1], [(6, 6), (6, 6), (6, 6)],
             "② 算子融合：9 个合成 3 个，计算量一点没少",
             "总时间 36 µs（省一半）", C["capacity"])
    timeline(axes[2], [(6, 18)],
             "③ CUDA Graph：把整串命令预先录制好，一次提交",
             "总时间 24 µs（发射只剩 1 次）", C["ok"])

    fig.text(0.5, 0.028,
             "三条对策解决的是同一个问题：让「发射次数」少下来。"
             "注意它们都没有让 GPU 算得更快一点",
             ha="center", fontsize=11.5, color=C["neutral"], fontweight="bold")

    fig.suptitle("图 6 · 三条对策：融合 / CUDA Graph / 持久化 kernel",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "fig6_three_cures.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    fig1_stack()
    fig2_async_timeline()
    fig3_overhead_ladder()
    fig4_queue_depth()
    fig5_bandwidth_vs_overhead()
    fig6_three_cures()
    for p in sorted(OUT.glob("*.png")):
        print("  ->", p.relative_to(OUT.parents[1]))
