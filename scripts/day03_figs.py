"""Day 03 配图（编号 = 正文阅读顺序）：
  fig1_peak_breakdown.png    峰值算力/带宽的构成：理论 vs 实测
  fig2_memory_design.png     带宽 = 位宽 × 每 pin 速率：GDDR 与 HBM 的两条路
  fig3_warp_reality.png      线程才是虚的，warp 才是硬件里真实的执行体
  fig4_sm_partition.png      一个 warp 在硬件上到底占了什么
  fig6_what_is_tile.png      Tile 是什么：分块 + 为什么对不齐会浪费
  fig7_tile_quantization.png 边长对不齐，性能就掉一截（实测）
  fig8_stride_layout.png     什么叫“相邻有用元素间隔”：内存布局示意
  fig9_cache_line.png        跨步访问暴露 cache line 大小（实测）
  fig10_occupancy.png         驻留 warp 与延迟隐藏

数据来自 labs/day03/gpu_anatomy.py 在 Mac mini M4 上的实测。
运行：uv run python scripts/day03_figs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, C

OUT = Path(__file__).resolve().parents[1] / "assets" / "day03"
OUT.mkdir(parents=True, exist_ok=True)


def fig1_peak_breakdown() -> None:
    """左：算力怎么乘出来的；右：理论与实测的差距在哪。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- 左：连乘的四个因子 ---
    factors = [("GPU 核数", 10), ("每核 ALU", 128), ("FMA = 2 次运算", 2),
               ("频率 (GHz)", 1.40)]
    running, labels, vals = 1.0, [], []
    for name, v in factors:
        running *= v
        labels.append(f"{name}\n×{v}")
        vals.append(running)
    x = np.arange(len(factors))
    ax1.bar(x, vals, color=[C["neutral"], C["memory"], C["accent"], C["compute"]],
            width=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("累乘结果（对数轴）")
    for i, v in enumerate(vals):
        txt = f"{v/1e3:.2f} T" if i == len(vals) - 1 else f"{v:,.0f}"
        ax1.text(i, v * 1.5, txt, ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylim(5, vals[-1] * 12)
    ax1.set_title("峰值算力 = 四个数连乘\n10 × 128 × 2 × 1.4 GHz = 3.58 TFLOPS",
                  fontsize=11.5, fontweight="bold")

    # --- 右：理论 vs 实测 ---
    items = ["峰值算力\n(TFLOPS)", "峰值带宽\n(GB/s)"]
    theo = np.array([3.58, 120.0])
    meas = np.array([3.56, 91.0])
    y = np.arange(len(items))
    w = 0.35
    ax2.barh(y + w / 2, theo, w, color=C["grid"], edgecolor=C["neutral"],
             label="架构参数算出的理论值")
    ax2.barh(y - w / 2, meas, w, color=C["memory"], label="实测")
    for i, (t, m) in enumerate(zip(theo, meas)):
        ax2.text(t * 1.02, i + w / 2, f"{t:g}", va="center", fontsize=9)
        ax2.text(m * 1.02, i - w / 2, f"{m:g}  ({m/t*100:.0f}%)", va="center",
                 fontsize=9, fontweight="bold", color=C["memory"])
    ax2.set_yticks(y); ax2.set_yticklabels(items)
    ax2.set_xscale("log"); ax2.set_xlim(1, 400)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("算力几乎跑满 (99%)，带宽只有 76%\n差距来自刷新/写分配/协议开销",
                  fontsize=11.5, fontweight="bold")

    fig.suptitle("图 1  Roofline 那两个数字，是从硬件规格乘出来的",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_peak_breakdown.png")
    plt.close(fig)


def fig2_memory_design() -> None:
    """带宽 = 位宽 × 每 pin 速率。GDDR 走"窄而快"，HBM 走"宽而慢"。"""
    # (名字, 总线位宽 bit, 每 pin 速率 Gbps, 容量 GB, 显存类型)
    gpus = [
        ("Mac mini M4", 128, 7.5, 16, "LPDDR5X"),
        ("RTX 4050 Laptop", 96, 16.0, 6, "GDDR6"),
        ("RTX 4090", 384, 21.0, 24, "GDDR6X"),
        ("A100 80GB", 5120, 3.2, 80, "HBM2e"),
        ("H100 SXM", 5120, 5.2, 80, "HBM3"),
        ("B200", 8192, 8.0, 192, "HBM3e"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    XLO, XHI, YLO, YHI = 2.2, 40, 36, 26000

    # 等带宽斜线：位宽 × 速率 = 常数
    rates = np.logspace(np.log10(XLO), np.log10(XHI), 100)
    for bw_gbs in (100, 200, 500, 1000, 2000, 5000, 8000):
        ax.plot(rates, bw_gbs * 8 / rates, color=C["grid"], lw=1, zorder=0)
        # 只在这条线真正落在画布内的那一段上打标签
        x_lo = max(XLO, bw_gbs * 8 / YHI)
        x_hi = min(XHI, bw_gbs * 8 / YLO)
        if x_hi <= x_lo:
            continue
        x_lab = (x_lo * x_hi) ** 0.5 * 1.35
        x_lab = min(x_lab, x_hi * 0.92)
        txt = f"{bw_gbs/1000:g} TB/s" if bw_gbs >= 1000 else f"{bw_gbs} GB/s"
        ax.text(x_lab, bw_gbs * 8 / x_lab, txt, fontsize=8, color=C["neutral"],
                va="center", ha="center", rotation=-33, rotation_mode="anchor",
                bbox=dict(fc="white", ec="none", pad=0.8), clip_on=True)

    for name, bits, rate, cap, kind in gpus:
        hbm = kind.startswith("HBM")
        col = C["accent"] if hbm else (C["compute"] if "GDDR" in kind else C["memory"])
        ax.scatter([rate], [bits], s=60 + cap * 3.2, color=col, alpha=0.85,
                   edgecolors="black", linewidths=1.1, zorder=5)
        ax.annotate(f"{name}\n{kind} · {bits}bit · {cap}GB",
                    (rate, bits), textcoords="offset points",
                    xytext=(0, 24 if hbm else -36), ha="center",
                    fontsize=8.5, fontweight="bold", color=col, zorder=6)

    ax.set_xscale("log"); ax.set_yscale("log", base=2)
    ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
    ax.set_xticks([2.5, 4, 8, 16, 32])
    ax.set_xticklabels(["2.5", "4", "8", "16", "32"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ax.set_yticklabels(["64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K"])
    ax.set_xlabel("每根数据线的速率 (Gbps)  →  越往右信号越快")
    ax.set_ylabel("总线位宽 (bit)  →  越往上线越多")

    ax.annotate("消费级显卡走这条路\n少量高速线（窄而快）",
                xy=(20, 340), xytext=(3.0, 900), fontsize=9.5,
                color=C["compute"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["compute"], lw=1.6))
    ax.annotate("数据中心 HBM 走这条路\n海量中速线（宽而慢）",
                xy=(3.4, 5120), xytext=(2.6, 1600), fontsize=9.5,
                color=C["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.6))

    ax.set_title("图 2  带宽 = 位宽 × 每 pin 速率（圆点大小 = 显存容量）\n"
                 "位宽由物理连线数决定，而连线数又决定了能挂几颗显存 → 容量与带宽是绑死的",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_memory_design.png")
    plt.close(fig)


def fig3_warp_reality() -> None:
    """活动掩码：被关掉的 lane 时钟照走，这就是"warp 才是真实执行体"的硬证据。"""
    from matplotlib.patches import Rectangle

    LANES = 32
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- 左：分支发散时的活动掩码 ----------------
    phases = [
        ("公共代码", 4, lambda l: True, C["ok"]),
        ("if 分支\n(lane 0-15)", 6, lambda l: l < 16, C["memory"]),
        ("else 分支\n(lane 16-31)", 6, lambda l: l >= 16, C["capacity"]),
        ("汇合后", 4, lambda l: True, C["ok"]),
    ]
    axA.set_xlim(-2.2, sum(p[1] for p in phases) + 0.4)
    axA.set_ylim(-8.5, LANES + 1.2)
    axA.axis("off")

    t = 0
    total_lane_cycles = 0
    useful_lane_cycles = 0
    for name, dur, pred, col in phases:
        for lane in range(LANES):
            on = pred(lane)
            axA.add_patch(Rectangle((t, lane), dur, 0.86,
                                    fc=col if on else C["grid"],
                                    ec="white", lw=0.35,
                                    alpha=1.0 if on else 0.55))
            total_lane_cycles += dur
            useful_lane_cycles += dur if on else 0
        axA.plot([t, t], [-0.4, LANES + 0.2], color="white", lw=1.6)
        axA.text(t + dur / 2, -0.9, name, ha="center", va="top", fontsize=9,
                 fontweight="bold", linespacing=1.4)
        t += dur

    for lane, lab in ((0, "lane 0"), (31, "lane 31")):
        axA.text(-0.35, lane + 0.4, lab, ha="right", va="center",
                 fontsize=8, color=C["neutral"])

    waste = 1 - useful_lane_cycles / total_lane_cycles
    axA.text(7, 24, "灰色 = 被掩码关掉的 lane\n时钟照走、周期照花",
             ha="center", va="center", fontsize=9.5,
             color=C["compute"], fontweight="bold", linespacing=1.7)
    axA.text(t / 2, -6.8,
             f"整体 lane 利用率 {(1-waste)*100:.0f}%  →  发散期间一半硬件在空转",
             ha="center", fontsize=10.5, fontweight="bold", color=C["compute"])
    axA.set_title("一个 warp 遇到 if/else 时，硬件真正在做什么",
                  fontsize=12, fontweight="bold", pad=10)

    # ---------------- 右：只启动 1 个线程 ----------------
    axB.set_xlim(-1.0, 7.6); axB.set_ylim(-8.5, LANES + 1.2)
    axB.axis("off")
    for lane in range(LANES):
        on = (lane == 0)
        axB.add_patch(Rectangle((0, lane), 6, 0.86,
                                fc=C["ok"] if on else C["grid"],
                                ec="white", lw=0.35, alpha=1.0 if on else 0.55))
    axB.text(3, -0.9, "kernel<<<1, 1>>>   只启动 1 个线程",
             ha="center", va="top", fontsize=9.5, fontweight="bold")
    axB.text(6.3, 0.4, "只有 lane 0 干活", ha="left", va="center",
             fontsize=9.5, color=C["ok"], fontweight="bold")
    axB.text(3, -2.6,
             "但硬件仍然调度了一整个 warp\n"
             "31 条 lane 全程被掩码关闭\n\n"
             "启动 1 个线程和 32 个线程\n耗时一模一样",
             ha="center", va="top", fontsize=10, linespacing=1.8,
             color=C["compute"], fontweight="bold")
    axB.set_title("最硬的证据", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("图 3  线程才是虚的，warp 才是硬件里真实的执行体",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_warp_reality.png")
    plt.close(fig)


def fig4_sm_partition() -> None:
    """一个 SM 分区：哪些晶体管归 warp 独占，哪些是所有 warp 分时复用的。"""
    from matplotlib.patches import Rectangle, FancyBboxPatch

    N_SLOT, N_LANE, ACTIVE = 12, 32, 0
    cmap = plt.get_cmap("tab20")
    wcol = [cmap(i % 20) for i in range(N_SLOT)]

    fig = plt.figure(figsize=(13, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    axT = fig.add_subplot(gs[1])
    ax.set_xlim(-0.2, 15.6); ax.set_ylim(-0.4, 11.4)
    ax.axis("off")

    LEFT, WIDTH = 0.4, 12.4
    sw = WIDTH / N_SLOT

    # ---- 上：warp 调度槽 ----
    for i in range(N_SLOT):
        x = LEFT + i * sw
        hot = (i == ACTIVE)
        ax.add_patch(Rectangle((x + 0.04, 9.1), sw - 0.08, 1.15,
                               fc=wcol[i], ec="black" if hot else "none",
                               lw=2.2 if hot else 0, alpha=0.95))
        ax.text(x + sw / 2, 9.9, f"w{i}", ha="center", fontsize=8.5,
                fontweight="bold", color="black")
        ax.text(x + sw / 2, 9.42, "PC\n掩码\n记分板", ha="center", va="center",
                fontsize=5.6, linespacing=1.25)
    ax.text(LEFT + WIDTH / 2, 10.55, "12 个 warp 调度槽（每个 warp 独占一个）",
            ha="center", fontsize=11, fontweight="bold")

    # ---- 中：寄存器堆 ----
    ax.add_patch(Rectangle((LEFT, 5.5), WIDTH, 2.6, fc="white",
                           ec=C["neutral"], lw=2))
    for i in range(N_SLOT):
        x = LEFT + i * sw
        ax.add_patch(Rectangle((x + 0.06, 5.58), sw - 0.12, 2.44,
                               fc=wcol[i], ec="white", lw=0.8, alpha=0.75))
        ax.text(x + sw / 2, 6.8, f"w{i} 的\n寄存器", ha="center", va="center",
                fontsize=6.6, linespacing=1.35)
    ax.text(LEFT + WIDTH / 2, 8.22,
            "寄存器堆 16384 个 32-bit（每个 warp 独占一段，驻留期间不还）",
            ha="center", fontsize=11, fontweight="bold")

    # ---- 下：ALU lane，灰色 = 共享 ----
    lw_ = WIDTH / N_LANE
    for i in range(N_LANE):
        x = LEFT + i * lw_
        ax.add_patch(Rectangle((x + 0.02, 2.3), lw_ - 0.04, 1.5,
                               fc=C["neutral"], ec="white", lw=0.6, alpha=0.9))
    ax.text(LEFT + WIDTH / 2, 3.05, "32 条 ALU lane", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(LEFT + WIDTH / 2, 4.25,
            "只有这一套！所有 12 个 warp 分时复用（每周期只服务 1 个）",
            ha="center", fontsize=11, fontweight="bold", color=C["compute"])
    ax.add_patch(Rectangle((LEFT, 0.75), WIDTH, 1.1, fc=C["capacity"],
                           ec="white", lw=0.6, alpha=0.85))
    ax.text(LEFT + WIDTH / 2, 1.3, "1 个 Tensor Core（同样是共享的）",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # ---- 本周期的数据通路 ----
    ax_ = LEFT + ACTIVE * sw + sw / 2
    ax.annotate("", xy=(ax_, 8.15), xytext=(ax_, 9.05),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.6))
    ax.annotate("", xy=(ax_, 3.9), xytext=(ax_, 5.45),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.6))
    ax.text(ax_ + 0.28, 8.72, "① 调度器本周期挑中 w0", fontsize=9.5,
            fontweight="bold", ha="left")
    ax.text(ax_ + 0.28, 5.02, "② 读 w0 自己那段寄存器 → 喂给 32 条 lane",
            fontsize=9.5, fontweight="bold", ha="left")

    # ---- 右侧归属标注 ----
    for y0, y1, txt, col in [
        (9.1, 10.25, "每个 warp\n独占", C["ok"]),
        (5.5, 8.1, "每个 warp\n独占", C["ok"]),
        (0.75, 3.8, "所有 warp\n分时共享", C["compute"]),
    ]:
        ax.add_patch(FancyBboxPatch((13.1, y0), 2.3, y1 - y0,
                                    boxstyle="round,pad=0.04",
                                    fc="white", ec=col, lw=2.2))
        ax.text(14.25, (y0 + y1) / 2, txt, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=col, linespacing=1.6)

    ax.set_title("一个 SM 分区里，warp 到底占了哪些晶体管\n"
                 "（Ada 架构：一个 SM = 4 个这样的分区）",
                 fontsize=12.5, fontweight="bold", pad=12)

    # ================= 下：分时复用的逐周期时间线 =================
    CYCLES = [3, 7, 0, 3, 9, 7, 1, 3]
    axT.set_xlim(-2.6, len(CYCLES) + 0.2)
    axT.set_ylim(-1.9, 3.2)
    axT.axis("off")

    for i, w in enumerate(CYCLES):
        axT.add_patch(Rectangle((i + 0.04, 2.05), 0.92, 0.75,
                                fc=wcol[w], ec="black", lw=1.1))
        axT.text(i + 0.5, 2.42, f"w{w}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold")
        # 32 条 lane 这一周期归谁用
        for l in range(N_LANE):
            axT.add_patch(Rectangle((i + 0.04 + l * 0.92 / N_LANE, 0.75),
                                    0.92 / N_LANE * 0.9, 1.0,
                                    fc=wcol[w], ec="none", alpha=0.75))
        axT.add_patch(Rectangle((i + 0.04, 0.75), 0.92, 1.0,
                                fc="none", ec=C["neutral"], lw=1.0))
        axT.text(i + 0.5, -0.42, f"第 {i+1} 周期", ha="center", fontsize=8.5,
                 color=C["neutral"])

    axT.text(-0.25, 2.42, "调度器挑中谁", ha="right", va="center",
             fontsize=10.5, fontweight="bold")
    axT.text(-0.25, 1.25, "32 条 ALU lane\n（同一套硬件）", ha="right",
             va="center", fontsize=10.5, fontweight="bold", linespacing=1.5)

    axT.text(len(CYCLES) / 2, -1.35,
             "同一套 lane，每个周期换一个 warp 用 —— 这就是「分时复用」\n"
             "注意：上排颜色一直在换，但下排硬件从头到尾就是那一套",
             ha="center", va="top", fontsize=10.5, fontweight="bold",
             color=C["compute"], linespacing=1.7)
    axT.set_title("那「分时复用」具体长什么样",
                  fontsize=12.5, fontweight="bold", pad=8)

    fig.suptitle("图 4  一个 warp 占的是「状态」，不是「算力」",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(OUT / "fig4_sm_partition.png", bbox_inches="tight")
    plt.close(fig)


def fig5_register_file() -> None:
    """寄存器堆是一块二维阵列：列归 lane，行归 warp。切换只改行地址，数据不动。"""
    from matplotlib.patches import Rectangle

    N_LANE, N_WARP, N_REG = 32, 3, 4
    cmap = plt.get_cmap("tab20")
    wcol = [cmap(i * 2) for i in range(N_WARP)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    def draw(ax, hot_warp, hot_reg, cycle_txt, instr):
        ax.set_xlim(-2.4, N_LANE + 0.4)
        ax.set_ylim(-4.3, N_WARP * N_REG + 1.9)
        ax.axis("off")

        for w in range(N_WARP):
            for r in range(N_REG):
                row = (N_WARP - 1 - w) * N_REG + (N_REG - 1 - r)
                hot = (w == hot_warp and r == hot_reg)
                for l in range(N_LANE):
                    ax.add_patch(Rectangle((l + 0.05, row + 0.08), 0.9, 0.84,
                                           fc=wcol[w],
                                           alpha=0.95 if hot else 0.22,
                                           ec="black" if hot else "white",
                                           lw=0.9 if hot else 0.3))
                ax.text(-0.35, row + 0.5, f"R{r}", ha="right",
                        va="center", fontsize=8.5,
                        fontweight="bold" if hot else "normal",
                        color="black" if hot else C["neutral"])
            # warp 的地址段
            top = (N_WARP - 1 - w) * N_REG + N_REG
            ax.plot([-1.25, -1.25], [top - N_REG + 0.1, top - 0.1],
                    color=wcol[w], lw=5, solid_capstyle="butt")
            ax.text(-1.45, top - N_REG / 2, f"w{w} 的\n地址段", ha="right",
                    va="center", fontsize=8.5, color=wcol[w], fontweight="bold",
                    linespacing=1.4)

        for l in (0, 1, 2, N_LANE - 1):
            ax.text(l + 0.5, -0.45, f"L{l}", ha="center", va="top",
                    fontsize=7.5, color=C["neutral"])
        ax.text(N_LANE / 2, -1.25, "32 列 = 32 条 lane（每列的连线是焊死的）",
                ha="center", va="top", fontsize=9.5, fontweight="bold",
                color=C["memory"])

        hot_row = (N_WARP - 1 - hot_warp) * N_REG + (N_REG - 1 - hot_reg)
        ax.annotate("", xy=(N_LANE + 0.2, hot_row + 0.5),
                    xytext=(N_LANE + 2.0, hot_row + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4))
        ax.set_title(cycle_txt, fontsize=11.5, fontweight="bold")
        ax.text(N_LANE / 2, -2.2, instr, ha="center", va="top", fontsize=10,
                family="monospace",
                bbox=dict(fc="white", ec=C["neutral"], lw=1.2, pad=4))

    draw(axes[0], 0, 2, "第 1 周期：w0 的 PC 指向这条指令",
         "FMA R2, R1, R0")
    draw(axes[1], 2, 0, "第 2 周期：w2 的 PC 在程序的另一处",
         "LD  R0, [addr]")

    axes[0].text(N_LANE / 2, -3.5,
                 "行地址 = w0 基址 + 2", ha="center", fontsize=10.5,
                 fontweight="bold", color=C["accent"])
    axes[1].text(N_LANE / 2, -3.5,
                 "行地址 = w2 基址 + 0", ha="center", fontsize=10.5,
                 fontweight="bold", color=C["accent"])

    fig.suptitle("图 5  为什么切换 warp 不用搬数据：货架只有一面，"
                 "换的只是「伸手够哪一格」\n"
                 "（基址由调度器给，寄存器号由该 warp 自己的 PC 决定 —— 两者独立）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_register_file.png")
    plt.close(fig)


def fig6_what_is_tile() -> None:
    """左：一个 tile 是怎么算出来的；右：边长对不齐时多出来的那一圈。"""
    from matplotlib.patches import Rectangle

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------- 左：分块矩阵乘的机制 ----------
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    n, cell = 4, 1.35
    cx, cy = 4.0, 1.2          # C 左下角
    hi_r, hi_c = 2, 1          # 高亮的 tile 行列

    # B 在上方
    for j in range(n):
        col = C["capacity"] if j == hi_c else "white"
        axR_x = cx + j * cell
        axL.add_patch(Rectangle((axR_x, cy + n * cell + 0.9), cell, cell,
                                fc=col, ec=C["neutral"], lw=1.1))
    axL.text(cx + n * cell / 2, cy + (n + 1) * cell + 1.15, "B",
             ha="center", fontsize=13, fontweight="bold")
    axL.text(cx + (hi_c + 0.5) * cell, cy + n * cell + 0.55,
             "取这一竖条", ha="center", fontsize=8.5,
             color=C["capacity"], fontweight="bold")

    # A 在左侧
    for i in range(n):
        col = C["memory"] if i == hi_r else "white"
        axL.add_patch(Rectangle((cx - cell - 0.9, cy + i * cell), cell, cell,
                                fc=col, ec=C["neutral"], lw=1.1))
    axL.text(cx - cell / 2 - 0.9, cy + n * cell + 0.35, "A",
             ha="center", fontsize=13, fontweight="bold")
    axL.text(cx - cell - 1.05, cy + (hi_r + 0.5) * cell, "取这一横条",
             ha="right", va="center", fontsize=8.5,
             color=C["memory"], fontweight="bold")

    # C
    for i in range(n):
        for j in range(n):
            hit = (i == hi_r and j == hi_c)
            axL.add_patch(Rectangle((cx + j * cell, cy + i * cell), cell, cell,
                                    fc=C["ok"] if hit else "white",
                                    ec=C["neutral"], lw=2.0 if hit else 1.1))
    axL.text(cx + n * cell / 2, cy - 0.55, "C = A @ B", ha="center",
             fontsize=13, fontweight="bold")
    axL.text(cx + (hi_c + 0.5) * cell, cy + (hi_r + 0.5) * cell, "一个\ntile",
             ha="center", va="center", fontsize=9, color="white",
             fontweight="bold")

    axL.annotate("", xy=(cx + hi_c * cell + 0.15, cy + (hi_r + 0.9) * cell),
                 xytext=(cx + hi_c * cell + 0.15, cy + n * cell + 0.85),
                 arrowprops=dict(arrowstyle="-|>", color=C["capacity"], lw=2.2))
    axL.annotate("", xy=(cx + hi_c * cell + 0.2, cy + (hi_r + 0.5) * cell),
                 xytext=(cx - 0.85, cy + (hi_r + 0.5) * cell),
                 arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=2.2))

    axL.set_title("Tile = 输出矩阵上的一个小方块\n"
                  "算它只需读 A 的一横条 + B 的一竖条",
                  fontsize=12, fontweight="bold")

    # ---------- 右：对齐 vs 不对齐 ----------
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis("off")

    def grid(x0, y0, size, full, part_frac, title, sub, col_ok, col_bad):
        m = full + (1 if part_frac else 0)
        s = size / m
        for i in range(m):
            for j in range(m):
                edge = (i == full or j == full) and part_frac
                w = s * (part_frac if j == full and part_frac else 1)
                h = s * (part_frac if i == full and part_frac else 1)
                axR.add_patch(Rectangle((x0 + j * s, y0 + i * s), s, s,
                                        fc=col_bad if edge else col_ok,
                                        ec="white", lw=0.9, alpha=0.85))
                if edge:                      # 画出这块里真正有用的部分
                    axR.add_patch(Rectangle((x0 + j * s, y0 + i * s), w, h,
                                            fc=col_ok, ec="none"))
        axR.text(x0 + size / 2, y0 + size + 0.45, title, ha="center",
                 fontsize=11.5, fontweight="bold")
        axR.text(x0 + size / 2, y0 - 0.45, sub, ha="center", va="top",
                 fontsize=9.5, linespacing=1.8)

    grid(0.5, 4.0, 3.9, 8, 0, "N = 1024",
         "8×8 = 64 个 tile\n每个都是满的\n零浪费",
         C["ok"], C["compute"])
    grid(5.6, 4.0, 3.9, 8, 1 / 128, "N = 1025",
         "9×9 = 81 个 tile\n多出的 17 个只用了 1/128\n硬件多干 27% 的活",
         C["ok"], C["compute"])

    axR.annotate("多出来这一圈\n红色部分全是空转",
                 xy=(9.35, 7.4), xytext=(5.6, 8.9), fontsize=9.5,
                 color=C["compute"], fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["compute"], lw=1.8))
    axR.set_title("边长只差 1，却要多起一整圈 tile", fontsize=12,
                  fontweight="bold")

    fig.suptitle("图 6  什么是 Tile：GPU 把大矩阵切成小方块来算",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_what_is_tile.png")
    plt.close(fig)


def fig7_tile_quantization() -> None:
    """M4 实测：边长 mod 128 == 0 时才跑满。"""
    data = [(1024, 3.45), (1025, 2.94), (1088, 3.35), (1152, 3.51),
            (1279, 2.95), (1280, 3.51), (1281, 2.88), (1536, 3.54),
            (2048, 3.58), (2049, 3.15)]
    names = [str(n) for n, _ in data]
    vals = np.array([v for _, v in data])
    aligned = np.array([n % 128 == 0 for n, _ in data])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(data))
    cols = [C["ok"] if a else C["compute"] for a in aligned]
    bars = ax.bar(x, vals, color=cols, width=0.62)
    for b, v, a in zip(bars, vals, aligned):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                ha="center", fontsize=8.5, fontweight="bold",
                color=C["ok"] if a else C["compute"])

    ax.axhline(vals.max(), color=C["neutral"], ls="--", lw=1.6)
    ax.text(len(data) - 0.4, vals.max() + 0.13, f"峰值 {vals.max():.2f} TFLOPS",
            ha="right", fontsize=9, color=C["neutral"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, vals.max() * 1.32)
    ax.set_xlabel("矩阵边长 N（绿色 = 128 的倍数）")
    ax.set_ylabel("实测 TFLOPS")

    for idx, txt in [(1, "1024→1025：只差 1，掉 15%"),
                     (6, "1280→1281：只差 1，掉 18%")]:
        ax.annotate(txt, xy=(idx, vals[idx] + 0.08),
                    xytext=(idx + 0.15, vals.max() * 1.19),
                    ha="center", fontsize=9.5, color=C["capacity"],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["capacity"], lw=1.8))

    ax.set_title("图 7  Tile 量化：GPU 按固定大小的块干活，边长对不齐就有硬件空转",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_tile_quantization.png")
    plt.close(fig)


def fig8_stride_layout() -> None:
    """把「相邻有用元素间隔」画出来：内存里哪些字节被搬了、哪些真被用了。"""
    from matplotlib.patches import Rectangle

    LINE_B = 64          # 假定 cache line = 64 字节
    ESZ = 2              # fp16
    N_ELEM = 64          # 画 64 个元素 = 128 字节 = 2 条 line
    W, H = 0.145, 0.52   # 每个元素的方块尺寸

    cases = [
        (1, "stride=1  间隔 2 B", "整列就是连续内存，64/64 都有用"),
        (4, "stride=4  间隔 8 B", "每条 line 里只有 1/4 有用"),
        (32, "stride=32  间隔 64 B", "每条 line 里只剩 1 个有用元素"),
        (64, "stride=64  间隔 128 B", "第 2 条 line 根本不用取 → 效率不再下降"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(-2.9, N_ELEM * W + 0.5)
    ax.set_ylim(-0.6, len(cases) * 1.25 + 0.5)
    ax.axis("off")

    for row, (stride, title, note) in enumerate(cases):
        y = (len(cases) - 1 - row) * 1.25
        elems_per_line = LINE_B // ESZ                 # 32 个元素一条 line
        for i in range(N_ELEM):
            useful = (i % stride == 0)
            line_idx = i // elems_per_line
            # 该条 line 里有没有需要的元素？没有就整条不取
            line_touched = any((k % stride == 0)
                               for k in range(line_idx * elems_per_line,
                                              (line_idx + 1) * elems_per_line))
            if not line_touched:
                fc, ec, hatch = "white", C["grid"], "//"
            elif useful:
                fc, ec, hatch = C["ok"], "black", None
            else:
                fc, ec, hatch = C["compute"], "none", None
            ax.add_patch(Rectangle((i * W, y), W * 0.92, H, fc=fc, ec=ec,
                                   lw=0.8 if useful else 0.3, hatch=hatch,
                                   alpha=1.0 if useful else 0.45))

        # cache line 分隔
        for b in (0, elems_per_line, N_ELEM):
            ax.plot([b * W - 0.01, b * W - 0.01], [y - 0.12, y + H + 0.12],
                    color=C["neutral"], lw=2)
        ax.text(-0.15, y + H / 2, title, ha="right", va="center",
                fontsize=10, fontweight="bold")
        ax.text(N_ELEM * W + 0.12, y + H / 2, note, ha="left", va="center",
                fontsize=9, color=C["neutral"])

    top = (len(cases) - 1) * 1.25 + H
    for b, lab in ((16, "第 1 条 cache line (64 B)"), (48, "第 2 条 cache line")):
        ax.text(b * W, top + 0.42, lab, ha="center", fontsize=9,
                color=C["neutral"], fontweight="bold")

    handles = [
        Rectangle((0, 0), 1, 1, fc=C["ok"], ec="black", label="真正要用的元素"),
        Rectangle((0, 0), 1, 1, fc=C["compute"], alpha=0.45,
                  label="被硬件顺带搬上来、但用不到"),
        Rectangle((0, 0), 1, 1, fc="white", ec=C["grid"], hatch="//",
                  label="整条 line 都用不到 → 压根不取"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5,
              bbox_to_anchor=(0.45, -0.13))

    fig.suptitle("图 8  「相邻有用元素间隔」是什么：读矩阵的一列时，内存里发生了什么",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_stride_layout.png")
    plt.close(fig)


def fig10_occupancy() -> None:
    """左：寄存器堆怎么被瓜分；右：驻留 warp 多寡如何决定延迟能否被藏住。"""
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.55, wspace=0.22)
    axL = fig.add_subplot(gs[:, 0])
    axT = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1])

    # ---------------- 左：寄存器堆的瓜分 ----------------
    TOTAL = 65536
    axL.set_xlim(0, 10); axL.set_ylim(-1.4, 11.4); axL.axis("off")

    for k, (R, x0, col) in enumerate([(32, 0.6, C["ok"]), (128, 5.4, C["compute"])]):
        per_warp = R * 32
        fit = min(48, TOTAL // per_warp)
        h = 9.6
        axL.add_patch(Rectangle((x0, 0.6), 3.4, h, fc="white",
                                ec=C["neutral"], lw=2))
        used_frac = fit * per_warp / TOTAL
        for w in range(fit):
            wy = 0.6 + w * (h * used_frac / fit)
            axL.add_patch(Rectangle((x0 + 0.08, wy + 0.02), 3.24,
                                    h * used_frac / fit - 0.04,
                                    fc=col, ec="white", lw=0.7, alpha=0.85))
        if used_frac < 1:
            axL.text(x0 + 1.7, 0.6 + h * used_frac + (h * (1 - used_frac)) / 2,
                     "剩下的寄存器\n空着也没用\n（已达 48 warp 上限）",
                     ha="center", va="center", fontsize=8.5, color=C["neutral"])
        axL.text(x0 + 1.7, h + 0.95, f"每线程 {R} 个寄存器",
                 ha="center", fontsize=10.5, fontweight="bold")
        axL.text(x0 + 1.7, -0.15,
                 f"1 warp 吃掉 {per_warp:,} 个槽\n"
                 f"→ 装得下 {fit} 个 warp\n"
                 f"→ 占用率 {fit/48*100:.0f}%",
                 ha="center", va="top", fontsize=8.5, color=col,
                 fontweight="bold", linespacing=1.6)

    axL.set_title("寄存器堆 65536 个槽，被驻留 warp 瓜分",
                  fontsize=11.5, fontweight="bold")

    # ---------------- 右：延迟隐藏时间线 ----------------
    COMPUTE, WAIT = 20, 200
    PERIOD = COMPUTE + WAIT

    def timeline(ax, n_warps, title):
        ax.set_xlim(0, PERIOD * 1.30)
        ax.set_ylim(-2.0, n_warps + 0.4)
        ax.axis("off")
        step = COMPUTE                     # 依次错开发射
        busy = set()
        for w in range(n_warps):
            y = n_warps - 1 - w
            t = w * step
            while t < PERIOD:
                ax.add_patch(Rectangle((t, y + 0.12), COMPUTE, 0.62,
                                       fc=C["compute"], ec="none"))
                ax.add_patch(Rectangle((t + COMPUTE, y + 0.28), WAIT, 0.3,
                                       fc=C["grid"], ec="none"))
                for c in range(int(t), int(t + COMPUTE)):
                    busy.add(c)
                t += PERIOD
            ax.text(-6, y + 0.42, f"w{w}", ha="right", va="center", fontsize=7.5,
                    color=C["neutral"])

        # SM 发射槽
        for c in range(int(PERIOD)):
            ax.add_patch(Rectangle((c, -1.5), 1.0, 0.85,
                                   fc=C["ok"] if c in busy else C["capacity"],
                                   ec="none"))
        util = len(busy) / PERIOD
        ax.text(-6, -1.07, "SM", ha="right", va="center", fontsize=8.5,
                fontweight="bold")
        ax.text(PERIOD * 1.03, -1.07, f"利用率 {util*100:.0f}%", ha="left",
                va="center", fontsize=10, fontweight="bold",
                color=C["ok"] if util > 0.9 else C["compute"])
        ax.set_title(title, fontsize=10.5, fontweight="bold")

    timeline(axT, 4, "只驻留 4 个 warp：都在等内存时，SM 无事可做（黄色 = 空转）")
    timeline(axB, 11, "驻留 11 个 warp：总有一个 warp 就绪，发射槽被填满")

    handles = [
        Rectangle((0, 0), 1, 1, fc=C["compute"], label="warp 在计算"),
        Rectangle((0, 0), 1, 1, fc=C["grid"], label="warp 在等内存（约 200 周期）"),
        Rectangle((0, 0), 1, 1, fc=C["ok"], label="SM 有指令可发"),
        Rectangle((0, 0), 1, 1, fc=C["capacity"], label="SM 空转"),
    ]
    axB.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.42, -0.02))

    fig.suptitle("图 10  驻留 warp：不是「没电的 warp」，而是「已经坐进 SM 的 warp」",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig10_occupancy.png", bbox_inches="tight")
    plt.close(fig)


def fig9_cache_line() -> None:
    """跨步扫描：曲线在哪里变平，哪里就是 cache line。"""
    stride_bytes = np.array([2, 4, 8, 16, 32, 64, 128])
    eff_bw = np.array([64.3, 48.1, 22.7, 12.4, 6.3, 3.1, 2.8])

    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.plot(stride_bytes, eff_bw, "o-", lw=2.5, ms=8, color=C["memory"], zorder=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(stride_bytes)
    ax.set_xticklabels([f"{b} B" for b in stride_bytes])
    ax.set_xlabel("相邻两个有用元素之间隔了多少字节")
    ax.set_ylabel("有效带宽 GB/s（只算有用数据）")

    for b, v in zip(stride_bytes, eff_bw):
        ax.annotate(f"{v:.1f}", (b, v), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, fontweight="bold")

    ax.axvspan(2, 64, color=C["memory"], alpha=0.07)
    ax.axvspan(64, 128, color=C["ok"], alpha=0.12)
    ax.text(8, 40, "每翻一倍，有效带宽就腰斩\n（一条 line 里的有用数据少了一半）",
            fontsize=9.5, color=C["memory"], fontweight="bold")
    ax.annotate("到这里不再腰斩了\n一条 line 里只剩 1 个有用元素\n→ cache line ≈ 64 字节",
                xy=(64, 3.1), xytext=(11, 4.2), fontsize=9.5,
                color=C["ok"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["ok"], lw=1.8))

    ax.set_ylim(2, 110)
    ax.set_title("图 9  只要 1 个数，硬件也得搬一整条 cache line（Mac mini M4 实测）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_cache_line.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"[style] CJK font = {setup()}")
    fig1_peak_breakdown()
    fig2_memory_design()
    fig3_warp_reality()
    fig4_sm_partition()
    fig5_register_file()
    fig6_what_is_tile()
    fig7_tile_quantization()
    fig8_stride_layout()
    fig9_cache_line()
    fig10_occupancy()
    for p in sorted(OUT.glob("*.png")):
        print(f"[ok] {p.relative_to(OUT.parents[1])}  ({p.stat().st_size/1024:.0f} KB)")
