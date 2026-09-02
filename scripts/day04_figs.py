"""Day 04 配图 · 存储层次与数据搬运

    .venv/bin/python scripts/day04_figs.py

图号 = 正文阅读顺序：
  fig1  存储金字塔（容量 / 延迟 / 带宽 三张脸）
  fig2  把延迟换算成人类时间
  fig3  为什么必然是金字塔（SRAM vs DRAM 单元 + 距离即时间）
  fig4  带宽阶梯（对数，标出相对显存的倍数）
  fig5  工作集扫描的三段模型
  fig6  搬 1 GB 数据，走哪条路要多久
  fig7  FlashAttention：让中间矩阵永不落地
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import C, setup

OUT = Path(__file__).resolve().parents[1] / "assets" / "day04"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- 共用数据表
# RTX 4050 Laptop (AD107, 20 SM @2.37 GHz, 96-bit GDDR6)
# 延迟单位 = GPU 周期；带宽单位 = GB/s
LEVELS = [
    # name,             容量,        容量B,      延迟周期, 带宽GB/s,  颜色
    ("寄存器",          "5 MB",     5e6,          1,      97000,  C["compute"]),
    ("Shared Mem / L1", "2.5 MB",   2.5e6,       30,       6100,  "#E4785C"),
    ("L2",              "24 MB",    24e6,       200,       1500,  C["capacity"]),
    ("显存 DRAM",       "6 GB",     6e9,        500,        192,  C["memory"]),
    ("主机内存(PCIe)",  "32 GB",    32e9,      4700,         16,  C["accent"]),
    ("NVMe SSD",        "1 TB",     1e12,    190000,          6,  C["neutral"]),
]


def _fmt_cycles(cyc: float) -> str:
    ns = cyc / 2.37
    if ns < 1000:
        return f"{cyc:,.0f} 周期\n≈ {ns:.1f} ns"
    return f"{cyc:,.0f} 周期\n≈ {ns/1000:.0f} µs"


# ------------------------------------------------------------------- 图 1
def fig1_pyramid() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    names = [x[0] for x in LEVELS]
    n = len(names)

    # --- 左：金字塔本体 ---
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(-0.05, 1.30)
    ax.set_ylim(-0.5, n + 0.4)
    for i, (name, cap, _, cyc, bw, col) in enumerate(LEVELS):
        y = n - 1 - i
        half = 0.09 + 0.40 * i / (n - 1)          # 越往下越宽
        ax.add_patch(Rectangle((0.5 - half, y), 2 * half, 0.82,
                               fc=col, ec="white", lw=1.6, alpha=0.92))
        short = {"Shared Mem / L1": "SMEM / L1",
                 "主机内存(PCIe)": "主机内存"}.get(name, name)
        ax.text(0.5, y + 0.41, short, ha="center", va="center",
                color="white", fontweight="bold",
                fontsize=9.5 if i < 2 else 11)
        ax.text(1.02, y + 0.41, cap, ha="left", va="center",
                fontsize=10, color=C["neutral"], fontweight="bold")
    ax.text(0.5, n + 0.05, "越往上：越快、越小、越贵",
            ha="center", fontsize=10.5, color=C["neutral"])
    ax.annotate("", xy=(-0.02, n - 0.2), xytext=(-0.02, 0.2),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C["compute"]))
    ax.text(-0.05, n / 2, "速\n度", ha="center", va="center",
            fontsize=10, color=C["compute"], fontweight="bold")
    ax.set_title("① 六层结构（RTX 4050 为例）")

    # --- 中：延迟 ---
    ax = axes[1]
    cyc = [x[3] for x in LEVELS]
    cols = [x[5] for x in LEVELS]
    ypos = np.arange(n)
    ax.barh(ypos, cyc, color=cols, alpha=0.92, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(ypos, names)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlim(0.5, 1e8)
    ax.set_xlabel("延迟（GPU 周期，对数轴）")
    for i, v in enumerate(cyc):
        ax.text(v * 1.6, i, _fmt_cycles(v).replace("\n", "  "),
                va="center", fontsize=9, color=C["neutral"], zorder=10)
    ax.set_title("② 取一个数要等多久")

    # --- 右：带宽 ---
    ax = axes[2]
    bws = [x[4] for x in LEVELS]
    ax.barh(ypos, bws, color=cols, alpha=0.92, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(ypos, names)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlim(2, 5e7)
    ax.set_xlabel("带宽（GB/s，对数轴）")
    for i, v in enumerate(bws):
        lab = f"{v/1000:.1f} TB/s" if v >= 1000 else f"{v:.0f} GB/s"
        ratio = f"{v/192:.0f}×" if v >= 192 else f"1/{192/v:.0f}"
        ax.text(v * 1.6, i, f"{lab}   ({ratio} 显存)",
                va="center", fontsize=9, color=C["neutral"], zorder=10)
    ax.axvline(192, color=C["memory"], ls="--", lw=1.4, alpha=0.7)
    ax.set_title("③ 每秒能搬多少")

    fig.text(0.50, 0.045, "从最上到最下：延迟差 19 万倍",
             ha="center", fontsize=11, color=C["compute"], fontweight="bold")
    fig.text(0.84, 0.045, "带宽差 1.6 万倍\n但 Roofline 的屋顶只画了显存那一层",
             ha="center", fontsize=10.5, color=C["memory"],
             fontweight="bold", linespacing=1.6)

    fig.suptitle("图 1 · 存储金字塔：容量、延迟、带宽是同一件事的三张脸",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(OUT / "fig1_pyramid.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_human_scale() -> None:
    """1 个 GPU 周期 = 1 秒，把延迟放大到人类能感受的尺度。"""
    rows = [
        ("寄存器",            1,        "伸手拿桌上的笔",                 C["compute"]),
        ("Shared Memory",     30,       "走到隔壁工位",                   "#E4785C"),
        ("L2",                200,      "下楼到大厅",                     C["capacity"]),
        ("显存 DRAM",         500,      "骑车去街对面的仓库",             C["memory"]),
        ("主机内存 (PCIe)",   4_700,    "开车去邻市取件",                 C["accent"]),
        ("NVMe SSD",          190_000,  "坐飞机出趟差",                   C["neutral"]),
        ("跨区域网络",        118_000_000, "读完一个本科学位",            "#8D6A9F"),
    ]

    def human(sec: float) -> str:
        if sec < 60:
            return f"{sec:.0f} 秒"
        if sec < 3600:
            return f"{sec/60:.0f} 分钟"
        if sec < 86400:
            return f"{sec/3600:.1f} 小时"
        if sec < 86400 * 365:
            return f"{sec/86400:.1f} 天"
        return f"{sec/86400/365:.1f} 年"

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    n = len(rows)
    ypos = np.arange(n)
    vals = [r[1] for r in rows]
    ax.barh(ypos, vals, color=[r[3] for r in rows], alpha=0.92, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_ylim(n - 0.4, -0.85)
    ax.set_xlim(0.5, 5e10)
    ax.set_xlabel("换算后的「人类时间」（对数轴）")

    for i, (_, cyc, story, _c) in enumerate(rows):
        ax.text(cyc * 1.6, i, f"{human(cyc)}   —   {story}",
                va="center", fontsize=10.2, color=C["neutral"],
                fontweight="bold", zorder=10)

    fig.text(0.5, 0.035,
             "换算规则：1 个 GPU 周期（0.42 ns）当作 1 秒　|　"
             "同一份数据，放对地方和放错地方，差的是「伸手」和「出差」",
             ha="center", fontsize=11, color=C["compute"], fontweight="bold")

    fig.suptitle("图 2 · 把 GPU 的延迟放慢 24 亿倍，你就能感受到差距",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig2_human_scale.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig3_why_pyramid() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    # --- 左：SRAM 6T vs DRAM 1T1C ---
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.add_patch(Rectangle((0.2, 6.0), 4.6, 3.7, fc="#FDF1F3",
                           ec=C["compute"], lw=2))
    ax.text(2.5, 9.25, "SRAM  ·  1 bit = 6 个晶体管", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C["compute"])
    for k in range(6):
        cx = 1.15 + (k % 3) * 1.35
        cy = 8.15 - (k // 3) * 1.0
        ax.add_patch(Rectangle((cx - 0.32, cy - 0.26), 0.64, 0.52,
                               fc=C["compute"], ec="white", lw=1.3, alpha=0.9))
    ax.text(2.5, 6.40, "通电就一直记着\n读出来不破坏 · 一步就取到",
            ha="center", va="center", fontsize=9.2,
            color=C["neutral"], linespacing=1.5)

    ax.add_patch(Rectangle((5.2, 6.0), 4.6, 3.7, fc="#EAF4F6",
                           ec=C["memory"], lw=2))
    ax.text(7.5, 9.25, "DRAM  ·  1 bit = 1 晶体管 + 1 电容", ha="center",
            va="center", fontsize=11.5, fontweight="bold", color=C["memory"])
    ax.add_patch(Rectangle((6.35, 7.60), 0.64, 0.52,
                           fc=C["memory"], ec="white", lw=1.3, alpha=0.9))
    ax.plot([7.85, 7.85], [7.55, 7.80], lw=3.2, color=C["memory"])
    ax.plot([7.85, 7.85], [8.00, 8.25], lw=3.2, color=C["memory"])
    ax.plot([6.99, 7.85], [7.86, 7.86], lw=1.5, color=C["memory"])
    ax.text(8.15, 7.9, "电容", fontsize=9, va="center", color=C["memory"])
    ax.text(7.5, 6.55, "电容会漏电 → 每几十 µs 必须刷新\n读出来电就没了 → 必须再写回",
            ha="center", va="center", fontsize=9.2,
            color=C["neutral"], linespacing=1.5)

    ax.text(5.0, 5.1,
            "同样的硅片面积，DRAM 能装下 ~100 倍的 bit\n"
            "→「又快又大」在物理上做不到，只能分层",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=C["compute"], linespacing=1.8)

    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 3.5, fc="#FBFCFD",
                           ec=C["grid"], lw=1.4))
    ax.text(5.0, 3.35, "第二个原因：距离就是时间",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=C["accent"])
    ax.text(5.0, 1.75,
            "一个周期 0.42 ns，电信号在芯片里大约只走 5 cm\n"
            "寄存器就在 ALU 旁边（微米级）→ 1 个周期够用\n"
            "L2 在芯片正中央、要仲裁全部 20 个 SM 的请求 → 200 个周期\n"
            "显存颗粒焊在 PCB 上、隔着 PHY 和协议 → 500 个周期",
            ha="center", va="center", fontsize=9.6,
            color=C["neutral"], linespacing=1.9)
    ax.set_title("① 为什么快的必然小")

    # --- 右：容量 vs 延迟 的取舍前沿 ---
    ax = axes[1]
    caps = [x[2] for x in LEVELS]
    cycs = [x[3] for x in LEVELS]
    for (name, _cap, cb, cy, _bw, col) in LEVELS:
        ax.scatter(cb, cy, s=210, color=col, zorder=5, edgecolor="white", lw=1.5)
    ax.plot(caps, cycs, color=C["neutral"], ls="--", lw=1.4, alpha=0.6, zorder=1)

    # (dx 倍数, dy 倍数, 水平对齐)
    offsets = [(1.7, 1.0, "left"), (1.5, 0.40, "left"), (1.7, 1.0, "left"),
               (0.6, 0.42, "center"), (0.55, 2.6, "center"), (0.30, 2.6, "center")]
    for (name, _cap, cb, cy, _bw, col), (ox, oy, ha) in zip(LEVELS, offsets):
        ax.text(cb * ox, cy * oy, name, fontsize=10, fontweight="bold",
                color=col, zorder=10, ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("容量（字节，对数轴）")
    ax.set_ylabel("延迟（GPU 周期，对数轴）")
    ax.set_xlim(1e6, 2e13)
    ax.set_ylim(0.25, 3e6)

    ax.add_patch(Rectangle((1e6, 0.25), 2e13, 4.0, fc=C["ok"], alpha=0.10))
    ax.text(3e11, 1.0, "「又大又快」这块区域是空的\n不是没人做，是做不出来",
            ha="center", va="center", fontsize=10.5, color=C["ok"],
            fontweight="bold", linespacing=1.7)
    ax.set_title("② 每一级都在同一条取舍曲线上")

    fig.suptitle("图 3 · 存储金字塔不是设计选择，是物理结果",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig3_why_pyramid.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig4_bandwidth_ladder() -> None:
    rows = [
        ("寄存器堆", 97000, "喂饱 12 TFLOPS 所必需"),
        ("Shared Memory", 6100, "32 bank × 4 B × 20 SM"),
        ("L2", 1500, "全卡共享"),
        ("显存 GDDR6", 192, "Roofline 屋顶画的就是这条"),
        ("PCIe 4.0 x8", 16, "去主机内存"),
        ("NVMe SSD", 6, "顺序读"),
        ("25 GbE 网络", 3, "跨机器"),
    ]
    cols = [C["compute"], "#E4785C", C["capacity"], C["memory"],
            C["accent"], C["neutral"], "#8D6A9F"]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    n = len(rows)
    ypos = np.arange(n)
    vals = [r[1] for r in rows]
    ax.barh(ypos, vals, color=cols, alpha=0.92, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlim(1, 4e7)
    ax.set_xlabel("带宽 GB/s（对数轴）")

    for i, (_name, v, note) in enumerate(rows):
        lab = f"{v/1000:,.1f} TB/s" if v >= 1000 else f"{v:,.0f} GB/s"
        ratio = f"{v/192:.0f}× 显存" if v >= 192 else f"显存的 1/{192/v:.0f}"
        ax.text(v * 1.6, i, f"{lab}    {ratio}    ·  {note}",
                va="center", fontsize=9.6, color=C["neutral"],
                fontweight="bold", zorder=10,
                bbox=dict(fc="white", ec="none", pad=1.5))

    ax.axvline(192, color=C["memory"], ls="--", lw=1.6, alpha=0.8)
    ax.text(192, -0.55, "Day 02 的屋顶", ha="center", fontsize=9.5,
            color=C["memory"], fontweight="bold")

    fig.text(0.5, 0.035,
             "Roofline 只有一层屋顶，因为它假设「数据都从显存来」。"
             "而所有 kernel 优化，本质上都是在把数据往上面那几根柱子挪。",
             ha="center", fontsize=11, color=C["compute"], fontweight="bold")

    fig.suptitle("图 4 · 带宽阶梯：从 97 TB/s 到 3 GB/s，跨 5 个数量级",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig4_bandwidth_ladder.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
def fig5_three_regimes() -> None:
    """工作集扫描曲线的三段解释：启动开销 / 命中缓存 / 落到 DRAM。"""
    t0 = 12e-6          # 每次 kernel 启动的固定开销 (s)
    b_cache = 190e9     # 命中片上缓存时的带宽
    b_dram = 90e9       # 落到 DRAM 时的带宽
    cache_bytes = 8e6   # 假想的缓存容量

    sizes = np.logspace(np.log10(4e3), np.log10(4e8), 400)
    bw_ceiling = np.where(sizes < cache_bytes, b_cache, b_dram)
    # 过渡区平滑一下，模拟部分命中
    trans = (sizes > cache_bytes * 0.5) & (sizes < cache_bytes * 4)
    frac = np.clip(np.log(sizes[trans] / (cache_bytes * 0.5)) / np.log(8), 0, 1)
    bw_ceiling[trans] = b_cache * (1 - frac) + b_dram * frac

    t = t0 + sizes / bw_ceiling
    eff = sizes / t / 1e9

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.plot(sizes / 1e6, eff, lw=2.8, color=C["memory"], zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("工作集大小（MB，对数轴）")
    ax.set_ylabel("实测到的有效带宽  GB/s")
    ax.set_ylim(0, 215)
    ax.set_xlim(0.004, 400)

    ax.axhline(b_cache / 1e9, color=C["capacity"], ls="--", lw=1.5)
    ax.text(0.006, b_cache / 1e9 + 5, "片上缓存带宽", fontsize=9.5,
            color=C["capacity"], fontweight="bold")
    ax.axhline(b_dram / 1e9, color=C["compute"], ls="--", lw=1.5)
    ax.text(0.006, b_dram / 1e9 + 5, "DRAM 带宽（真正的屋顶）", fontsize=9.5,
            color=C["compute"], fontweight="bold")

    for lo, hi, col in ((0.004, 1.2, C["accent"]), (1.2, 8.0, C["capacity"]),
                        (8.0, 400, C["compute"])):
        ax.axvspan(lo, hi, color=col, alpha=0.07, zorder=0)

    ann = [
        (0.055, 30, "① 太小\n时间被 kernel 启动开销吃掉\n"
                    r"$t \approx t_0$，带宽 $\approx S/t_0$" "\n斜率固定，纯线性上升", C["accent"]),
        (2.3, 118, "② 刚好\n整个工作集都待在片上缓存里\n"
                   "第二次访问根本没碰 DRAM\n→ 测出来比「峰值」还高", C["capacity"]),
        (45, 118, "③ 太大\n缓存装不下，每次都得去 DRAM\n"
                  "→ 这才是 Roofline 的那个数", C["compute"]),
    ]
    for x, y, txt, col in ann:
        ax.text(x, y, txt, fontsize=9.8, color=col, fontweight="bold",
                ha="center", va="center", linespacing=1.7, zorder=10,
                bbox=dict(fc="white", ec=col, lw=1.2, alpha=0.93,
                          boxstyle="round,pad=0.45"))

    ax.annotate("", xy=(8.0, 96), xytext=(8.0, 60),
                arrowprops=dict(arrowstyle="->", lw=2, color=C["neutral"]))
    ax.text(8.6, 52, "拐点位置 ≈ 这一级缓存的容量\n（厂商不公布，但你能测出来）",
            fontsize=9.5, color=C["neutral"], fontweight="bold", linespacing=1.6)

    ax.set_title(r"模型：$t = t_0 + S / B$，实测带宽 $= S/t$ —— 三段全部来自这一个公式",
                 fontsize=11.5)
    fig.suptitle("图 5 · Day 02 那条奇怪的曲线，终于说得通了",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig5_three_regimes.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 6
def fig6_move_1gb() -> None:
    """同样搬 1 GB，走不同的路要多久。"""
    rows = [
        ("显存 到 SM\n(GDDR6 192 GB/s)", 1e9 / 192e9, C["memory"]),
        ("统一内存 CPU 到 GPU\n(M4，实测约 28 GB/s)", 1e9 / 28e9, C["ok"]),
        ("PCIe 4.0 x8 · pinned\n(约 13 GB/s)", 1e9 / 13e9, C["accent"]),
        ("PCIe 4.0 x8 · pageable\n(约 6 GB/s，多一次 CPU 拷贝)", 1e9 / 6e9, "#8D6A9F"),
        ("NVMe SSD 顺序读\n(约 6 GB/s)", 1e9 / 6e9, C["neutral"]),
        ("25 GbE 网络\n(约 3 GB/s)", 1e9 / 3e9, "#B0755E"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    n = len(rows)
    ypos = np.arange(n)
    vals = [r[1] * 1000 for r in rows]           # ms
    ax.barh(ypos, vals, color=[r[2] for r in rows], alpha=0.92, height=0.6)
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(n - 0.4, -1.0)
    ax.set_xlim(0, 400)
    ax.set_xlabel("搬 1 GB 需要的时间（ms）")

    base = vals[0]
    for i, v in enumerate(vals):
        ax.text(v + 6, i, f"{v:.0f} ms      {v/base:.0f}× 显存",
                va="center", fontsize=10, color=C["neutral"],
                fontweight="bold", zorder=10,
                bbox=dict(fc="white", ec="none", pad=1.5))

    ax.axvline(24, color=C["compute"], ls="--", lw=1.8)
    ax.text(30, -0.62, "主线案例的 TPOT 预算 = 24 ms",
            fontsize=10, color=C["compute"], fontweight="bold", va="center")

    fig.text(0.5, 0.035,
             "所有 offload 方案（KV offload / 权重分层加载 / ZeRO-Offload）的成败都在这张图上："
             "省下的显存，值不值这几十倍的时间",
             ha="center", fontsize=11, color=C["accent"], fontweight="bold")

    fig.suptitle("图 6 · 同样一份 1 GB，走哪条路差 60 倍",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig6_move_1gb.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 7
def fig7_flashattention() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    n_tok, d_head, heads, b = 4096, 128, 32, 2
    qkvo = 4 * n_tok * d_head * b * heads / 1e9          # GB
    s_mat = n_tok * n_tok * b * heads / 1e9              # GB（一次）

    # --- 左：朴素实现 ---
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(Rectangle((0.4, 0.5), 9.2, 2.3, fc="#EAF4F6",
                           ec=C["memory"], lw=2))
    ax.text(5, 1.65, "HBM / 显存", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C["memory"])
    ax.add_patch(Rectangle((0.4, 6.6), 9.2, 2.3, fc="#FDF1F3",
                           ec=C["compute"], lw=2))
    ax.text(5, 7.75, "SM 片上（Shared Memory 只有 100 KB）", ha="center",
            va="center", fontsize=13, fontweight="bold", color=C["compute"])

    steps = [
        (1.4, "读 Q,K", "↑"), (3.0, "写 S", "↓"),
        (4.6, "读 S", "↑"), (6.2, "写 P", "↓"),
        (7.8, "读 P,V", "↑"), (9.0, "写 O", "↓"),
    ]
    for x, lab, direc in steps:
        y0, y1 = (2.9, 6.5) if direc == "↑" else (6.5, 2.9)
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                     arrowstyle="-|>", mutation_scale=15,
                                     lw=2.0, color=C["neutral"], alpha=0.85))
        ax.text(x, 4.7, lab, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=C["neutral"], rotation=90,
                bbox=dict(fc="white", ec="none", pad=1.5))

    ax.text(5, 3.55, f"中间矩阵 S / P 各 {s_mat:.2f} GB，来回过 4 遍 HBM",
            ha="center", fontsize=10.5, color=C["compute"], fontweight="bold")
    ax.text(5, 9.55, f"HBM 总流量 ≈ {qkvo + 4 * s_mat:.1f} GB"
                     f"    @192 GB/s = {(qkvo + 4*s_mat)/192*1000:.0f} ms",
            ha="center", fontsize=12, fontweight="bold", color=C["compute"])
    ax.set_title("① 朴素 Attention：把 n×n 矩阵写出去再读回来")

    # --- 右：FlashAttention ---
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(Rectangle((0.4, 0.5), 9.2, 2.3, fc="#EAF4F6",
                           ec=C["memory"], lw=2))
    ax.text(5, 1.65, "HBM / 显存", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C["memory"])
    ax.add_patch(Rectangle((0.4, 6.6), 9.2, 2.3, fc="#FDF1F3",
                           ec=C["compute"], lw=2))
    ax.text(2.0, 7.75, "SM 片上", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C["compute"])

    for x, lab, direc, ha in ((2.2, "读 Q,K,V 的一个块", "↑", "left"),
                              (9.2, "写 O 的一个块", "↓", "right")):
        y0, y1 = (2.9, 6.5) if direc == "↑" else (6.5, 2.9)
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                     arrowstyle="-|>", mutation_scale=16,
                                     lw=2.4, color=C["ok"]))
        dx = 0.32 if ha == "left" else -0.32
        ax.text(x + dx, 4.7, lab, ha=ha, va="center", fontsize=9.8,
                fontweight="bold", color=C["ok"], rotation=90)

    ax.add_patch(Rectangle((3.7, 6.95), 5.4, 1.6, fc=C["capacity"],
                           ec="white", lw=1.5, alpha=0.9))
    ax.text(6.4, 7.75, "S 的一个小块在这里生老病死\n算 S 块 → softmax → 乘 V",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#3A2A00", linespacing=1.7)

    ax.text(5.6, 3.55, "S 从来没有在 HBM 里出现过", ha="center",
            fontsize=11, color=C["ok"], fontweight="bold")
    ax.text(5, 9.55, f"HBM 总流量 ≈ {qkvo:.2f} GB"
                     f"    @192 GB/s = {qkvo/192*1000:.1f} ms",
            ha="center", fontsize=12, fontweight="bold", color=C["ok"])
    ax.set_title("② FlashAttention：把中间结果关在片上")

    fig.suptitle(f"图 7 · 同样的数学、同样的 FLOPs，HBM 流量差 "
                 f"{(qkvo + 4*s_mat)/qkvo:.0f} 倍"
                 f"（n={n_tok}, {heads} 头, fp16）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    fig.savefig(OUT / "fig7_flashattention.png")
    plt.close(fig)


if __name__ == "__main__":
    font = setup()
    print("CJK font:", font)
    fig1_pyramid()
    fig2_human_scale()
    fig3_why_pyramid()
    fig4_bandwidth_ladder()
    fig5_three_regimes()
    fig6_move_1gb()
    fig7_flashattention()
    for p in sorted(OUT.glob("*.png")):
        print("  ->", p.relative_to(OUT.parents[1]))
