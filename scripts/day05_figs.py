"""Day 05 配图 · 数值格式全解

    .venv/bin/python scripts/day05_figs.py

图号 = 正文阅读顺序：
  fig1  七种格式的位分布
  fig2  动态范围：FP16 为什么会炸
  fig3  精度阶梯：间隔随数值增大，以及「数不动了」的临界点
  fig4  指数域的特殊编码：零 / 次正规 / 正规 / inf / NaN
  fig5  浮点的对数刻度 vs 整数的均匀刻度
  fig6  离群值：per-tensor 量化为什么会崩
  fig7  主线案例：精度就是在选屋顶
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _style import C, setup

OUT = Path(__file__).resolve().parents[1] / "assets" / "day05"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 格式定义表
# name, 指数位 e, 尾数位 m, bias, 备注
FORMATS = [
    ("FP32",      8, 23, 127,  "训练/推理的老标准"),
    ("TF32",      8, 10, 127,  "A100 起，Tensor Core 内部用"),
    ("BF16",      8,  7, 127,  "训练首选：范围同 FP32"),
    ("FP16",      5, 10,  15,  "推理常用：范围窄，精度高"),
    ("FP8 E5M2",  5,  2,  15,  "训练时存梯度"),
    ("FP8 E4M3",  4,  3,   7,  "训练/推理时存权重与激活"),
]


def fmt_max(e: int, m: int, bias: int, name: str = "") -> float:
    """最大可表示的正规数。"""
    if name == "FP8 E4M3":                      # OCP 变体：无 inf，尾数全 1 是 NaN
        return (2 - 2 ** -(m - 1)) * 2.0 ** (2 ** e - 1 - bias)
    emax = 2 ** e - 2 - bias                    # 指数域全 1 留给 inf/NaN
    return (2 - 2.0 ** -m) * 2.0 ** emax


def fmt_min_normal(bias: int) -> float:
    return 2.0 ** (1 - bias)


def fmt_min_subnormal(m: int, bias: int) -> float:
    return 2.0 ** (1 - bias - m)


def fmt_eps(m: int) -> float:
    return 2.0 ** -m


# ------------------------------------------------------------------- 图 1
def fig1_bit_layout() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    n = len(FORMATS)
    for i, (name, e, m, bias, note) in enumerate(FORMATS):
        y = n - 1 - i
        total = 1 + e + m
        ax.add_patch(Rectangle((0, y), 1, 0.62, fc=C["neutral"], ec="white", lw=1.2))
        ax.add_patch(Rectangle((1, y), e, 0.62, fc=C["compute"], ec="white", lw=1.2))
        ax.add_patch(Rectangle((1 + e, y), m, 0.62, fc=C["memory"], ec="white", lw=1.2))

        ax.text(1 + e / 2, y + 0.31, f"指数 {e}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
        if m >= 3:
            ax.text(1 + e + m / 2, y + 0.31, f"尾数 {m}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
        else:
            ax.text(1 + e + m / 2, y + 0.78, f"尾数 {m}", ha="center", va="bottom",
                    color=C["memory"], fontweight="bold", fontsize=9)

        ax.text(-0.5, y + 0.31, name, ha="right", va="center",
                fontsize=12, fontweight="bold")
        digits = np.log10(2 ** (m + 1))
        ax.text(total + 0.5, y + 0.31,
                f"{total} bit  ·  十进制有效数字 ≈ {digits:.1f} 位  ·  {note}",
                ha="left", va="center", fontsize=9.5, color=C["neutral"])

    ax.set_xlim(-6.5, 46)
    ax.set_ylim(-0.9, n + 0.2)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 8, 16, 24, 32])
    ax.set_xlabel("第几个 bit")
    ax.grid(axis="y", visible=False)

    for x, lab, col in ((0.5, "符号", C["neutral"]), (5, "指数 → 管【范围】", C["compute"]),
                        (20, "尾数 → 管【精度】", C["memory"])):
        ax.text(x, n + 0.02, lab, ha="center", fontsize=11,
                fontweight="bold", color=col)

    fig.text(0.5, 0.035,
             "同样 16 bit，BF16 把位数给了指数（范围大），FP16 把位数给了尾数（精度高）"
             " —— 这一个取舍决定了它们各自的用途",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 1 · 一个浮点数的三段结构：符号 / 指数 / 尾数",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig1_bit_layout.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_dynamic_range() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    rows = [f for f in FORMATS]
    n = len(rows)
    cols = [C["capacity"], "#C79A2E", C["ok"], C["compute"], "#B0755E", C["accent"]]

    for i, ((name, e, m, bias, _note), col) in enumerate(zip(rows, cols)):
        y = n - 1 - i
        lo = fmt_min_subnormal(m, bias)
        lo_n = fmt_min_normal(bias)
        hi = fmt_max(e, m, bias, name)
        ax.plot([lo, hi], [y, y], lw=9, color=col, solid_capstyle="butt", alpha=0.35)
        ax.plot([lo_n, hi], [y, y], lw=9, color=col, solid_capstyle="butt")
        ax.text(hi * 4, y, f"{hi:.3g}", va="center", fontsize=9.5,
                color=col, fontweight="bold")
        ax.text(lo / 5, y, f"{lo:.2g}", va="center", ha="right", fontsize=9.5,
                color=col, fontweight="bold")

    ax.axvspan(1e-10, 1e-2, color=C["memory"], alpha=0.13, zorder=0)
    ax.text(1e-6, n - 0.4, "训练时梯度的典型分布区间",
            ha="center", fontsize=10.5, color=C["memory"], fontweight="bold")

    ax.axvline(fmt_min_subnormal(10, 15), color=C["compute"], ls="--", lw=1.8)
    ax.annotate("FP16 到 5.96e-8 以下\n全部变成 0",
                xy=(6e-8, 2.0), xytext=(2e-24, 1.1),
                fontsize=10, color=C["compute"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["compute"]))
    ax.axvline(65504, color=C["compute"], ls="--", lw=1.8)
    ax.annotate("FP16 超过 65504\n就是 inf",
                xy=(65504, 3.4), xytext=(1e17, 4.4),
                fontsize=10, color=C["compute"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["compute"]))

    ax.set_xscale("log")
    ax.set_xlim(1e-53, 1e46)
    ax.set_ylim(-0.8, n + 0.05)
    ax.set_yticks(np.arange(n)[::-1], [r[0] for r in rows], fontsize=11.5)
    ax.set_xlabel("可表示的正数范围（对数轴）")
    ax.grid(axis="y", visible=False)

    fig.text(0.5, 0.035,
             "浅色段 = 次正规数（能表示但精度已经在掉）　|　"
             "BF16 和 FP32 的范围完全一样，这就是它训练更稳的全部原因",
             ha="center", fontsize=11, color=C["ok"], fontweight="bold")

    fig.suptitle("图 2 · 动态范围：FP16 两头都够不着，BF16 够得着",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig2_dynamic_range.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig3_precision_gap() -> None:
    """相邻两个可表示数之间的间隔（ULP），随数值大小指数增长。"""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    show = [("FP32", 23, C["capacity"]), ("BF16", 7, C["ok"]),
            ("FP16", 10, C["compute"])]
    vals = np.logspace(-3, 8, 400)

    ax = axes[0]
    for name, m, col in show:
        ulp = 2.0 ** (np.floor(np.log2(vals)) - m)
        ax.plot(vals, ulp, lw=2.4, color=col, label=name)
    ax.axhline(1.0, color=C["neutral"], ls="--", lw=1.6)
    ax.text(2e-3, 1.35, "间隔 = 1：再加 1 就加不动了", fontsize=10,
            color=C["neutral"], fontweight="bold")
    for name, m, col, x in (("FP16", 10, C["compute"], 2048),
                            ("BF16", 7, C["ok"], 256),
                            ("FP32", 23, C["capacity"], 2 ** 24)):
        ax.scatter([x], [1.0], s=130, color=col, zorder=6,
                   edgecolor="white", lw=1.5)
        ax.annotate(f"{name}\n{x:,}", xy=(x, 1.0), xytext=(x, 4e-4 if name != "FP32" else 6e-6),
                    fontsize=9.5, color=col, fontweight="bold", ha="center",
                    linespacing=1.5,
                    arrowprops=dict(arrowstyle="->", lw=1.4, color=col))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("数值大小")
    ax.set_ylabel("相邻两个可表示数的间隔（ULP）")
    ax.set_ylim(1e-9, 1e3)
    ax.legend(loc="upper left")
    ax.set_title("① 数越大，能分辨的最小差别越粗")

    # --- 右：累加 1 的实际后果 ---
    ax = axes[1]
    steps = np.arange(1, 5000)
    for name, m, col in show[1:]:
        acc, curve = 0.0, []
        limit = 2.0 ** (m + 1)
        for _ in steps:
            ulp = 2.0 ** (np.floor(np.log2(acc)) - m) if acc > 0 else 1.0
            acc = acc + 1.0 if ulp <= 1.0 else acc
            curve.append(acc)
            if acc >= limit * 4:
                break
        curve = np.array(curve + [curve[-1]] * (len(steps) - len(curve)))
        ax.plot(steps, curve, lw=2.6, color=col, label=name)
    ax.plot(steps, steps, lw=2.0, color=C["capacity"], ls=":", label="FP32（理想）")

    ax.axhline(2048, color=C["compute"], ls="--", lw=1.3, alpha=0.7)
    ax.text(4900, 2160, "FP16 卡在 2048", ha="right", fontsize=10,
            color=C["compute"], fontweight="bold")
    ax.axhline(256, color=C["ok"], ls="--", lw=1.3, alpha=0.7)
    ax.text(4900, 380, "BF16 卡在 256", ha="right", fontsize=10,
            color=C["ok"], fontweight="bold")

    ax.set_xlabel("已经加了多少个 1")
    ax.set_ylabel("累加器里的值")
    ax.set_ylim(0, 5000)
    ax.legend(loc="upper left")
    ax.set_title("② 从 0 开始一个一个加 1，会停在哪")

    fig.text(0.5, 0.035,
             "这就是为什么 Tensor Core 吃 FP16 输入、却一律用 FP32 累加 —— "
             "不然一个 4096 长度的点积根本加不完",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 3 · 精度：大数会「吃掉」小数",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig3_precision_gap.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig4_special_values() -> None:
    """以 FP16 为例，指数域的取值决定这个数是什么。"""
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 8.3)

    blocks = [
        (0.3, 2.3, "指数域 = 00000", C["accent"],
         "尾数 = 0  →  ±0\n尾数 ≠ 0  →  次正规数\n（没有隐含的 1，精度在衰减）"),
        (2.8, 4.3, "指数域 = 00001 ~ 11110", C["ok"],
         "正规数\n$(-1)^S \\times 1.M \\times 2^{E-15}$\n99.99% 的时间待在这里"),
        (7.3, 2.4, "指数域 = 11111", C["compute"],
         "尾数 = 0  →  ±inf\n尾数 ≠ 0  →  NaN\n（溢出和非法运算的归宿）"),
    ]
    for x0, w, title, col, body in blocks:
        ax.add_patch(Rectangle((x0, 3.4), w, 4.4, fc=col, ec="white",
                               lw=2, alpha=0.14))
        ax.add_patch(Rectangle((x0, 6.9), w, 0.9, fc=col, ec="white", lw=2))
        ax.text(x0 + w / 2, 7.35, title, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11.5)
        ax.text(x0 + w / 2, 5.2, body, ha="center", va="center",
                fontsize=10.5, color=C["neutral"], linespacing=2.0)

    ax.annotate("", xy=(9.9, 2.9), xytext=(0.3, 2.9),
                arrowprops=dict(arrowstyle="->", lw=2, color=C["neutral"]))
    ax.text(5.0, 2.35, "指数域从 00000 数到 11111（FP16 是 5 位）",
            ha="center", fontsize=10.5, color=C["neutral"])

    ax.text(5.0, 1.15,
            "两个坑：① 指数域全 0 和全 1 被「征用」了，所以真正能用的指数少了 2 个\n"
            "② NaN 会传染 —— 任何和 NaN 的运算结果都是 NaN，"
            "所以 loss 一旦变 NaN，整个训练就废了",
            ha="center", va="center", fontsize=11, color=C["compute"],
            fontweight="bold", linespacing=1.9)

    fig.suptitle("图 4 · 指数域的三种特殊编码（以 FP16 为例）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    fig.savefig(OUT / "fig4_special_values.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
def fig5_float_vs_int() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2))

    # 一个玩具浮点：1 位符号、3 位指数、2 位尾数
    ticks = []
    for e in range(0, 8):
        for mm in range(4):
            ticks.append((1 + mm / 4) * 2.0 ** (e - 3))
    ticks = np.array([t for t in ticks if t <= 16])

    ax = axes[0]
    ax.vlines(ticks, 0, 1, color=C["compute"], lw=1.6)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(0, 1.9)
    ax.set_yticks([])
    ax.set_title("① 浮点：刻度是【对数】排布的 —— 0 附近极密，远处极疏")
    ax.text(0.35, 1.35, "这里挤了一大堆刻度", fontsize=10,
            color=C["compute"], fontweight="bold")
    ax.text(11.5, 1.35, "这里两个刻度之间隔了 2", fontsize=10,
            color=C["compute"], fontweight="bold")
    ax.set_xlabel("数值")

    ax = axes[1]
    iticks = np.linspace(0, 16, 33)
    ax.vlines(iticks, 0, 1, color=C["memory"], lw=1.6)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(0, 1.9)
    ax.set_yticks([])
    ax.set_title("② 定点整数：刻度是【均匀】排布的 —— 每一格都一样宽")
    ax.text(8, 1.35, "间隔恒定 = scale，全程一视同仁", fontsize=10,
            ha="center", color=C["memory"], fontweight="bold")
    ax.set_xlabel("数值")

    fig.text(0.5, 0.035,
             "选浮点还是定点，取决于你的数据长什么样：\n"
             "梯度横跨十几个数量级 → 必须用浮点　|　"
             "权重集中在一个窄区间 → 定点更省位数",
             ha="center", fontsize=11.5, color=C["accent"],
             fontweight="bold", linespacing=1.7)

    fig.suptitle("图 5 · 浮点和整数的根本区别：刻度怎么排",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(OUT / "fig5_float_vs_int.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 6
def fig6_outlier() -> None:
    """INT4 对称量化：per-tensor 遇到离群值会直接崩。"""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, 128)
    x[57] = 20.0                                # 一个离群值
    QMAX = 7                                    # INT4 对称：-7 ~ +7

    def quant(v, s):
        return np.clip(np.round(v / s), -QMAX, QMAX) * s

    s_tensor = np.abs(x).max() / QMAX
    y_tensor = quant(x, s_tensor)

    g = 32
    y_group = x.copy()
    for i in range(0, len(x), g):
        seg = x[i:i + g]
        y_group[i:i + g] = quant(seg, np.abs(seg).max() / QMAX)

    clean = np.ones(128, bool)
    clean[(57 // g) * g:(57 // g) * g + g] = False   # 排掉含离群值的那一组
    err_t = np.abs(y_tensor - x)[clean].mean()
    err_g = np.abs(y_group - x)[clean].mean()
    zeroed = int((y_tensor == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    ax = axes[0]
    ax.bar(np.arange(128), x, color=C["memory"], width=0.9)
    ax.bar([57], [x[57]], color=C["compute"], width=2.0)
    ax.annotate("一个离群值\n比其余大 20 多倍", xy=(57, 20), xytext=(92, 13),
                fontsize=10.5, color=C["compute"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["compute"]))
    ax.set_xlabel("张量里的第几个元素")
    ax.set_ylabel("数值")
    ax.set_title("① LLM 的激活里真的长这样")

    ax = axes[1]
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], color=C["neutral"], ls="--", lw=1.4,
            label="理想（无损）")
    ax.scatter(x[clean], y_tensor[clean], s=42, color=C["compute"], alpha=0.85,
               label=f"per-tensor（1 个 scale）平均误差 {err_t:.3f}")
    ax.scatter(x[clean], y_group[clean], s=42, color=C["ok"], alpha=0.85,
               marker="D", label=f"per-group（每 {g} 个 1 个 scale）平均误差 {err_g:.3f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("原值")
    ax.set_ylabel("INT4 量化再还原之后的值")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.annotate(f"per-tensor 把 {zeroed}/128 个数\n全部压成了 0",
                xy=(0.62, 0.0), xytext=(0.52, -0.62),
                fontsize=10.5, color=C["compute"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["compute"]))
    ax.set_title("② 同样是 INT4，还原出来的东西完全不同")

    fig.text(0.5, 0.035,
             f"per-tensor 的 scale 被那一个离群值撑大了 20 多倍，"
             f"于是其余数字全落进了同一个量化档位　→　平均误差大 {err_t/err_g:.0f} 倍",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 6 · 为什么 LLM 量化必须分组：一个离群值就能毁掉整个张量",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig6_outlier.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 7
def fig7_precision_ladder() -> None:
    """主线案例：7B 模型在不同精度下的显存与 TPOT。"""
    rows = [
        ("FP32", 4.0, C["accent"]),
        ("BF16 / FP16", 2.0, C["capacity"]),
        ("FP8 / INT8", 1.0, C["ok"]),
        ("INT4", 0.5, C["compute"]),
    ]
    params, bw = 7e9, 192e9

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    names = [r[0] for r in rows]
    gb = [params * r[1] / 1e9 for r in rows]
    ax.bar(names, gb, color=[r[2] for r in rows], width=0.6)
    ax.axhline(6, color=C["compute"], ls="--", lw=2)
    ax.text(-0.42, 7.2, "RTX 4050 只有 6 GB", ha="left", fontsize=11,
            color=C["compute"], fontweight="bold")
    for i, v in enumerate(gb):
        ok = "装得下" if v < 5 else "装不下"
        ax.text(i, v + 0.8, f"{v:.1f} GB\n{ok}", ha="center", fontsize=10.5,
                fontweight="bold", color=C["neutral"], linespacing=1.5, zorder=10)
    ax.set_ylabel("7B 模型的权重占用 (GB)")
    ax.set_ylim(0, 34)
    ax.set_title("① 能不能跑：容量说了算")

    ax = axes[1]
    tpot = [params * r[1] / bw * 1000 for r in rows]
    ax.bar(names, tpot, color=[r[2] for r in rows], width=0.6)
    for i, v in enumerate(tpot):
        ax.text(i, v + 4, f"{v:.0f} ms\n{1000/v:.0f} tok/s", ha="center",
                fontsize=10.5, fontweight="bold", color=C["neutral"],
                linespacing=1.5, zorder=10)
    ax.set_ylabel("decode 每 token 最短耗时 (ms)")
    ax.set_ylim(0, 178)
    ax.set_title("② 跑多快：字节数说了算（Day 01 的公式）")

    fig.text(0.5, 0.035,
             "精度减半 = 权重减半 = 每 token 搬运时间减半。"
             "前四天的每一个公式，分母或分子上都挂着这个「字节/参数」",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")

    fig.suptitle("图 7 · 选精度，就是同时在选「装不装得下」和「跑多快」",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(OUT / "fig7_precision_ladder.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    fig1_bit_layout()
    fig2_dynamic_range()
    fig3_precision_gap()
    fig4_special_values()
    fig5_float_vs_int()
    fig6_outlier()
    fig7_precision_ladder()
    for p in sorted(OUT.glob("*.png")):
        print("  ->", p.relative_to(OUT.parents[1]))
