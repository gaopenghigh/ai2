"""附加课 X02 配图 · 矩阵与形状

    .venv/bin/python scripts/extra_x02_figs.py

图号 = 正文阅读顺序：
  fig1  矩阵的两种身份：一叠数据 vs 一个变换
  fig2  矩阵乘的三种读法
  fig3  形状推演：中间那个维度被消掉
  fig4  行主序内存布局：为什么 nn.Linear 存 (out, in)
  fig5  算术强度 ≈ batch size
  fig6  一层 Transformer 的完整形状链
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import C, setup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "extra" / "x02"
OUT.mkdir(parents=True, exist_ok=True)


def _grid(ax, x0, y0, w, h, nr, nc, fc, ec="white", lw=1.2, alpha=0.9):
    cw, ch = w / nc, h / nr
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + h - (i + 1) * ch), cw, ch,
                                   fc=fc, ec=ec, lw=lw, alpha=alpha))


# ------------------------------------------------------------------- 图 1
def fig1_two_kinds() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # --- 左：数据矩阵 ---
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.3, "① 一叠数据（激活）", ha="center", fontsize=13.5,
            fontweight="bold", color=C["memory"])
    _grid(ax, 2.2, 3.2, 5.6, 4.4, 4, 8, C["memory"])
    for i, lab in enumerate(["token 1", "token 2", "token 3", "token 4"]):
        ax.text(2.0, 7.6 - (i + 0.5) * 1.1, lab, ha="right", va="center",
                fontsize=10.5, color=C["memory"], fontweight="bold")
    ax.annotate("", xy=(7.8, 2.85), xytext=(2.2, 2.85),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color=C["neutral"]))
    ax.text(5, 2.35, "d = 每个 token 有多少个数", ha="center", fontsize=11,
            color=C["neutral"], fontweight="bold")
    ax.text(5, 1.35,
            "形状 (T, d)：一行一个 token\n"
            "「有多少个东西」在前，「每个多少数」在后",
            ha="center", va="center", fontsize=11, color=C["memory"],
            fontweight="bold", linespacing=1.8)

    # --- 右：变换矩阵 ---
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.3, "② 一个变换（权重）", ha="center", fontsize=13.5,
            fontweight="bold", color=C["compute"])
    _grid(ax, 2.6, 3.2, 4.8, 4.4, 6, 6, C["compute"])
    ax.annotate("", xy=(2.35, 7.6), xytext=(2.35, 3.2),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color=C["neutral"]))
    ax.text(1.9, 5.4, "输入\n维度", ha="center", va="center", fontsize=10.5,
            color=C["neutral"], fontweight="bold", linespacing=1.6)
    ax.annotate("", xy=(7.4, 2.85), xytext=(2.6, 2.85),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color=C["neutral"]))
    ax.text(5, 2.35, "输出维度", ha="center", fontsize=10.5,
            color=C["neutral"], fontweight="bold")
    ax.text(5, 1.35,
            "形状 (in, out)：一列一个「输出神经元的配方」\n"
            "它不是数据，是「把向量搬到另一个空间」的规则",
            ha="center", va="center", fontsize=11, color=C["compute"],
            fontweight="bold", linespacing=1.8)

    fig.text(0.5, 0.035,
             "同样叫「矩阵」，含义完全不同。分清这两种身份，"
             "是看懂一切形状的第一步",
             ha="center", fontsize=12, color=C["accent"], fontweight="bold")
    fig.suptitle("图 1 · 矩阵有两种完全不同的身份", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "fig1_two_kinds.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig2_three_readings() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    n, k, m = 4, 3, 5

    for idx, (ax, title, sub, col) in enumerate(zip(
            axes,
            ["① 行 × 列", "② 列的线性组合", "③ 外积求和"],
            ["每个格子 = 一个点积", "输出的一列 = 输入各列的加权和",
             "沿中间维度一段一段累加"],
            [C["memory"], C["ok"], C["compute"]])):
        ax.set_axis_off()
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 9)

        # A (n,k)
        _grid(ax, 0.6, 4.0, 2.4, 3.2, n, k, C["neutral"], alpha=0.28)
        ax.text(1.8, 7.55, f"A ({n},{k})", ha="center", fontsize=10.5,
                fontweight="bold", color=C["neutral"])
        # B (k,m)
        _grid(ax, 4.0, 4.4, 4.0, 2.4, k, m, C["neutral"], alpha=0.28)
        ax.text(6.0, 7.15, f"B ({k},{m})", ha="center", fontsize=10.5,
                fontweight="bold", color=C["neutral"])
        # C (n,m)
        _grid(ax, 8.9, 4.0, 3.6, 3.2, n, m, C["neutral"], alpha=0.28)
        ax.text(10.7, 7.55, f"C ({n},{m})", ha="center", fontsize=10.5,
                fontweight="bold", color=C["neutral"])
        ax.text(3.4, 5.6, "@", fontsize=17, ha="center", va="center",
                color=C["neutral"], fontweight="bold")
        ax.text(8.4, 5.6, "=", fontsize=17, ha="center", va="center",
                color=C["neutral"], fontweight="bold")

        cwA, chA = 2.4 / k, 3.2 / n
        cwB, chB = 4.0 / m, 2.4 / k
        cwC, chC = 3.6 / m, 3.2 / n

        if idx == 0:                                   # 高亮 A 一行、B 一列、C 一格
            _grid(ax, 0.6, 4.0 + 3.2 - 2 * chA, 2.4, chA, 1, k, col)
            _grid(ax, 4.0 + 2 * cwB, 4.4, cwB, 2.4, k, 1, col)
            _grid(ax, 8.9 + 2 * cwC, 4.0 + 3.2 - 2 * chC, cwC, chC, 1, 1, col)
        elif idx == 1:                                 # 高亮 A 全部列、B 一列、C 一列
            _grid(ax, 0.6, 4.0, 2.4, 3.2, n, k, col, alpha=0.55)
            _grid(ax, 4.0 + 2 * cwB, 4.4, cwB, 2.4, k, 1, col)
            _grid(ax, 8.9 + 2 * cwC, 4.0, cwC, 3.2, n, 1, col)
        else:                                          # 高亮 A 一列、B 一行、C 全部
            _grid(ax, 0.6 + 1 * cwA, 4.0, cwA, 3.2, n, 1, col)
            _grid(ax, 4.0, 4.4 + 2.4 - 2 * chB, 4.0, chB, 1, m, col)
            _grid(ax, 8.9, 4.0, 3.6, 3.2, n, m, col, alpha=0.55)

        ax.set_title(title, fontsize=13)
        ax.text(6.4, 2.9, sub, ha="center", va="center", fontsize=11.5,
                color=col, fontweight="bold")
        ax.text(6.4, 1.7,
                ["复用为零 → 算术强度 0.5",
                 "「矩阵 = 线性变换」的本意",
                 "★ 硬件真正的做法（Day 03 §6）"][idx],
                ha="center", va="center", fontsize=10.5, color=C["neutral"])

    fig.suptitle("图 2 · 同一个矩阵乘，三种读法", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig2_three_readings.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig3_shape_rule() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    cw, chh, y0 = 9.0, 3.0, 6.0
    groups = [(8, "n", "k"), (38, "k", "m"), (68, "n", "m")]
    kill = {(0, 1), (1, 0)}                      # 要被消掉的那两个格子

    centers = {}
    for gi, (x0, a, b) in enumerate(groups):
        for ci, lab in enumerate((a, b)):
            bad = (gi, ci) in kill
            fc = "#FBE3E7" if bad else "#EAF4F6"
            ec = C["compute"] if bad else C["memory"]
            ax.add_patch(Rectangle((x0 + ci * cw, y0), cw, chh,
                                   fc=fc, ec=ec, lw=2.4))
            ax.text(x0 + ci * cw + cw / 2, y0 + chh / 2, lab, ha="center",
                    va="center", fontsize=25, fontweight="bold",
                    color=ec, family="monospace")
            centers[(gi, ci)] = x0 + ci * cw + cw / 2

    ax.text(32.5, y0 + chh / 2, "@", ha="center", va="center", fontsize=24,
            color=C["neutral"], fontweight="bold")
    ax.text(62.5, y0 + chh / 2, "=", ha="center", va="center", fontsize=24,
            color=C["neutral"], fontweight="bold")

    # 中间那两个消掉
    ax.annotate("这两个必须相等，然后一起消失",
                xy=(centers[(0, 1)], y0 - 0.15), xytext=(35, 3.4),
                ha="center", va="center", fontsize=13, color=C["compute"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=2, color=C["compute"]))
    ax.annotate("", xy=(centers[(1, 0)], y0 - 0.15), xytext=(35, 3.8),
                arrowprops=dict(arrowstyle="->", lw=2, color=C["compute"]))

    # 两头留下：直线箭头 + 竖直引线，避免弧线跑出画布
    def keep(cell_from, cell_to, y_line, y_label, label):
        x1, x2 = centers[cell_from], centers[cell_to]
        for x in (x1, x2):
            ax.plot([x, x], [y0 if y_line < y0 else y0 + chh, y_line],
                    lw=1.4, ls=":", color=C["ok"])
        ax.annotate("", xy=(x2, y_line), xytext=(x1, y_line),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C["ok"],
                                    mutation_scale=17))
        ax.text((x1 + x2) / 2, y_label, label, ha="center", va="center",
                fontsize=12, color=C["ok"], fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.5))

    keep((0, 0), (2, 0), y0 + chh + 0.7, y0 + chh + 1.5, "左边的第一个维度，原样留下")
    keep((1, 1), (2, 1), y0 - 0.7, y0 - 1.5, "右边的最后一个维度，原样留下")

    ax.text(50, 1.2, "记忆口诀：中间对上，中间消掉，两头留下",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["accent"],
            bbox=dict(fc="#F6F0FA", ec=C["accent"], lw=2,
                      boxstyle="round,pad=0.6"))

    fig.suptitle("图 3 · 形状推演只有一条规则", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig3_shape_rule.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig4_row_major() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.6))
    out_n, in_n = 3, 5

    def draw(ax, shape_txt, nr, nc, order_label, good, note):
        ax.set_axis_off()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10)
        H, cw = 4.6, 3.0
        ch = H / nr
        x0, y0 = 12, 3.4
        cols = [C["ok"], C["capacity"], C["accent"], "#8D6A9F", "#2E7D6E"]
        for i in range(nr):
            for j in range(nc):
                col = cols[i % len(cols)] if nr == out_n else cols[j % len(cols)]
                ax.add_patch(Rectangle((x0 + j * cw, y0 + H - (i + 1) * ch),
                                       cw, ch, fc=col, ec="white", lw=1.4,
                                       alpha=0.9))
        ax.text(x0 - 1.2, y0 + H / 2, shape_txt, ha="right", va="center",
                fontsize=12, fontweight="bold", color=C["neutral"])

        mx = x0 + nc * cw + 6
        ax.text(mx, y0 + H + 0.4, "内存里实际是这么一条排下去的：",
                fontsize=10.5, color=C["neutral"], fontweight="bold")
        for t in range(nr * nc):
            i, j = divmod(t, nc)
            col = cols[i % len(cols)] if nr == out_n else cols[j % len(cols)]
            ax.add_patch(Rectangle((mx + t * 2.6, y0 + 1.4), 2.4, 2.0,
                                   fc=col, ec="white", lw=1.2, alpha=0.9))
        ax.text(mx, y0 + 0.5, order_label, fontsize=10.5,
                color=C["ok"] if good else C["compute"], fontweight="bold")
        ax.text(2, 1.3, note, fontsize=11.5, va="center",
                color=C["ok"] if good else C["compute"], fontweight="bold")

    draw(axes[0], "weight\n(out=3, in=5)", out_n, in_n,
         "同色 = 同一个输出神经元的权重，连续摆在一起", True,
         "PyTorch 的做法：一行 = 一个输出神经元的全部权重 → 算 y[i] 时顺序读一整段")
    draw(axes[1], "weight\n(in=5, out=3)", in_n, out_n,
         "同色 = 同一个输出神经元的权重，被打散了", False,
         "如果存成 (in, out)：一个输出神经元的权重跳着放 → 每读一个浪费一条 cache line")

    fig.text(0.5, 0.03,
             "所以 nn.Linear 存 (out, in)、前向算 x @ weight.T —— "
             "而转置在 GEMM 里只是个标志位，不花钱",
             ha="center", fontsize=11.5, color=C["accent"], fontweight="bold")
    fig.suptitle("图 4 · 为什么 nn.Linear 存的是转置（接 Day 03 §7 / Day 04）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "fig4_row_major.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 5
def fig5_batch_ai() -> None:
    B = np.array([1, 2, 8, 32, 128, 512, 2048])
    ai = np.array([1.0, 2.0, 8.0, 31.5, 120.5, 409.6, 1024.0])
    tf = np.array([0.08, 0.15, 0.57, 1.98, 2.84, 3.65, 3.59])
    peak, ridge = 3.5, 41

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    ax.loglog(B, ai, "o-", lw=2.6, ms=10, color=C["memory"], label="实测算术强度")
    ax.loglog(B, B, "--", lw=2.0, color=C["compute"], label="理论值 = batch size")
    ax.axhline(ridge, color=C["accent"], ls=":", lw=2)
    ax.text(1.2, ridge * 1.25, f"机器平衡点 {ridge}", fontsize=10.5,
            color=C["accent"], fontweight="bold")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("算术强度 (FLOP/Byte)")
    ax.legend(loc="lower right")
    ax.set_title("① 算术强度就等于 batch size")

    ax = axes[1]
    ax.semilogx(B, tf, "o-", lw=2.8, ms=10, color=C["ok"])
    ax.axhline(peak, color=C["compute"], ls="--", lw=2)
    ax.text(1.2, peak * 0.93, "峰值算力 3.5 TFLOPS", fontsize=10.5,
            color=C["compute"], fontweight="bold", va="top")
    ax.axvline(ridge, color=C["accent"], ls=":", lw=2)
    ax.annotate("B ≈ 41 之后\n才真正撞上算力墙",
                xy=(ridge, 2.15), xytext=(4.2, 2.45),
                fontsize=11, color=C["accent"], fontweight="bold",
                ha="center", va="center", linespacing=1.7,
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["accent"]))
    ax.set_xlabel("batch size B")
    ax.set_ylabel("实测算力 (TFLOPS)")
    ax.set_ylim(0, 4.1)
    ax.set_title("② 在越过平衡点之前，加大 batch 几乎是「免费」的")

    fig.text(0.5, 0.035,
             r"$\mathrm{AI} = \dfrac{2Bdk}{(Bd + dk + Bk)\,b} \approx \dfrac{2B}{b}$"
             "　　fp16 时 b=2，于是 AI ≈ B　—— 这就是 Day 01 实验 2 的全部原因",
             ha="center", fontsize=12.5, color=C["accent"], fontweight="bold")
    fig.suptitle("图 5 · 为什么必须写成矩阵：权重被复用了 B 次（Mac mini M4 实测）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(OUT / "fig5_batch_ai.png")
    plt.close(fig)


# ------------------------------------------------------------------- 图 6
def fig6_layer_shapes() -> None:
    steps = [
        ("输入 x", "(T, d)", C["memory"], "T 个 token"),
        ("x @ Wq/Wk/Wv", "(T, d)", C["ok"], "(T,d)@(d,d)"),
        ("拆头", "(h, T, dh)", C["capacity"], "d → (h, dh)"),
        ("Q @ K.T", "(h, T, T)", C["compute"], "dh 被消掉"),
        ("softmax @ V", "(h, T, dh)", C["capacity"], "T 被消掉"),
        ("合头", "(T, d)", C["ok"], "(h,dh) → d"),
        ("@ Wo", "(T, d)", C["memory"], "★ 回到输入形状"),
    ]

    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    w = 11.5
    for i, (name, shape, col, note) in enumerate(steps):
        x = 2 + i * 13.8
        ax.add_patch(Rectangle((x, 4.6), w, 3.0, fc=col, ec="white", lw=2,
                               alpha=0.92))
        ax.text(x + w / 2, 6.6, name, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, 5.4, shape, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", family="monospace")
        ax.text(x + w / 2, 3.7, note, ha="center", va="center", fontsize=9.2,
                color=C["neutral"])
        if i:
            ax.add_patch(FancyArrowPatch((x - 2.1, 6.1), (x - 0.4, 6.1),
                                         arrowstyle="-|>", mutation_scale=15,
                                         lw=2, color=C["neutral"]))

    ax.add_patch(FancyArrowPatch((7.7, 8.0), (92.0, 8.0), arrowstyle="<|-|>",
                                 mutation_scale=16, lw=2.2, color=C["accent"],
                                 connectionstyle="arc3,rad=-0.14"))
    ax.text(50, 9.4, "首尾形状完全一样 → 残差能直接相加 → 才能摞 32 层",
            ha="center", fontsize=12.5, color=C["accent"], fontweight="bold")

    ax.text(50, 1.6,
            "全程只有两件事在发生：① 矩阵乘把「中间那个维度」消掉　"
            "② reshape/transpose 只是换个方式看同一堆数（不搬数据）",
            ha="center", va="center", fontsize=11.5, color=C["neutral"],
            fontweight="bold")

    fig.suptitle("图 6 · 一层注意力的完整形状链", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(OUT / "fig6_layer_shapes.png")
    plt.close(fig)


if __name__ == "__main__":
    print("CJK font:", setup())
    fig1_two_kinds()
    fig2_three_readings()
    fig3_shape_rule()
    fig4_row_major()
    fig5_batch_ai()
    fig6_layer_shapes()
    for p in sorted(OUT.glob("*.png")):
        print("  ->", p.relative_to(ROOT))
