"""Day 01 配图生成（编号 = 正文中的阅读顺序）：
  fig1_gemm_vs_gemv.png          §3.2  Prefill/Decode 的形状差异
  fig2_causal_mask.png           §3.4  因果掩码与只留最后一行
  fig3_memory_wall.png           §5    算力 vs 显存带宽 的剪刀差
  fig4_arithmetic_intensity.png  §6    Prefill / Decode 的算术强度定位
  fig5_token_budget.png          §7    两台机器的能力边界

运行：uv run python scripts/day01_figs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, C

OUT = Path(__file__).resolve().parents[1] / "assets" / "day01"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 硬件常数
# 本课程两台机器（数值为厂商标称峰值，Day 07 会实测对比）
M4 = dict(name="Apple M4 (Mac mini)", tflops_fp16=4.3, bw=120.0, vram=16.0)
RTX4050 = dict(name="RTX 4050 Laptop", tflops_fp16=24.0, bw=192.0, vram=6.0)


def fig3_memory_wall() -> None:
    """NVIDIA 数据中心 GPU 七年演进：算力涨得比带宽快得多。"""
    gpus = ["P100\n2016", "V100\n2017", "A100\n2020", "H100\n2022", "B200\n2024"]
    tflops = np.array([21.2, 125.0, 312.0, 989.0, 2250.0])   # 稠密 FP16(Tensor Core)
    bw = np.array([732.0, 900.0, 1555.0, 3350.0, 8000.0])    # GB/s

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(gpus))
    ax1.plot(x, tflops / tflops[0], "o-", lw=2.5, ms=8,
             color=C["compute"], label="算力 FP16 (相对 P100)")
    ax1.plot(x, bw / bw[0], "s-", lw=2.5, ms=8,
             color=C["memory"], label="显存带宽 (相对 P100)")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(gpus)
    ax1.set_ylabel("相对 P100 的倍数（对数轴）")
    ax1.set_title("算力涨了 106×，带宽只涨了 11×")
    ax1.legend(loc="upper left")
    last = len(gpus) - 1
    ax1.annotate(f"{tflops[last]/tflops[0]:.0f}×", (x[last], tflops[last] / tflops[0]),
                 textcoords="offset points", xytext=(-4, 12),
                 ha="right", color=C["compute"], fontweight="bold")
    ax1.annotate(f"{bw[last]/bw[0]:.0f}×", (x[last], bw[last] / bw[0]),
                 textcoords="offset points", xytext=(-4, 12),
                 ha="right", color=C["memory"], fontweight="bold")

    ratio = tflops * 1e12 / (bw * 1e9)   # ops per byte
    bars = ax2.bar(x, ratio, color=C["accent"], width=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(gpus)
    ax2.set_ylabel("机器平衡点 (FLOP / Byte)")
    ax2.set_title("想喂饱 GPU，每读 1 字节要算多少次？")
    for b, v in zip(bars, ratio):
        ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}",
                 ha="center", fontweight="bold", color=C["accent"])
    ax2.set_ylim(0, max(ratio) * 1.45)
    ax2.axhline(1, color=C["compute"], ls="--", lw=1.5)
    ax2.annotate("自回归 decode 的算术强度 ≈ 1\n（差了两个数量级）",
                 xy=(0.15, 6), xytext=(1.55, 340),
                 color=C["compute"], fontsize=9, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["compute"], lw=1.4,
                                 connectionstyle="arc3,rad=0.15"))

    fig.suptitle("图 3  内存墙：这就是大模型推理慢的物理根源", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_memory_wall.png")
    plt.close(fig)


def fig5_token_budget() -> None:
    """7B 模型：生成 1 个 token 的理论最短耗时 = 权重字节数 / 显存带宽。"""
    params = 7e9
    schemes = [("FP16\n(2 B/参数)", 2.0), ("INT8\n(1 B/参数)", 1.0),
               ("INT4\n(0.5 B/参数)", 0.5)]
    machines = [M4, RTX4050]
    colors = [C["memory"], C["compute"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    width = 0.35
    x = np.arange(len(schemes))

    for i, (m, col) in enumerate(zip(machines, colors)):
        toks = []
        for _, bpp in schemes:
            gb = params * bpp / 1e9
            ms = gb / m["bw"] * 1000
            toks.append(1000 / ms)
        bars = ax1.bar(x + (i - 0.5) * width, toks, width, label=m["name"], color=col)
        for b, (label, bpp), t in zip(bars, schemes, toks):
            gb = params * bpp / 1e9
            fits = gb < m["vram"] * 0.85
            ax1.text(b.get_x() + b.get_width() / 2, t + 1.5,
                     f"{t:.0f}" + ("" if fits else " OOM"),
                     ha="center", fontsize=9, fontweight="bold",
                     color=col if fits else C["neutral"])

    ax1.set_xticks(x); ax1.set_xticklabels([s[0] for s in schemes])
    ax1.set_ylabel("理论上限 (token/s)")
    ax1.set_title("7B 模型 · batch=1 解码速度上限\n(OOM = 装不进显存)")
    ax1.legend()
    ax1.set_ylim(0, 65)

    # 右图：显存容量 vs 模型大小
    sizes = {"7B FP16": 14.0, "7B INT8": 7.0, "7B INT4": 3.5,
             "13B INT4": 6.5, "3B FP16": 6.0}
    names = list(sizes.keys())
    vals = [sizes[n] for n in names]
    y = np.arange(len(names))
    ax2.barh(y, vals, color=C["capacity"], height=0.55)
    ax2.set_yticks(y); ax2.set_yticklabels(names)
    ax2.set_ylim(len(names) - 0.4, -1.15)
    ax2.set_xlabel("仅权重占用 (GB)")
    ax2.axvline(RTX4050["vram"], color=C["compute"], lw=2, ls="--")
    ax2.text(RTX4050["vram"], -0.75, "RTX 4050\n6 GB", ha="center", va="center",
             color=C["compute"], fontsize=9, fontweight="bold",
             bbox=dict(fc="white", ec="none", pad=1.5))
    ax2.axvline(11.0, color=C["memory"], lw=2, ls="--")
    ax2.text(11.0, -0.75, "M4 可用\n≈ 11 GB", ha="center", va="center",
             color=C["memory"], fontsize=9, fontweight="bold",
             bbox=dict(fc="white", ec="none", pad=1.5))
    ax2.set_xlim(0, 16)
    ax2.set_title("容量墙：能不能跑，先看装不装得下")

    fig.suptitle("图 5  两台机器的能力边界（权重搬运是 decode 的唯一瓶颈）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_token_budget.png")
    plt.close(fig)


def fig4_arithmetic_intensity() -> None:
    """算术强度 = FLOPs / Bytes，决定你撞哪堵墙。"""
    cases = [
        ("Decode\nbatch=1", 1),
        ("Decode\nbatch=8", 8),
        ("Decode\nbatch=32", 32),
        ("Decode\nbatch=128", 128),
        ("Prefill\n2000 tok", 2000),
    ]
    labels = [c[0] for c in cases]
    ai = np.array([c[1] for c in cases], dtype=float)  # FP16 权重下 AI ≈ 有效 token 数

    ridge_m4 = M4["tflops_fp16"] * 1e12 / (M4["bw"] * 1e9)
    ridge_4050 = RTX4050["tflops_fp16"] * 1e12 / (RTX4050["bw"] * 1e9)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(cases))
    cols = [C["memory"] if v < ridge_4050 else C["compute"] for v in ai]
    bars = ax.bar(x, ai, color=cols, width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("算术强度 (FLOP / Byte，对数轴)")
    ax.set_ylim(0.5, 6000)
    ax.set_xlim(-1.15, len(cases) - 0.5)

    for b, v in zip(bars, ai):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f}",
                ha="center", fontweight="bold", fontsize=9)

    ax.axhline(ridge_m4, color=C["accent"], ls="--", lw=2)
    ax.text(-1.08, ridge_m4 * 1.12, f"M4 平衡点 ≈ {ridge_m4:.0f}",
            color=C["accent"], fontsize=9, fontweight="bold")
    ax.axhline(ridge_4050, color=C["ok"], ls="--", lw=2)
    ax.text(-1.08, ridge_4050 * 1.12, f"RTX 4050 平衡点 ≈ {ridge_4050:.0f}",
            color=C["ok"], fontsize=9, fontweight="bold")

    ax.text(-1.08, 900, "平衡点以上：算力受限（GPU 真在算）", fontsize=10,
            color=C["compute"], ha="left", fontweight="bold", zorder=10)
    ax.text(-1.08, 380, "平衡点以下：带宽受限（GPU 在等内存）", fontsize=10,
            color=C["memory"], ha="left", fontweight="bold", zorder=10)

    ax.set_title("图 4  同一个模型，Prefill 与 Decode 撞的是两堵完全不同的墙",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_arithmetic_intensity.png")
    plt.close(fig)


def fig1_gemm_vs_gemv() -> None:
    """Prefill 与 Decode 的形状差异：同一个 W，一个被用 2000 次，一个只用 1 次。"""
    from matplotlib.patches import Rectangle, FancyBboxPatch

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    def draw(ax, n_rows_label, x_h, title, tag, tag_color, cost_text):
        ax.set_xlim(0, 10); ax.set_ylim(1.05, 4.6)
        ax.axis("off")
        cy = 2.9  # 矩阵竖直中心

        # X
        ax.add_patch(Rectangle((0.4, cy - x_h / 2), 1.3, x_h,
                               fc=tag_color, ec="black", lw=1.5, alpha=0.85))
        ax.text(1.05, cy + x_h / 2 + 0.14, "输入 X", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(1.05, cy - x_h / 2 - 0.14, n_rows_label, ha="center", va="top",
                fontsize=9)

        ax.text(2.05, cy, "@", ha="center", va="center", fontsize=20, fontweight="bold")

        # W —— 两幅图里完全一样大，这是重点
        ax.add_patch(Rectangle((2.5, cy - 1.05), 2.1, 2.1,
                               fc=C["capacity"], ec="black", lw=2.5))
        ax.text(3.55, cy + 0.18, "权重 W", ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(3.55, cy - 0.32, "4096 × 4096\n(整个模型 3.5 GB)",
                ha="center", va="center", fontsize=9)
        ax.text(3.55, cy - 1.05 - 0.14, "← 每次前向都必须从显存完整读一遍 →",
                ha="center", va="top", fontsize=9, color=C["neutral"])

        ax.text(4.95, cy, "=", ha="center", va="center", fontsize=20, fontweight="bold")

        # Out
        ax.add_patch(Rectangle((5.4, cy - x_h / 2), 1.3, x_h,
                               fc=tag_color, ec="black", lw=1.5, alpha=0.5))
        ax.text(6.05, cy + x_h / 2 + 0.14, "输出", ha="center", va="bottom", fontsize=10)

        # 右侧成本框
        ax.add_patch(FancyBboxPatch((7.25, cy - 1.0), 2.5, 2.0,
                                    boxstyle="round,pad=0.08",
                                    fc="white", ec=tag_color, lw=2))
        ax.text(8.5, cy + 0.55, tag, ha="center", va="center",
                fontsize=12, fontweight="bold", color=tag_color)
        ax.text(8.5, cy - 0.25, cost_text, ha="center", va="center", fontsize=9.5)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=2)

    draw(axes[0], "2000 行\n(prompt 的 2000 个 token)", 2.3,
         "Prefill · 矩阵 × 矩阵 (GEMM)", "算力受限", C["compute"],
         "读 3.5 GB 权重\n算了 2000 个 token\n\n每个 token 摊到\n1.75 MB 的搬运")
    draw(axes[1], "1 行\n(刚生成的那 1 个 token)", 0.22,
         "Decode · 向量 × 矩阵 (GEMV)", "带宽受限", C["memory"],
         "读 3.5 GB 权重\n只算了 1 个 token\n\n每个 token 摊到\n3500 MB 的搬运")

    fig.suptitle("图 1  同一块权重，两种用法，成本相差 2000 倍",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_gemm_vs_gemv.png")
    plt.close(fig)


def fig2_causal_mask() -> None:
    """为什么 prompt 可以并行算：因果掩码 + 只保留最后一行的预测。"""
    toks = ["今", "天", "天", "气"]
    n = len(toks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 左：因果掩码 ---
    from matplotlib.colors import ListedColormap

    mask = np.tril(np.ones((n, n)))
    ax1.imshow(mask, cmap=ListedColormap(["#E4EAF0", C["memory"]]), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ok = j <= i
            ax1.text(j, i, "看得见" if ok else "挡住", ha="center", va="center",
                     fontsize=8.5, color="white" if ok else C["neutral"],
                     fontweight="bold" if ok else "normal")
    ax1.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=2.5)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(toks)
    ax1.set_yticks(range(n)); ax1.set_yticklabels([f"第{i}行\n「{t}」" for i, t in enumerate(toks)])
    ax1.set_xlabel("能注意到哪些 token")
    ax1.set_title("因果掩码：第 i 行只能看 ≤ i\n所以并行算 = 串行算，结果一样",
                  fontsize=11, fontweight="bold")
    ax1.grid(False)
    ax1.tick_params(length=0)

    # --- 右：输出的 4 行，只有最后一行有用 ---
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 4.4); ax2.axis("off")
    rows = [
        ("第0行", "看完「今」", "预测下一个", "「天」— 已知，丢弃", False),
        ("第1行", "看完「今天」", "预测下一个", "「天」— 已知，丢弃", False),
        ("第2行", "看完「今天天」", "预测下一个", "「气」— 已知，丢弃", False),
        ("第3行", "看完「今天天气」", "预测下一个", "「很」← 这就是首个输出！", True),
    ]
    for k, (name, seen, _, res, keep) in enumerate(rows):
        y = 3.55 - k * 0.85
        col = C["ok"] if keep else C["grid"]
        ax2.add_patch(plt.Rectangle((0.2, y - 0.32), 9.5, 0.66,
                                    fc=col, alpha=0.35 if keep else 0.5,
                                    ec=C["ok"] if keep else C["neutral"],
                                    lw=2 if keep else 0.8))
        ax2.text(0.5, y, name, fontsize=9.5, va="center", fontweight="bold")
        ax2.text(1.6, y, seen, fontsize=9.5, va="center")
        ax2.text(5.0, y, "→", fontsize=11, va="center")
        ax2.text(5.5, y, res, fontsize=9.5, va="center",
                 fontweight="bold" if keep else "normal",
                 color=C["ok"] if keep else C["neutral"])
    ax2.text(5.0, 0.28, "前 3 行的预测虽然丢弃，但它们算出的 K/V 全部存进了 KV Cache",
             ha="center", fontsize=9.5, color=C["memory"], fontweight="bold")
    ax2.set_title("Prefill 一次输出 4 行，只有最后一行是我们要的",
                  fontsize=11, fontweight="bold")

    fig.suptitle("图 2  Prefill 内部发生了什么（prompt =「今天天气」）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_causal_mask.png")
    plt.close(fig)


if __name__ == "__main__":
    font = setup()
    print(f"[style] CJK font = {font}")
    fig1_gemm_vs_gemv()
    fig2_causal_mask()
    fig3_memory_wall()
    fig4_arithmetic_intensity()
    fig5_token_budget()
    for p in sorted(OUT.glob("*.png")):
        print(f"[ok] {p.relative_to(OUT.parents[1])}  ({p.stat().st_size/1024:.0f} KB)")
