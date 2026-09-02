"""Day 02 实验 · 画出你自己机器的 Roofline

做三件事：
  1. 实测峰值算力与带宽 → 确定「屋顶」的形状
  2. 测一批真实算子的算术强度与实际性能 → 把它们打到屋顶图上
  3. 输出一张 results/roofline_<设备>.png

用法：
    uv run python labs/day02/roofline.py
    uv run python labs/day02/roofline.py --n 4096   # 改矩阵规模
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "labs" / "day01"))

import matplotlib.pyplot as plt          # noqa: E402
from _style import setup, C              # noqa: E402
from probe import pick_device, timeit    # noqa: E402

console = Console()


@dataclass
class Op:
    name: str
    fn: Callable[[], object]
    flops: float          # 理论浮点运算次数
    bytes: float          # 理论必须搬运的字节数
    note: str = ""

    @property
    def ai(self) -> float:
        return self.flops / self.bytes


def build_ops(dev: torch.device, dtype: torch.dtype, n: int) -> list[Op]:
    """构造一批算子，每个都精确标注它的 FLOPs 与访存量。"""
    esz = torch.tensor([], dtype=dtype).element_size()
    m = 2 ** 24                                   # 逐元素算子的向量长度

    x = torch.randn(m, device=dev, dtype=dtype)
    y = torch.randn(m, device=dev, dtype=dtype)
    w = torch.randn(n, n, device=dev, dtype=dtype)
    rows = torch.randn(4096, n, device=dev, dtype=dtype)

    ops: list[Op] = [
        Op("x + y", lambda: x + y,
           flops=m, bytes=3 * m * esz, note="逐元素：读2写1"),
        Op("relu(x)", lambda: torch.relu(x),
           flops=m, bytes=2 * m * esz, note="逐元素：读1写1"),
        Op("x.sum()", lambda: x.sum(),
           flops=m, bytes=1 * m * esz, note="归约：只读不写"),
        Op("softmax(行长4096)", lambda: torch.softmax(rows, dim=-1),
           flops=5 * rows.numel(), bytes=2 * rows.numel() * esz,
           note="max/exp/sum/div"),
    ]

    # GEMM 家族：同一个权重 W(n,n)，只改 batch 行数 → 算术强度 ≈ batch
    for b in (1, 8, 64, 512, 4096):
        a = torch.randn(b, n, device=dev, dtype=dtype)
        ops.append(Op(
            ("GEMV B=1" if b == 1 else f"GEMM B={b}"),
            (lambda a=a: a @ w),
            flops=2.0 * b * n * n,
            bytes=(n * n + 2 * b * n) * esz,
            note=f"({b},{n}) @ ({n},{n})",
        ))
    return ops


def measure_peaks(dev: torch.device, dtype: torch.dtype) -> tuple[float, float]:
    """峰值算力(FLOP/s)与峰值带宽(Byte/s)——屋顶的两条边。"""
    esz = torch.tensor([], dtype=dtype).element_size()
    best_f = 0.0
    for k in (2048, 4096):
        a = torch.randn(k, k, device=dev, dtype=dtype)
        b = torch.randn(k, k, device=dev, dtype=dtype)
        best_f = max(best_f, 2.0 * k ** 3 / timeit(lambda: a @ b, dev))
        del a, b
    best_b = 0.0
    for m in (2 ** 24, 2 ** 26):
        u = torch.randn(m, device=dev, dtype=dtype)
        v = torch.randn(m, device=dev, dtype=dtype)
        best_b = max(best_b, 3 * m * esz / timeit(lambda: u + v, dev))
        del u, v
    return best_f, best_b


def latency_sweep(dev: torch.device, dtype: torch.dtype) -> list[tuple[int, float]]:
    """带宽随数据量的变化：小 kernel 喂不饱内存系统（Little's Law）。"""
    esz = torch.tensor([], dtype=dtype).element_size()
    out = []
    for m in (2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26):
        u = torch.randn(m, device=dev, dtype=dtype)
        v = torch.randn(m, device=dev, dtype=dtype)
        t = timeit(lambda: u + v, dev, warmup=20, iters=100)
        out.append((3 * m * esz, 3 * m * esz / t))
        del u, v
    return out


def plot(peak_f: float, peak_b: float, ops: list[Op], perf: list[float],
         sweep: list[tuple[int, float]], dev_name: str, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    ridge = peak_f / peak_b
    ai_grid = np.logspace(-2, 4, 400)
    roof = np.minimum(peak_f, ai_grid * peak_b) / 1e12

    ax.plot(ai_grid, roof, lw=3, color=C["neutral"], zorder=3, label="Roofline 上限")
    ax.fill_between(ai_grid, 1e-4, roof, color=C["grid"], alpha=0.35, zorder=0)
    ax.axvline(ridge, color=C["accent"], ls="--", lw=1.8, zorder=2)
    ax.text(ridge * 1.1, peak_f / 1e12 * 0.045,
            f"平衡点\n{ridge:.0f} FLOP/Byte", color=C["accent"],
            fontsize=9, fontweight="bold")

    ax.text(0.03, peak_f / 1e12 * 0.30, "带宽受限区\n（斜坡：性能 = AI × 带宽）",
            fontsize=9.5, color=C["memory"], fontweight="bold")
    ax.text(ridge * 1.4, peak_f / 1e12 * 1.85, "算力受限区（平顶）",
            fontsize=9.5, color=C["compute"], fontweight="bold")

    for i, (op, p) in enumerate(zip(ops, perf)):
        col = C["memory"] if op.ai < ridge else C["compute"]
        ax.scatter(op.ai, p / 1e12, s=70, color=col, zorder=5,
                   edgecolors="black", linewidths=0.8)
        # 左右交错，避免相邻点的标签被对方的圆点盖住
        right = i % 2 == 0
        ax.annotate(op.name, (op.ai, p / 1e12), textcoords="offset points",
                    xytext=(11, 9) if right else (-11, -16),
                    ha="left" if right else "right",
                    fontsize=8.5, color=col, fontweight="bold")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.02, 5000)
    ax.set_ylim(peak_f / 1e12 * 5e-4, peak_f / 1e12 * 3)
    ax.set_xlabel("算术强度 AI (FLOP / Byte)")
    ax.set_ylabel("实测性能 (TFLOPS)")
    ax.set_title(f"{dev_name}\n峰值 {peak_f/1e12:.2f} TFLOPS · {peak_b/1e9:.0f} GB/s",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    # 右图：带宽随数据量爬升，三个区间讲三件不同的事
    bx = np.array([s[0] for s in sweep]) / 1024
    by = np.array([s[1] for s in sweep]) / 1e9
    peak_idx = int(by.argmax())
    ax2.plot(bx, by, "o-", lw=2.2, ms=6, color=C["memory"], zorder=3)
    ax2.axhline(peak_b / 1e9, color=C["neutral"], ls="--", lw=1.8)
    ax2.text(bx[-1], peak_b / 1e9 * 0.90,
             f"DRAM 带宽 {peak_b/1e9:.0f} GB/s", fontsize=9, ha="right",
             color=C["neutral"], fontweight="bold",
             bbox=dict(fc="white", ec="none", pad=1.2))

    ax2.scatter([bx[peak_idx]], [by[peak_idx]], s=140, facecolor="none",
                edgecolors=C["capacity"], linewidths=2.5, zorder=6)
    ax2.annotate(f"② 刚好装进缓存\n   {by[peak_idx]:.0f} GB/s",
                 xy=(bx[peak_idx], by[peak_idx]),
                 xytext=(bx[peak_idx] * 0.05, by[peak_idx] * 0.97),
                 fontsize=9, color=C["capacity"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["capacity"], lw=1.5))
    ax2.annotate("① 太小：kernel 启动开销\n   + 并行度不足",
                 xy=(bx[0], by[0]), xytext=(bx[0] * 1.4, by.max() * 0.42),
                 fontsize=9, color=C["compute"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["compute"], lw=1.5))
    ax2.text(bx[-1], by.max() * 0.12, "③ 足够大：落到真实 DRAM 带宽",
             fontsize=9, ha="right", color=C["memory"], fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_ylim(0, by.max() * 1.15)
    ax2.set_xlabel("单次 kernel 搬运量 (KB，对数轴)")
    ax2.set_ylabel("实测带宽 (GB/s)")
    ax2.set_title("为什么探测带宽必须用大数组\n（同一个 x+y，只改规模）",
                  fontsize=11, fontweight="bold")

    fig.suptitle("你这台机器的 Roofline", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="GEMM 方阵边长")
    args = ap.parse_args()

    setup()
    dev, dev_name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {dev_name}  |  {dtype}\n")

    console.rule("[bold]① 测出屋顶")
    peak_f, peak_b = measure_peaks(dev, dtype)
    ridge = peak_f / peak_b
    console.print(f"  峰值算力  [bold]{peak_f/1e12:.2f} TFLOPS[/bold]")
    console.print(f"  峰值带宽  [bold]{peak_b/1e9:.1f} GB/s[/bold]")
    console.print(f"  平衡点    [bold cyan]{ridge:.0f} FLOP/Byte[/bold cyan]\n")

    console.rule("[bold]② 把真实算子打到屋顶图上")
    ops = build_ops(dev, dtype, args.n)
    perf = [op.flops / timeit(op.fn, dev, warmup=10, iters=30) for op in ops]

    # 屋顶必须罩住所有观测点：任何算子跑得比探针快，说明探针低估了峰值
    peak_f = max(peak_f, *perf)
    ridge = peak_f / peak_b

    tbl = Table("算子", "形状 / 说明", "AI\n(FLOP/Byte)", "屋顶允许", "实测",
                "达成率", "撞哪堵墙")
    for op, p in zip(ops, perf):
        ceil = min(peak_f, op.ai * peak_b)
        wall = "带宽" if op.ai < ridge else "算力"
        col = "cyan" if wall == "带宽" else "red"
        tbl.add_row(op.name, op.note, f"{op.ai:.2f}",
                    f"{ceil/1e12:.3f} T", f"{p/1e12:.3f} T",
                    f"{p/ceil*100:.0f}%", f"[{col}]{wall}[/{col}]")
    console.print(tbl)
    console.print(f"  [dim]修正后的峰值算力 {peak_f/1e12:.2f} TFLOPS，"
                  f"平衡点 {ridge:.0f} FLOP/Byte[/dim]")

    console.rule("[bold]③ 带宽随数据量的爬升")
    sweep = latency_sweep(dev, dtype)

    out = ROOT / "results" / f"roofline_{dev.type}.png"
    out.parent.mkdir(exist_ok=True)
    plot(peak_f, peak_b, ops, perf, sweep, dev_name, out)
    console.print(f"\n[bold green]图已保存[/bold green] → {out.relative_to(ROOT)}\n")

    console.print(
        "[bold yellow]怎么读这张表：[/bold yellow]\n"
        "  · [bold]AI < 平衡点[/bold] 的算子，无论你怎么优化计算，都不可能超过「屋顶允许」那一列。\n"
        "  · [bold]达成率[/bold] 接近 100% = 这个算子已经写到极限，瓶颈是物理不是代码。\n"
        "  · 达成率很低 = 还有优化空间（或者撞上了第三堵墙：并行度不足）。\n"
    )


if __name__ == "__main__":
    main()
