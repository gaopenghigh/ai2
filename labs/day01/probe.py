"""Day 01 实验 · 亲手撞一次"内存墙"

三个实验：
  exp1  实测你这台机器的峰值算力 (TFLOPS) 与显存带宽 (GB/s) → 算出"机器平衡点"
  exp2  同一个权重矩阵，batch 从 1 涨到 512：FLOPs 涨 512×，耗时几乎不变（!）
  exp3  用实测数据预测 7B 模型的 decode 速度上限，并与真实框架对比

跨平台：Apple Silicon (mps) / NVIDIA (cuda) / CPU 自动适配。

用法：
    uv run python labs/day01/probe.py            # 全跑
    uv run python labs/day01/probe.py --exp 2    # 只跑实验 2
"""

from __future__ import annotations

import argparse
import time

import torch
from rich.console import Console
from rich.table import Table

console = Console()


# ----------------------------------------------------------------- 设备与计时
def pick_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    return torch.device("cpu"), "CPU"


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def timeit(fn, dev: torch.device, warmup: int = 5, iters: int = 20) -> float:
    """返回单次调用的平均耗时（秒）。"""
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------- 实验 1
def exp1_peak(dev: torch.device, dtype: torch.dtype) -> tuple[float, float]:
    """测峰值算力与带宽。

    算力：大方阵 GEMM。FLOPs = 2*N^3，算术强度 ~N/3，足够高 → 撞算力墙。
    带宽：超大向量的逐元素加法。每元素读 2 次写 1 次 → 算术强度 ~1/12，必撞带宽墙。
    """
    console.rule("[bold]实验 1 · 实测峰值算力与带宽")

    # --- 峰值算力 ---
    best_tflops = 0.0
    tbl = Table("矩阵规模 N", "FLOPs", "耗时", "实测 TFLOPS", title="GEMM 扫描（算力探测）")
    for n in (512, 1024, 2048, 4096):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        t = timeit(lambda: a @ b, dev)
        flops = 2.0 * n ** 3
        tflops = flops / t / 1e12
        best_tflops = max(best_tflops, tflops)
        tbl.add_row(str(n), f"{flops/1e9:.1f} G", f"{t*1e3:.3f} ms", f"{tflops:.2f}")
        del a, b
    console.print(tbl)

    # --- 峰值带宽 ---
    best_bw = 0.0
    tbl = Table("向量元素数", "搬运字节", "耗时", "实测 GB/s", title="逐元素加法扫描（带宽探测）")
    for m in (2 ** 22, 2 ** 24, 2 ** 26):
        x = torch.randn(m, device=dev, dtype=dtype)
        y = torch.randn(m, device=dev, dtype=dtype)
        t = timeit(lambda: x + y, dev)
        moved = 3 * m * x.element_size()          # 读 x、读 y、写 out
        bw = moved / t / 1e9
        best_bw = max(best_bw, bw)
        tbl.add_row(f"{m:,}", f"{moved/1e6:.0f} MB", f"{t*1e3:.3f} ms", f"{bw:.1f}")
        del x, y
    console.print(tbl)

    ridge = best_tflops * 1e12 / (best_bw * 1e9)
    console.print(
        f"\n[bold]机器平衡点（ridge point）= 峰值算力 / 峰值带宽 = "
        f"[cyan]{ridge:.0f} FLOP/Byte[/cyan][/bold]"
    )
    console.print(
        "  → 一个算子的[bold]算术强度[/bold]低于这个数，它就是[bold cyan]带宽受限[/bold cyan]的；"
        "高于这个数，才是[bold red]算力受限[/bold red]的。\n"
    )
    return best_tflops, best_bw


# --------------------------------------------------------------------- 实验 2
def exp2_batch_is_free(dev: torch.device, dtype: torch.dtype) -> None:
    """核心洞察：decode 阶段，加大 batch 几乎是"免费"的。

    固定权重 W: (4096, 4096)，输入 X: (B, 4096)。
    B 从 1 涨到 512，FLOPs 涨 512 倍，但只要还没撞到算力墙，
    耗时几乎不变 —— 因为时间全花在"把 W 从显存搬到片上"。
    """
    console.rule("[bold]实验 2 · batch 增大 512 倍，耗时却几乎不变")

    d = 4096
    w = torch.randn(d, d, device=dev, dtype=dtype)
    w_bytes = w.numel() * w.element_size()

    tbl = Table("batch B", "FLOPs", "耗时", "相对 B=1", "实测 TFLOPS",
                "算术强度\n(FLOP/Byte)", title=f"X(B,{d}) @ W({d},{d})，W 固定不变")
    base = None
    for bsz in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        x = torch.randn(bsz, d, device=dev, dtype=dtype)
        t = timeit(lambda: x @ w, dev, warmup=10, iters=50)
        base = base or t
        flops = 2.0 * bsz * d * d
        ai = flops / w_bytes                       # 权重搬运主导，忽略激活
        tbl.add_row(str(bsz), f"{flops/1e9:.2f} G", f"{t*1e6:.0f} µs",
                    f"{t/base:.2f}×", f"{flops/t/1e12:.2f}", f"{ai:.1f}")
        del x
    console.print(tbl)
    console.print(
        "\n[bold yellow]读懂这张表：[/bold yellow]\n"
        "  · 「相对 B=1」这一列长时间贴近 1.0 → 多算 100 倍，几乎没多花时间。\n"
        "  · 这就是[bold]连续批处理 (continuous batching) 能把吞吐提升一个数量级[/bold]的物理根源。\n"
        "  · 直到算术强度越过机器平衡点，耗时才开始随 B 线性增长（真正撞上算力墙）。\n"
    )


# --------------------------------------------------------------------- 实验 3
def exp3_predict_7b(bw_gbs: float) -> None:
    """用实测带宽预测 7B 模型 decode 速度上限。"""
    console.rule("[bold]实验 3 · 预测 7B 模型的 decode 速度上限")

    params = 7e9
    tbl = Table("权重精度", "字节/参数", "权重总量", "每 token 最短耗时",
                "理论上限 (token/s)", title=f"基于实测带宽 {bw_gbs:.0f} GB/s")
    for name, bpp in (("FP16 / BF16", 2.0), ("INT8", 1.0), ("INT4", 0.5)):
        gb = params * bpp / 1e9
        ms = gb / bw_gbs * 1000
        tbl.add_row(name, f"{bpp} B", f"{gb:.1f} GB", f"{ms:.1f} ms", f"{1000/ms:.1f}")
    console.print(tbl)
    console.print(
        "\n[bold yellow]为什么这个上限如此之硬：[/bold yellow]\n"
        "  生成 1 个 token，必须把[bold]每一个权重都从显存读一遍[/bold]（batch=1 时每个权重只用 1 次）。\n"
        "  所以：t_min = 权重字节数 / 显存带宽。这是物理下界，任何优化都绕不过 ——\n"
        "  除非你 ① 减少字节数（量化）② 增大 batch（摊薄）③ 一次多出几个 token（投机解码）。\n"
        "  [dim]这三条路，正是 Week 3/5/8 的全部内容。[/dim]\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, choices=[1, 2, 3], help="只跑指定实验")
    args = ap.parse_args()

    dev, name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {name}  |  "
                  f"[bold green]dtype[/bold green]: {dtype}  |  "
                  f"[bold green]torch[/bold green]: {torch.__version__}\n")

    tflops, bw = 0.0, 0.0
    if args.exp in (None, 1):
        tflops, bw = exp1_peak(dev, dtype)
    if args.exp in (None, 2):
        exp2_batch_is_free(dev, dtype)
    if args.exp in (None, 3):
        if bw == 0.0:
            _, bw = exp1_peak(dev, dtype)
        exp3_predict_7b(bw)


if __name__ == "__main__":
    main()
