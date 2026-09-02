"""Day 03 实验 · 屋顶那两个数字是怎么来的

四件事：
  A  用架构参数手推峰值算力/带宽，和 Day 02 实测对账
  B  Tile 量化：矩阵边长差 1，性能可能差 30% —— 因为硬件按固定块干活
  C  访存粒度：跨步访问暴露 cache line 大小
  D  低精度的收益：fp32 vs fp16 vs bf16

用法：
    uv run python labs/day03/gpu_anatomy.py
    uv run python labs/day03/gpu_anatomy.py --exp B
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from probe import pick_device, sync, timeit  # noqa: E402

console = Console()


# ------------------------------------------------------------------ 架构参数表
@dataclass
class Arch:
    name: str
    cores: int              # SM 数 / GPU core 数
    alu_per_core: int       # 每个核里的 FP32 ALU 数
    clock_ghz: float        # 加速频率
    bus_bits: int           # 显存总线宽度
    mem_gbps: float         # 每根数据线每秒传输的 Gb
    tc_factor: float        # Tensor Core 相对 FP32 的倍数（无则 1）
    warp: int               # 一个 warp / SIMD-group 多少线程

    @property
    def peak_fp32(self) -> float:
        # 每个 ALU 每周期做 1 次 FMA = 2 次浮点运算
        return self.cores * self.alu_per_core * 2 * self.clock_ghz * 1e9

    @property
    def peak_fp16(self) -> float:
        return self.peak_fp32 * self.tc_factor

    @property
    def bandwidth(self) -> float:
        return self.bus_bits / 8 * self.mem_gbps * 1e9


ARCHS = {
    "mps": Arch("Apple M4 (10-core GPU)", cores=10, alu_per_core=128,
                clock_ghz=1.40, bus_bits=128, mem_gbps=7.5,
                tc_factor=1.0, warp=32),
    "cuda": Arch("RTX 4050 Laptop (AD107)", cores=20, alu_per_core=128,
                 clock_ghz=2.37, bus_bits=96, mem_gbps=16.0,
                 tc_factor=2.0, warp=32),
}


def exp_a(dev: torch.device, dtype: torch.dtype) -> None:
    """从架构参数推峰值，再和实测对账。"""
    console.rule("[bold]A · 屋顶的两个数字，是从架构参数乘出来的")

    arch = ARCHS.get(dev.type)
    if arch is None:
        console.print("[yellow]CPU 模式跳过架构核算[/yellow]\n")
        return

    console.print(f"[bold]{arch.name}[/bold]\n")
    console.print("[bold cyan]峰值算力 = 核数 × 每核ALU数 × 2(FMA) × 频率[/bold cyan]")
    console.print(f"  = {arch.cores} × {arch.alu_per_core} × 2 × {arch.clock_ghz} GHz"
                  f"  =  [bold]{arch.peak_fp32/1e12:.2f} TFLOPS[/bold] (FP32)")
    if arch.tc_factor > 1:
        console.print(f"  × {arch.tc_factor:.0f} (Tensor Core, FP16)"
                      f"          =  [bold]{arch.peak_fp16/1e12:.2f} TFLOPS[/bold]")

    console.print("\n[bold cyan]峰值带宽 = 总线宽度 ÷ 8 × 每线速率[/bold cyan]")
    console.print(f"  = {arch.bus_bits} bit ÷ 8 × {arch.mem_gbps} Gbps"
                  f"  =  [bold]{arch.bandwidth/1e9:.0f} GB/s[/bold]\n")

    # 实测
    esz = torch.tensor([], dtype=dtype).element_size()
    k = 4096
    a = torch.randn(k, k, device=dev, dtype=dtype)
    b = torch.randn(k, k, device=dev, dtype=dtype)
    meas_f = 2.0 * k ** 3 / timeit(lambda: a @ b, dev)
    m = 2 ** 26
    u = torch.randn(m, device=dev, dtype=dtype)
    v = torch.randn(m, device=dev, dtype=dtype)
    meas_b = 3 * m * esz / timeit(lambda: u + v, dev)

    theo_f = arch.peak_fp16 if dtype == torch.float16 else arch.peak_fp32
    tbl = Table("指标", "架构参数算出的理论值", "实测", "达成率", title="对账")
    tbl.add_row("峰值算力", f"{theo_f/1e12:.2f} TFLOPS",
                f"{meas_f/1e12:.2f} TFLOPS", f"{meas_f/theo_f*100:.0f}%")
    tbl.add_row("峰值带宽", f"{arch.bandwidth/1e9:.0f} GB/s",
                f"{meas_b/1e9:.0f} GB/s", f"{meas_b/arch.bandwidth*100:.0f}%")
    console.print(tbl)
    console.print(
        "\n  [dim]算力达成率通常 80~95%（降频、指令开销）；"
        "带宽达成率通常 70~85%（刷新、写分配、总线协议开销）。[/dim]\n"
    )


def exp_b(dev: torch.device, dtype: torch.dtype) -> None:
    """Tile 量化：边长不是硬件块大小的整数倍，就有一整排单元在空转。"""
    console.rule("[bold]B · 矩阵边长差 1，性能可能差 30%")

    sizes = [1024, 1025, 1088, 1152, 1279, 1280, 1281, 1536, 2048, 2049]
    tbl = Table("边长 N", "N mod 64", "N mod 128", "耗时", "实测 TFLOPS", "相对最好",
                title="C = A @ B，A/B/C 都是 N×N")
    res = []
    for n in sizes:
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        t = timeit(lambda: a @ b, dev, warmup=5, iters=20)
        res.append((n, 2.0 * n ** 3 / t / 1e12, t))
        del a, b
    best = max(r[1] for r in res)
    for n, tf, t in res:
        mark = "[green]✔[/green]" if n % 128 == 0 else ("[yellow]~[/yellow]" if n % 64 == 0 else "[red]✘[/red]")
        tbl.add_row(str(n), f"{n%64}", f"{n%128} {mark}", f"{t*1e3:.2f} ms",
                    f"{tf:.2f}", f"{tf/best*100:.0f}%")
    console.print(tbl)
    console.print(
        "\n[bold yellow]怎么读：[/bold yellow]\n"
        "  · GPU 不是一个数一个数地算，而是按 [bold]固定大小的块[/bold]（典型 128×128 或 64×64）搬运和计算。\n"
        "  · 边长不是块大小的整数倍时，最后一块只用了一部分 —— [bold]剩下的硬件在空转[/bold]。\n"
        "  · 这就是为什么模型的 hidden_size 几乎总是 128 的倍数（4096、5120、8192…）。\n"
    )


def exp_c(dev: torch.device, dtype: torch.dtype) -> None:
    """访存粒度：显存最小搬运单位是一整条 cache line，不是一个数。"""
    console.rule("[bold]C · 只要 1 个数，硬件也得搬一整条 cache line")

    esz = torch.tensor([], dtype=dtype).element_size()
    rows = 2 ** 21
    tbl = Table("跨步 stride", "跨步字节", "用到的数据", "耗时", "有效带宽\n(按有用数据)",
                "相对 stride=1", title=f"x[:, 0].sum()，x 形状 ({rows}, stride)")
    base = None
    for stride in (1, 2, 4, 8, 16, 32, 64):
        x = torch.randn(rows, stride, device=dev, dtype=dtype)
        col = x[:, 0]
        t = timeit(lambda: col.sum(), dev, warmup=10, iters=30)
        useful = rows * esz
        bw = useful / t / 1e9
        base = base or bw
        tbl.add_row(str(stride), f"{stride*esz} B", f"{useful/1e6:.0f} MB",
                    f"{t*1e3:.2f} ms", f"{bw:.1f} GB/s", f"{bw/base*100:.0f}%")
        del x, col
    console.print(tbl)
    console.print(
        "\n[bold yellow]怎么读：[/bold yellow]\n"
        "  · 有效带宽[bold]一路下跌，然后在某个 stride 之后不再跌[/bold]。\n"
        "  · 那个拐点处的「跨步字节」≈ [bold cyan]cache line 大小[/bold cyan]"
        "（NVIDIA 通常 32/128 B，Apple 通常 64/128 B）。\n"
        "  · 一旦跨步超过一条 line，每读 1 个数就浪费一整条 line —— 这叫[bold]访存不合并[/bold]。\n"
        "  · [dim]所以 KV Cache 的内存布局（Day 31 PagedAttention）才这么讲究。[/dim]\n"
    )


def exp_d(dev: torch.device) -> None:
    """低精度不只是省显存，更是换了一套更快的计算单元。"""
    console.rule("[bold]D · 换精度 = 抬高屋顶（Day 02 的第③个方向）")

    k = 4096
    tbl = Table("精度", "字节/元素", "耗时", "实测 TFLOPS", "相对 FP32",
                title=f"({k},{k}) @ ({k},{k})")
    base = None
    for name, dt in (("FP32", torch.float32), ("FP16", torch.float16),
                     ("BF16", torch.bfloat16)):
        try:
            a = torch.randn(k, k, device=dev, dtype=dt)
            b = torch.randn(k, k, device=dev, dtype=dt)
            t = timeit(lambda: a @ b, dev, warmup=5, iters=15)
            tf = 2.0 * k ** 3 / t / 1e12
            base = base or tf
            tbl.add_row(name, str(a.element_size()), f"{t*1e3:.2f} ms",
                        f"{tf:.2f}", f"{tf/base:.2f}×")
            del a, b
        except Exception as e:                     # 某些后端不支持某精度
            tbl.add_row(name, "-", "不支持", "-", f"[dim]{type(e).__name__}[/dim]")
    console.print(tbl)
    console.print(
        "\n[bold yellow]怎么读：[/bold yellow]\n"
        "  · NVIDIA 上 FP16 通常是 FP32 的 [bold]2~8 倍[/bold] —— 那不是"
        "「少搬一半数据」能解释的，是 [bold cyan]Tensor Core[/bold cyan] 在干活。\n"
        "  · Tensor Core 一条指令算完一整个小矩阵块，而普通 ALU 一次只做一个乘加。\n"
        "  · 但它[bold]只吃矩阵乘[/bold] —— 逐元素算子（LayerNorm/GELU）完全用不上，"
        "所以它们的屋顶从来没被抬高过。\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "D"], help="只跑某个实验")
    args = ap.parse_args()

    dev, name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {name}  |  {dtype}\n")

    if args.exp in (None, "A"):
        exp_a(dev, dtype)
    if args.exp in (None, "B"):
        exp_b(dev, dtype)
    if args.exp in (None, "C"):
        exp_c(dev, dtype)
    if args.exp in (None, "D"):
        exp_d(dev)


if __name__ == "__main__":
    main()
