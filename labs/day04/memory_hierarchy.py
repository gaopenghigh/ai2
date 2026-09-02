"""Day 04 实验 · 把存储金字塔逐级测出来

四件事：
  A  工作集扫描：同一个 kernel，只改数据大小，带宽会走出三段台阶
     → 拟合 t = t0 + S/B，量出 ① kernel 启动开销 ② 缓存拐点 ③ 真正的 DRAM 带宽
  B  顺序 vs 随机：搬同样多的字节，随机访问慢多少？（带宽 vs 延迟的分水岭）
  C  跨越 PCIe / 统一内存：CPU 和 GPU 之间搬数据有多贵，pinned 内存值不值
  D  融合的价值：同一串逐元素运算，多读几遍显存 vs 只读一遍

用法：
    uv run python labs/day04/memory_hierarchy.py
    uv run python labs/day04/memory_hierarchy.py --exp A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from probe import pick_device, sync, timeit  # noqa: E402

console = Console()
RESULTS = ROOT / "results"


# --------------------------------------------------------------------- 实验 A
def exp_a_working_set(dev: torch.device, dtype: torch.dtype) -> None:
    """固定 kernel，只改工作集大小，看带宽怎么变。

    用 y.copy_(x)：每个元素读 1 次写 1 次，算术强度 = 0，纯粹考验访存。
    工作集 = x 和 y 加起来的字节数（这才是缓存需要装下的东西）。
    """
    console.rule("[bold]A · 工作集扫描：带宽不是一个数，是一条曲线")

    esz = torch.empty(0, dtype=dtype).element_size()
    rows: list[tuple[int, float]] = []          # (bytes, seconds)
    for kb in (4, 16, 64, 256, 1024, 2048, 4096, 8192,
               16384, 32768, 65536, 131072, 262144):
        total = kb * 1024
        n = total // (2 * esz)                  # x 和 y 各一半
        if n < 8:
            continue
        x = torch.randn(n, device=dev, dtype=dtype)
        y = torch.empty_like(x)
        t = timeit(lambda: y.copy_(x), dev, warmup=20, iters=100)
        rows.append((total, t))
        del x, y

    # ① 启动开销：最小的那几个尺寸，时间里几乎全是固定开销
    small = sorted(t for b, t in rows if b <= 256 * 1024)
    t0 = small[len(small) // 2]
    # ③ 稳态带宽：最大的两个尺寸之间的斜率（斜率天然消掉了 t0）
    (b1, t1), (b2, t2) = rows[-3], rows[-1]
    b_dram = (b2 - b1) / (t2 - t1) / 1e9

    tbl = Table("工作集", "单次耗时", "有效带宽\nS/t", "扣掉启动开销\nS/(t−t0)",
                "落在哪一段",
                title="y.copy_(x) —— 工作集 = (x + y) 的字节数")

    knee, peak_eff = 0, 0.0
    for total, t in rows:
        unit = f"{total/1024:.0f} KB" if total < 1024 ** 2 else f"{total/1024**2:.0f} MB"
        raw = total / t / 1e9
        if t < 1.8 * t0:                        # 减法不稳，不报
            eff_s, seg = "—", "① 启动开销主导"
        else:
            eff = total / (t - t0) / 1e9
            eff_s = f"{eff:7.0f}"
            if eff > b_dram * 1.15:
                seg = "② 命中片上缓存"
                knee, peak_eff = max(knee, total), max(peak_eff, eff)
            else:
                seg = "③ 落到 DRAM"
        tbl.add_row(unit, f"{t*1e6:8.1f} µs", f"{raw:7.1f}", eff_s, seg)
    console.print(tbl)

    console.print(
        f"\n[bold]从这条曲线上能直接读出三个硬件参数：[/bold]\n"
        f"  ① [cyan]kernel 启动开销 t0 ≈ {t0*1e6:.1f} µs[/cyan] "
        f"—— 最小的那几行耗时几乎不随数据量变，那就是它\n"
        f"  ② [yellow]片上缓存带宽最高冲到 {peak_eff:.0f} GB/s，"
        f"缓存拐点在 {knee/1024**2:.0f} ~ {2*knee/1024**2:.0f} MB 之间[/yellow]\n"
        f"  ③ [red]稳态带宽 ≈ {b_dram:.0f} GB/s[/red] "
        f"—— 工作集远大于缓存后，这才是 Roofline 屋顶的那个数\n"
    )
    console.print(
        "[dim]模型：t = t0 + S/B。第 4 列把 t0 扣掉，剩下的就是纯粹的搬运速度，\n"
        "于是「缓存段」和「DRAM 段」的台阶一下子就露出来了。\n"
        "拐点位置 ≈ 这一级缓存的容量 —— 厂商往往不公布，但你刚刚把它测出来了。[/dim]\n"
    )


# --------------------------------------------------------------------- 实验 B
def exp_b_random_access(dev: torch.device, dtype: torch.dtype) -> None:
    """固定取 100 万个数，只改「它们散落在多大的范围里」。"""
    console.rule("[bold]B · 局部性：取的数一样多，散得越开越慢")

    k = 1 << 18                                 # 每次都取 26 万个元素
    esz = torch.empty(0, dtype=dtype).element_size()

    tbl = Table("源数组大小", "顺序取\n每个 ns", "随机取\n每个 ns", "随机 / 顺序",
                "解读",
                title=f"gather {k:,} 个元素，只改源数组有多大")

    ratios: list[tuple[int, float]] = []
    for mb in (0.5, 1, 2, 4, 8, 16, 32, 64, 256):
        n = int(mb * 1024 ** 2) // esz
        if n < k:
            continue
        src = torch.randn(n, device=dev, dtype=dtype)
        seq = torch.arange(k, device=dev)
        rnd = torch.randint(0, n, (k,), device=dev)

        t_seq = timeit(lambda: src[seq], dev, warmup=10, iters=30)
        t_rnd = timeit(lambda: src[rnd], dev, warmup=10, iters=30)
        ratio = t_rnd / t_seq
        ratios.append((int(mb), ratio))
        hint = "整个数组都在缓存里" if ratio < 1.5 else (
            "开始装不下了" if ratio < 3 else "每次都要跑一趟 DRAM")
        tbl.add_row(f"{mb:g} MB", f"{t_seq/k*1e9:8.2f}", f"{t_rnd/k*1e9:8.2f}",
                    f"{ratio:5.2f}×", hint)
        del src, seq, rnd
    console.print(tbl)

    console.print(
        "\n[bold yellow]这张表的关键在于「随机 / 顺序」这一列会随源数组变大而变大：[/bold yellow]\n"
        "  · 源数组小的时候，随机和顺序[bold]几乎一样快[/bold] —— 反正整个数组都在缓存里，\n"
        "    随便怎么跳都命中。\n"
        "  · 源数组大到装不下缓存，随机访问就开始[bold]每次都跑一趟 DRAM[/bold]：\n"
        "    一条 cache line 搬回 64 字节，只用掉 2 字节，其余全是废料。\n"
        f"  · [bold]所以「随机访问慢」这句话是不完整的[/bold] ——\n"
        "    慢的不是随机，是[bold]工作集超出了缓存[/bold]。这就是「局部性」的全部含义。\n\n"
        "[dim]注意绝对值别当真：索引张量本身是 int64，每个元素要额外读 8 字节，\n"
        "所以这里测的是相对关系，不是绝对带宽。[/dim]\n"
    )


# --------------------------------------------------------------------- 实验 C
def exp_c_host_device(dev: torch.device, dtype: torch.dtype) -> None:
    """CPU 和 GPU 之间搬数据有多贵。"""
    console.rule("[bold]C · 跨过那道墙：CPU 和 GPU 之间搬 256 MB")

    if dev.type == "cpu":
        console.print("[yellow]CPU 模式跳过[/yellow]\n")
        return

    mb = 256
    n = mb * 1024 * 1024 // torch.empty(0, dtype=dtype).element_size()
    nbytes = mb * 1024 ** 2

    host = torch.randn(n, dtype=dtype)                 # pageable
    devt = torch.empty(n, device=dev, dtype=dtype)

    tbl = Table("方向 / 方式", "耗时", "带宽 GB/s", "对比同卡显存",
                title=f"搬运 {mb} MB")

    # 设备内部拷贝，作为参照
    dst = torch.empty_like(devt)
    t_dev = timeit(lambda: dst.copy_(devt), dev, warmup=5, iters=20)
    bw_dev = 2 * nbytes / t_dev / 1e9                  # 读+写
    tbl.add_row("显存 → 显存（读+写）", f"{t_dev*1e3:.2f} ms", f"{bw_dev:.1f}", "1.00×")

    t = timeit(lambda: devt.copy_(host), dev, warmup=3, iters=10)
    tbl.add_row("主机 → GPU（pageable）", f"{t*1e3:.2f} ms",
                f"{nbytes/t/1e9:.1f}", f"{(nbytes/t)/(2*nbytes/t_dev):.2f}×")

    if dev.type == "cuda":
        pinned = torch.randn(n, dtype=dtype).pin_memory()
        t = timeit(lambda: devt.copy_(pinned, non_blocking=False), dev,
                   warmup=3, iters=10)
        tbl.add_row("主机 → GPU（pinned）", f"{t*1e3:.2f} ms",
                    f"{nbytes/t/1e9:.1f}", f"{(nbytes/t)/(2*nbytes/t_dev):.2f}×")
        del pinned

    out = torch.empty(n, dtype=dtype)
    t = timeit(lambda: out.copy_(devt), dev, warmup=3, iters=10)
    tbl.add_row("GPU → 主机（pageable）", f"{t*1e3:.2f} ms",
                f"{nbytes/t/1e9:.1f}", f"{(nbytes/t)/(2*nbytes/t_dev):.2f}×")
    console.print(tbl)

    if dev.type == "mps":
        console.print(
            "\n[bold yellow]Apple 统一内存的一个反直觉点：[/bold yellow]\n"
            "  CPU 和 GPU 用的是[bold]同一块物理内存[/bold]，理论上根本不需要拷贝。\n"
            "  但 `.to('mps')` 还是花了时间 —— 因为 PyTorch 的 CPU 张量分配在普通堆页上，\n"
            "  不是 Metal buffer，跨过去必须真拷一次。\n"
            "  [bold]统一内存省掉的是「过 PCIe」，不是「拷贝」本身。[/bold]\n"
            "  想真正零拷贝，得让框架一开始就在共享缓冲区里分配（MLX 就是这么做的）。\n"
        )
    else:
        console.print(
            "\n[bold yellow]pinned（页锁定）内存为什么更快：[/bold yellow]\n"
            "  普通内存页可能被操作系统换出，DMA 引擎不敢直接读，\n"
            "  于是驱动要先把数据拷进一块内部的锁页缓冲区，再发起 DMA ——[bold]多搬了一遍[/bold]。\n"
            "  pin_memory() 把页钉死，DMA 直接读，省掉那一次拷贝。\n"
            "  代价：钉住的页不能被换出，开太多会挤占系统内存。\n"
        )

    console.print(
        "[bold]记住这个比值[/bold]：主机 ↔ GPU 的带宽只有显存带宽的几分之一到十几分之一。\n"
        "任何「把东西挪到 CPU 内存去省显存」的方案，都要先在这个比值上算清账。\n"
    )
    del host, devt, dst, out


# --------------------------------------------------------------------- 实验 D
def exp_d_fusion(dev: torch.device, dtype: torch.dtype) -> None:
    """同一串逐元素运算：分步做 vs 融合成一个 kernel。"""
    console.rule("[bold]D · 融合的价值：数学一样，显存流量差 4 倍")

    n = 32 * 1024 * 1024
    x = torch.randn(n, device=dev, dtype=dtype)
    esz = x.element_size()

    def eager():
        a = x * 2.0
        b = a + 1.0
        c = torch.relu(b)
        return c * c

    fused = None
    try:
        fused = torch.compile(eager, mode="max-autotune-no-cudagraphs")
        fused()                                        # 触发编译
        sync(dev)
    except Exception as e:                             # noqa: BLE001
        console.print(f"[yellow]torch.compile 不可用，跳过融合对比：{e}[/yellow]")
        fused = None

    tbl = Table("实现", "显存流量（估算）", "耗时", "等效带宽 GB/s",
                title=f"y = relu(2x+1)^2，x 有 {n:,} 个元素（{n*esz/1024**2:.0f} MB）")

    t_eager = timeit(eager, dev, warmup=5, iters=20)
    # 4 个 kernel，每个读 1 个数组写 1 个数组
    bytes_eager = 4 * 2 * n * esz
    tbl.add_row("eager（4 个独立 kernel）", f"{bytes_eager/1024**2:.0f} MB",
                f"{t_eager*1e3:.2f} ms", f"{bytes_eager/t_eager/1e9:.0f}")

    bytes_fused = 2 * n * esz                          # 只读一遍 x，写一遍结果
    speedup = None
    if fused is not None:
        t_fused = timeit(fused, dev, warmup=5, iters=20)
        speedup = t_eager / t_fused
        tbl.add_row("torch.compile（融合成 1 个）",
                    f"{bytes_fused/1024**2:.0f} MB",
                    f"{t_fused*1e3:.2f} ms", f"{bytes_fused/t_fused/1e9:.0f}")
    console.print(tbl)

    console.print(
        f"\n[bold yellow]算一笔账：[/bold yellow]\n"
        f"  eager 每一步都要[bold]把整个数组写回显存、再读回来[/bold]，\n"
        f"  4 步就是 {bytes_eager/1024**2:.0f} MB 的流量；\n"
        f"  融合之后，一个元素读进寄存器，四步一口气算完再写出去 ——"
        f" 只有 {bytes_fused/1024**2:.0f} MB。\n"
        f"  [bold]FLOPs 一模一样，省下的全是搬运。[/bold]\n"
    )
    if speedup:
        console.print(
            f"[bold]注意最后一列：两行的「等效带宽」几乎一样大[/bold] —— 两者都已经把显存带宽跑满了！\n"
            f"  换句话说：融合[bold]没有把屋顶抬高一点点[/bold]，它只是把要搬的东西减少了 4 倍，\n"
            f"  于是快了 [cyan]{speedup:.1f}×[/cyan]。这就是 Day 02 说的②号方向在最小尺度上的样子。\n"
        )
    console.print(
        "  这就是 Week 10 讲编译器（TorchInductor / Triton）时的核心动机，\n"
        "  也是 FlashAttention 的思想在最小尺度上的样子：[bold]让中间结果不要落地[/bold]。\n"
    )
    del x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list("ABCD"), help="只跑指定实验")
    args = ap.parse_args()

    dev, name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {name}  |  "
                  f"[bold green]dtype[/bold green]: {dtype}  |  "
                  f"[bold green]torch[/bold green]: {torch.__version__}\n")

    if args.exp in (None, "A"):
        exp_a_working_set(dev, dtype)
    if args.exp in (None, "B"):
        exp_b_random_access(dev, dtype)
    if args.exp in (None, "C"):
        exp_c_host_device(dev, dtype)
    if args.exp in (None, "D"):
        exp_d_fusion(dev, dtype)


if __name__ == "__main__":
    main()
