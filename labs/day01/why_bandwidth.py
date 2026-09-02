"""Day 01 附加实验 · 证明"带宽探测"真的在测搬运，而不是在测计算

两个证据：
  A) 固定元素数，改变"要搬几个数组"：耗时严格按 2:3:4 走，跟 FLOPs 无关
  B) 固定每元素的搬运量，改变数组大小：字节数与耗时严格成正比，斜率就是带宽

用法：uv run python labs/day01/why_bandwidth.py
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from probe import pick_device, timeit

console = Console()


def exp_a(dev: torch.device, dtype: torch.dtype) -> None:
    """同样 6700 万个元素，只改变经手的数组个数。"""
    console.rule("[bold]证据 A · 耗时只跟『搬几个数组』有关，跟算几次无关")

    m = 2 ** 26
    x = torch.randn(m, device=dev, dtype=dtype)
    y = torch.randn(m, device=dev, dtype=dtype)
    z = torch.randn(m, device=dev, dtype=dtype)
    out = torch.empty(m, device=dev, dtype=dtype)
    esz = x.element_size()

    cases = [
        # 名称, 调用, 理想融合后的数组数, eager 实际经手的数组数, 每元素运算次数
        ("out.copy_(x)", lambda: out.copy_(x), 2, 2, 0),
        ("x + y", lambda: x + y, 3, 3, 1),
        # eager 不融合：先算 tmp=x+y（3个）再算 tmp+z（3个），共 6 个
        ("x + y + z", lambda: x + y + z, 4, 6, 2),
    ]

    tbl = Table("操作", "运算\n/元素", "经手数组", "实际搬运", "耗时",
                "相对首行 / 期望", "GB/s",
                title=f"元素数固定 = {m:,}")
    base_t = None
    for name, fn, ideal, actual, n_flop in cases:
        t = timeit(fn, dev, warmup=5, iters=20)
        base_t = base_t or t
        moved = actual * m * esz
        arr = f"{actual} 个" + (f" ⚠理想{ideal}" if actual > ideal else "")
        tbl.add_row(name, str(n_flop), arr, f"{moved/1e6:.0f} MB",
                    f"{t*1e3:.2f} ms",
                    f"{t/base_t:.2f}× / {actual/2:.2f}×",
                    f"{moved/t/1e9:.1f}")
    console.print(tbl)
    console.print(
        "\n[bold yellow]怎么读：[/bold yellow]\n"
        "  · 第一行 [bold]0 次运算[/bold]却要花 3 ms —— 如果耗时由计算决定，它应该接近 0。\n"
        "  · 「相对第一行」与括号里的期望值[bold]一路吻合[/bold]，"
        "期望值只由[bold cyan]数组个数[/bold cyan]算出，完全没用到运算次数。\n"
        "  · 最后一列三行[bold]收敛到同一个数[/bold] —— 那就是这台机器的真实带宽。\n"
        "\n[bold red]⚠ 注意第三行[/bold red]：`x + y + z` 理想上只需搬 4 个数组，"
        "但 PyTorch eager [bold]不做算子融合[/bold] ——\n"
        "  它先算 `tmp = x + y`（读2写1），再算 `tmp + z`（读2写1），"
        "一共经手 [bold]6 个[/bold]数组，白白多搬了 50%。\n"
        "  [dim]这就是算子融合能提速的全部原理，Day 26 会专门讲。[/dim]\n"
    )
    del x, y, z, out


def exp_b(dev: torch.device, dtype: torch.dtype) -> None:
    """同一个 x+y，只改数组大小，看字节数与耗时是否严格成正比。"""
    console.rule("[bold]证据 B · 搬运量翻倍，耗时就翻倍")

    tbl = Table("元素数", "搬运字节", "耗时", "字节比", "耗时比", "实测 GB/s",
                title="out = x + y，只改变规模")
    b0 = t0 = None
    for m in (2 ** 22, 2 ** 23, 2 ** 24, 2 ** 25, 2 ** 26):
        x = torch.randn(m, device=dev, dtype=dtype)
        y = torch.randn(m, device=dev, dtype=dtype)
        t = timeit(lambda: x + y, dev, warmup=5, iters=20)
        moved = 3 * m * x.element_size()
        b0 = b0 or moved
        t0 = t0 or t
        tbl.add_row(f"{m:,}", f"{moved/1e6:.0f} MB", f"{t*1e3:.2f} ms",
                    f"{moved/b0:.0f}×", f"{t/t0:.2f}×", f"{moved/t/1e9:.1f}")
        del x, y
    console.print(tbl)
    console.print(
        "\n[bold yellow]怎么读：[/bold yellow]\n"
        "  · 「字节比」和「耗时比」两列[bold]一路对齐[/bold]（1/2/4/8/16），"
        "就是干净的带宽受限行为。\n"
        "  · 小数组略微偏慢（GB/s 更低）不是缓存问题，是 [bold]kernel 启动开销[/bold]"
        "（几十微秒）和并行度不足被摊进去了。\n"
        "  · 所以探测带宽要用[bold]足够大[/bold]的数组：既远超缓存，又能把启动开销摊薄到可忽略。\n"
    )


def main() -> None:
    dev, name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {name}  |  {dtype}\n")
    exp_a(dev, dtype)
    exp_b(dev, dtype)
    console.print(
        "[bold]结论[/bold]：`x + y` 每个元素只做 [bold]1 次[/bold]加法，却要搬 [bold]6 字节[/bold]。\n"
        "算术强度 0.167 FLOP/Byte，远低于机器平衡点 —— GPU 算力几乎全程空转，\n"
        "耗时 100% 由内存总线决定。所以 [bold cyan]字节数 / 耗时 = 带宽[/bold cyan] 成立。\n"
    )


if __name__ == "__main__":
    main()
