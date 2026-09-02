"""Day 06 实验 · 从 `y = model(x)` 到 GPU 指令，中间那几层各花了多少

五件事：
  A  开销阶梯：纯 Python → 元数据 → 最小 kernel → 大 kernel，把固定开销剥出来
  B  异步：不 sync 量出来的时间是假的；以及一次同步有多贵
  C  分发路径：同一个加法的四种写法，小张量上差多少
  D  一层 Transformer 的账：发了多少个算子，软件开销占多大比例
  E  torch.profiler：学会读三个数，判断 CPU-bound 还是 GPU-bound

用法：
    uv run python labs/day06/stack_profile.py
    uv run python labs/day06/stack_profile.py --exp A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from probe import pick_device, sync  # noqa: E402

console = Console()


def bench(fn, dev: torch.device, n: int = 2000, warmup: int = 100,
          do_sync: bool = True) -> float:
    """返回单次调用的平均耗时（微秒）。"""
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if do_sync:
        sync(dev)
    return (time.perf_counter() - t0) / n * 1e6


# --------------------------------------------------------------------- 实验 A
def exp_a_overhead_ladder(dev: torch.device) -> None:
    console.rule("[bold]A · 开销阶梯：一次调用的时间都花在哪一层")

    x1 = torch.randn(1, device=dev)
    y1 = torch.randn(1, device=dev)
    mid = torch.randn(1024, 1024, device=dev)
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)

    def noop() -> None:
        pass

    rows = [
        ("① 纯 Python 空函数", lambda: noop(), 20000, "解释器自己"),
        ("② 读元数据 x.shape", lambda: x1.shape, 20000, "不进 C++"),
        ("③ 元数据操作 x.view(1)", lambda: x1.view(1), 20000, "进了 C++，不发 kernel"),
        ("④ 1 元素相加", lambda: x1 + y1, 3000, "★ 第一次真正发 kernel"),
        ("⑤ 1024² 相加 (12 MB)", lambda: mid + mid, 500, "开始有真实工作量"),
        ("⑥ 4096² 相加 (200 MB)", lambda: a + b, 200, "数据搬运主导"),
    ]

    tbl = Table("这一步做了什么", "不 sync\nCPU 提交", "sync\n端到端",
                "比上一行多", "说明", title=f"设备 {dev.type}")
    prev = 0.0
    submit = launch = 0.0
    for name, fn, n, note in rows:
        t_nosync = bench(fn, dev, n=n, warmup=max(50, n // 20), do_sync=False)
        t_sync = bench(fn, dev, n=n, warmup=max(50, n // 20), do_sync=True)
        tbl.add_row(name, f"{t_nosync:9.3f} µs", f"{t_sync:9.3f} µs",
                    f"+{t_sync - prev:8.3f} µs", note)
        if name.startswith("④"):
            submit, launch = t_nosync, t_sync
        prev = t_sync
    console.print(tbl)

    console.print(
        f"\n[bold yellow]三件事：[/bold yellow]\n"
        f"  ① ①→③ 全在 CPU 上，加起来不到 1 µs。[bold]Python 慢，慢不在这里。[/bold]\n"
        f"  ② [bold]③→④ 这一跳最关键[/bold]：第一次真正「把命令交给 GPU」。\n"
        f"     CPU 侧提交要 [cyan]{submit:.1f} µs[/cyan]，端到端要 [cyan]{launch:.1f} µs[/cyan]。\n"
        f"     [bold]这就是 Day 04 实验 A 里那个 t0 的真身。[/bold]\n"
        f"  ③ ④→⑥ 才是真正的计算/搬运时间，随数据量增长。\n\n"
        f"[bold]结论：任何 kernel 都要先交约 {launch:.0f} µs 的「过路费」，和它算多少完全无关。[/bold]\n"
        f"  所以「把 10 个小算子合成 1 个」能省下 9 份过路费 —— 这就是算子融合的全部动机。\n"
    )


# --------------------------------------------------------------------- 实验 B
def exp_b_async(dev: torch.device) -> None:
    console.rule("[bold]B · 异步：CPU 和 GPU 是两条独立的时间线")

    if dev.type == "cpu":
        console.print("[yellow]CPU 上没有异步队列，跳过[/yellow]\n")
        return

    a = torch.randn(2048, 2048, device=dev)

    t_nosync = bench(lambda: a @ a, dev, n=50, warmup=20, do_sync=False)
    t_sync = bench(lambda: a @ a, dev, n=50, warmup=20, do_sync=True)

    tbl = Table("计时方式", "量出来的耗时", "这个数到底是什么",
                title="同一个 2048³ 矩阵乘，测 50 次取平均")
    tbl.add_row("循环里不 sync", f"{t_nosync:9.1f} µs",
                "只是【CPU 把命令塞进队列】的时间")
    tbl.add_row("循环后 sync", f"{t_sync:9.1f} µs", "GPU 真正算完的时间")
    tbl.add_row("[bold]比值[/bold]", f"[bold]{t_sync / t_nosync:9.1f}×[/bold]",
                "CPU 跑得比 GPU 快这么多倍")
    console.print(tbl)

    # ---- 队列深度 ----
    small = torch.randn(256, 256, device=dev)
    tbl = Table("连发多少个小 kernel", "总耗时", "平均每个", "说明",
                title="连续发射同一个小算子，中间不 sync，最后 sync 一次")
    for n in (1, 10, 100, 1000):
        for _ in range(20):
            small + small
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            small + small
        sync(dev)
        total = (time.perf_counter() - t0) * 1e6
        note = "这一行几乎全是【同步本身】的代价" if n == 1 else "开销被摊薄"
        tbl.add_row(f"{n:>5}", f"{total:10.1f} µs", f"{total/n:8.2f} µs", note)
    console.print(tbl)

    # ---- 隐式同步的陷阱（用小算子，同步开销才看得出来）----
    n = 300
    s = torch.randn(512, 512, device=dev)
    for _ in range(30):
        s @ s
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        s @ s
    sync(dev)
    t_pipeline = (time.perf_counter() - t0) / n * 1e6

    t0 = time.perf_counter()
    for _ in range(n):
        (s @ s)[0, 0].item()                     # 每一步都强制同步
    t_item = (time.perf_counter() - t0) / n * 1e6

    tbl = Table("循环里写了什么", "每步耗时", "变慢了多少",
                title=f"隐式同步的代价（同样 {n} 次 512³ 矩阵乘）")
    tbl.add_row("y = s @ s", f"{t_pipeline:9.1f} µs", "—")
    tbl.add_row("y = s @ s; y[0,0].item()", f"{t_item:9.1f} µs",
                f"[bold]{t_item / t_pipeline:.1f}×[/bold]")
    console.print(tbl)

    console.print(
        "\n[bold yellow]三个必须记住的点：[/bold yellow]\n"
        "  ① [bold]`a @ a` 这一行在 CPU 上做的事是「把命令塞进队列」，然后立刻返回[/bold]。\n"
        "     GPU 什么时候算，Python 完全不知道。所以[bold]不 sync 就计时，量到的是投递速度[/bold]。\n"
        "     这就是 Day 01 起每次计时都要 sync() 的原因 —— 今天终于说清楚了。\n\n"
        "  ② [bold]一次同步本身就很贵[/bold]（看队列深度表第一行）：要提交命令缓冲区、\n"
        "     等 GPU 跑完、再等完成通知回到 CPU，是一次完整的往返。\n\n"
        "  ③ [bold]这些写法会偷偷触发同步[/bold]，在训练/推理循环里是性能杀手：\n"
        "     `.item()` · `.cpu()` · `.numpy()` · `print(tensor)` · `if tensor > 0` ·\n"
        "     `float(loss)` · `tensor.tolist()` —— 任何把张量当 Python 数值用的地方。\n"
        "     [bold]正确做法：把 loss 攒在 GPU 上的列表里，几百步才同步打印一次。[/bold]\n"
    )


# --------------------------------------------------------------------- 实验 C
def exp_c_dispatch(dev: torch.device) -> None:
    console.rule("[bold]C · 分发路径：同一个加法的四种写法")

    for n, tag in ((1024, "小张量 1024 个元素（4 KB）—— 固定开销主导"),
                   (1 << 22, "大张量 4M 个元素（16 MB）—— 数据搬运主导")):
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        out = torch.empty(n, device=dev)

        variants = [
            ("a + b", lambda: a + b, "Python __add__ → dispatcher → aten::add"),
            ("torch.add(a, b)", lambda: torch.add(a, b), "少一层 Python 魔术方法"),
            ("torch.add(a, b, out=out)", lambda: torch.add(a, b, out=out),
             "复用输出张量，省一次分配"),
            ("a.add_(b)", lambda: a.add_(b), "原地写回，也省分配"),
        ]

        iters = 3000 if n <= 4096 else 300
        times = [bench(fn, dev, n=iters, warmup=max(50, iters // 10))
                 for _n, fn, _c in variants]
        fastest = min(times)
        tbl = Table("写法", "平均耗时", "相对最快", "分发路径", title=tag)
        for (name, _fn, note), us in zip(variants, times):
            tbl.add_row(name, f"{us:9.2f} µs", f"{us/fastest:5.2f}×", note)
        console.print(tbl)
        del a, b, out

    console.print(
        "\n[bold yellow]结果可能和你预期的不一样：四种写法差别不到 10%。[/bold yellow]\n"
        "  网上常说的「用 `out=` / 原地操作能提速」，在 PyTorch 里[bold]效果很有限[/bold]。\n\n"
        "  [bold]为什么？回到图 1 那张剖面图[/bold]：\n"
        "  Python 包装 + dispatcher 加起来只有 ~1 µs，\n"
        "  而过路费的大头在[bold]运行时 + 驱动 + 命令队列[/bold]那两三层 —— 换写法碰不到它们。\n\n"
        "  [bold]所以想省这笔钱，唯一的办法是「少发几次」，不是「换个姿势发」。[/bold]\n"
        "  这就把我们直接推向了 §8 的三条对策：融合、CUDA Graph、持久化 kernel。\n\n"
        "[dim]（`out=` 和原地操作仍然值得写 —— 它们真正省的是显存峰值和分配器压力，\n"
        "在显存紧张时很关键，只是别指望它提速。）[/dim]\n"
    )


# --------------------------------------------------------------------- 实验 D
def _layer_factory(dev: torch.device, dtype: torch.dtype, d: int):
    w_qkv = torch.randn(d, 3 * d, device=dev, dtype=dtype)
    w_o = torch.randn(d, d, device=dev, dtype=dtype)
    w_up = torch.randn(d, 4 * d, device=dev, dtype=dtype)
    w_dn = torch.randn(4 * d, d, device=dev, dtype=dtype)
    g1 = torch.randn(d, device=dev, dtype=dtype)
    g2 = torch.randn(d, device=dev, dtype=dtype)
    weight_bytes = sum(t.numel() * t.element_size()
                       for t in (w_qkv, w_o, w_up, w_dn))

    def layer(x: torch.Tensor) -> torch.Tensor:
        h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g1   # RMSNorm
        qkv = h @ w_qkv
        q, k, v = qkv.chunk(3, dim=-1)
        att = torch.softmax(q * k * 0.125, dim=-1) * v                   # 简化的注意力
        x = x + att @ w_o
        h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g2
        h = torch.nn.functional.silu(h @ w_up)
        return x + h @ w_dn

    return layer, weight_bytes


def exp_d_layer_budget(dev: torch.device, dtype: torch.dtype) -> None:
    console.rule("[bold]D · 一层 Transformer 的账：软件开销到底占多少")

    d = 4096
    layer, wbytes = _layer_factory(dev, dtype, d)
    flops_per_token = 2 * (3 * d * d + d * d + 4 * d * d + 4 * d * d)
    peak = 3.5e12 if dev.type == "mps" else 24e12
    bw = 91e9 if dev.type == "mps" else 192e9

    tbl = Table("batch", "每步耗时", "有效算力\nTFLOPS", "算力\n达成率",
                "光搬权重\n就要多久", "占比", "谁是瓶颈",
                title=f"一层 Transformer（d={d}，权重 {wbytes/1e6:.0f} MB）")
    for bsz in (1, 4, 16, 64, 256):
        x = torch.randn(bsz, d, device=dev, dtype=dtype)
        us = bench(lambda: layer(x), dev, n=50, warmup=20)
        tf = flops_per_token * bsz / (us * 1e-6) / 1e12
        rate = tf / peak * 100
        bw_us = wbytes / bw * 1e6
        who = "带宽（权重搬运）" if rate < 50 else "算力"
        tbl.add_row(str(bsz), f"{us:9.1f} µs", f"{tf:8.2f}", f"{rate:6.1f}%",
                    f"{bw_us:8.1f} µs", f"{bw_us/us*100:5.1f}%", who)
        del x
    console.print(tbl)

    # ---- 数一数这一层发了多少个算子 ----
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(1, d, device=dev, dtype=dtype)
    for _ in range(10):
        layer(x)
    sync(dev)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        layer(x)
        sync(dev)
    ops = [(e.key, e.count) for e in prof.key_averages() if e.key.startswith("aten::")]
    n_ops = sum(c for _k, c in ops)

    tbl = Table("算子", "调用次数",
                title=f"这一层一共调用了 {n_ops} 次 aten 算子（含组合算子，是 kernel 数的上界）")
    for k, c in sorted(ops, key=lambda t: -t[1])[:10]:
        tbl.add_row(k, str(c))
    console.print(tbl)

    # ---- 外推：GPU 越快，软件开销占比越高 ----
    launch_us = 6.0
    over_us = n_ops * launch_us
    tbl = Table("机器", "显存带宽", "一层权重搬运", f"{n_ops} 个算子的\n发射开销",
                "软件开销占比",
                title=f"同一份代码换到不同的卡上（假设每个算子固定开销 {launch_us} µs）")
    for name, b in (("Mac mini M4", 91e9), ("RTX 4050", 192e9),
                    ("RTX 4090", 1008e9), ("A100 80G", 2039e9),
                    ("H100 SXM", 3350e9)):
        move_us = wbytes / b * 1e6
        tbl.add_row(name, f"{b/1e9:.0f} GB/s", f"{move_us:8.1f} µs",
                    f"{over_us:8.1f} µs", f"{over_us/(move_us+over_us)*100:5.1f}%")
    console.print(tbl)

    console.print(
        f"\n[bold yellow]这三张表连起来说了一件很重要的事：[/bold yellow]\n"
        f"  · 在 [bold]M4 上 batch=1 时，软件开销只占几个百分点[/bold] —— "
        f"瓶颈是带宽（{wbytes/1e6:.0f} MB 权重必须全读一遍）。\n"
        f"  · 但[bold]显存带宽越快，同样这几十个 kernel 的发射开销占比就越高[/bold]。\n"
        f"    在 H100 上，权重搬运只要一百多微秒，而发射开销一分不少。\n\n"
        f"  [bold]软件开销占比 = N×t_launch / (权重字节/带宽 + N×t_launch)[/bold]\n\n"
        f"  所以 [bold]CUDA Graph、算子融合、持久化 kernel 在慢卡上可有可无，"
        f"在快卡上是必需品[/bold]。\n"
        f"  这也解释了一个常见困惑：「为什么同一份代码换到 H100 上，"
        f"加速远达不到带宽比？」\n"
        f"  —— 因为软件开销那一项，一点都没变小。\n"
    )


# --------------------------------------------------------------------- 实验 E
def exp_e_profiler(dev: torch.device, dtype: torch.dtype) -> None:
    console.rule("[bold]E · torch.profiler：判断 CPU-bound 还是 GPU-bound")

    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    dev_name = None
    for attr, key in (("CUDA", "cuda"), ("XPU", "xpu"), ("MPS", "mps")):
        if dev.type == key and hasattr(ProfilerActivity, attr):
            acts.append(getattr(ProfilerActivity, attr))
            dev_name = key
    if dev.type != "cpu" and dev_name is None:
        console.print(
            f"[yellow]⚠️  这个 PyTorch 版本没有 ProfilerActivity.{dev.type.upper()}"
            " —— 只能拿到 CPU 侧数据。[/yellow]\n"
            "[yellow]   本实验的完整版请在 RTX 4050 上跑，"
            "那里能看到真实的 CUDA kernel 时间。[/yellow]\n"
        )

    d, bsz = 2048, 8
    layer, _ = _layer_factory(dev, dtype, d)
    x = torch.randn(bsz, d, device=dev, dtype=dtype)

    for _ in range(20):
        layer(x)
    sync(dev)

    with profile(activities=acts) as prof:
        for _ in range(20):
            layer(x)
        sync(dev)

    ka = prof.key_averages()
    console.print(ka.table(sort_by="self_cpu_time_total", row_limit=12))

    cpu_us = sum(e.self_cpu_time_total for e in ka)
    gpu_us = sum(getattr(e, "self_device_time_total", 0.0) or 0.0 for e in ka)

    tbl = Table("指标", "数值", "含义", title="CPU-bound 还是 GPU-bound")
    tbl.add_row("Self CPU 总和", f"{cpu_us:10.1f} µs", "分发 + 参数检查 + 发射")
    tbl.add_row("Self Device 总和", f"{gpu_us:10.1f} µs",
                "GPU 真正执行" + ("" if gpu_us else "（本设备采不到）"))
    if gpu_us > 0:
        tbl.add_row("[bold]判定[/bold]",
                    "[bold red]CPU-bound[/bold red]" if cpu_us > gpu_us
                    else "[bold]GPU-bound[/bold]", "谁大听谁的")
    console.print(tbl)

    console.print(
        "\n[bold yellow]profiler 表里三列，三种完全不同的含义：[/bold yellow]\n"
        "  · [bold]Self CPU[/bold]     —— 这个算子[bold]自己[/bold]在 CPU 上花的时间，不含子算子。\n"
        "    [bold]只有这一列可以求和[/bold]，加起来就是 CPU 侧的总开销。\n"
        "  · [bold]CPU total[/bold]    —— 含子算子。`aten::matmul` 会把 `aten::mm` 的时间算进去，\n"
        "    [bold]拿它求和一定重复计算[/bold]。它只适合看「这个高层算子一共花了多久」。\n"
        "  · [bold]Self Device[/bold]  —— GPU 上真正执行的时间。\n\n"
        "[bold]判据（在 4050 上才能完整用上）：[/bold]\n"
        "  · Self CPU 总和 [bold]>[/bold] Self Device 总和 → [bold red]CPU-bound[/bold red]：\n"
        "    GPU 饿着等命令。对策：算子融合 / torch.compile / CUDA Graph / 加大 batch。\n"
        "  · Self CPU 总和 [bold]<[/bold] Self Device 总和 → [bold]GPU-bound[/bold]：\n"
        "    回到 Day 02 的 Roofline，去分是算力受限还是带宽受限。\n\n"
        "[dim]⚠️ Mac 上还有一个坑：MPS 的某些算子会在 CPU 侧阻塞等待，\n"
        "所以 Self CPU 这一列会把 GPU 时间也算进去，数值不能直接和 CUDA 上的比。[/dim]\n\n"
        "[dim]另外留意 aten::empty / aten::empty_like / aten::to / cudaLaunchKernel 这些条目 ——\n"
        "它们不是你的计算，是软件栈自己的开销。占比高就是优化信号。[/dim]\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list("ABCDE"), help="只跑指定实验")
    args = ap.parse_args()

    dev, name = pick_device()
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    console.print(f"[bold green]设备[/bold green]: {name}  |  "
                  f"[bold green]torch[/bold green]: {torch.__version__}\n")

    if args.exp in (None, "A"):
        exp_a_overhead_ladder(dev)
    if args.exp in (None, "B"):
        exp_b_async(dev)
    if args.exp in (None, "C"):
        exp_c_dispatch(dev)
    if args.exp in (None, "D"):
        exp_d_layer_budget(dev, dtype)
    if args.exp in (None, "E"):
        exp_e_profiler(dev, dtype)


if __name__ == "__main__":
    main()
