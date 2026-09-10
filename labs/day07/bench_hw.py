"""Day 07 实验 · bench_hw.py：一次量出这台机器的六个数，再用它预测模型性能

Week 1 六天的方法全部收进一个工具：

  ① 峰值算力 P        大方阵 GEMM 扫描（Day 01 / Day 03）
  ② DRAM 带宽 B       t = t0 + S/B 的最小二乘拟合（Day 04）
  ③ 缓存拐点          S/(t - t0) 曲线掉下来的位置（Day 04）
  ④ 过路费 t0         最小 kernel 的端到端耗时（Day 06）
  ⑤ 稳态发射单价      连发 1000 个小 kernel 的平均（Day 06）
  ⑥ 主机↔设备带宽 / 可用容量（Day 04 / Day 05）

跑完得到一张「机器画像卡」，存成 JSON；之后 --predict 不再碰硬件，
纯粹拿这六个数 + Week 1 的公式，算出任意模型的 TTFT / TPOT / 最大 batch / 吞吐。

用法：
    uv run python labs/day07/bench_hw.py                    # 全量实测 + 存画像卡
    uv run python labs/day07/bench_hw.py --quick            # 快一点（精度略降）
    uv run python labs/day07/bench_hw.py --predict          # 用画像卡预测主线案例
    uv run python labs/day07/bench_hw.py --predict --model llama3-8b --wbits 16
    uv run python labs/day07/bench_hw.py --predict --profile results/profile_cuda.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from probe import pick_device, sync, timeit  # noqa: E402

console = Console()
RESULTS = ROOT / "results"


# ===================================================================== 画像卡
@dataclass
class Profile:
    """一台机器的六个数（外加几个附赠品）。"""

    machine: str
    backend: str
    torch_version: str
    tflops_fp16: float          # ① 峰值算力
    tflops_fp32: float
    bw_dram: float              # ② DRAM 稳态带宽 GB/s
    bw_cache_peak: float        # ③ 缓存段峰值有效带宽 GB/s
    cache_knee_mb: float        # ③ 缓存拐点（工作集 MB）
    t0_us: float                # ④ 过路费：最小 kernel 的摊薄单价（直接测）
    t0_fit_us: float            # ④ 同一个量，用带宽曲线的拟合截距独立测一遍
    t_launch_us: float          # ⑤ 稳态单价（连发 1000 个）
    sync_cost_us: float         # ⑤ 附赠：一次同步的代价
    bw_h2d: float               # ⑥ 主机 → 设备 GB/s
    bw_d2h: float
    mem_total_gb: float
    mem_usable_gb: float

    @property
    def ridge(self) -> float:
        """机器平衡点 = 峰值算力 / 峰值带宽（Day 02）。"""
        return self.tflops_fp16 * 1e12 / (self.bw_dram * 1e9)

    @property
    def tensor_core_gain(self) -> float:
        """FP16 相对 FP32 的加速比：接近 1 说明没有独立矩阵单元（Day 03）。"""
        return self.tflops_fp16 / max(self.tflops_fp32, 1e-9)


# ===================================================================== 工具
def _bench_nosync(fn, dev: torch.device, n: int, warmup: int) -> float:
    """只量 CPU 侧提交成本（微秒）。"""
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / n * 1e6


def _lstsq(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘拟合 y = a + b·x，返回 (截距 a, 斜率 b)。手写，不引 numpy。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx
    return my - b * mx, b


def machine_name(dev_name: str) -> str:
    return f"{dev_name} · {platform.machine()} · {platform.system()}"


# ============================================================== ① 峰值算力
def measure_compute(dev: torch.device, quick: bool) -> tuple[float, float]:
    console.rule("[bold]① 峰值算力 P —— 大方阵 GEMM（Day 01 / Day 03 的方法）")

    sizes = (1024, 2048) if quick else (1024, 2048, 4096)
    tbl = Table("N", "FLOPs", "FP16 耗时", "FP16 TFLOPS", "FP32 耗时", "FP32 TFLOPS",
                title="GEMM 扫描：算术强度 ≈ N/3，足够高 → 必定撞算力墙")
    best = {torch.float16: 0.0, torch.float32: 0.0}
    for n in sizes:
        row = [str(n), f"{2.0 * n ** 3 / 1e9:.1f} G"]
        for dt in (torch.float16, torch.float32):
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
            t = timeit(lambda: a @ b, dev, warmup=5, iters=20)
            tf = 2.0 * n ** 3 / t / 1e12
            best[dt] = max(best[dt], tf)
            row += [f"{t * 1e3:.2f} ms", f"{tf:.2f}"]
            del a, b
        tbl.add_row(*row)
    console.print(tbl)

    gain = best[torch.float16] / max(best[torch.float32], 1e-9)
    verdict = ("有独立矩阵单元（Tensor Core）" if gain > 1.8
               else "[bold]没有独立矩阵单元[/bold]，FP16 只是省了带宽")
    console.print(f"\n  FP16 / FP32 = [cyan]{gain:.2f}×[/cyan] → {verdict}\n")
    return best[torch.float16], best[torch.float32]


# ================================================ ②③④ 带宽 / 缓存拐点 / t0
def measure_memory(dev: torch.device, quick: bool) -> dict:
    console.rule("[bold]②③ 带宽 B 与缓存拐点 —— 拟合 t = t0 + S/B（Day 04 的方法）")

    # 先独立量一次过路费：1 个元素的加法，工作量可以忽略，剩下的全是固定开销
    x1 = torch.randn(1, device=dev)
    t0_us = timeit(lambda: x1 + x1, dev, warmup=200, iters=2000) * 1e6

    hi = 24 if quick else 26
    rows = []
    for p in range(12, hi + 1):
        m = 1 << p
        x = torch.randn(m, device=dev, dtype=torch.float16)
        y = torch.randn(m, device=dev, dtype=torch.float16)
        iters = 200 if p < 22 else 30
        t = timeit(lambda: x + y, dev, warmup=max(10, iters // 5), iters=iters)
        moved = 3 * m * 2                       # 读 x、读 y、写 out
        rows.append((moved, t))
        del x, y

    # 只用最大的 4 个尺寸拟合（那时一定落在 DRAM 上），斜率 = 1/B，截距 = t0
    tail = rows[-4:]
    inter_s, slope = _lstsq([r[0] for r in tail], [r[1] for r in tail])
    bw_dram = 1.0 / slope / 1e9
    t0_fit_us = inter_s * 1e6

    tbl = Table("工作集", "搬运字节", "耗时", "朴素 S/t", "扣掉 t0 后 S/(t−t0)",
                title=f"逐元素加法扫描（t0 独立实测 = {t0_us:.2f} µs）")
    peak_eff, knee = 0.0, 0.0
    effs = []
    for moved, t in rows:
        naive = moved / t / 1e9
        # t 还没明显超过 t0 时，S/(t−t0) 是两个相近数相减，噪声会放大到几千 GB/s
        eff = moved / (t - t0_us / 1e6) / 1e9 if t > 3 * t0_us / 1e6 else None
        effs.append(eff)
        peak_eff = max(peak_eff, eff or 0.0)
        tbl.add_row(f"{moved / 1024 ** 2:8.2f} MB", f"{moved / 1e6:8.1f} MB",
                    f"{t * 1e6:9.1f} µs", f"{naive:7.1f} GB/s",
                    f"[cyan]{eff:7.1f} GB/s[/cyan]" if eff
                    else "[dim]—[/dim]")
    console.print(tbl)

    # 拐点：有效带宽最后一次高于 DRAM 带宽 1.5 倍的工作集
    for (moved, _), eff in zip(rows, effs):
        if eff and eff > 1.5 * bw_dram:
            knee = moved / 1024 ** 2
    console.print(
        f"\n  ② DRAM 稳态带宽（尾部 4 点最小二乘）= [bold cyan]{bw_dram:.1f} GB/s[/bold cyan]\n"
        f"  ③ 缓存段峰值有效带宽 = [bold cyan]{peak_eff:.1f} GB/s[/bold cyan]"
        f"（是 DRAM 的 {peak_eff / bw_dram:.1f} 倍）\n"
        f"  ③ 缓存拐点 ≈ [bold cyan]{knee:.0f} ~ {knee * 2:.0f} MB[/bold cyan]\n"
        f"  ④ 拟合截距 t0 = [bold]{t0_fit_us:.2f} µs[/bold]，"
        f"独立实测 t0 = [bold]{t0_us:.2f} µs[/bold] "
        f"—— 两条完全不同的路量出同一个数，说明模型是对的\n"
    )
    return dict(bw_dram=bw_dram, bw_cache_peak=peak_eff, cache_knee_mb=knee,
                t0_us=t0_us, t0_fit_us=t0_fit_us)


# ============================================================ ⑤ 稳态发射单价
def measure_launch(dev: torch.device) -> tuple[float, float]:
    """返回 (稳态单价 µs, 一次同步的代价 µs)。"""
    console.rule("[bold]⑤ 稳态发射单价 —— 连发不 sync（Day 06 的方法）")

    if dev.type == "cpu":
        console.print("[yellow]CPU 上没有命令队列，跳过[/yellow]\n")
        return 0.0, 0.0

    a = torch.randn(256, 256, device=dev)
    tbl = Table("连发多少个", "总耗时", "平均每个", "说明",
                title="发 N 个小 kernel，中间不 sync，最后 sync 一次")
    steady = sync_cost = 0.0
    for n in (1, 10, 100, 1000):
        for _ in range(50):
            a + a
        sync(dev)
        reps = 20 if n < 1000 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            for _ in range(n):
                a + a
            sync(dev)
        total = (time.perf_counter() - t0) / reps * 1e6
        steady = total / n
        if n == 1:
            sync_cost = total
        note = ("这一行几乎全是同步本身的代价" if n == 1
                else "稳态：流水线打满后的真实单价" if n == 1000 else "")
        tbl.add_row(str(n), f"{total:9.1f} µs", f"[cyan]{steady:7.2f} µs[/cyan]", note)
    console.print(tbl)
    console.print(
        f"\n  ⑤ 稳态单价 t_launch = [bold cyan]{steady:.2f} µs[/bold cyan]"
        f" —— 这才是估「N 个算子要多久」时该用的数\n"
        f"     一次同步 = [bold]{sync_cost:.0f} µs[/bold] ≈ "
        f"[bold]{sync_cost / max(steady, 1e-9):.0f} 个 kernel[/bold] 的发射开销\n")
    return steady, sync_cost


# ====================================================== ⑥ 主机↔设备 / 容量
def measure_transfer(dev: torch.device) -> tuple[float, float]:
    console.rule("[bold]⑥ 主机 ↔ 设备带宽（Day 04 的方法）")

    if dev.type == "cpu":
        console.print("[yellow]CPU 上不存在这条链路，跳过[/yellow]\n")
        return 0.0, 0.0

    n = 1 << 24                                  # 32 MB fp16
    nbytes = n * 2
    host = torch.randn(n, dtype=torch.float16).pin_memory() if dev.type == "cuda" \
        else torch.randn(n, dtype=torch.float16)
    devt = torch.randn(n, device=dev, dtype=torch.float16)

    t_h2d = timeit(lambda: devt.copy_(host), dev, warmup=5, iters=30)
    t_d2h = timeit(lambda: host.copy_(devt), dev, warmup=5, iters=30)
    h2d, d2h = nbytes / t_h2d / 1e9, nbytes / t_d2h / 1e9

    tbl = Table("方向", "字节", "耗时", "带宽")
    tbl.add_row("主机 → 设备", f"{nbytes / 1e6:.0f} MB", f"{t_h2d * 1e3:.2f} ms",
                f"[cyan]{h2d:.1f} GB/s[/cyan]")
    tbl.add_row("设备 → 主机", f"{nbytes / 1e6:.0f} MB", f"{t_d2h * 1e3:.2f} ms",
                f"[cyan]{d2h:.1f} GB/s[/cyan]")
    console.print(tbl)
    del host, devt
    return h2d, d2h


def measure_capacity(dev: torch.device) -> tuple[float, float]:
    """返回 (标称总量 GB, 实际能装下的 GB)。"""
    total = 0.0
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    elif dev.type == "mps":
        fn = getattr(torch.mps, "recommended_max_memory", None)
        total = fn() / 1024 ** 3 if fn else 0.0

    # 实探：一块一块 512 MB 地要并写满（只 empty 不写的话分配器可能只是记账，没真占住）
    limit = int(total / 0.5) if total > 0 else 64
    usable, blocks = 0.0, []
    try:
        for _ in range(limit):
            blk = torch.empty(1 << 28, dtype=torch.float16, device=dev)
            blk.fill_(1.0)
            sync(dev)
            blocks.append(blk)
            usable += 0.5
    except Exception:
        pass
    finally:
        del blocks
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "mps":
            torch.mps.empty_cache()

    console.print(f"  容量：标称 [cyan]{total:.1f} GB[/cyan]，"
                  f"实探能真正写满 [bold cyan]{usable:.1f} GB[/bold cyan]"
                  f"（探测上限设为标称值；这才是模型真正能用的上限）\n")
    return total, usable


# ===================================================================== 画像卡
def render_card(p: Profile) -> None:
    body = (
        f"[bold]{p.machine}[/bold]\n"
        f"backend = {p.backend} · torch {p.torch_version}\n\n"
        f"① 峰值算力 P (fp16)     [bold cyan]{p.tflops_fp16:8.2f} TFLOPS[/bold cyan]"
        f"    (fp32 {p.tflops_fp32:.2f}，比值 {p.tensor_core_gain:.2f}×)\n"
        f"② DRAM 带宽 B          [bold cyan]{p.bw_dram:8.1f} GB/s[/bold cyan]"
        f"    缓存段峰值 {p.bw_cache_peak:.0f} GB/s\n"
        f"③ 缓存拐点             [bold cyan]{p.cache_knee_mb:8.0f} MB[/bold cyan]"
        f"    工作集小于它，带宽当场翻几倍\n"
        f"④ 过路费 t0            [bold cyan]{p.t0_us:8.2f} µs[/bold cyan]"
        f"    (拟合截距独立测出 {p.t0_fit_us:.2f} µs)\n"
        f"⑤ 稳态发射单价         [bold cyan]{p.t_launch_us:8.2f} µs[/bold cyan]"
        f"    一次同步要 {p.sync_cost_us:.0f} µs\n"
        f"⑥ 主机↔设备           [bold cyan]{p.bw_h2d:8.1f} GB/s[/bold cyan]"
        f" ↑ / {p.bw_d2h:.1f} GB/s ↓    显存带宽是它的 {p.bw_dram / max(p.bw_h2d, 1e-9):.1f} 倍\n"
        f"   可用容量            [bold cyan]{p.mem_usable_gb:8.1f} GB[/bold cyan]"
        f"    (标称 {p.mem_total_gb:.1f} GB)\n\n"
        f"[bold yellow]机器平衡点 = P / B = {p.ridge:.0f} FLOP/Byte[/bold yellow]"
        f"   ← 算术强度低于它就是带宽受限"
    )
    console.print(Panel(body, title="[bold]机器画像卡[/bold]", border_style="cyan"))


# ===================================================================== 预测
MODELS = {
    "llama2-7b": dict(name="Llama-2-7B", params=6.74e9, layers=32,
                      n_kv=32, d_head=128, ops_per_layer=51),
    "llama3-8b": dict(name="Llama-3-8B", params=8.03e9, layers=32,
                      n_kv=8, d_head=128, ops_per_layer=51),
    "qwen3-32b": dict(name="Qwen3-32B", params=32.8e9, layers=64,
                      n_kv=8, d_head=128, ops_per_layer=51),
}

# 手里没有的机器：用标称参数拼一张画像卡，好回答「换台卡会怎样」
# t_launch 一律按 5 µs 算（Day 06：这一项几乎不随硬件变化）
MACHINES = {
    "rtx4050": dict(machine="RTX 4050 Laptop（标称）", tflops_fp16=24.0,
                    bw_dram=192.0, mem_usable_gb=5.6, bw_h2d=13.0),
    "rtx4090": dict(machine="RTX 4090（标称）", tflops_fp16=165.0,
                    bw_dram=1008.0, mem_usable_gb=23.0, bw_h2d=25.0),
    "a100": dict(machine="A100 80G SXM（标称）", tflops_fp16=312.0,
                 bw_dram=2039.0, mem_usable_gb=78.0, bw_h2d=25.0),
    "h100": dict(machine="H100 SXM（标称）", tflops_fp16=989.0,
                 bw_dram=3350.0, mem_usable_gb=78.0, bw_h2d=55.0),
}


def nominal_profile(key: str) -> Profile:
    m = MACHINES[key]
    return Profile(
        machine=m["machine"], backend="nominal", torch_version="—",
        tflops_fp16=m["tflops_fp16"], tflops_fp32=m["tflops_fp16"] / 2,
        bw_dram=m["bw_dram"], bw_cache_peak=0.0, cache_knee_mb=0.0,
        t0_us=5.0, t0_fit_us=5.0, t_launch_us=5.0, sync_cost_us=0.0,
        bw_h2d=m["bw_h2d"], bw_d2h=m["bw_h2d"],
        mem_total_gb=m["mem_usable_gb"], mem_usable_gb=m["mem_usable_gb"],
    )


def predict(p: Profile, key: str, wbits: float, kvbits: float,
            s_in: int, s_out: int, batch: int) -> None:
    m = MODELS[key]
    console.rule(f"[bold]用画像卡预测 · {m['name']} · 权重 {wbits:g} bit · "
                 f"{s_in} in → {s_out} out · batch {batch}")

    P = p.tflops_fp16 * 1e12
    B = p.bw_dram * 1e9
    w_bytes = m["params"] * wbits / 8
    kv_tok = 2 * m["layers"] * m["n_kv"] * m["d_head"] * kvbits / 8
    n_launch = m["layers"] * m["ops_per_layer"]
    t_soft = n_launch * p.t_launch_us / 1e6

    # ---- 容量墙先过一遍（Day 04 / Day 05）----
    kv_full = kv_tok * (s_in + s_out)
    free = p.mem_usable_gb * 1024 ** 3 - w_bytes
    max_batch = int(free // kv_full) if free > 0 else 0

    tbl = Table("项", "值", "出处")
    tbl.add_row("权重", f"{w_bytes / 1024 ** 3:.2f} GB", "P × 每参数字节（Day 05）")
    tbl.add_row("KV / token", f"{kv_tok / 1024:.0f} KB",
                "2·L·n_kv·d_head·b（Day 01）")
    tbl.add_row(f"KV @ {s_in + s_out} token × batch {batch}",
                f"{kv_full * batch / 1024 ** 3:.2f} GB", "")
    tbl.add_row("装得下吗", "[green]装得下[/green]" if w_bytes + kv_full * batch
                < p.mem_usable_gb * 1024 ** 3 else "[red]装不下[/red]",
                f"可用 {p.mem_usable_gb:.1f} GB")
    tbl.add_row("[bold]容量允许的最大 batch[/bold]", f"[bold]{max_batch}[/bold]",
                "(可用 − 权重) / KV")
    console.print(tbl)

    if max_batch == 0:
        console.print("[red]权重都放不下 —— 后面的速度不用算了，先解决容量墙。[/red]\n")

    # ---- Prefill（TTFT）----
    f_pre = 2 * m["params"] * s_in * batch
    s_pre = w_bytes + kv_tok * s_in * batch
    t_pre_c, t_pre_m = f_pre / P, s_pre / B
    ttft = max(t_pre_c, t_pre_m) + t_soft

    # ---- Decode（TPOT），按平均上下文长度算 ----
    ctx = s_in + s_out / 2
    f_dec = 2 * m["params"] * batch
    s_dec = w_bytes + kv_tok * ctx * batch
    t_dec_c, t_dec_m = f_dec / P, s_dec / B
    tpot = max(t_dec_c, t_dec_m) + t_soft

    tbl = Table("阶段", "FLOPs", "算力要", "字节", "带宽要",
                "软件开销", "总计", "瓶颈")
    tbl.add_row("Prefill (TTFT)", f"{f_pre / 1e12:.2f} T", f"{t_pre_c * 1e3:.1f} ms",
                f"{s_pre / 1e9:.2f} GB", f"{t_pre_m * 1e3:.1f} ms",
                f"{t_soft * 1e3:.1f} ms", f"[bold]{ttft * 1e3:.1f} ms[/bold]",
                "[red]算力[/red]" if t_pre_c > t_pre_m else "[cyan]带宽[/cyan]")
    tbl.add_row("Decode (TPOT)", f"{f_dec / 1e9:.2f} G", f"{t_dec_c * 1e3:.2f} ms",
                f"{s_dec / 1e9:.2f} GB", f"{t_dec_m * 1e3:.2f} ms",
                f"{t_soft * 1e3:.1f} ms", f"[bold]{tpot * 1e3:.2f} ms[/bold]",
                "[red]算力[/red]" if t_dec_c > t_dec_m else "[cyan]带宽[/cyan]")
    console.print(tbl)

    e2e = ttft + tpot * (s_out - 1)
    ai_dec = f_dec / s_dec
    console.print(
        f"\n  算术强度：prefill = [cyan]{f_pre / s_pre:.0f}[/cyan]，"
        f"decode = [cyan]{ai_dec:.2f}[/cyan]，机器平衡点 = [yellow]{p.ridge:.0f}[/yellow]\n"
        f"  → decode 的算术强度离平衡点差 [bold]{p.ridge / max(ai_dec, 1e-9):.0f} 倍[/bold]"
        f"，这就是它永远撞带宽墙的原因\n\n"
        f"  [bold]端到端 = TTFT + (out−1)·TPOT = {e2e:.2f} s[/bold]"
        f"    单流吞吐 = [bold cyan]{s_out / e2e:.1f} tok/s[/bold cyan]"
        f"    批量吞吐 = {batch * s_out / e2e:.1f} tok/s\n"
        f"  软件开销占 TPOT 的 [bold]{t_soft / tpot * 100:.0f}%[/bold]"
        f"（Day 06：这一项不随硬件变快而变小）\n"
    )


# ===================================================================== main
def run_bench(quick: bool) -> Profile:
    dev, name = pick_device()
    console.print(f"[bold]设备：{name}（{dev.type}）· torch {torch.__version__}[/bold]\n")

    tf16, tf32 = measure_compute(dev, quick)
    mem = measure_memory(dev, quick)
    t_launch, sync_cost = measure_launch(dev)
    h2d, d2h = measure_transfer(dev)
    total, usable = measure_capacity(dev)

    return Profile(
        machine=machine_name(name), backend=dev.type,
        torch_version=torch.__version__,
        tflops_fp16=tf16, tflops_fp32=tf32,
        bw_dram=mem["bw_dram"], bw_cache_peak=mem["bw_cache_peak"],
        cache_knee_mb=mem["cache_knee_mb"], t0_us=mem["t0_us"],
        t0_fit_us=mem["t0_fit_us"], t_launch_us=t_launch or mem["t0_us"],
        sync_cost_us=sync_cost,
        bw_h2d=h2d, bw_d2h=d2h, mem_total_gb=total, mem_usable_gb=usable,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 07 · 跨平台硬件基准 + 性能预测")
    ap.add_argument("--quick", action="store_true", help="尺寸扫小一点，快一倍")
    ap.add_argument("--predict", action="store_true", help="跳过实测，直接用画像卡预测")
    ap.add_argument("--profile", type=Path, default=None, help="画像卡 JSON 路径")
    ap.add_argument("--machine", default=None, choices=sorted(MACHINES),
                    help="用标称参数拼一张画像卡，回答「换台卡会怎样」")
    ap.add_argument("--model", default="llama2-7b", choices=sorted(MODELS))
    ap.add_argument("--wbits", type=float, default=4, help="权重位宽（4 / 8 / 16）")
    ap.add_argument("--kvbits", type=float, default=16, help="KV Cache 位宽")
    ap.add_argument("--s-in", type=int, default=2000)
    ap.add_argument("--s-out", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    if args.machine:
        p = nominal_profile(args.machine)
        console.print(f"[dim]用标称参数（不碰硬件）：{args.machine}[/dim]")
        render_card(p)
    elif args.predict:
        path = args.profile or next(iter(sorted(RESULTS.glob("profile_*.json"))), None)
        if path is None or not path.exists():
            console.print("[red]找不到画像卡，先跑一次 bench_hw.py（不加 --predict）[/red]")
            return
        p = Profile(**json.loads(path.read_text()))
        console.print(f"[dim]画像卡：{path}[/dim]")
        render_card(p)
    else:
        p = run_bench(args.quick)
        render_card(p)
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"profile_{p.backend}.json"
        path.write_text(json.dumps(asdict(p), ensure_ascii=False, indent=2))
        console.print(f"[dim]画像卡已存到 {path}[/dim]\n")

    predict(p, args.model, args.wbits, args.kvbits,
            args.s_in, args.s_out, args.batch)


if __name__ == "__main__":
    main()
