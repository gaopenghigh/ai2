"""Day 05 实验 · 把浮点数拆开看

五件事：
  A  手写位解析器：把一个数拆成 符号 / 指数 / 尾数，再按公式还原回去
  B  边界实测：每种格式的 max / 最小正规数 / 最小次正规数 / eps，和 torch.finfo 对账
  C  累加灾难：从 0 开始一个一个加 1，FP16 会停在哪？怎么救
  D  溢出与下溢：模拟梯度分布，看 FP16 丢了多少，loss scaling 能不能救回来
  E  量化：INT8 / INT4，per-tensor 和 per-group 差多少（含离群值）

用法：
    uv run python labs/day05/numerics.py
    uv run python labs/day05/numerics.py --exp C
"""

from __future__ import annotations

import argparse
import struct

import torch
from rich.console import Console
from rich.table import Table

console = Console()


# =============================================================== A 位解析器
def f32_bits(x: float) -> str:
    return f"{struct.unpack('>I', struct.pack('>f', x))[0]:032b}"


def f16_bits(x: float) -> str:
    return f"{struct.unpack('>H', struct.pack('>e', x))[0]:016b}"


def bf16_bits(x: float) -> str:
    """BF16 就是 FP32 的高 16 位（带舍入）。"""
    v = torch.tensor([x], dtype=torch.float32).to(torch.bfloat16)
    u = v.to(torch.float32).view(torch.int32).item() & 0xFFFFFFFF
    return f"{(u >> 16) & 0xFFFF:016b}"


def decode(bits: str, e_bits: int, m_bits: int) -> tuple[str, float]:
    """按 IEEE 754 的定义，把二进制串还原成数值。"""
    s = int(bits[0])
    e_field = int(bits[1:1 + e_bits], 2)
    m_field = int(bits[1 + e_bits:], 2)
    bias = 2 ** (e_bits - 1) - 1

    if e_field == 0:
        kind = "±0" if m_field == 0 else "次正规数"
        val = (-1) ** s * (m_field / 2 ** m_bits) * 2.0 ** (1 - bias)
    elif e_field == 2 ** e_bits - 1:
        return ("±inf" if m_field == 0 else "NaN"), float("nan")
    else:
        kind = "正规数"
        val = (-1) ** s * (1 + m_field / 2 ** m_bits) * 2.0 ** (e_field - bias)
    return kind, val


def exp_a_bit_parser() -> None:
    console.rule("[bold]A · 把一个浮点数拆开看")

    for x in (1.0, 0.1, -2.5, 65504.0, 1e-8):
        console.print(f"\n[bold cyan]x = {x}[/bold cyan]")
        tbl = Table("格式", "符号", "指数域", "实际指数", "尾数域", "类型",
                    "还原出来的值", show_edge=False)
        for name, bits, e_b, m_b in (
            ("FP32", f32_bits(x), 8, 23),
            ("BF16", bf16_bits(x), 8, 7),
            ("FP16", f16_bits(min(max(x, -65504.0), 65504.0)), 5, 10),
        ):
            kind, val = decode(bits, e_b, m_b)
            e_field = int(bits[1:1 + e_b], 2)
            bias = 2 ** (e_b - 1) - 1
            tbl.add_row(
                name, bits[0], bits[1:1 + e_b],
                f"{e_field}−{bias} = {e_field - bias}",
                bits[1 + e_b:], kind, f"{val!r}",
            )
        console.print(tbl)

    console.print(
        "\n[bold yellow]三件事值得停下来看：[/bold yellow]\n"
        "  · [bold]0.1 在任何二进制浮点里都存不准[/bold]（尾数是无限循环的 1100…），\n"
        "    FP32 存成 0.100000001490116…，这就是 0.1+0.2 != 0.3 的全部原因。\n"
        "  · [bold]BF16 的指数域和 FP32 一模一样[/bold]，只是尾数被砍掉了 16 位 ——\n"
        "    所以 FP32 转 BF16 基本就是「把低 16 位扔掉」，硬件实现极其便宜。\n"
        "  · [bold]1e-8 在 FP16 里直接变成了 0[/bold]（连最小次正规数 5.96e-8 都够不着），\n"
        "    而 BF16 毫发无损。这一条今天会反复出现。\n"
    )


# =============================================================== B 边界实测
FORMATS = [
    ("FP32", torch.float32, 8, 23),
    ("BF16", torch.bfloat16, 8, 7),
    ("FP16", torch.float16, 5, 10),
]
for _n, _a in (("FP8 E4M3", "float8_e4m3fn"), ("FP8 E5M2", "float8_e5m2")):
    if hasattr(torch, _a):
        FORMATS.append((_n, getattr(torch, _a), 4 if "E4" in _n else 5,
                        3 if "E4" in _n else 2))


def exp_b_limits() -> None:
    console.rule("[bold]B · 每种格式的四个边界，手推 vs torch.finfo")

    tbl = Table("格式 (1+e+m)", "最大值", "最小正规数", "最小次正规数",
                "eps = 2^-m", "十进制\n有效位", "和 finfo\n对得上",
                title="全部只用 (指数位 e, 尾数位 m) 两个数就能推出来")

    for name, dt, e, m in FORMATS:
        bias = 2 ** (e - 1) - 1
        if name == "FP8 E4M3":                       # OCP 变体：没有 inf
            mx = (2 - 2 ** -(m - 1)) * 2.0 ** (2 ** e - 1 - bias)
        else:
            mx = (2 - 2.0 ** -m) * 2.0 ** (2 ** e - 2 - bias)
        fi = torch.finfo(dt)
        tbl.add_row(
            f"{name} ({1+e+m}={1}+{e}+{m})", f"{mx:.4g}",
            f"{2.0 ** (1 - bias):.4g}", f"{2.0 ** (1 - bias - m):.4g}",
            f"{2.0 ** -m:.4g}", f"{(m + 1) * 0.30103:.1f}",
            "是" if abs(mx / fi.max - 1) < 1e-6 else f"否({fi.max:.3g})",
        )
    console.print(tbl)

    console.print(
        "\n[bold yellow]三个公式，背下来就够用：[/bold yellow]\n"
        "  最大值      = (2 − 2^−m) × 2^(2^e − 2 − bias)      ← 指数域全 1 要留给 inf/NaN\n"
        "  最小正规数  = 2^(1 − bias)\n"
        "  机器 epsilon = 2^−m      ← 1.0 后面的下一个数和 1.0 差多少\n\n"
        "  [bold]注意 BF16 的 eps 是 FP16 的 8 倍[/bold]（0.0078 vs 0.00098）——\n"
        "  同样 16 位，BF16 拿范围换掉了 3 位精度。\n"
    )


# =============================================================== C 累加灾难
def exp_c_accumulate() -> None:
    console.rule("[bold]C · 从 0 开始一个一个加 1，会停在哪")

    tbl = Table("dtype", "尾数位 m", "手推停住的位置 2^(m+1)", "实测停住的位置",
                title="acc = acc + 1，一直加到加不动为止")

    for name, dt, m in (("FP16", torch.float16, 10), ("BF16", torch.bfloat16, 7)):
        acc = torch.zeros((), dtype=dt)
        one = torch.ones((), dtype=dt)
        for _ in range(1 << 20):
            prev = acc.clone()
            acc = acc + one
            if bool((acc == prev).item()):
                break
        tbl.add_row(name, str(m), f"{2 ** (m + 1):,}", f"{acc.item():,.0f}")
    tbl.add_row("FP32", "23", f"{2 ** 24:,}", "太慢，不实测（同一个公式）")
    console.print(tbl)

    console.print(
        "\n[bold]为什么正好停在 2^(m+1)：[/bold]\n"
        "  数值到了 2^(m+1) 时，相邻两个可表示数的间隔正好变成 2，\n"
        "  这时 acc + 1 落在两个刻度[bold]正中间[/bold]，按「四舍六入五取偶」又被拉回原处。\n"
    )

    # ---- 同样一堆 1，换个加法就救回来了 ----
    n = 1 << 16
    x = torch.ones(n, dtype=torch.float16)
    tbl = Table("求和方式", "结果", "错在哪",
                title=f"把 {n:,} 个 1 加起来（正确答案 {n:,}）")

    acc = torch.zeros((), dtype=torch.float16)
    for v in x:
        acc = acc + v
    tbl.add_row("朴素逐个累加", f"{acc.item():,.0f}", "精度不够：加到 2048 就加不动了")

    s_tree = x.sum().item()
    tbl.add_row("torch.sum（树形累加，仍是 FP16）", f"{s_tree:,}",
                "范围不够：65536 > 65504，直接 inf")
    tbl.add_row("torch.sum(dtype=float32)（升位累加）",
                f"{x.sum(dtype=torch.float32).item():,.0f}", "正确")
    console.print(tbl)

    console.print(
        "\n[bold yellow]这三行正好演示了半精度的两种翻车方式：[/bold yellow]\n"
        "  · 第 1 行栽在[bold]精度[/bold]（尾数不够，小数被大数吃掉）；\n"
        "  · 第 2 行栽在[bold]范围[/bold]（树形求和治好了精度，但 65536 越过了 65504）；\n"
        "  · 只有[bold]升到 FP32 累加[/bold]，两个问题一起解决。\n\n"
        "[bold]三条现实推论：[/bold]\n"
        "  · [bold]Tensor Core 吃 FP16 输入，但一律用 FP32 累加[/bold] ——\n"
        "    不然一个长度 4096 的点积根本加不完。\n"
        "  · 归约类算子（sum / mean / softmax / LayerNorm）在生产代码里[bold]必须显式升位[/bold]。\n"
        "    PyTorch 不同后端的默认行为并不一致，别赌 —— 写 `dtype=torch.float32`。\n"
        "  · 树形（成对）求和把精度误差从 O(n) 降到 O(log n)，\n"
        "    因为每次相加的两个数量级总是接近的 —— [bold]但它对溢出无能为力[/bold]。\n"
    )


# =============================================================== D 溢出下溢
def exp_d_range() -> None:
    console.rule("[bold]D · 模拟梯度：FP16 会丢掉多少，loss scaling 能不能救")

    torch.manual_seed(0)
    n = 1_000_000
    # 梯度的量级在 1e-11 ~ 1e-3 之间对数均匀分布（真实训练里就是这个量级）
    g = 10.0 ** (torch.rand(n) * 8 - 11)

    def report(t: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
        zero = (t == 0).float().mean().item() * 100
        inf = (~torch.isfinite(t)).float().mean().item() * 100
        alive = (t != 0) & torch.isfinite(t)
        rel = ((t[alive].float() - ref[alive]).abs() / ref[alive]).mean().item()
        return zero, inf, rel

    tbl = Table("做法", "下溢成 0", "溢出成 inf", "活下来那些的\n平均相对误差", "点评",
                title=f"{n:,} 个梯度，量级横跨 1e-11 ~ 1e-3")

    for name, dt, note in (("直接转 FP16", torch.float16, "一半没了，而且是静默的"),
                           ("直接转 BF16", torch.bfloat16, "一个不丢，零调参")):
        z, i, r = report(g.to(dt), g)
        tbl.add_row(name, f"{z:.1f}%", f"{i:.1f}%", f"{r:.2e}", note)

    notes = {12: "不再归零，但大多还在次正规区",
             16: "追平 BF16",
             24: "拿到了 FP16 该有的精度",
             28: "放大过头，开始撞 65504"}
    for k in (12, 16, 24, 28):
        scale = 2 ** k
        t = (g * scale).to(torch.float16)
        back = t.float() / scale
        z = ((back == 0) & torch.isfinite(back)).float().mean().item() * 100
        inf = (~torch.isfinite(back)).float().mean().item() * 100
        alive = (back != 0) & torch.isfinite(back)
        r = ((back[alive] - g[alive]).abs() / g[alive]).mean().item()
        tbl.add_row(f"FP16 + loss scaling ×2^{k}", f"{z:.1f}%",
                    f"{inf:.1f}%", f"{r:.2e}", notes[k])
    console.print(tbl)

    console.print(
        "\n[bold yellow]这张表其实讲了三件事：[/bold yellow]\n"
        "  ① [bold]直接转 FP16，43% 的梯度变成 0[/bold] —— 那些参数这一步根本没更新。\n"
        "     而且它是[bold]静默的[/bold]：不报错、不出 NaN，只是模型悄悄学不动了。\n"
        "     侥幸活下来的那些也大多落在[bold]次正规区[/bold]，相对误差高达 5%（次正规数精度逐位衰减）。\n\n"
        "  ② [bold]BF16 一个都没丢[/bold]，因为它的范围和 FP32 完全一样宽。\n"
        "     相对误差稳定在 1.4e-3 ≈ eps/4，[bold]不需要任何调参[/bold]。\n\n"
        "  ③ [bold]loss scaling 不只是防下溢，还要把数推进「正规数」区间[/bold]。\n"
        "     ×2^12 时还有很多值卡在次正规区，误差反而不如 BF16；\n"
        "     ×2^24 才拿到 1.75e-4 —— [bold]比 BF16 好 8 倍，这才是 FP16 应有的精度[/bold]；\n"
        "     再往上放就撞 65504 的天花板了。\n\n"
        "[bold]结论：FP16 能比 BF16 更精确，但前提是你把 scale 调对，而且窗口很窄。[/bold]\n"
        "真实框架用[bold]动态 loss scaling[/bold]（出 inf 就减半、连续几百步正常就翻倍）来自动找这个窗口。\n"
        "[bold]BF16 用 3 位精度换掉了这整套机制 —— 这就是它成为训练默认值的原因。[/bold]\n"
    )


# =============================================================== E 量化
def _quant_dequant(w: torch.Tensor, qmax: int, group: int | None) -> torch.Tensor:
    """对称量化再反量化。group=None 表示 per-tensor。"""
    if group is None:
        s = w.abs().max() / qmax
        return torch.clamp(torch.round(w / s), -qmax, qmax) * s
    shp = w.shape
    v = w.reshape(-1, group)
    s = v.abs().amax(dim=1, keepdim=True) / qmax
    return (torch.clamp(torch.round(v / s), -qmax, qmax) * s).reshape(shp)


def exp_e_quantization() -> None:
    console.rule("[bold]E · 量化：per-tensor vs per-group，以及离群值的杀伤力")

    torch.manual_seed(0)
    d = 2048
    w = torch.randn(d, d) * 0.02                      # 典型的权重分布
    x = torch.randn(d, 64)
    ref = w @ x

    for tag, ww in (("干净的权重", w), ("注入 20 倍离群值后", None)):
        if ww is None:
            ww = w.clone()
            ww[7, 11] = 20 * w.abs().max()            # 一个离群值
            ref = ww @ x
        tbl = Table("方案", "平均量化误差", "矩阵乘输出的相对误差", "被压成 0 的比例",
                    title=f"{tag}（{d}×{d}）")
        for name, qmax, group in (("INT8 per-tensor", 127, None),
                                  ("INT8 per-group 128", 127, 128),
                                  ("INT4 per-tensor", 7, None),
                                  ("INT4 per-group 128", 7, 128),
                                  ("INT4 per-group 32", 7, 32)):
            wq = _quant_dequant(ww, qmax, group)
            e = (wq - ww).abs().mean().item()
            rel = (((wq @ x) - ref).norm() / ref.norm()).item()
            z = (wq == 0).float().mean().item() * 100
            tbl.add_row(name, f"{e:.2e}", f"{rel*100:.3f}%", f"{z:.1f}%")
        console.print(tbl)
        w, ref = ww, ref

    console.print(
        "\n[bold yellow]四个结论：[/bold yellow]\n"
        "  ① 权重干净的时候，[bold]INT8 per-tensor 就已经够用[/bold]（输出误差约 1%）。\n"
        "  ② [bold]一旦出现离群值，per-tensor 立刻崩掉[/bold] —— INT8 从 1% 掉到 23%，\n"
        "     INT4 更是[bold]把 100% 的权重全压成了 0[/bold]，输出误差 99.9%（模型彻底废了）。\n"
        "  ③ [bold]per-group 完全不受影响[/bold]（0.66% 和 11.8%，和干净时一模一样）——\n"
        "     损害被关在了那一个组里。这就是 GPTQ / AWQ / GGUF 全都用 group size 32~128 的原因：\n"
        "     每 128 个数多存 1 个 FP16 scale（64 字节里多 2 字节，[bold]+3.1% 体积[/bold]），\n"
        "     换来的是「能不能用」。\n"
        "  ④ 但注意 [bold]INT4 per-group 的输出误差仍有 ~12%[/bold] ——\n"
        "     光靠「四舍五入到最近的格子」是不够的。真正能用的 INT4 方案还需要\n"
        "     校准数据 + 误差补偿（GPTQ）或者按重要性缩放（AWQ），[dim]Week 5 会推导。[/dim]\n\n"
        "  [dim]额外代价：group 越小，反量化时要读的 scale 越多 —— 又回到 Day 04 的搬运账。[/dim]\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list("ABCDE"), help="只跑指定实验")
    args = ap.parse_args()

    console.print(f"[bold green]torch[/bold green]: {torch.__version__}  |  "
                  "[dim]本实验全部在 CPU 上跑，和设备无关[/dim]\n")

    if args.exp in (None, "A"):
        exp_a_bit_parser()
    if args.exp in (None, "B"):
        exp_b_limits()
    if args.exp in (None, "C"):
        exp_c_accumulate()
    if args.exp in (None, "D"):
        exp_d_range()
    if args.exp in (None, "E"):
        exp_e_quantization()


if __name__ == "__main__":
    main()
