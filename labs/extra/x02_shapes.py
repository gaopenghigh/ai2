"""附加课 X02 实验 · 矩阵与形状

五件事：
  A  手写矩阵乘的三种读法（行×列 / 列的线性组合 / 外积求和），结果必须一模一样
  B  形状追踪器：让一个 token 走过一层注意力，把每一步的形状打出来
  C  nn.Linear 的真相：它存的到底是什么形状，前向到底算的是什么
  D  转置的代价：连续 vs 非连续，实测差多少（接 Day 03/04）
  E  batch 与算术强度：实测 AI ≈ batch size，并看它什么时候越过平衡点

用法：
    uv run python labs/extra/x02_shapes.py
    uv run python labs/extra/x02_shapes.py --exp A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "day01"))
from probe import pick_device, sync  # noqa: E402

console = Console()


# =============================================================== A 三种读法
def matmul_row_col(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """读法一：C[i,j] = A 的第 i 行 · B 的第 j 列。教科书版。"""
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, f"中间那个维度对不上：{k} vs {k2}"
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = sum(A[i, t] * B[t, j] for t in range(k))   # 一个点积
    return C


def matmul_col_combo(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """读法二：C 的第 j 列 = A 的各列按 B[:,j] 加权求和。变换的视角。"""
    n, k = A.shape
    _k, m = B.shape
    C = np.zeros((n, m))
    for j in range(m):
        for t in range(k):
            C[:, j] += A[:, t] * B[t, j]      # A 的第 t 列，权重是 B[t,j]
    return C


def matmul_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """读法三：C = Σ_t (A 的第 t 列) ⊗ (B 的第 t 行)。硬件分块的视角。"""
    n, k = A.shape
    _k, m = B.shape
    C = np.zeros((n, m))
    for t in range(k):
        C += np.outer(A[:, t], B[t, :])       # 一次外积，得到一个完整的 n×m
    return C


def exp_a_three_readings() -> None:
    console.rule("[bold]A · 矩阵乘的三种读法，结果必须一模一样")

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(3, 5))
    ref = A @ B

    tbl = Table("读法", "一句话概括", "和 numpy 的最大误差", "这个视角对应什么",
                title=f"A{A.shape} @ B{B.shape} = C{ref.shape}")
    for name, fn, idea, use in (
        ("① 行 × 列", matmul_row_col, "C[i,j] = A 第 i 行 · B 第 j 列",
         "教科书 / X01 的点积"),
        ("② 列的线性组合", matmul_col_combo, "C 的第 j 列 = A 各列的加权和",
         "「矩阵 = 一个变换」"),
        ("③ 外积求和", matmul_outer, "C = Σ (A 的一列) ⊗ (B 的一行)",
         "[bold]硬件分块[/bold]（Day 03 §6）"),
    ):
        out = fn(A, B)
        tbl.add_row(name, idea, f"{np.abs(out - ref).max():.2e}", use)
    console.print(tbl)

    console.print(
        "\n[bold yellow]三种读法是同一个计算，但对应三种完全不同的思维方式：[/bold yellow]\n"
        "  ① [bold]行 × 列[/bold] —— 每个输出格子独立地做一个点积。\n"
        "     好处：直观。坏处：n×m 个点积各读各的，[bold]数据复用为零[/bold]（X01 §7.5 的 0.5）。\n\n"
        "  ② [bold]列的线性组合[/bold] —— 输出的每一列，都是输入各列的一种「配方」。\n"
        "     这才是「矩阵 = 一个线性变换」的真正含义。\n\n"
        "  ③ [bold]外积求和[/bold] —— 沿着中间那个维度 k 一段一段地累加。\n"
        "     [bold]这正是 Day 03 Q4 里「沿 K 方向切成 32 列一段」的做法[/bold]：\n"
        "     每段只需搬进 A 的几列和 B 的几行，就能更新整个输出块。\n"
    )

    # 读法二的直接后果：B 的第 j 列只管 C 的第 j 列，所以能竖着劈开分给多张卡
    B1, B2 = B[:, :2], B[:, 2:]
    split = np.concatenate([A @ B1, A @ B2], axis=1)
    console.print(
        "[bold]验证读法二的一个直接后果 —— 权重可以按列切开：[/bold]\n"
        f"  A @ [B₁ | B₂]  vs  [A@B₁ | A@B₂]   最大误差 = {np.abs(split - ref).max():.1e}\n"
        "  这就是张量并行里的[bold]列并行[/bold]，也是结构化剪枝能「整列砍掉」的依据。\n"
    )

    # 读法三为什么是硬件唯一选择：数一数「搬多少个数」换来「多少次乘加」
    n = m = 128
    k = 4096
    b = 2                                    # fp16
    reuse = Table("读法", "一次读进", "一次干的乘加", "复用（乘加/数）", "fp16 算术强度",
                  title=f"数据复用对比（n=m={n}, k={k}）")
    reuse.add_row("① 行 × 列（算一个格子）", f"{2*k}", f"{k}",
                  f"{k/(2*k):.1f}", f"{2*k/(2*k*b):.1f}  [dim]← X01 那个 0.5[/dim]")
    reuse.add_row("③ 外积（做一层）", f"[bold]{n+m}[/bold]", f"[bold]{n*m}[/bold]",
                  f"[bold]{n*m/(n+m):.0f}[/bold]",
                  f"[bold]{2*n*m/((n+m)*b):.0f}[/bold]  [dim]← 越过 ridge 41/125[/dim]")
    console.print(reuse)
    console.print(
        f"  同样的计算量，读法③ [bold]少搬 {(n*m/(n+m))/0.5:.0f} 倍[/bold]的数据。\n"
        f"  代价：{n}×{m} 的 C 块 = {n*m} 个累加器全程不下片，"
        f"256 线程平摊每人 {n*m//256} 个 —— [bold]这就是 Day 03 Q2 里寄存器先撞墙的原因[/bold]。\n"
    )

    # 证明里用了「求和可任意分组」，但浮点加法不满足结合律 —— 看看代价
    def same_bits(x: np.ndarray, y: np.ndarray) -> bool:
        return np.array_equal(x.view(np.int64), y.view(np.int64))

    r1, r2, r3 = matmul_row_col(A, B), matmul_col_combo(A, B), matmul_outer(A, B)
    console.print(
        "[bold]数学上严格相等，浮点下呢？[/bold]\n"
        f"  (1.0 + 1e16) + (-1e16) = {(1.0 + 1e16) + -1e16}"
        f"      1.0 + (1e16 + (-1e16)) = {1.0 + (1e16 + -1e16)}"
        "   [dim]← 浮点加法不满足结合律[/dim]\n"
        f"  ① vs ② 逐位相同：{same_bits(r1, r2)}   ② vs ③ 逐位相同：{same_bits(r2, r3)}"
        "   [dim]← 三种读法对 t 的累加顺序一样[/dim]\n"
        f"  ① vs numpy@ 逐位相同：{same_bits(r1, ref)}"
        "        [dim]← BLAS 分块/向量化/多线程，改了求和顺序[/dim]\n"
        "  [bold yellow]这就是换 kernel、改 batch size、开 TF32 会让输出微微变化的原因。[/bold yellow]\n"
    )


# =============================================================== B 形状追踪
class ShapeTracer:
    def __init__(self, title: str) -> None:
        self.tbl = Table("步骤", "干了什么", "输入形状", "权重形状", "输出形状",
                         "这些数是什么", title=title)
        self.n = 0

    def step(self, what: str, x_in, w, x_out, note: str) -> None:
        self.n += 1
        fmt = lambda t: "—" if t is None else str(tuple(t.shape))   # noqa: E731
        self.tbl.add_row(str(self.n), what, fmt(x_in), fmt(w), fmt(x_out), note)


def exp_b_shape_tracer() -> None:
    console.rule("[bold]B · 形状追踪器：一句话走过一层注意力")

    T, d, h = 4, 64, 8            # 4 个 token，特征 64 维，8 个头
    dh = d // h                   # 每个头 8 维
    torch.manual_seed(0)

    x = torch.randn(T, d)
    w_q = torch.randn(d, d)
    w_k = torch.randn(d, d)
    w_v = torch.randn(d, d)
    w_o = torch.randn(d, d)

    tr = ShapeTracer(f"T={T} 个 token · d={d} 维 · h={h} 个头 · 每头 dh={dh} 维")

    tr.step("输入", None, None, x, "一行一个 token")
    q = x @ w_q
    tr.step("Q 投影  x @ Wq", x, w_q, q, "中间的 d 被消掉")
    k = x @ w_k
    tr.step("K 投影  x @ Wk", x, w_k, k, "同上")
    v = x @ w_v
    tr.step("V 投影  x @ Wv", x, w_v, v, "同上")

    qh = q.view(T, h, dh).transpose(0, 1)
    tr.step("拆头 view+transpose", q, None, qh, "d 拆成 (h,dh)，h 提到最前")
    kh = k.view(T, h, dh).transpose(0, 1)
    vh = v.view(T, h, dh).transpose(0, 1)

    att = qh @ kh.transpose(-2, -1)
    tr.step("QKᵀ  批量矩阵乘", qh, None, att, "h 只是「有多少批」")
    att = att / dh ** 0.5
    tr.step("除 √dh", att, None, att, "形状不变（X01 §6.2）")
    att = att.softmax(-1)
    tr.step("softmax(-1)", att, None, att, "沿最后一维归一化")

    oh = att @ vh
    tr.step("@ V", att, None, oh, "中间的 T 被消掉")
    o = oh.transpose(0, 1).reshape(T, d)
    tr.step("合头 transpose+reshape", oh, None, o, "h 塞回最后一维")
    y = o @ w_o
    tr.step("O 投影  o @ Wo", o, w_o, y, "★ 回到输入的形状")

    console.print(tr.tbl)

    console.print(
        "\n[bold yellow]看懂这张表只需要两条规则：[/bold yellow]\n"
        "  ① [bold]最后一维 = 「每个东西有多少个数」，前面所有维度 = 「有多少个东西」[/bold]\n"
        "     `(T, d)` = T 个 token，每个 d 个数\n"
        "     `(h, T, dh)` = h 个头，每个头下面 T 个 token，每个 token dh 个数\n\n"
        "  ② [bold]矩阵乘只动最后两维[/bold]，前面的维度原样保留（这叫「批量矩阵乘」）。\n"
        "     `(h,T,dh) @ (h,dh,T)` → `(h,T,T)`，h 从头到尾没参与运算。\n\n"
        "  [bold]★ 注意第 1、11 步的形状完全一样 —— 这是残差连接能直接相加的前提。[/bold]\n"
        "  一层 Transformer 进去什么形状、出来就是什么形状，所以可以摞 32 层。\n"
    )


# =============================================================== C nn.Linear
def exp_c_linear_truth() -> None:
    console.rule("[bold]C · nn.Linear 存的到底是什么形状")

    d_in, d_out, T = 5, 3, 4
    torch.manual_seed(0)
    lin = torch.nn.Linear(d_in, d_out, bias=True)
    x = torch.randn(T, d_in)

    tbl = Table("对象", "形状", "说明", title=f"nn.Linear(in={d_in}, out={d_out})")
    tbl.add_row("输入 x", str(tuple(x.shape)), f"{T} 个样本，每个 {d_in} 维")
    tbl.add_row("[bold]lin.weight[/bold]", f"[bold]{tuple(lin.weight.shape)}[/bold]",
                "[bold]注意是 (out, in)，不是 (in, out)！[/bold]")
    tbl.add_row("lin.bias", str(tuple(lin.bias.shape)), "每个输出神经元一个偏置")
    tbl.add_row("输出 lin(x)", str(tuple(lin(x).shape)), f"{T} 个样本，每个 {d_out} 维")
    console.print(tbl)

    y_ref = lin(x)
    y_manual = x @ lin.weight.T + lin.bias
    console.print(
        f"\n手动算 [cyan]x @ weight.T + bias[/cyan]，"
        f"和 lin(x) 的最大误差 = [bold]{(y_manual - y_ref).abs().max().item():.2e}[/bold]"
        "  → 完全一致\n"
    )

    console.print(
        "[bold yellow]为什么要存转置？两个理由：[/bold yellow]\n"
        "  ① [bold]数学习惯是列向量[/bold]：教科书写 y = Wx，x 是竖着的 (in,1)，W 是 (out,in)。\n"
        "     PyTorch 沿用了 W 的形状，但把数据改成了「一行一个样本」。\n\n"
        "  ② [bold]更实际的理由：访存[/bold]（接 Day 03 §7 / Day 04）。\n"
        "     张量是[bold]按行连续存储[/bold]的。存成 (out, in) 意味着\n"
        "     [bold]「第 i 个输出神经元的全部权重」在内存里是连续的一段[/bold]。\n"
        "     算 y[i] 时正好顺序读一整段 —— cache line 一个字节都不浪费。\n\n"
        "  ③ [bold]转置本身不花钱[/bold]：底层 GEMM 有 transA/transB 两个标志位，\n"
        "     `x @ W.T` 只是把标志置上，[bold]不会真的搬数据[/bold]。下面实验 D 会验证。\n"
    )


# =============================================================== D 转置的代价
def exp_d_transpose_cost(dev: torch.device) -> None:
    console.rule("[bold]D · 转置到底花不花钱")

    n = 4096
    torch.manual_seed(0)
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    x = torch.randn(n, n, device=dev, dtype=dtype)
    w = torch.randn(n, n, device=dev, dtype=dtype)      # (out, in)，模仿 nn.Linear
    w_t = w.T.contiguous()                              # 提前转置好并连续化

    def bench(fn, iters=20):
        for _ in range(5):
            fn()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(dev)
        return (time.perf_counter() - t0) / iters * 1e3

    t_lazy = bench(lambda: x @ w.T)
    t_pre = bench(lambda: x @ w_t)
    t_mat = bench(lambda: x @ w)

    tbl = Table("写法", "耗时", "相对", "说明", title=f"{n}×{n} 矩阵乘")
    tbl.add_row("x @ w.T（懒转置）", f"{t_lazy:7.2f} ms", f"{t_lazy/t_mat:5.2f}×",
                "只设了个标志位，没搬数据")
    tbl.add_row("x @ w_t（提前转置好）", f"{t_pre:7.2f} ms", f"{t_pre/t_mat:5.2f}×",
                "数据已经是连续的")
    tbl.add_row("x @ w（不转置，作参照）", f"{t_mat:7.2f} ms", "1.00×", "基准")
    console.print(tbl)

    # ---- 但「显式搬一次」是要花钱的 ----
    t_contig = bench(lambda: w.T.contiguous(), iters=20)
    bw = bytes_moved = 2 * n * n * x.element_size()
    eff = bytes_moved / (t_contig / 1e3) / 1e9
    console.print(
        f"\n而真正[bold]搬一遍数据[/bold]（`w.T.contiguous()`）要 "
        f"[bold]{t_contig:.2f} ms[/bold]，"
        f"等效带宽只有 [bold]{eff:.0f} GB/s[/bold]\n"
    )

    console.print(
        "[bold yellow]结论：[/bold yellow]\n"
        "  · [bold]`.T` 本身是免费的[/bold] —— 它只改了张量的 stride 元数据，一个字节都没动。\n"
        "  · [bold]`.contiguous()` 才要钱[/bold] —— 那是一次实打实的全量读写（Day 04 的账）。\n"
        f"  · 而且注意它只跑到 {eff:.0f} GB/s，[bold]远低于峰值[/bold] ——\n"
        "    因为转置是「读一行、写一列」，写入端完全不连续，\n"
        "    [bold]每条 cache line 只用上几个字节[/bold]（Day 04 §6.3 的局部性问题）。\n"
        "  · 所以框架里到处是 `.transpose()` / `.permute()`，但你很少看到 `.contiguous()`。\n"
        "    [bold]什么时候不得不加？当下一个算子要求连续内存时（比如 `.view()`）。[/bold]\n\n"
        "  [dim]顺带：三种写法耗时接近，说明 GEMM 确实是靠标志位处理转置的。\n"
        "  如果差得多，说明这个后端在某些组合上要退化成先转置再算。[/dim]\n"
    )


# =============================================================== E batch 与 AI
def exp_e_batch_ai(dev: torch.device) -> None:
    console.rule("[bold]E · 算术强度 ≈ batch size：一条公式解释 Day 01 实验 2")

    d = k = 4096
    dtype = torch.float32 if dev.type == "cpu" else torch.float16
    esz = torch.empty(0, dtype=dtype).element_size()
    w = torch.randn(d, k, device=dev, dtype=dtype)

    peak = 3.5e12 if dev.type == "mps" else 24e12
    ridge = 41 if dev.type == "mps" else 125

    tbl = Table("batch B", "FLOPs", "搬运字节", "算术强度\n(理论)", "实测 TFLOPS",
                "达成率", "在屋顶的哪一侧",
                title=f"X(B,{d}) @ W({d},{k})，机器平衡点 ≈ {ridge}")
    for B in (1, 2, 8, 32, 128, 512, 2048):
        x = torch.randn(B, d, device=dev, dtype=dtype)
        for _ in range(10):
            x @ w
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(30):
            x @ w
        sync(dev)
        t = (time.perf_counter() - t0) / 30

        flops = 2 * B * d * k
        nbytes = (B * d + d * k + B * k) * esz
        ai = flops / nbytes
        tf = flops / t / 1e12
        side = "带宽受限" if ai < ridge else "算力受限"
        tbl.add_row(f"{B:>5}", f"{flops/1e9:8.2f} G", f"{nbytes/1e6:7.1f} MB",
                    f"{ai:8.1f}", f"{tf:8.2f}", f"{tf/(peak/1e12)*100:5.1f}%", side)
        del x
    console.print(tbl)

    console.print(
        "\n[bold yellow]看第 4 列：算术强度几乎就等于 batch size。[/bold yellow]\n"
        "  推导只有两行。设 X 是 (B,d)、W 是 (d,k)、每个数 b 字节：\n\n"
        "      FLOPs = 2·B·d·k\n"
        "      Bytes = (B·d + d·k + B·k)·b   ← 当 B ≪ d,k 时，中间那项（权重）压倒性地大\n"
        "            ≈ d·k·b\n"
        "      AI   ≈ 2·B·d·k / (d·k·b) = [bold]2B / b[/bold]\n\n"
        "  fp16 时 b=2，所以 [bold]AI ≈ B[/bold]。\n\n"
        "[bold]于是 Day 01 实验 2「batch 涨 512 倍，耗时几乎不变」就完全说得通了：[/bold]\n"
        "  B 很小的时候，时间全花在[bold]把 W 搬进来[/bold]，而 W 的大小和 B 无关。\n"
        "  加大 B 只是让这块搬进来的 W [bold]被多用几次[/bold] —— 几乎不花额外时间。\n"
        f"  直到 AI 越过平衡点 {ridge}（也就是 B ≈ {ridge}），才开始真正撞算力墙。\n\n"
        "[bold]★ 这就是「为什么一切都要写成矩阵」的完整答案：[/bold]\n"
        "  矩阵乘不是为了写起来好看，是为了[bold]让搬进来的权重被复用 B 次[/bold]。\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list("ABCDE"), help="只跑指定实验")
    args = ap.parse_args()

    dev, name = pick_device()
    console.print(f"[bold green]X02[/bold green] 矩阵与形状  |  "
                  f"[bold green]设备[/bold green]: {name}\n")

    if args.exp in (None, "A"):
        exp_a_three_readings()
    if args.exp in (None, "B"):
        exp_b_shape_tracer()
    if args.exp in (None, "C"):
        exp_c_linear_truth()
    if args.exp in (None, "D"):
        exp_d_transpose_cost(dev)
    if args.exp in (None, "E"):
        exp_e_batch_ai(dev)


if __name__ == "__main__":
    main()
