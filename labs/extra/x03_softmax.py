"""附加 X03：稳定 softmax、温度与概率；全部使用 CPU 小张量。

    uv run python labs/extra/x03_softmax.py
    uv run python labs/extra/x03_softmax.py --exp B
    uv run python labs/extra/x03_softmax.py --check
"""

from __future__ import annotations

import argparse
import math

import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def centered_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """沿最后一维减最大值；允许 -inf mask，但拒绝整行被 mask。"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not logits.is_floating_point():
        raise ValueError("logits must have a floating-point dtype")
    if logits.ndim == 0 or logits.numel() == 0:
        raise ValueError("logits must contain nonempty rows")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("NaN and +inf logits are not supported")
    work = logits if logits.dtype == torch.float64 else logits.float()
    maximum = work.amax(dim=-1, keepdim=True)
    if torch.isneginf(maximum).any():
        raise ValueError("each row must contain at least one unmasked finite logit")
    return (work - maximum) / temperature


def stable_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """返回 FP32 概率；FP64 输入保留 FP64，输入形状不变。"""
    shifted = centered_logits(logits, temperature)
    weights = shifted.exp()
    return weights / weights.sum(dim=-1, keepdim=True)


def stable_log_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    shifted = centered_logits(logits, temperature)
    return shifted - shifted.exp().sum(dim=-1, keepdim=True).log()


def exp_a_temperature() -> None:
    console.rule("[bold]A · 分数差、概率比与温度")
    logits = torch.tensor([0.0, math.log(2), math.log(4)], dtype=torch.float64)
    table = Table("温度", "p(A)", "p(B)", "p(C)", "p(C)/p(A)", "熵 (nat)")
    for temperature in (0.5, 1.0, 2.0, 10.0):
        probs = stable_softmax(logits, temperature)
        entropy = -(probs * stable_log_softmax(logits, temperature)).sum().item()
        table.add_row(str(temperature), *(f"{value:.6f}" for value in probs.tolist()),
                      f"{(probs[2] / probs[0]).item():.4f}", f"{entropy:.4f}")
    console.print(table)
    shifted = stable_softmax(logits + 1000)
    error = (shifted - stable_softmax(logits)).abs().max().item()
    console.print(f"共同加 1000，最大概率差 = {error:.2e}（浮点舍入范围内）")
    console.print("温度改变分布，不改变 argmax；只有采样时才引入随机性。")


def exp_b_numerics() -> None:
    console.rule("[bold]B · 溢出、下溢与 log-softmax")
    table = Table("输入", "直接 exp 再除总和", "稳定 softmax (FP32)")
    cases = [
        ("FP16 [10, 11, 12]", torch.tensor([10.0, 11.0, 12.0], dtype=torch.float16)),
        ("FP32 [-1002, -1001, -1000]", torch.tensor([-1002.0, -1001.0, -1000.0])),
    ]
    for label, logits in cases:
        weights = logits.exp()
        naive = weights / weights.sum()
        table.add_row(label, Text(str(naive.tolist())),
                      Text(str([round(value, 6) for value in stable_softmax(logits).tolist()])))
    console.print(table)

    logits = torch.tensor([-1000.0, 0.0])
    console.print(f"log(softmax([-1000, 0])) = {stable_softmax(logits).log().tolist()}")
    console.print(f"log_softmax([-1000, 0]) = {stable_log_softmax(logits).tolist()}")
    console.print("目标类别为第 0 项：先算概率再取 log 得到 inf 损失，直接 log-softmax 得到 1000。")

    ones = torch.ones(70000, dtype=torch.float16)
    console.print(f"70000 个 exp(0) 的 FP16 总和 = {ones.sum().item()}；"
                  f"FP32 总和 = {ones.sum(dtype=torch.float32).item():.0f}")
    logits = torch.tensor([2048.0, 2049.0])
    console.print(f"原始 [2048, 2049] 的概率 = {stable_softmax(logits).tolist()}")
    console.print(f"先转 FP16 得 {logits.half().tolist()}，再升 FP32 的概率 = "
                  f"{stable_softmax(logits.half()).tolist()}")
    console.print("升精度能保护后续计算，但不能找回输入中已经丢失的分数差。")


def exp_c_rows_and_masks() -> None:
    console.rule("[bold]C · 沿候选维归一化，mask 必须先于 softmax")
    scores = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]])
    allowed = torch.ones(3, 3, dtype=torch.bool).tril()
    probs = stable_softmax(scores.masked_fill(~allowed, -torch.inf))
    table = Table("Query 行", "Key 0", "Key 1", "Key 2", "行和")
    for index, row in enumerate(probs):
        table.add_row(str(index), *(f"{value:.6f}" for value in row.tolist()),
                      f"{row.sum().item():.6f}")
    console.print(table)
    wrong = stable_softmax(scores).masked_fill(~allowed, 0)
    console.print(f"先 softmax 后直接清零，行和变成 {wrong.sum(dim=-1).tolist()}")
    try:
        stable_softmax(torch.full((1, 3), -torch.inf))
    except ValueError as error:
        console.print(f"全 mask 行：明确报错：{error}")


def verify() -> None:
    generator = torch.Generator().manual_seed(3)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for shape in ((5,), (2, 5), (2, 3, 5)):
            logits = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
            work = logits if dtype == torch.float64 else logits.float()
            for temperature in (0.5, 1.0, 2.0):
                probs = stable_softmax(logits, temperature)
                reference = torch.softmax(work / temperature, dim=-1)
                torch.testing.assert_close(probs, reference)
                torch.testing.assert_close(stable_log_softmax(logits, temperature),
                                           torch.log_softmax(work / temperature, dim=-1))
                torch.testing.assert_close(probs.sum(dim=-1),
                                           torch.ones(shape[:-1], dtype=work.dtype))
                assert probs.shape == logits.shape

    logits = torch.tensor([0.0, math.log(2), math.log(4)], dtype=torch.float64)
    torch.testing.assert_close(stable_softmax(logits), logits.new_tensor([1, 2, 4]) / 7)
    torch.testing.assert_close(stable_softmax(logits + 1000), stable_softmax(logits))
    torch.testing.assert_close(stable_softmax(logits, 0.5), logits.new_tensor([1, 4, 16]) / 21)
    torch.testing.assert_close(stable_softmax(logits, 2),
                               logits.new_tensor([1, math.sqrt(2), 2]) / (3 + math.sqrt(2)))

    masked = torch.tensor([[1.0, -torch.inf, 2.0], [-torch.inf, 3.0, -torch.inf]])
    torch.testing.assert_close(stable_softmax(masked), torch.softmax(masked, dim=-1))
    assert (stable_softmax(masked)[torch.isneginf(masked)] == 0).all()
    torch.testing.assert_close(stable_log_softmax(masked), torch.log_softmax(masked, dim=-1))
    extreme = torch.tensor([[-1000.0, 0.0], [1000.0, 1001.0]])
    targets = torch.tensor([0, 1])
    losses = -stable_log_softmax(extreme)[torch.arange(2), targets]
    torch.testing.assert_close(losses, torch.nn.functional.cross_entropy(
        extreme, targets, reduction="none"))
    assert losses[0].item() == 1000.0

    differentiable = logits.clone().requires_grad_()
    loss = -stable_log_softmax(differentiable, 2.0)[1]
    loss.backward()
    expected = (stable_softmax(differentiable.detach(), 2.0) - logits.new_tensor([0, 1, 0])) / 2
    torch.testing.assert_close(differentiable.grad, expected)

    invalid_inputs = [torch.tensor(1.0), torch.empty(2, 0), torch.tensor([1, 2]),
                      torch.tensor([float("nan"), 0.0]), torch.tensor([float("inf"), 0.0]),
                      torch.tensor([[0.0, 1.0], [-torch.inf, -torch.inf]])]
    invalid_cases = [(value, 1.0) for value in invalid_inputs]
    invalid_cases += [(logits, value) for value in (0.0, -1.0, math.inf, math.nan)]
    for function in (stable_softmax, stable_log_softmax):
        for value, temperature in invalid_cases:
            try:
                function(value, temperature)
            except ValueError:
                continue
            raise AssertionError("invalid input accepted")
    print("PASS: 36 dtype/shape/temperature cases; shift, ratios, masks, "
          "log-loss, gradient and 20 invalid-input checks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify()
        return
    for name, function in (("A", exp_a_temperature), ("B", exp_b_numerics),
                           ("C", exp_c_rows_and_masks)):
        if args.exp in ("all", name):
            function()


if __name__ == "__main__":
    main()