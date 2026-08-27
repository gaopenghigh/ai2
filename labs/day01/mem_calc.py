"""Day 01 加餐 · 真实模型的 KV Cache 显存计算器

回答两个问题：
  1. 推理时，每多一个 token 要多花多少显存？
  2. 撑住 1M context 要多少显存？

所有模型参数来自 HuggingFace 上的真实 config.json（见每个条目的 source 字段）。

用法：
    uv run python labs/day01/mem_calc.py              # 全模型对照表
    uv run python labs/day01/mem_calc.py --ctx 128000 # 指定上下文长度
    uv run python labs/day01/mem_calc.py --budget 6   # 给定显存预算(GB)反推能存多少 token
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()

KB, MB, GB = 1024, 1024 ** 2, 1024 ** 3


# --------------------------------------------------------------------- 模型定义
@dataclass
class Model:
    name: str
    params_b: float                 # 总参数量（十亿）
    active_b: float                 # 每 token 实际激活的参数量（稠密模型 = 总量）
    layers: int
    kv_bytes_per_elem: float        # 生产环境常用的 KV dtype 字节数
    arch: str                       # 架构简述
    source: str

    # --- GQA / MHA 路线 ---
    n_kv_heads: int | None = None
    head_dim: int | None = None
    full_attn_layers: int | None = None   # 混合架构：只有这些层的 KV 会随上下文增长

    # --- MLA 路线 ---
    mla_latent: int | None = None         # 压缩后的 KV 隐向量维度
    mla_rope: int | None = None           # 解耦 RoPE key 维度
    compress_ratios: list[int] = field(default_factory=list)  # 每层的 KV 压缩倍率

    # --- 线性注意力的常数态（不随上下文增长）---
    const_state_bytes: int = 0

    def kv_elems_per_token(self) -> float:
        """每个 token 在 KV Cache 里占多少个数（不含 dtype）。"""
        if self.mla_latent is not None:
            per_layer = self.mla_latent + (self.mla_rope or 0)
            if self.compress_ratios:
                # ratio=0 表示该层不压缩
                return sum(per_layer / (r if r > 0 else 1) for r in self.compress_ratios)
            return per_layer * self.layers
        n_layers = self.full_attn_layers or self.layers
        return 2 * n_layers * self.n_kv_heads * self.head_dim   # K 和 V 各一份

    def kv_bytes_per_token(self) -> float:
        return self.kv_elems_per_token() * self.kv_bytes_per_elem

    def kv_bytes(self, ctx: int) -> float:
        return self.kv_bytes_per_token() * ctx + self.const_state_bytes

    def weight_gb(self, bytes_per_param: float) -> float:
        return self.params_b * 1e9 * bytes_per_param / GB


MODELS: list[Model] = [
    Model(
        name="Llama-2-7B",
        params_b=6.74, active_b=6.74, layers=32,
        n_kv_heads=32, head_dim=128, kv_bytes_per_elem=2,
        arch="MHA（每个 Q 头配一个 KV 头）",
        source="meta-llama/Llama-2-7b-hf",
    ),
    Model(
        name="Qwen3-32B",
        params_b=32.8, active_b=32.8, layers=64,
        n_kv_heads=8, head_dim=128, kv_bytes_per_elem=2,
        arch="GQA 8 KV 头（64 Q 头共享）",
        source="Qwen/Qwen3-32B",
    ),
    Model(
        name="Qwen3-30B-A3B",
        params_b=30.5, active_b=3.3, layers=48,
        n_kv_heads=4, head_dim=128, kv_bytes_per_elem=2,
        arch="MoE 128 专家取 8 + GQA 4 KV 头",
        source="Qwen/Qwen3-30B-A3B",
    ),
    Model(
        name="Qwen3.8-27B",
        params_b=28.0, active_b=28.0, layers=64,
        n_kv_heads=4, head_dim=256, kv_bytes_per_elem=2,
        full_attn_layers=16,                 # layer_types 里 64 层只有 1/4 是 full_attention
        const_state_bytes=48 * 48 * 128 * 128 * 4,   # 48 个线性注意力层的 fp32 循环状态
        arch="混合：48 层线性注意力 + 16 层全注意力",
        source="Qwen/Qwen3.8-27B (text_config)",
    ),
    Model(
        name="DeepSeek-V3",
        params_b=671, active_b=37, layers=61,
        mla_latent=512, mla_rope=64, kv_bytes_per_elem=1,
        arch="MLA（KV 压成 512 维隐向量）",
        source="deepseek-ai/DeepSeek-V3",
    ),
    Model(
        name="DeepSeek-V4-Flash-0731",
        params_b=304, active_b=13, layers=43,
        mla_latent=512, mla_rope=64, kv_bytes_per_elem=1,
        # config.json 的 compress_ratios：前 2 层和后 3 层不压缩，中间 CSA(4×) 与 HCA(128×) 交替
        compress_ratios=[0, 0] + [4, 128] * 19 + [0, 0, 0],
        arch="MLA + CSA(4×) / HCA(128×) 混合压缩",
        source="deepseek-ai/DeepSeek-V4-Flash-0731",
    ),
]

GPUS = [
    ("RTX 4050 Laptop", 6), ("Mac mini M4 可用", 11), ("RTX 4090", 24),
    ("A100 / H100", 80), ("8×H100 整机", 640), ("4×GB300 整机", 1152),
]


def human(b: float) -> str:
    for unit, size in (("TB", GB * 1024), ("GB", GB), ("MB", MB), ("KB", KB)):
        if b >= size:
            return f"{b / size:.1f} {unit}"
    return f"{b:.0f} B"


def table_overview(ctx: int) -> None:
    t = Table(title=f"KV Cache 显存：每 token 开销 与 {ctx:,} token 上下文总量",
              show_lines=True)
    t.add_column("模型", style="bold")
    t.add_column("架构")
    t.add_column("KV\ndtype", justify="center")
    t.add_column("每 token\nKV", justify="right")
    t.add_column(f"{ctx//1000}K ctx\nKV 总量", justify="right")
    t.add_column("相对 Llama-2-7B", justify="right")

    base = MODELS[0].kv_bytes_per_token()
    for m in MODELS:
        per = m.kv_bytes_per_token()
        dt = {1: "FP8", 2: "BF16"}[int(m.kv_bytes_per_elem)]
        t.add_row(m.name, m.arch, dt, human(per), human(m.kv_bytes(ctx)),
                  f"1/{base/per:.0f}" if per < base else "1×")
    console.print(t)


def table_1m() -> None:
    t = Table(title="撑住 1M token 上下文，光 KV Cache 要多少显存（单条序列）",
              show_lines=True)
    t.add_column("模型", style="bold")
    t.add_column("1M ctx KV", justify="right")
    t.add_column("权重 (原生精度)", justify="right")
    t.add_column("KV + 权重", justify="right")
    t.add_column("最小可行硬件", justify="left")

    ctx = 1_048_576
    for m in MODELS:
        kv = m.kv_bytes(ctx)
        # MoE 大模型按 FP8 存权重，中小模型按 BF16
        bpp = 1.0 if m.params_b > 100 else 2.0
        w = m.weight_gb(bpp) * GB
        total = kv + w
        fit = next((f"{n} ({g} GB)" for n, g in GPUS if total < g * GB * 0.9), "需要多机")
        t.add_row(m.name, human(kv), human(w), human(total), fit)
    console.print(t)
    console.print(
        "\n[dim]注：权重可被多个并发请求共享，KV Cache 每条序列一份 —— "
        "所以并发数几乎完全由 KV 容量决定。[/dim]\n"
    )


def table_budget(budget_gb: float) -> None:
    t = Table(title=f"给定 {budget_gb} GB 显存[b]全部[/b]用于 KV Cache，能存多少 token")
    t.add_column("模型", style="bold")
    t.add_column("可存 token 数", justify="right")
    t.add_column("等价于", justify="left")

    for m in MODELS:
        n = int(budget_gb * GB / m.kv_bytes_per_token())
        t.add_row(m.name, f"{n:,}", f"{n/1000:.0f}K 上下文 × 1 并发")
    console.print(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=128_000, help="上下文长度")
    ap.add_argument("--budget", type=float, help="显存预算 (GB)")
    args = ap.parse_args()

    console.print()
    table_overview(args.ctx)
    console.print()
    table_1m()
    if args.budget:
        table_budget(args.budget)

    console.print(
        "[bold yellow]一句话[/bold yellow]："
        "四年时间，同样 1M 上下文的 KV 开销降了近 [bold]100 倍[/bold] —— "
        "MHA → GQA → 混合线性注意力 → MLA → CSA/HCA。\n"
        "[dim]这不是算力进步带来的，全部来自架构对『内存墙』的针对性设计。[/dim]\n"
    )


if __name__ == "__main__":
    main()
