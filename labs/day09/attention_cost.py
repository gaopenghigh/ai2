"""Day 09: attention FLOPs and tensor payloads, without loading model weights.

    uv run python labs/day09/attention_cost.py
    uv run python labs/day09/attention_cost.py --mode decode --seq-len 2048
    uv run python labs/day09/attention_cost.py --kv-heads 8
    uv run python labs/day09/attention_cost.py --demo
    uv run python labs/day09/attention_cost.py --check
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day08"))
from param_count import Config, layer_counts

PROJECTIONS = ("Q projection", "K projection", "V projection", "O projection")


@dataclass(frozen=True)
class Workload:
    batch: int = 1
    seq_len: int = 2048
    mode: str = "prefill"
    element_bytes: int = 2

    def __post_init__(self) -> None:
        if any(value <= 0 for value in (self.batch, self.seq_len, self.element_bytes)):
            raise ValueError("batch, seq_len and element_bytes must be positive.")
        if self.mode not in ("prefill", "decode"):
            raise ValueError("mode must be prefill or decode.")

    @property
    def queries(self) -> int:
        return self.seq_len if self.mode == "prefill" else 1

    @property
    def causal_pairs(self) -> int:
        if self.mode == "prefill":
            return self.seq_len * (self.seq_len + 1) // 2
        return self.seq_len


def flop_counts(config: Config, work: Workload, *, causal: bool = False) -> dict[str, int]:
    """Count matrix products at two FLOPs per multiply-accumulate."""
    parameters = layer_counts(config)
    counts = {
        name: 2 * work.batch * work.queries * parameters[name]
        for name in PROJECTIONS
    }
    pairs = work.causal_pairs if causal else work.queries * work.seq_len
    product = 2 * work.batch * pairs * config.width
    return {
        "Q projection": counts["Q projection"],
        "K projection": counts["K projection"],
        "V projection": counts["V projection"],
        "Q @ K.T": product,
        "P @ V": product,
        "O projection": counts["O projection"],
    }


def payload_bytes(config: Config, work: Workload) -> dict[str, int]:
    """Logical tensor payloads, not allocator peaks or memory traffic."""
    b, q, t, s = work.batch, work.queries, work.seq_len, work.element_bytes
    return {
        "Q (one layer)": b * q * config.width * s,
        "K + V (one layer, all visible tokens)": 2 * b * t * config.kv_width * s,
        "One dense scores OR P tensor (one layer)": b * config.heads * q * t * s,
        "KV cache (all layers, after this step)": 2 * b * t * config.kv_width * s * config.layers,
    }


def tiny_inputs(kv_heads: int):
    import torch

    generator = torch.Generator().manual_seed(9)
    config = Config(width=16, layers=2, heads=4, kv_heads=kv_heads)
    x = torch.randn(2, 5, config.width, generator=generator, dtype=torch.float64)

    def project(heads: int):
        weight = torch.randn(config.width, heads * 4, generator=generator,
                             dtype=torch.float64) / config.width ** 0.5
        return (x @ weight).reshape(2, 5, heads, 4).transpose(1, 2)

    return config, project(config.heads), project(kv_heads), project(kv_heads)


def attend(q, k, v, allowed):
    import torch

    # Only this tiny teaching implementation expands grouped K/V in memory.
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    probabilities = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    return scores, probabilities, probabilities @ v


def demo() -> None:
    import torch

    config, q, k, v = tiny_inputs(kv_heads=4)
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()
    scores, probabilities, context = attend(q, k, v, allowed)
    merged = context.transpose(1, 2).reshape(2, 5, config.width)
    print("Tiny CPU / FP64 demo: B=2, T=5, d=16, h=4; no model download.")
    for name, tensor in (("Q", q), ("K", k), ("V", v), ("scores", scores),
                         ("P", probabilities), ("P @ V", context), ("merged heads", merged)):
        print(f"  {name:16s} shape={str(tuple(tensor.shape)):18s} numel={tensor.numel()}")
    print("\nP for batch 0, head 0 (rows=query, columns=key):")
    for row in probabilities[0, 0]:
        print("  " + " ".join(f"{value:.4f}" for value in row.tolist()))
    _, _, last = attend(q[:, :, -1:], k, v, torch.ones(1, 5, dtype=torch.bool))
    torch.testing.assert_close(last, context[:, :, -1:])
    print(f"Row sums: {probabilities[0, 0].sum(dim=-1).tolist()}")
    print(f"Cached last-row vs full causal attention max error: "
          f"{(last - context[:, :, -1:]).abs().max().item():.3e}")
    print("This checks attention outputs, not a full Transformer or GPU latency.")


def verify() -> None:
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    config = Config()
    prefill = Workload()
    dense = flop_counts(config, prefill)
    useful = flop_counts(config, prefill, causal=True)
    assert sum(dense[name] for name in PROJECTIONS) == 274_877_906_944
    assert dense["Q @ K.T"] + dense["P @ V"] == 68_719_476_736
    assert useful["Q @ K.T"] + useful["P @ V"] == 34_376_515_584
    assert sum(dense.values()) == 343_597_383_680
    assert sum(flop_counts(config, Workload(mode="decode")).values()) == 167_772_160
    assert payload_bytes(config, prefill)["KV cache (all layers, after this step)"] == 1024 ** 3
    grouped = Config(kv_heads=8)
    assert payload_bytes(grouped, prefill)["KV cache (all layers, after this step)"] == 1024 ** 3 // 4
    assert flop_counts(grouped, prefill)["Q @ K.T"] == dense["Q @ K.T"]

    for kv_heads in (1, 2, 4):
        config, q, k, v = tiny_inputs(kv_heads)
        allowed = torch.ones(5, 5, dtype=torch.bool).tril()
        scores, probabilities, full = attend(q, k, v, allowed)
        k_expanded = k.repeat_interleave(config.heads // kv_heads, dim=1)
        v_expanded = v.repeat_interleave(config.heads // kv_heads, dim=1)
        reference = scaled_dot_product_attention(q, k_expanded, v_expanded,
                                                dropout_p=0.0, is_causal=True)
        torch.testing.assert_close(full, reference)
        torch.testing.assert_close(probabilities.sum(dim=-1),
                                   torch.ones(2, 4, 5, dtype=torch.float64))
        assert torch.count_nonzero(probabilities[..., ~allowed]).item() == 0
        assert full.shape == q.shape

        work = Workload(batch=2, seq_len=5, element_bytes=q.element_size())
        payload = payload_bytes(config, work)
        assert payload["Q (one layer)"] == q.numel() * q.element_size()
        assert payload["K + V (one layer, all visible tokens)"] == (k.numel() + v.numel()) * k.element_size()
        assert payload["One dense scores OR P tensor (one layer)"] == scores.numel() * scores.element_size()
        assert flop_counts(config, work)["Q @ K.T"] == 2 * scores.numel() * q.shape[-1]
        assert flop_counts(config, work, causal=True)["P @ V"] == 2 * 2 * 4 * int(allowed.sum()) * 4
        counts = flop_counts(config, work)
        assert counts["Q projection"] == counts["O projection"] == 2 * 2 * 5 * 16 * 16
        assert counts["K projection"] == counts["V projection"] == 2 * 2 * 5 * 16 * kv_heads * 4

        for index in range(5):
            cached_k, cached_v = k[:, :, :index + 1], v[:, :, :index + 1]
            _, _, step = attend(q[:, :, index:index + 1], cached_k, cached_v,
                                torch.ones(1, index + 1, dtype=torch.bool))
            torch.testing.assert_close(step, full[:, :, index:index + 1])
            work = Workload(batch=2, seq_len=index + 1, mode="decode")
            assert flop_counts(config, work) == flop_counts(config, work, causal=True)
            for name in PROJECTIONS:
                assert flop_counts(config, work)[name] * 5 == counts[name]

        # Cached chunks need an offset causal mask, not an upper-left triangle.
        chunk_mask = torch.arange(5)[None, :] <= torch.arange(3, 5)[:, None]
        _, _, chunk = attend(q[:, :, 3:], k, v, chunk_mask)
        reference = scaled_dot_product_attention(q[:, :, 3:], k_expanded, v_expanded,
                                                attn_mask=chunk_mask, dropout_p=0.0)
        torch.testing.assert_close(chunk, reference)
        torch.testing.assert_close(chunk, full[:, :, 3:])

    for length in (1, 2, 17, 2048):
        for batch in (1, 3):
            config = Config()
            work = Workload(batch=batch, seq_len=length)
            assert work.causal_pairs == sum(range(1, length + 1))
            expected = 8 * batch * length * config.width ** 2 + 4 * batch * length ** 2 * config.width
            assert sum(flop_counts(config, work).values()) == expected
            assert sum(flop_counts(config, work, causal=True).values()) <= expected

    for element_bytes in (1, 2, 4, 8):
        config = Config()
        payload = payload_bytes(config, Workload(element_bytes=element_bytes))
        base = payload_bytes(config, Workload(element_bytes=1))
        assert payload == {name: value * element_bytes for name, value in base.items()}

    for invalid in ({"batch": 0}, {"seq_len": -1}, {"element_bytes": 0}, {"mode": "train"}):
        try:
            Workload(**invalid)
        except ValueError:
            continue
        raise AssertionError(f"Invalid workload accepted: {invalid}")
    print("PASS: reference FLOPs/payloads, MHA/GQA/MQA vs SDPA, causal row sums,")
    print("      15 cached steps, 3 offset chunks, length/batch scaling, invalid workloads.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Visible K/V tokens, including the current token in decode.")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--element-bytes", type=int, choices=(1, 2, 4, 8), default=2,
                        help="Bytes per stored tensor element; accounting only, not demo dtype.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--demo", action="store_true", help="Run a fixed tiny CPU shape demo.")
    actions.add_argument("--check", action="store_true", help="Run fixed CPU correctness checks.")
    args = parser.parse_args()
    try:
        config = Config(width=args.width, layers=args.layers, heads=args.heads,
                        kv_heads=args.kv_heads)
        work = Workload(batch=args.batch, seq_len=args.seq_len, mode=args.mode,
                        element_bytes=args.element_bytes)
    except ValueError as error:
        parser.error(str(error))
    if args.check:
        verify()
        return
    if args.demo:
        demo()
        return

    print(f"{work.mode}: B={work.batch}, Tq={work.queries}, Tk={work.seq_len}, "
          f"d={config.width}, h={config.heads}, h_kv={config.kv_heads}, L={config.layers}")
    print("FLOPs: 2 per multiply-accumulate; attention only, forward only.")
    if work.mode == "decode":
        print("Tk includes current token; history K/V are cached, not projected again.")
    dense, useful = flop_counts(config, work), flop_counts(config, work, causal=True)
    print(f"\n{'Per layer':20s} {'Dense FLOPs':>20s} {'Causal useful FLOPs':>22s}")
    for name in dense:
        print(f"{name:20s} {dense[name]:20,d} {useful[name]:22,d}")
    for name, multiplier in (("TOTAL / layer", 1), ("TOTAL / all layers", config.layers)):
        print(f"{name:20s} {sum(dense.values()) * multiplier:20,d} "
              f"{sum(useful.values()) * multiplier:22,d}")
    print(f"All layers: dense={sum(dense.values()) * config.layers / 1e12:.6f} TFLOPs; "
          f"causal useful={sum(useful.values()) * config.layers / 1e12:.6f} TFLOPs")
    print(f"\nTensor payloads at {work.element_bytes} bytes/element:")
    for name, size in payload_bytes(config, work).items():
        print(f"  {name:44s} {size:16,d} bytes  {size / 1024 ** 3:.6f} GiB")
    print("\nExcludes scale/mask/softmax, RoPE, norm, FFN, LM head and backward.")
    print("Causal useful FLOPs are not measured hardware instructions or a speedup guarantee.")
    print("Payloads are not peak memory: scores may be fused away; dtypes/workspace may differ.")


if __name__ == "__main__":
    main()
