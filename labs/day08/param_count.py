"""Day 08: count bias-free, RMSNorm decoder parameters without loading weights.

    python labs/day08/param_count.py
    python labs/day08/param_count.py --kv-heads 8
    python labs/day08/param_count.py --ffn dense --intermediate 16384
    uv run python labs/day08/param_count.py --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    width: int = 4096
    layers: int = 32
    heads: int = 32
    kv_heads: int = 32
    intermediate: int = 11008
    vocab: int = 32000
    ffn: str = "swiglu"
    tied: bool = False

    def __post_init__(self) -> None:
        dimensions = (self.width, self.layers, self.heads, self.kv_heads,
                      self.intermediate, self.vocab)
        if any(value <= 0 for value in dimensions):
            raise ValueError("All dimensions must be positive.")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width must divide into heads; heads into KV groups.")
        if self.ffn not in ("dense", "swiglu"):
            raise ValueError("ffn must be dense or swiglu.")

    @property
    def kv_width(self) -> int:
        return self.kv_heads * (self.width // self.heads)


def layer_counts(config: Config) -> dict[str, int]:
    counts = {
        "Q projection": config.width ** 2,
        "K projection": config.width * config.kv_width,
        "V projection": config.width * config.kv_width,
        "O projection": config.width ** 2,
        "FFN up": config.width * config.intermediate,
        "FFN down": config.width * config.intermediate,
    }
    if config.ffn == "swiglu":
        counts["FFN gate"] = config.width * config.intermediate
    counts["Two RMSNorm scales"] = 2 * config.width
    return counts


def parameter_counts(config: Config) -> dict[str, int]:
    return {
        "Token embedding": config.vocab * config.width,
        "Decoder blocks": config.layers * sum(layer_counts(config).values()),
        "Final RMSNorm": config.width,
        "Extra LM head": 0 if config.tied else config.vocab * config.width,
    }


def verify() -> None:
    import torch
    from torch import nn

    for ffn in ("dense", "swiglu"):
        for kv_heads in (1, 2, 4):
            for tied in (False, True):
                config = Config(width=16, layers=2, heads=4, kv_heads=kv_heads,
                                intermediate=40, vocab=23, ffn=ffn, tied=tied)
                with torch.device("meta"):
                    model = nn.Module()
                    model.embedding = nn.Embedding(config.vocab, config.width)
                    model.blocks = nn.ModuleList()
                    for _ in range(config.layers):
                        block = nn.ModuleDict({
                            "q": nn.Linear(config.width, config.width, bias=False),
                            "k": nn.Linear(config.width, config.kv_width, bias=False),
                            "v": nn.Linear(config.width, config.kv_width, bias=False),
                            "o": nn.Linear(config.width, config.width, bias=False),
                            "up": nn.Linear(config.width, config.intermediate, bias=False),
                            "down": nn.Linear(config.intermediate, config.width, bias=False),
                            "norm1": nn.RMSNorm(config.width),
                            "norm2": nn.RMSNorm(config.width),
                        })
                        if config.ffn == "swiglu":
                            block["gate"] = nn.Linear(config.width, config.intermediate,
                                                      bias=False)
                        model.blocks.append(block)
                    model.norm = nn.RMSNorm(config.width)
                    model.head = nn.Linear(config.width, config.vocab, bias=False)
                    if config.tied:
                        model.head.weight = model.embedding.weight
                actual = sum(parameter.numel() for parameter in model.parameters())
                assert actual == sum(parameter_counts(config).values()), config

    assert sum(parameter_counts(Config()).values()) == 6_738_415_616
    assert sum(parameter_counts(Config(kv_heads=8)).values()) == 5_933_109_248
    for invalid in ({"width": 0}, {"heads": 3}, {"kv_heads": 5}):
        try:
            Config(**invalid)
        except ValueError:
            continue
        raise AssertionError(f"Invalid config accepted: {invalid}")
    print("PASS: 12 meta-module counts, 2 reference totals, 3 invalid configs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--ffn", choices=("dense", "swiglu"), default="swiglu")
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify()
        return
    try:
        config = Config(**{key: value for key, value in vars(args).items() if key != "check"})
    except ValueError as error:
        parser.error(str(error))

    print(f"Bias-free decoder / RMSNorm / {config.ffn} / tied={config.tied}")
    print(f"d={config.width}, L={config.layers}, h={config.heads}, "
          f"h_kv={config.kv_heads}, m={config.intermediate}, V={config.vocab}")
    print("\nPer layer:")
    for name, count in layer_counts(config).items():
        print(f"  {name:24s} {count:16,d}")
    print("\nWhole model (unique parameters):")
    counts = parameter_counts(config)
    for name, count in counts.items():
        print(f"  {name:24s} {count:16,d}")
    total = sum(counts.values())
    print(f"  {'TOTAL':24s} {total:16,d}  ({total / 1e9:.6f} B)")
    print(f"\n12*d^2*L (block approximation): {12 * config.width ** 2 * config.layers:,}")
    for bits in (16, 4):
        storage = total * bits / 8
        print(f"Ideal {bits}-bit weight payload: {storage / 1e9:.3f} GB / "
              f"{storage / 1024 ** 3:.3f} GiB")
    matrix_parameters = sum(layer_counts(config).values()) - 2 * config.width
    linear_flops = 2 * (config.layers * matrix_parameters + config.vocab * config.width)
    print(f"Dense projections per token (incl. LM head): {linear_flops:,} FLOPs")
    print("Excludes attention products, norms, nonlinearities and sampling.")
    print("Weight payload excludes quantization metadata, KV, activations and workspace.")


if __name__ == "__main__":
    main()