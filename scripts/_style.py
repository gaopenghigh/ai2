"""课程通用绘图样式：中文字体自动探测 + 统一配色。

所有 scripts/dayXX_*.py 都 `from _style import setup, C` 使用。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 统一配色（色盲友好）
C = {
    "compute": "#D1495B",   # 算力 / compute-bound
    "memory": "#00798C",    # 带宽 / memory-bound
    "capacity": "#EDAE49",  # 容量
    "neutral": "#66717E",
    "accent": "#7B4B94",
    "ok": "#2E933C",
    "grid": "#D8DEE4",
    "bg": "#FFFFFF",
}

_CJK_CANDIDATES = [
    "PingFang SC", "Heiti SC", "Songti SC", "STHeiti",
    "Hiragino Sans GB", "Arial Unicode MS",
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
]


def pick_cjk_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def setup(dpi: int = 160) -> str | None:
    """配置全局 rcParams，返回实际使用的中文字体名（None 表示未找到）。"""
    font = pick_cjk_font()
    family = ([font] if font else []) + ["DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": family,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": C["bg"],
        "axes.facecolor": C["bg"],
        "axes.edgecolor": C["neutral"],
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "font.size": 10,
    })
    return font


if __name__ == "__main__":
    print("CJK font:", setup())
