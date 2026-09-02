"""附加课 X01 实验 · 向量、点积、以及"意思"是怎么变成几何的

四件事：
  A  手写点积 / 范数 / 余弦相似度，和 numpy 对账
  B  验证几何恒等式  a·b = |a||b|cos θ
  C  从零训一组词向量：共现统计 → PPMI → SVD，看"意思"长出几何结构
  D  高维的反直觉：随机向量几乎两两垂直（这条会在 X06 解释 √d 时用到）

全部在 CPU 上跑，不需要联网、不需要下载模型。

用法：
    uv run python labs/extra/x01_vectors.py
    uv run python labs/extra/x01_vectors.py --exp C
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


# =============================================================== A 手写基本运算
def dot(a: list[float], b: list[float]) -> float:
    """点积 = 逐元素相乘再全部加起来。就这么简单。"""
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(a: list[float]) -> float:
    """向量的长度 = 各分量平方和再开方（勾股定理的高维版）。"""
    return math.sqrt(dot(a, a))


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 = 把长度除掉，只留方向。"""
    return dot(a, b) / (norm(a) * norm(b))


def exp_a_basics() -> None:
    console.rule("[bold]A · 手写点积 / 范数 / 余弦，和 numpy 对账")

    pairs = [
        ("完全同向", [1, 2, 3, 4], [2, 4, 6, 8]),
        ("有点像", [1, 2, 3, 4], [2, 1, 4, 3]),
        ("互相垂直", [1, 0, 1, 0], [0, 1, 0, 1]),
        ("完全反向", [1, 2, 3, 4], [-1, -2, -3, -4]),
        ("长度差很多", [1, 2, 3, 4], [100, 200, 300, 400]),
    ]

    tbl = Table("这两个向量", "a", "b", "a·b", "|a|", "|b|", "cos θ", "夹角",
                title="点积和余弦，全部手写实现")
    for name, a, b in pairs:
        c = cosine(a, b)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        assert abs(dot(a, b) - float(np.dot(a, b))) < 1e-9      # 和 numpy 对账
        tbl.add_row(name, str(a), str(b), f"{dot(a,b):8.2f}",
                    f"{norm(a):6.2f}", f"{norm(b):7.2f}",
                    f"{c:6.3f}", f"{ang:5.1f}°")
    console.print(tbl)

    console.print(
        "\n[bold yellow]看最后两行，这是今天最重要的一个区别：[/bold yellow]\n"
        "  · [bold]「完全反向」的点积是 -30[/bold]，"
        "而[bold]「长度差很多」的点积是 3000[/bold]。\n"
        "  · 但后者的 cos θ = 1.0 —— 它们方向完全一样，只是一个长 100 倍。\n\n"
        "  [bold]点积同时被「方向」和「长度」影响；余弦只看方向。[/bold]\n"
        "  想问「这两个词意思像不像」→ 用余弦。\n"
        "  想问「这个特征在这条数据上有多强」→ 用点积（长度就是强度）。\n"
    )


# =============================================================== B 几何恒等式
def exp_b_geometry() -> None:
    console.rule("[bold]B · 验证 a·b = |a||b|cos θ")

    rng = np.random.default_rng(0)
    tbl = Table("维度 d", "a·b（直接算）", "|a||b|cos θ（几何算）", "差值",
                title="随机取 5 组向量，两种算法结果必须一样")
    for d in (2, 3, 4, 16, 4096):
        a = rng.normal(size=d)
        b = rng.normal(size=d)
        lhs = float(a @ b)
        cos = lhs / (np.linalg.norm(a) * np.linalg.norm(b))
        rhs = float(np.linalg.norm(a) * np.linalg.norm(b) * cos)
        tbl.add_row(str(d), f"{lhs:12.6f}", f"{rhs:12.6f}", f"{abs(lhs-rhs):.2e}")
    console.print(tbl)

    console.print(
        "\n[bold yellow]这个恒等式为什么重要：[/bold yellow]\n"
        "  左边 [bold]a·b = Σ aᵢbᵢ[/bold] 是「一堆乘加」—— GPU 最擅长的事（Day 03 的 FMA）。\n"
        "  右边 [bold]|a||b|cos θ[/bold] 是「两个箭头夹角多大」—— 人能想象的几何。\n\n"
        "  [bold]同一件事的两副面孔：一副给硬件看，一副给你看。[/bold]\n"
        "  整个注意力机制，就是在反复做这件事 —— 一次算几百万个夹角。\n"
    )


# =============================================================== C 训一组词向量
# 同类词共享一批模板（所以会像），但每个词还有自己的专属句子（所以不会一模一样）
ANIMAL_TEMPLATES = [
    "{w} eats {f}", "the {w} eats the {f}", "my {w} likes {f}",
    "a small {w} sleeps here", "the {w} is very cute",
    "i feed my {w} some {f}", "that {w} looks hungry",
]
VEHICLE_TEMPLATES = [
    "{w} drives on the road", "the {w} drives very fast",
    "my {w} needs some fuel", "a big {w} carries people",
    "the {w} is very fast", "i park my {w} outside",
    "that {w} looks expensive",
]
FOODS = ["meat", "bread", "rice", "fish"]

# 每个词的专属上下文 —— 这是让「猫和小猫」比「猫和鸟」更像的关键
FLAVOR = {
    "cat":    ["the cat meows softly", "my cat purrs a lot", "the cat climbs a tree"],
    "kitten": ["the kitten meows softly", "my kitten purrs a lot", "a tiny kitten plays"],
    "dog":    ["the dog barks loudly", "my dog wags its tail", "the dog guards the house"],
    "puppy":  ["the puppy barks loudly", "my puppy wags its tail", "a tiny puppy plays"],
    "bird":   ["the bird flies away", "my bird sings a song", "the bird builds a nest"],
    "car":    ["the car parks in a garage", "my car has four wheels"],
    "truck":  ["the truck hauls heavy cargo", "my truck has six wheels"],
    "bus":    ["the bus stops at a station", "my bus carries many people"],
    "bike":   ["the bike has two wheels", "i pedal my bike uphill"],
    "train":  ["the train stops at a station", "the train runs on rails"],
}
ANIMALS = ["cat", "kitten", "dog", "puppy", "bird"]
VEHICLES = ["car", "truck", "bus", "bike", "train"]


def build_corpus() -> list[list[str]]:
    sents: list[list[str]] = []
    for w in ANIMALS:
        for tpl in ANIMAL_TEMPLATES:
            if "{f}" in tpl:
                sents += [tpl.format(w=w, f=f).split() for f in FOODS]
            else:
                sents.append(tpl.format(w=w).split())
    for w in VEHICLES:
        sents += [tpl.format(w=w).split() for tpl in VEHICLE_TEMPLATES]
    for w, extra in FLAVOR.items():
        sents += [s.split() for s in extra for _ in range(3)]   # 专属句子加权
    return sents


def train_embeddings(dim: int = 8, window: int = 2):
    """共现计数 → PPMI → SVD。这是词向量最原始、也最容易看懂的做法。"""
    sents = build_corpus()
    vocab = sorted({w for s in sents for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    # ① 数共现：两个词在同一个窗口里出现过几次
    co = np.zeros((n, n))
    for s in sents:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i != j:
                    co[idx[w], idx[s[j]]] += 1

    # ② PPMI：把"因为常见所以共现多"的部分除掉，只留下"超出偶然的关联"
    total = co.sum()
    p_ij = co / total
    p_i = co.sum(1, keepdims=True) / total
    p_j = co.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij / (p_i * p_j))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)

    # ③ SVD 降维：从 n 维压到 dim 维，保留最主要的几个"方向"
    u, s, _vt = np.linalg.svd(ppmi)
    emb = u[:, :dim] * s[:dim]
    return vocab, idx, emb, co


def _pairs(xs: list[str]):
    return [(xs[i], xs[j]) for i in range(len(xs)) for j in range(i + 1, len(xs))]


def exp_c_embeddings() -> None:
    console.rule("[bold]C · 从零训一组词向量：看「意思」长出几何结构")

    vocab, idx, emb, co = train_embeddings()
    console.print(f"[dim]语料 {len(build_corpus())} 句 · 词表 {len(vocab)} 个词 · "
                  f"向量维度 {emb.shape[1]}[/dim]\n")

    def cos(w1: str, w2: str) -> float:
        a, b = emb[idx[w1]], emb[idx[w2]]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    nouns = ANIMALS + VEHICLES
    tbl = Table("词 1", "词 2", "余弦相似度", "它们是什么关系",
                title="模型从来没被告诉过谁是动物谁是车")
    for w1, w2 in (("cat", "kitten"), ("dog", "puppy"), ("cat", "dog"),
                   ("car", "truck"), ("bus", "train"),
                   ("cat", "car"), ("dog", "truck"), ("bird", "bike")):
        c = cos(w1, w2)
        same = (w1 in ANIMALS) == (w2 in ANIMALS)
        tbl.add_row(w1, w2, f"[{'green' if c > 0.5 else 'red'}]{c:6.3f}[/]",
                    "同类" if same else "[bold]★ 不同类[/bold]")

    within = [cos(a, b) for a, b in
              [(x, y) for x, y in _pairs(ANIMALS)] + [(x, y) for x, y in _pairs(VEHICLES)]]
    across = [cos(a, v) for a in ANIMALS for v in VEHICLES]
    tbl.add_row("[bold]类内平均[/bold]", "", f"[bold green]{np.mean(within):6.3f}[/bold green]",
                f"{len(within)} 对")
    tbl.add_row("[bold]类间平均[/bold]", "", f"[bold red]{np.mean(across):6.3f}[/bold red]",
                f"{len(across)} 对")
    console.print(tbl)

    # 最近邻（只在这 10 个名词里找，这样结构看得最清楚）
    tbl = Table("查询词", "最相似的 3 个名词",
                title="最近邻（只在 5 个动物 + 5 个交通工具里比）")
    for q in nouns:
        sims = [(w, cos(q, w)) for w in nouns if w != q]
        sims.sort(key=lambda t: -t[1])
        top = "   ".join(f"{w} ({c:.2f})" for w, c in sims[:3])
        hit = all((w in ANIMALS) == (q in ANIMALS) for w, _ in sims[:3])
        tbl.add_row(q, f"[green]{top}[/green]" if hit else top)
    console.print(tbl)

    console.print(
        "\n[bold yellow]这里发生了什么？[/bold yellow]\n"
        "  我们[bold]从头到尾没有告诉模型任何词的含义[/bold]，\n"
        "  只是数了「哪些词经常出现在一起」，然后压缩到 8 维。\n\n"
        "  [bold]结果类内相似度 0.9+，类间相似度 0.1 左右 —— 相差一个数量级。[/bold]\n"
        "  动物自动聚成一团，车自动聚成一团。\n\n"
        "  这就是[bold]分布式假说（distributional hypothesis）[/bold]：\n"
        "  [italic]「一个词的意思，由它周围出现的词决定。」[/italic]\n"
        "  —— Firth, 1957。整个 embedding 的思想都是这一句话。\n\n"
        "  真实模型（GPT / Llama）的 embedding 是[bold]训出来的[/bold]而不是数出来的，\n"
        "  但目标完全一样：[bold]让「意思相近」变成「方向相近」[/bold]。\n\n"
        "[dim]⚠️ 诚实地说：这个玩具语料只有 200 多句、而且很规整，\n"
        "所以「大类」分得很开，但「cat 和 kitten 应该比 cat 和 bird 更近」这种细粒度关系并不可靠。\n"
        "真实模型要用几千亿 token 才能学出稳定的细结构。[/dim]\n"
    )


# =============================================================== D 高维的反直觉
def exp_d_high_dim() -> None:
    console.rule("[bold]D · 高维的反直觉：随机向量几乎两两垂直")

    rng = np.random.default_rng(0)
    tbl = Table("维度 d", "cos θ 的标准差", "1/√d", "|cos θ| > 0.1 的比例",
                "夹角落在 80°~100° 的比例",
                title="每个维度取 20000 对随机向量")
    for d in (2, 3, 8, 64, 512, 4096):
        a = rng.normal(size=(20000, d))
        b = rng.normal(size=(20000, d))
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        tbl.add_row(f"{d:>5}", f"{c.std():8.4f}", f"{1/math.sqrt(d):8.4f}",
                    f"{(np.abs(c) > 0.1).mean()*100:7.1f}%",
                    f"{((ang > 80) & (ang < 100)).mean()*100:7.1f}%")
    console.print(tbl)

    console.print(
        "\n[bold yellow]两条结论，后面都会用到：[/bold yellow]\n"
        "  ① [bold]cos θ 的标准差正好等于 1/√d[/bold]（第 2、3 列一模一样）。\n"
        "     维度越高，随机两个向量越接近垂直 —— d=4096 时 99.9% 的夹角都在 90° 附近。\n\n"
        "  ② [bold]这是好事[/bold]：高维空间「很空」，可以塞下海量互不干扰的方向。\n"
        "     4096 维里能放下几万个近似正交的词向量，这就是词表能有 15 万的原因。\n\n"
        "  [bold]★ 但它也带来一个麻烦[/bold]：点积 a·b ≈ |a||b|·(1/√d) 的量级随 d 变化，\n"
        "  d 越大，QKᵀ 的方差越大，softmax 就越容易饱和。\n"
        "  [bold]这正是注意力公式里那个 1/√d 的来历 —— X06 会完整推一遍。[/bold]\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list("ABCD"), help="只跑指定实验")
    args = ap.parse_args()

    console.print("[bold green]X01[/bold green] 向量与点积  |  "
                  "[dim]纯 CPU / numpy，不需要联网[/dim]\n")

    if args.exp in (None, "A"):
        exp_a_basics()
    if args.exp in (None, "B"):
        exp_b_geometry()
    if args.exp in (None, "C"):
        exp_c_embeddings()
    if args.exp in (None, "D"):
        exp_d_high_dim()


if __name__ == "__main__":
    main()
