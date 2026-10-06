# Day 09 · 我的笔记

正文：[注意力的计算图与复杂度](day09-attention-complexity.md)。

> 每个数字都先说清：哪个算子、哪几层、prefill 还是 decode。

## 1. 两类计算

**Attention 中哪些计算有参数？哪些没有？各自随 $T$ 怎样增长？**


**为什么说平方成本是“按内容取信息”的代价？**


补全计算图：

```text
X：
   ↓ Q/K/V 投影 + 拆头
Q：               K：               V：
   ↓ Q @ Kᵀ
scores：
   ↓ 缩放 + mask + softmax
P：                 （行 = ____，列 = ____，softmax 沿 ____）
   ↓ P @ V
context：
   ↓ 合头 + O 投影
输出：
```

## 2. FLOPs

计数约定：一次乘加 = ______ FLOPs。

| MHA 每层 | Prefill | 已有缓存的单步 decode |
|---|---|---|
| Q/K/V/O 投影 | | |
| 两次配对，完整 | | |
| 两次配对，因果有效 | | — |

**头数为什么从公式里消失了？**


**交叉点 $T=$ ______。Llama-2-7B 在 2K 上下文时，主要成本在哪？**


**长度从 2048 翻倍到 4096，Attention 计算量变成几倍？为什么不是 4 倍？**


**因果 mask 让配对数减半，为什么总计算量只少约 10%？**


## 3. 内存

| | 一层分数表 | 全部层 KV Cache |
|---|---|---|
| 公式 | | |
| 随 $T$ 增长方式 | | |
| 生命周期 | | |
| 能否避免 | | |

## 4. Decode 与 KV Cache

**KV Cache 成立的两个条件：**

1.
2.

**为什么不缓存历史 Q 和 P？用“第 4 步、第 5 步各用到了什么”来说明：**


**为什么旧位置的隐藏状态也不需要缓存，只存 K/V 就够了？**


**decode 配对计算的算术强度约为多少？为什么受带宽限制？**


**增大 batch 对投影和对配对计算的效果有什么不同？**


**GQA 减少了什么？没有减少什么？为什么它对 decode 有效？**


## 5. 手算对账

| 题目 | 我的手算 | 计算器 / 正文 |
|---|---:|---:|
| Q1 完整总计 | | 18,432 |
| Q1 因果有效总计 | | 17,664 |
| Q2 交叉点 | | 16,384 |
| Q3 MHA KV Cache | | 4 GiB |
| Q3 分数表 | | 2 GiB |
| Q3 GQA KV Cache | | 1 GiB |

## 6. 实验记录

```bash
uv run python labs/day09/attention_cost.py
uv run python labs/day09/attention_cost.py --mode decode --seq-len 2048
uv run python labs/day09/attention_cost.py --demo
uv run python labs/day09/attention_cost.py --check
```

**PyTorch 版本：**

**`--demo` 中缓存单步与完整前向最后一行的最大误差：**

**`--check` 是否通过？**

## 7. 思考题与下一讲

**T1：**


**T2：**


**T3：**


**写 Day 10 的 Attention 时，我最容易写错的维度或 mask：**


**我还没讲清楚的一点：**
