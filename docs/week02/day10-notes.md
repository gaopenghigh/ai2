# Day 10 · 我的笔记

正文：[从零手写 GPT](day10-minigpt.md)。

> 先不看代码，凭记忆写出每一步；写不出来的地方，就是还没真正理解的地方。

## 1. 模型是什么

**`config.json` 和 `model.safetensors` 各包含什么？“加载模型”具体做了什么？**


**为什么 `k_proj.weight` 的形状是 (192, 576)，而不是 (576, 576)？**


**为什么文件里没有 `lm_head.weight`？**


## 2. 不看代码，写出前向

```text
MiniGPT.forward：


Block.forward：


Attention.forward（标出每一步的形状）：
  q：               k：               v：
  RoPE：
  缓存：
  GQA：
  scores：
  mask：
  softmax：
  输出：
```

**RoPE 为什么只作用于 q 和 k？**


**mask 为什么要按绝对位置构造？换成 `torch.tril` 在哪种情况下会出错？**


**生成循环中，带不带缓存的差别在哪一行？**


## 3. 对账记录

```bash
uv run python labs/day10/minigpt.py --check --offline
uv run python labs/day10/minigpt.py --check
```

| 检查 | 回答什么问题 | 我的结果 |
|---|---|---|
| 随机小模型 vs transformers | | |
| 真实权重 vs transformers | | |
| 参数量 vs Day 08 计算器 | | |
| prefill + decode vs 完整前向 | | |
| 生成一致性 | | |

**为什么用 FP32 对账？缓存版本的误差为什么不是 0？**


## 4. 故意改错

| 实验 | 我的预测：哪些检查会失败 | 实际失败项 | 为什么 |
|---|---|---|---|
| A · `repeat` 替代 `repeat_interleave` | | | |
| B · 删掉 mask | | | |
| C · RoPE 只旋转 q | | | |

**实验 C 说明了什么？**


## 5. KV Cache 实测

```bash
uv run python labs/day10/minigpt.py
uv run python labs/day10/minigpt.py --max-new-tokens 200
uv run python labs/day10/minigpt.py --device mps
```

| 设备 | 生成长度 | 带缓存 token/s | 不带缓存 token/s | 倍数 |
|---|---:|---:|---:|---:|
| CPU | 48 | | | |
| CPU | 200 | | | |
| MPS | 48 | | | |

**为什么短序列时缓存收益不大？把一步的耗时拆成三部分：**

1.
2.
3.

## 6. 思考题

**T1：**


**T2（至少三处浪费，以及哪一天解决）：**


**T3：SmolLM2 在 8K 上下文的 KV Cache = ______，约为 FP16 权重的 ______。**


**我还没讲清楚的一点：**
