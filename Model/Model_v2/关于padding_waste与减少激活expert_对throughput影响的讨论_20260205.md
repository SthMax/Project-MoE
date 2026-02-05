# 关于 padding waste 与“减少激活 expert”对 throughput 影响的讨论（vLLM / fused MoE）

> 日期：2026-02-05  
> 主题：在 vLLM fused MoE 中，padding waste 怎么理解、对应代码是什么；以及为什么“减少激活 expert”可能比单纯优化 padding waste 更有效。

---

## 1) 直觉澄清：padding waste 看似只是“整除后的余数”，但乘上“活跃专家数 × 两次 GEMM”可能不小

你的直觉是：

> padding 上能节省的开销，本质就是一个整除问题后的余数，那么总的算下来似乎也不会太多。

这个直觉在 **每个 expert 的 token 数 $n_e$ 足够大且不碎** 时成立——因为单个 expert 的 padding 上限确实是 $B-1$（$B=\text{BLOCK\_SIZE\_M}$）。

但在 MoE 推理里，决定总浪费的是：

- **活跃专家数** $E_\text{active}$（本 batch 有 token 路由到的 expert 数）
- 很多 expert 是否落在 **小桶区**（尤其是 $n_e < B$）
- padding 不只是影响“一次操作”，而是放大整个 fused MLP 链条（至少两次 GEMM + activation + 读写）

对单个 expert，padding 的定义是：

$$
\text{padded}(n_e)=\left\lceil\frac{n_e}{B}\right\rceil\cdot B,\quad
\text{waste}(e)=\text{padded}(n_e)-n_e
$$

总浪费是：

$$
\text{waste\_total}\approx \sum_{e\in\text{active}}\left(\left\lceil\frac{n_e}{B}\right\rceil\cdot B-n_e\right)
$$

### 关键反例：当 $n_e < B$ 时，浪费不是“小余数”，而是“几乎整块”

如果很多 expert 被点亮但每个只分到少量 token（例如 1~10），那么 $B=64$ 时每个 expert 的浪费接近 50~63 个 token-expert pair。

此时 padding waste 的本质不是“余数小”，而是：

> 块大小固定（为 kernel 友好），但分布太碎导致大量 block 被迫填充。

---

## 2) “减少激活 expert 会不会更有效？”——很可能是，而且它和 padding waste 本质上是同一根杠杆

你提出：

> 或许减少激活的 expert 可以增大 throughput？而且效果会比 padding waste 方向的优化更好？

结论倾向：**是的，减少激活 expert 往往是更强的结构性杠杆**。

原因是它会同时带来：

- padding waste 下降（token 更集中，$n_e$ 更接近整块）
- regroup/permute/sort 的碎片化下降
- EP 场景下 all2all 的碎片度可能下降（更少 peer 有非零发送）

与其把“padding waste”看成需要单独优化的目标，更准确的表述是：

> padding waste 是“分布碎片化”的一个 **kernel 级 proxy**；减少激活 expert 则是直接减少碎片化。

---

## 3) 硬约束：不能无脑减少 expert（否则等价于改 top-k / 影响质量）

在 vLLM 的 routing contract 下（尤其 fused kernel 路径）：

- 每个 token 需要输出 top-k experts（比如 k=8/10）
- routing 来自 router logits（允许小幅修正，但不能彻底不尊重 logits）

因此“减少激活 expert”如果通过强制把 token 都挤到少量 experts，可能导致质量明显下降。

### 更安全的做法：候选内集中（两阶段 top-k’ → 重排）

一种工程上更稳定的集中策略是：

1. 先取每个 token 的 top-k’（k’ 比 k 大一点，如 k+4 或 2k），保证语义候选足够。
2. 在候选内用一个“集中度惩罚 / 块对齐惩罚 / padding proxy”重排，输出最终 top-k。

这样集中发生在相对安全的候选池里，质量风险可控。

---

## 4) 什么时候 padding waste 真会大？一个很实用的判断标准

只需要看两个统计指标（无需 profiler，就能快速看方向）：

1. **小桶比例**：有多少 experts 满足 $n_e < B$（每 batch 或 EMA）。
2. **活跃专家数**：$E_\text{active}$（至少一个 token 路由到它的 expert 数）。

当你看到：

- $E_\text{active}$ 明显增大
- 且大量 experts 落在 $n_e < B$

那么 padding waste 往往不是“小余数”，而是会以近似线性的方式侵蚀吞吐。

---

## 5) 一句话结论（把两种观点统一起来）

- “padding waste 是余数问题”这句话只在分布不碎时成立；真正危险的是大量 $n_e < B$ 的小桶 experts。
- “减少激活 expert”很可能比单纯抠 padding 余数更有效，因为它同时减少 padding、碎片化与通信碎片。
- 最佳实践通常是：**用 padding-aware penalty 作为候选内集中的约束/引导**，而不是把 waste 当唯一目标。
