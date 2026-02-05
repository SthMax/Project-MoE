# PriceMoE 实验吞吐不升的复盘（以及为何 Fluid‑SemMoE 更贴近真实瓶颈）

> 日期：2026-02-05  
> 背景：你反馈 PriceMoE 的实验结果很糟糕，并没有提升 throughput；你给出的关键解释是：
>
> - **Expert Compute**：GPU 上的 FFN 是高度优化的 batched matrix multiplication，服务时间与 batch size 线性相关，**无 per-token 排队**。

下面把我的分析与下一步建议整理成可直接引用/落地的版本。

---

## 1) “无 per-token 排队”这句话：对，但只覆盖了瓶颈的一部分

在 vLLM fused MoE 路径里，**Expert Compute 是 batched GEMM**，意味着：

- token 会按 expert regroup 成块，然后执行 W1/W2 等矩阵乘法；
- 这一步更像吞吐机器：token 多一点通常更能吃满 GPU；
- 因此，**用 PriceMoE 把 token 从热门 expert 挪到冷门 expert**，很可能只是把一个大 GEMM 拆成多个更碎的 GEMM。
  - 这不一定更快，甚至可能更慢（更差的 tile 利用率 + 分组开销）。

但 vLLM 的 MoE 并不是“只有 GEMM”。即使 expert compute 端没有 per-token 排队，仍然存在会主导 throughput/P99 的“同步/排队/无效工作”来源：

1. **All-to-All / AllGather 类通信同步屏障（尤其 EP）**：天然是 tail/同步点。
2. **dispatch/permute/sort/pad**：把 token 按 expert 重排并 padding 到 block size 的过程。
   - padding 会带来“无效算力”。
3. **设备级 KV cache/显存准入导致的 admission / batch 形态变化**：这是系统层吞吐上限来源。

因此更准确的表述是：

> PriceMoE 更敏感于“某些 expert 计算显著更慢”的问题；但在 vLLM fused MoE 的现实里，throughput 更可能受 **通信同步 + dispatch/padding + device 准入**支配，而不是某个 expert 上的排队。

---

## 2) 为什么 PriceMoE 可能反而伤 throughput（即使“更均衡”）

即使没有 per-token 排队，PriceMoE 的“均衡”也可能引入更高的系统成本。

### (A) padding/对齐成本上升

fused MoE 经常需要把每个 expert 的 token 数 padding 到 `BLOCK_SIZE_M`。

如果 PriceMoE 把 token 更平均地摊到更多 experts 上，可能出现：

- 更多 experts 被点亮，但每个 expert 的 token 很少；
- 每个 expert 都要 padding 到 block 的整数倍；

结果是：**总 padding（无效 token）上升 → 总 GEMM 工作量上升 → throughput 下降**。

这在 top-k 较大、E 很大时更明显。

### (B) regroup 更碎，grouped GEMM 利用率下降

把大 batch 切碎到更多 expert 组会导致：

- 每个 expert 的矩阵更瘦小；
- kernel 调度与资源利用变差；

即便没有排队，也可能变慢。

### (C) 通信量不降反升（EP 场景）

如果 PriceMoE 让 token 更平均地分布到更多 device/EP rank：

- 原本“多数 token 本地算”的情况变成“更多 token 跨卡 all2all”；
- 或者消息更碎（更多 peer、更多小包）

→ 通信时间变长，吞吐下降。

---

## 3) 结论：不是否定动态路由，而是 PriceMoE 的“价格变量”需要换

你现在的观察本质上在支持 `Fluid_Sem_MoE v0.2` 的核心洞见：

- **影子价格不该定义在 expert queue（或其近似）**
- 更应该定义在 **device admission / 通信瓶颈/显存准入** 上

但不需要一口气跳到 Fluid‑SemMoE 全套（语义可行域 + Q_d + Z_d）；可以用“最小变更”先把 throughput 拉回来。

---

## 4) 从 PriceMoE 平滑升级的两条工程路线（优先追 throughput）

### 路线 1（最小变更）：padding-aware 的 PriceMoE（用“无效工作量”做 price）

不要用 $q_e$（expert token count）做价格，而用更贴近 fused kernel 成本的 proxy：**padding waste**。

设 $n_e$ 为本 batch routed 到 expert $e$ 的 token 数，$B$ 为 `BLOCK_SIZE_M`，则

$$
\text{waste}(e) \approx \left\lceil \frac{n_e}{B} \right\rceil \cdot B - n_e.
$$

路由时惩罚“点亮更多但很稀疏的 experts”，直觉是：

- **适度集中**往往比“平均分散”更省 padding；
- 更接近真实吞吐瓶颈（无效算力）。

### 路线 2（通信方向）：device-level cost proxy（Fluid‑SemMoE 的简化工程版）

直接把路由目标从 expert 维迁移到 device 维，用可观测 proxy 近似：

- $Q_d \approx$ 本 batch 发往 device $d$ 的 token/bytes（EMA）；
- $g_d$ 先简化成“本地/跨节点/跨机架”等分层常数；
- $Z_d$ 用 KV cache 使用率 proxy（例如 block 数/ratio 的 EMA）。

然后做：

$$
 d^* = \arg\min_d \{ Q_d + \eta g_d + V Z_d \}.
$$

这条路线优化的对象更贴近你看到的实际瓶颈（通信尾 + 准入压力）。

---

## 5) 两个低成本“判定实验”：1 天内把瓶颈原因坐实

你现在不缺理论，缺的是“到底卡哪儿”的证据。可以先做两组统计（不需要大改系统）：

### 实验 1：统计 padding waste 是否变大

- 每个 batch 记录 $n_e$（你已有 `topk_ids` hook，非常容易）；
- 用实际 `BLOCK_SIZE_M` 算总 waste；
- 对比 baseline vs PriceMoE。

如果 waste 上升且吞吐下降，基本坐实 (A)。

### 实验 2：统计跨 device 的 all2all send bytes 是否更大/更碎（EP）

- 在 all2all 前统计每个 peer 的 send counts/bytes；
- 对比 baseline vs PriceMoE。

如果 token 被打散到更多 peer 或小包增多，通信尾会变差，坐实 (C)。

---

## 6) 一句话总结

- 你说的“**expert compute 无 per-token 排队**”在 fused MoE 的计算阶段基本成立；
- 这意味着用 expert queue 做影子价格（PriceMoE 原型）**很可能对 throughput 不敏感**，甚至会因 padding/碎片化/通信而变差；
- 下一步正确方向是：把价格迁移到 **padding-aware** 或 **device/通信/准入-aware** 的 proxy 上（Fluid‑SemMoE 的工程化简版）。
