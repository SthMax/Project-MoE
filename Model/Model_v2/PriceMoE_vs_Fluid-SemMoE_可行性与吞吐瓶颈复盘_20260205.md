# PriceMoE 实验吞吐不升的复盘（以及为何 Fluid‑SemMoE 更贴近真实瓶颈）

> 日期：2026-02-05  
> 背景：你反馈 PriceMoE 的实验结果很糟糕，并没有提升 throughput；你给出的关键解释是：
>
> - **Expert Compute**：GPU 上的 FFN 是高度优化的 batched matrix multiplication，服务时间与 batch size 线性相关，**无 per-token 排队**。

下面把我的分析与下一步建议整理成可直接引用/落地的版本。

---

## 1) “无 per-token 排队”这句话：对，但只覆盖了瓶颈的一部分

## 0) 🗺️ 一张图看懂：从“现象”到“可验证的瓶颈假设”再到“可实施干预”

> 这张图的目标是把整个思路变成一个闭环：**测什么 → 判断哪里卡 → 改哪里 → 用数据验收**。

> 你的 Markdown 预览不显示 Mermaid，多半是当前预览器/插件没启用 Mermaid 渲染。
> 为了让这份文档**不依赖任何插件也能读**，这里提供一个“纯文本版总览图”。

```text
                         ┌────────────────────────────────────────────────┐
                         │  现象：PriceMoE 吞吐差 / P99 变差               │
                         └────────────────────────────────────────────────┘
                                           │
                                           v
┌───────────────────────────────────────────────────────────────────────────┐
│  瓶颈假设（可证伪）= 你到底是在“多算了”还是“等同步了”还是“被准入卡住了”？     │
├───────────────────────────────────────────────────────────────────────────┤
│ H1  Padding/Block 对齐 → 无效 token-expert pair ↑ → GEMM/访存量 ↑          │
│     观测：num_tokens_post_padded, waste_total, small_bucket_ratio         │
│ H2  Regroup/Permute/Sort → 碎片化/索引/重排开销 ↑                          │
│     观测：E_active, n_e 分布, (可选) permute/regroup 时间                  │
│ H3  EP 通信 All2All → send_bytes 更碎/peer 更多 → 同步 tail ↑              │
│     观测：per-peer send_bytes 分布, nonzero_peers, all2all 时间            │
│ H4  Admission/KV/显存准入 → 可服务 batch 变小/波动 ↑ → 吞吐上限下降       │
│     观测：prefill/decode batch size, 等待队列/调度周期, KV 使用率           │
└───────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌───────────────────────────────────────────────────────────────────────────┐
│  干预映射（先做最小可逆改动）                                                  │
├───────────────────────────────────────────────────────────────────────────┤
│ 若 H1/H2 主导：P1 候选内集中（top-k' → penalty 重排 → top-k）              │
│   目标：E_active ↓、small buckets ↓、waste_total ↓（吞吐↑，质量守门）       │
│ 若 H3/H4 主导：P2 device-level proxy（通信/准入-aware，工程简化版）         │
│   目标：nonzero_peers ↓ / tail ↓、batch 形态稳定、P99↓                      │
└───────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌───────────────────────────────────────────────────────────────────────────┐
│  统一验收（必须同时看“结果 + 解释”）                                         │
│    tokens/s ↑   +   P95/P99 ↓   +   质量≈   +   proxy（waste/bytes 等）↓   │
└───────────────────────────────────────────────────────────────────────────┘
```

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

下面这张图专门解释 **padding 在 GPU kernel 里的位置**，以及它为什么和 routing（top-k、active experts）强耦合：

> 同理，为了避免 Mermaid 不显示，这里也给一个“纯文本版 padding↔routing↔GPU kernel”示意。

```text
┌─────────────────────────────── Routing（top-k） ───────────────────────────────┐
│ 输入：token hidden states  X[T, H]                                              │
│ 计算：router logits  L[T, E]                                                    │
│ 输出：topk_ids[T, k], topk_weights[T, k]                                        │
└────────────────────────────────────────────────────────────────────────────────┘
                       │
                       v
┌────────────────────────── Binning / Regroup（按 expert 聚合） ─────────────────┐
│ 统计：n_e = #tokens routed to expert e  （由 topk_ids 直接计数得到）            │
│ 派生：E_active = |{ e | n_e > 0 }|                                              │
│ 直觉：E_active ↑ 往往意味着更多“小桶”出现（很多 n_e 很小）                     │
└────────────────────────────────────────────────────────────────────────────────┘
                       │
                       v
┌──────────────────────── Padding / Align（GPU kernel 的块结构约束） ─────────────┐
│ 常量：B = BLOCK_SIZE_M                                                          │
│ 对齐：padded(n_e) = ceil(n_e / B) * B                                           │
│ 总量：num_tokens_raw = Σ_e n_e = T * k                                          │
│      num_tokens_post_padded = Σ_e padded(n_e)                                   │
│ 浪费：waste_total = num_tokens_post_padded - num_tokens_raw                     │
│ 关键：当存在大量 0 < n_e < B 时，每个这样的 expert 都会“至少占用 1 个 block”    │
└────────────────────────────────────────────────────────────────────────────────┘
                       │
                       v
┌────────────────────────── GPU Fused Experts Kernel（真正耗时点） ───────────────┐
│ 计算：两次 GEMM（W1/W2）+ activation + 读写                                     │
│ 工作量：≈ num_tokens_post_padded  （而不是 num_tokens_raw）                     │
│ 后果：waste_total ↑ → 多算的 GEMM/访存 ↑ → throughput ↓ / P99 ↑                 │
└────────────────────────────────────────────────────────────────────────────────┘

一句话总结耦合关系：routing 让 token 更“分散” → E_active ↑ + 小桶 ↑ → padding ↑ → num_tokens_post_padded ↑ → kernel 工作量 ↑。
```

读图要点：

- routing 决定了每个 expert 的 token 数 $n_e$；
- kernel 的 `BLOCK_SIZE_M (=B)` 要求每个 expert 的 token 维按块对齐，于是得到 `padded(n_e)`；
- 总工作量的关键不是 $\sum n_e$，而是 $\sum \text{padded}(n_e)$，差值就是 **padding waste**；
- 当 routing 让更多 experts 被点亮（$E_{active}$ ↑）且很多 expert 落入小桶（$n_e < B$）时，$\sum \text{padded}(n_e)$ 会快速膨胀，吞吐就会掉。

源码锚点（从“对齐”到“kernel 工作量”）：

- `external/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py::moe_align_block_size`（以及 `moe_permute_unpermute.py` / `fused_moe.py` 中调用处）会产出 `num_tokens_post_padded`。
- `external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` 的 Triton kernel 内部会读取 `num_tokens_post_padded`，并用 `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: ...` 这类边界判断来裁剪无效 block——因此 `num_tokens_post_padded` 直接定义了“实际执行的 block 数/工作量”。

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

---

## 7) 实验应该怎么设计与实施（从“判定”到“最终效果”的最短路径）

下面给一个 **可直接落地到 vLLM/本仓库** 的实验方案：先判定瓶颈，再做最小改动干预，最后验收。

### 7.1 实验目标与验收口径（最重要的 4 个数）

1. **吞吐**：tokens/s（prefill + decode 分开统计更好）
2. **尾延迟**：P95/P99（按 request 或按 step，一定要固定口径）
3. **质量守门**：与 baseline 的输出差异（最少做：同 prompt 的生成差异率 / 或困惑度/准确率的代理指标）
4. **机制解释指标（proxy）**：
   - padding：`waste_total = num_tokens_post_padded - (T * K)`（或按 expert 求和的等价统计）
   - 稀疏碎片化：`E_active`、`n_e < B` 的 expert 比例
   - 通信：每 peer `send_bytes` 分布、nonzero peers 数、all2all 时间

> 经验上：要想“能解释”，至少要同时记录 (1)(2) 和一个 proxy；否则只能看到涨跌但很难知道是不是走对方向。

### 7.2 对照组（A/B/C）怎么设，避免误判

- **A: Baseline**（原始 vLLM routing，不开 PriceMoE）
- **B: PriceMoE 当前实现**（你现在的设置，吞吐差的那套）
- **C: PriceMoE + 只做统计埋点**（不改路由，只加 log/metric，确保埋点本身不引入明显开销）

如果后续做干预，再在 C 的基础上加：

- **D: padding-aware / active-experts-aware 候选内集中**（最小改动干预）
- **E: device-level proxy 路由/准入**（更系统的干预，放到第二阶段）

### 7.3 Phase 1（1 天内）：把“到底卡哪”坐实 —— 只做统计，不改策略

#### (1) padding / 对齐相关指标（强烈建议）

**埋点位置建议**（对应 fused MoE 真实工作量）：

- 统计 `topk_ids` 后，先得到每个 expert 的 token 计数 $n_e$（按 batch）
- 用配置里的 `BLOCK_SIZE_M` 计算：
  - `padded(n_e) = ceil(n_e / B) * B`
  - `waste_total = sum_e padded(n_e) - sum_e n_e`
- 同步记录：`E_active = #(n_e > 0)`、`small_bucket_ratio = #(0 < n_e < B) / E_active`

判定标准（简单粗暴但有效）：

- 若 **B(PriceMoE)** 相比 **A(Baseline)**：
  - `waste_total` 显著 ↑（或 small_bucket_ratio ↑ 且 E_active ↑）
  - 同时 throughput ↓ / P99 ↑

则说明 PriceMoE 的“均衡”在 fused kernel 下更像“碎片化”，H1/H2 很可能成立。

#### (2) EP 通信相关指标（如果你在跑 EP）

**埋点建议**：在 all2all 前后统计每个 peer 的 `send_bytes`，以及 all2all kernel/collective 的耗时。

判定标准：

- 若 PriceMoE 使 nonzero peers ↑、`send_bytes` 分布更碎（更多小包），且 all2all 时间/尾部 ↑，则 H3 成立。

#### (3) admission / batching 形态（如果你怀疑是调度/显存准入）

记录每 step 的：prefill/decode 的 batch size（或 num_seqs）、等待队列长度、KV cache 占用（哪怕是粗粒度 ratio）。

若 PriceMoE 触发更频繁的 batch 变小/波动/回退，则 H4 成立。

### 7.4 Phase 2（2~3 天）：做“最小可逆”的策略改动，验证能否把吞吐拉回去

#### 方案 D：候选内集中（top-k' → penalty 重排 → top-k）

核心思想：**不改变 router logits 的大方向**，只在候选集合内做轻量重排，把 token 往更少的 experts 聚合，从而降低 `E_active`、small buckets、padding waste。

实现要点（工程约束友好）：

- 对每个 token，先取 top-k'（例如 k' = k + 4 或 2k）
- 在这 k' 个候选里，按下式调整得分：

$$
  ext{score}(e) = \text{logit}(e) - \lambda \cdot \widehat{\text{padCost}}(e)
$$

其中 $\widehat{\text{padCost}}(e)$ 可以用“当前 batch 该 expert 的计数预测”构造（例如先根据原始 top-k 得到一个 n_e 估计，再对候选做一轮重排）。

验收：

- throughput 至少回到 baseline 附近/更好
- `waste_total`、`E_active`、small_bucket_ratio 明显下降
- 质量守门指标不明显变差

#### 方案 E：device-level proxy（第二阶段再做）

当 Phase 1 明确 H3/H4 主导时，再推进这条线：把 price 定义在 device/communication/admission 上，而不是在 expert 上。

### 7.5 实施细节：如何保证实验结果可信

- 固定：模型版本、并行配置（TP/PP/EP）、prompt 集合、请求到达模式、随机种子（尽量）
- 每组至少跑：
  - warmup（让 CUDA graph/缓存稳定）
  - 多次重复取均值/方差
- 记录：同一份日志里同时有吞吐/尾延迟/proxy（三者缺一就很难解释）

