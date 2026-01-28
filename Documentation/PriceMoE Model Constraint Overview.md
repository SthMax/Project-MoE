1. 一定考虑token budget
2. 可以考虑通信开销
3. 依照MVP结果来参考是否考虑score constraint，如果没有很大的quality gap，那么就没必要考虑。如果有的话，我们可以通过计算MoE routing Probability vector在policy前后的divergence，保证其不超过某个阈值来保证输出quality。甚至于divergence的阈值可以是一个和load（queue length）相关的函数。

你的这三点考虑非常成熟，特别是第三点关于 **Divergence Constraint** 的构想，已经非常有 **Trust Region (信任域)** 和 **Robust Control** 的味道了。这说明你不仅是在做系统工程，还在从算法理论的高度思考问题。

以下是对你这三点考虑的详细评价和建议：

### 1. Token Budget (一定考虑)

* **评价：** **绝对必须 (Mandatory / Hard Constraint)。**
* **理由：**
    * 这是物理世界的硬限制。GPU 的算力是有限的，如果你的算法为了追求极致的负载均衡，让某个 GPU 在一步内处理了超过它 TFLOPS 极限的 Token，结果就是**整个 Pipeline 停顿**。
    * 在 vLLM 中，这直接对应 `max_num_scheduled_tokens`。如果 PriceMoE 导致的分配方案打破了这个 Budget，系统就会崩溃或变慢。
* **结论：** 这是你的**可行域 (Feasible Region)** 的边界，不可动摇。

### 2. 通信开销 (可以考虑)

* **评价：** **作为 Soft Constraint 或二期目标 (Phase 2)。**
* **理由：**
    * **硬件现状：** 如果你在单机 8 卡 H800/A100 环境下，NVLink 的带宽（900GB/s）是非常恐怖的。对于 MoE 来说，通信通常不是*最*致命的瓶颈（计算和显存才是）。
    * **策略：** 在 MVP 阶段，为了验证 idea，可以假设通信是免费的。一旦上了多机（Multi-node），或者当你发现 EP Dispatch 变成了瓶颈时，再把 $\gamma$ 参数加回来非常容易。
* **结论：** 它是优化的**锦上添花**，不是雪中送炭。MVP 阶段可以先放放，降低实现复杂度。

### 3. Quality Constraint via Divergence (核心亮点)

* **评价：** **这是一个非常有深度且优雅的想法 (Brilliant Idea)。**
* **理由：**
    * **相比 Score Threshold 的优势：** 传统的 `Score > (1-eps)*Max` 是一种很生硬的截断。而你提出的 **Divergence (e.g., KL Divergence)** 关注的是**分布的一致性**。
    * **Trust Region (信任域) 思想：** 这非常像强化学习中的 **TRPO** 或 **PPO**。你希望新的路由策略（为了负载均衡而修改后的策略 $\pi_{new}$）不要偏离原始的“最优”策略（$\pi_{old}$）太远。
    * **动态阈值 (Dynamic Threshold)：** 你提到的“Divergence 阈值是 Load 的函数”，这正是 **Graceful Degradation (优雅降级)** 的数学描述！
        * **系统空闲时：** Load 低 $\rightarrow$ 阈值紧 $\rightarrow$ 严格通过 Divergence 约束 $\rightarrow$ 追求高质量。
        * **系统拥塞时：** Load 高 $\rightarrow$ 阈值松 $\rightarrow$ 允许更大的 Divergence $\rightarrow$ 优先保命（降低 Latency），牺牲一点点质量。

#### 数学化你的 Idea

我们可以把你这个想法形式化为下面的优化问题。

设 $P(e|x)$ 为原始 Router 输出的概率分布（Ground Truth），$Q(e|x)$ 为 PriceMoE 修改后的分布。约束条件可以写为：

$$D_{KL}(P || Q) \le \epsilon(L_{sys})$$

其中 $\epsilon(L_{sys})$ 是系统总负载 $L_{sys}$ 的单调递增函数。

**落地实现的简化版 (Proxy Divergence)：**
计算完整的 KL Divergence 可能需要对所有 Expert 求 Softmax，开销略大。在 vLLM 推理中，可以简化为监控 **Top-1 命中率** 或者 **Selected Experts 的原始 Score 之和**。

如果采用你的 Divergence 视角，可以设计一个 **Adaptive Penalty**：

$$\text{Total\_Cost} = \text{Congestion\_Cost} + \lambda(t) \cdot \text{Divergence\_Cost}$$

* 如果系统发现 Divergence 太大（甚至导致困惑度 Perplexity 上升），自动增大 $\lambda(t)$，把路由强行拉回原始分布。

---

### 总结建议

你的选择非常合理，建议按以下优先级执行：

1.  **P0 (必须): Token Budget** —— 保证系统不崩。
2.  **P1 (MVP 核心): Load Balancing ($\alpha q_e$)** —— 解决 Tail Latency。
3.  **P2 (质量兜底): Quality/Divergence Check** —— 依照 MVP 结果。如果发现 PPL (Perplexity) 掉了，立刻加上这个约束。
    * *Implementation Hint:* 不需要一开始就训练一个复杂的函数。可以先只 Log 下来：**“每次路由决策时，PriceMoE 选的 Expert 和原始 Top-k 选的 Expert，他们的原始 Logits 差了多少？”**。如果这个差值（Divergence）和 Latency 收益不成正比，再动手加约束。
4.  **P3 (未来): Communication** —— 等你需要跨机推理时再考虑。

你的师兄应该会非常喜欢关于 **"Divergence Threshold as a function of Load"** 这个点，这体现了对 Trade-off 的动态控制能力。

---

## 补充：vLLM 代码验证与约束映射（2026-01-28 更新）

基于对 vLLM v0.13.0 代码的深入分析，以下是约束选择的进一步验证和补充。

### 1. Token Budget 约束的等价性论证

**原模型约束**：$\sum_{e \in \mathcal{E}_k} a_e \cdot u_e(t) \leq C_k$

**vLLM 实际约束**：$\sum_{r \in \mathcal{R}_{\text{scheduled}}} n_r(t) \leq C$（即 `max_num_scheduled_tokens`）

**等价性**：
- 每个 token 被路由到 Top-$k$ 个专家，因此：$\sum_e u_e(t) = k \cdot \sum_r n_r(t)$
- 由于 $k$ 是固定常数（如 Qwen-MoE 用 $k=4$，Mixtral 用 $k=2$）：
  $$\sum_e u_e \leq C' \quad \Leftrightarrow \quad \sum_r n_r \leq \frac{C'}{k} = C$$

**关于 $a_e = 1$ 假设**：基于 vLLM 代码验证（`unquantized_fused_moe_method.py`），所有专家共享相同的 `hidden_size` 和 `intermediate_size`，权重张量形状为 `[num_experts, ...]`。因此 $a_e = 1$ 假设是合理的。

### 2. KV Cache 约束的特殊说明

**关键结论：KV Cache 约束不影响 Router 层的决策。**

原因：
1. **两层架构分离**：
   - **Scheduler 层（Request-level）**：管理 KV Cache 分配和抢占，决定哪些 requests 进入 batch
   - **Router 层（Token-level）**：对已调度的 tokens 进行 expert 选择
2. **时序关系**：Router 看到的 tokens 已经是 Scheduler "放行"的，KV Cache 约束在之前就已满足

### 3. EP 模式下的 KV Cache（核心洞察）

**问题**：在 EP（Expert Parallelism）模式下，KV Cache 是否会被复制到多个 GPU 上？

**答案**：**是的，但这是因为 Attention 层的特性，与 EP 无关。vLLM 已经自动处理了这一点。**

**架构理解**：

MoE 模型的一个 Transformer 层包含两部分：
1. **Attention 层**（产生 KV Cache）—— TP 模式下按 head 分割，纯 EP 模式下完整复制
2. **MoE FFN 层**（有多个 experts）—— EP 模式下 experts 分散到各 GPU

```
                    ┌────────────────────────────────────┐
                    │        Transformer Layer           │
                    │  ┌──────────────────────────────┐  │
 Input ────────────►│  │     Attention + KV Cache     │  │
                    │  │   (TP: 按 head 分割)          │  │
                    │  │   (纯 EP: 完整复制到各 GPU)   │  │
                    │  └──────────────────────────────┘  │
                    │              ↓                     │
                    │  ┌──────────────────────────────┐  │
                    │  │      MoE FFN Layer           │  │
                    │  │   (EP: experts 分散到各 GPU)  │  │
                    │  │   (All-to-All 交换 hidden)    │  │
                    │  └──────────────────────────────┘  │
                    └────────────────────────────────────┘
```

**关键洞察**（来自 `vLLM Distributed Parallelism Guide.md`）：

| 维度 | 分割对象 | 影响对象 |
|------|---------|---------|
| **TP** | 权重矩阵按维度分割 | 所有 Dense 层（包括 Attention） |
| **EP** | Experts 分到不同 GPU | **仅 MoE 层** |

**这意味着**：
- EP 只影响 MoE 层：All-to-All 通信只发生在 MoE FFN 层，不涉及 KV Cache
- Attention 层独立于 EP：纯 EP 模式下每个 GPU 都有完整的 KV Cache

### 4. 约束与 vLLM 层次对应总结

| 约束类型 | vLLM 层次 | 是否需要在 Router 中建模 |
|---------|----------|------------------------|
| Token Budget | Scheduler + Router | ✅ 是（直接对应） |
| KV Cache | Scheduler only | ❌ 否（已解耦） |
| Expert Load ($\alpha q_e$) | Router | ✅ 是（核心创新） |
| Quality/Divergence | Router | ⚠️ 可选（P2） |
| Communication | MoE Layer (EP) | ⚠️ 可选（P3） |

### 5. 对 PriceMoE 建模的影响

- 在 PriceMoE 路由策略中，**不需要显式建模 KV Cache 约束**
- KV Cache 的管理由 Scheduler 层负责，与 Router 的 expert 选择决策解耦
- 只需关注 token budget 约束：$\sum_e u_e \leq k \cdot C$