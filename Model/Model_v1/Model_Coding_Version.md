# PriceMoE Final Model

> 本文档整合 PriceMoE 模型的最终版本，包含系统背景、数学建模、约束设计和策略导出。

---

## 目录

1. [Background: vLLM MoE 推理系统的架构理解](#1-background-vllm-moe-推理系统的架构理解)
2. [Mathematical Model](#2-mathematical-model)
3. [Constraints](#3-constraints)
4. [Policy Derivation](#4-policy-derivation)
5. [Implementation Mapping](#5-implementation-mapping)

---

## 1. Background: vLLM MoE 推理系统的架构理解

### 1.1 两层调度架构：Request-level 建模，Token-level 应用

PriceMoE 采用**两层调度架构**，这是理解整个系统的关键：

| 层次 | 决策对象 | 约束类型 | vLLM 代码对应 |
|------|---------|---------|--------------|
| **Scheduler 层（Request-level）** | 请求 $r$ | Token budget, KV cache | `scheduler.schedule()` |
| **Router 层（Token-level）** | Token $j$ → Expert $e$ | 质量约束，负载均衡 | `FusedMoE._select_experts()` |

#### 为什么是 Request-level 建模？

**1. 调度视角**

vLLM Scheduler 的决策流程：
```
对于每个 request r ∈ running_queue:
    决定：本步调度多少 tokens → num_scheduled_tokens[r]
    受约束：∑_r n_r ≤ C  (token budget)
           blocks_needed(r) ≤ free_blocks (KV cache)
```

调度器**不关心**每个 token 去哪个专家，只关心：
- 这个请求能调度吗？
- 能调度多少 tokens？

**2. 约束语义**

约束的累加对象决定了建模层次：

| 约束 | 累加对象 | 语义 |
|------|---------|------|
| $\sum_r n_r \leq C$ | 请求 | "本步最多处理多少 tokens" |
| $\sum_r \text{blocks}(r) \leq B$ | 请求 | "所有活跃请求的 KV cache 总量" |

如果是 Token-level 建模，约束会变成 $\sum_j w_j(t) \leq W$（每个 token 的等待时间），但这在 vLLM 中**不存在**。

**3. 队列的含义**

| 建模层次 | 队列语义 | vLLM 对应 |
|---------|---------|----------|
| **Request-level** | 请求在 `waiting`/`running` 队列中等待 | ✅ 存在 |
| **Token-level** | Token 在某专家队列中等待处理 | ❌ 不存在 |

#### 为什么策略应用是 Token-level？

虽然建模是 Request-level，但**PriceMoE 的路由策略**必须在 Token-level 应用：

```python
# 在每个 MoE 层的 forward 中
def _select_experts(hidden_states, router_logits):
    # hidden_states: (T, D) - T 个 tokens
    # router_logits: (T, E) - 每个 token 对每个专家的分数
    
    # PriceMoE 修正评分（per-token, per-expert）
    adjusted_logits = router_logits - α * q_e  # 拥塞惩罚
    
    # 为每个 token 选择 Top-k 专家
    topk_weights, topk_ids = torch.topk(adjusted_logits, k=top_k, dim=-1)
```

原因：
- 同一个请求的不同 tokens 可能有不同的 router logits
- 每个 token 独立选择自己的 Top-k 专家
- 无法在 request-level 预先决定"这个请求的所有 tokens 都去专家 X"

---

### 1.2 "所有 Tokens 同时经过所有 MoE 层"的含义

这是理解 vLLM MoE 推理的关键概念。

#### 代码层面的理解

在 `model_runner.py` 中，模型的 forward 是**一次性**完成的：

```python
# model_runner.py L1047-1050
hidden_states = self.model(
    input_ids=input_batch.input_ids,  # 所有调度的 tokens
    positions=positions,
)
```

这意味着：
1. **Batch 内的所有 tokens 一起进入模型**
2. **每个 MoE 层是一个同步点**：所有 tokens 同时经过该层
3. **没有"token 排队等待专家"的概念**：一个 token 选中专家后，立即计算

#### 图示：单次 Forward 的执行流程

```
时间 →
─────────────────────────────────────────────────────────────────────►

Step t: Scheduler 调度 N 个请求，共 T 个 tokens
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Model Forward                               │
│                                                                    │
│  Embedding Layer: T tokens → hidden_states (T, D)                 │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Transformer Layer 0                                         │  │
│  │    Attention: 所有 T tokens 同时计算                         │  │
│  │    MoE Layer:                                                 │  │
│  │      - Router: 为每个 token 计算 scores (T, E)              │  │
│  │      - Top-k: 每个 token 选择 k 个专家                      │  │
│  │      - Expert Compute: 所有选中的 (token, expert) 对并行计算 │  │
│  │      - Combine: 加权合并专家输出                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Transformer Layer 1 ... Layer L-1                           │  │
│  │    (同上，所有 tokens 同时经过)                               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│        │                                                           │
│        ▼                                                           │
│  Output: hidden_states (T, D) → Sampling                          │
└───────────────────────────────────────────────────────────────────┘
```

#### 关键洞察

**没有"token 等待专家"的概念**：
- 在经典排队论中，任务到达后可能需要等待服务器空闲
- 在 MoE 推理中，**所有 token-expert 计算在一次 forward 中并行完成**
- 热门专家不会让 tokens "排队"，而是让**该 batch 的计算时间变长**

---

### 1.3 热门专家负载与 Throughput 的关系

#### 问题：为什么减少热门专家负载能提高 throughput？

既然没有"token 排队等待专家"，为什么专家负载不均会影响性能？

#### 答案：Batch 计算时间的瓶颈效应

在 FusedMoE 的实现中（来自 `fused_moe.py`），专家计算是**分组并行**的：

```python
# 简化的 FusedMoE 计算逻辑
for expert_id in range(num_experts):
    tokens_for_this_expert = topk_ids == expert_id  # 选中该专家的 tokens
    if tokens_for_this_expert.any():
        # 该专家的计算时间 ∝ tokens_for_this_expert.sum()
        expert_output[tokens_for_this_expert] = expert_ffn(
            hidden_states[tokens_for_this_expert]
        )
```

**瓶颈效应**：
- 假设专家 0 被 100 个 tokens 选中，专家 1 被 10 个 tokens 选中
- 专家 0 的计算时间 >> 专家 1 的计算时间
- **整个 MoE 层的耗时取决于最慢的专家**

$$T_{\text{MoE}} = \max_e T_e = \max_e f(u_e)$$

其中 $u_e$ 是专家 $e$ 处理的 token 数，$f(\cdot)$ 是计算时间函数（近似线性，但有 batching overhead）。

#### Throughput 的定义

$$\text{Throughput} = \frac{\text{Total tokens processed}}{\text{Total time}} = \frac{T}{\sum_{\ell=1}^{L} T_{\text{MoE}}^{(\ell)} + T_{\text{other}}}$$

其中 $L$ 是 MoE 层数，$T_{\text{other}}$ 是 attention 等其他计算的时间。

#### 负载均衡的收益

如果负载完全均衡：
$$u_e = \frac{k \cdot T}{E} \quad \forall e$$

则：
$$T_{\text{MoE}} = f\left(\frac{k \cdot T}{E}\right) \quad \text{（最优）}$$

如果负载不均衡（存在热门专家）：
$$u_{\text{hot}} \gg \frac{k \cdot T}{E}$$

则：
$$T_{\text{MoE}} = f(u_{\text{hot}}) \gg f\left(\frac{k \cdot T}{E}\right) \quad \text{（次优）}$$

**结论**：减少热门专家负载 → 降低 $\max_e u_e$ → 降低 $T_{\text{MoE}}$ → 提高 throughput

---

### 1.4 Virtual Queue $q_e(t)$ 的物理意义

在经典排队论中，队列长度表示"等待服务的任务数"。但在 MoE 中没有真正的等待，那么 $q_e(t)$ 的物理意义是什么？

#### 重新定义 Virtual Queue

$$q_e(t) = \text{专家 } e \text{ 在 batch } t \text{ 中被选中的 token 数}$$

或者采用指数移动平均（EMA）的形式：
$$q_e(t) = \beta \cdot q_e(t-1) + (1-\beta) \cdot u_e(t)$$

其中 $u_e(t) = \sum_{j \in \mathcal{B}(t)} \mathbb{I}\{e \in \text{Top-k}(j)\}$。

#### 物理意义：拥塞信号

$q_e(t)$ 不是"等待队列"，而是**拥塞程度的度量**：
- $q_e(t)$ 高 → 专家 $e$ 是热门专家 → 该专家是瓶颈
- $q_e(t)$ 低 → 专家 $e$ 利用率低 → 有冗余容量

#### 影子价格的解释

在 PriceMoE 的修正评分中：
$$\tilde{S}_{je}(t) = s_{je} - \alpha q_e(t)$$

$\alpha q_e(t)$ 可以解释为：
- **拥塞惩罚**：选择热门专家会增加瓶颈，降低整体效率
- **边际成本**：多一个 token 选择专家 $e$ 带来的系统开销增量

这与网络拥塞控制中的 backpressure routing 思想一致。

---

### 1.5 两层架构的关系图

```
                    ┌─────────────────────────────────────┐
                    │         Scheduler (Request-level)   │
                    │                                     │
                    │  约束: ∑_r n_r ≤ C                  │
                    │        ∑_r blocks(r) ≤ B            │
                    │                                     │
                    │  决策: 哪些请求？每个请求多少tokens？  │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ SchedulerOutput
                                   │ {req_id: num_tokens}
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Model Forward                                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  MoE Layer (Token-level)                        │  │
│  │                                                                 │  │
│  │  输入: hidden_states (T, D) ← 来自所有调度请求的 tokens        │  │
│  │                                                                 │  │
│  │  Router: 为每个 token j 计算 score(j, e)                       │  │
│  │                                                                 │  │
│  │  PriceMoE 策略应用:                                             │  │
│  │    S̃_je = s_je - α·q_e - γ·comm_penalty - δ·activation_cost   │  │
│  │    topk_ids[j] = argmax_k S̃_je                                │  │
│  │                                                                 │  │
│  │  约束（隐式）: 质量约束 (1-ε)·max                               │  │
│  │               负载均衡（通过 q_e 惩罚实现）                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  两层的耦合点:                                                        │
│  - 专家负载 q_e(t) 是 token-level 路由决策的累积结果                 │
│  - 可扩展: Scheduler 可感知 q_e，在 request-level 做预调度优化       │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 1.6 小结

| 概念 | 理解 |
|------|------|
| **建模层次** | Request-level（约束和调度决策） |
| **策略应用层次** | Token-level（路由决策） |
| **队列 $q_e(t)$** | 不是真正的等待队列，而是专家拥塞程度的度量 |
| **"同时经过"** | Batch 内所有 tokens 在每个 MoE 层同步计算，无等待 |
| **热门专家的影响** | 不是让 tokens 排队，而是增加 batch 计算时间（瓶颈效应） |
| **优化目标** | 减少 $\max_e u_e$ → 降低 MoE 层耗时 → 提高 throughput |

---

## 2. Mathematical Model

*（待补充：从 PriceMoE_Final_Model_v2.1.md 整合）*

---

## 3. Constraints

*（待补充：从 Constraint Survey.md 和 Feasibility Analysis 整合）*

---

## 4. Policy Derivation

*（待补充：影子价格推导和 PriceMoE 路由规则）*

---

## 5. Implementation Mapping

*（待补充：从 PriceMoE Implementation Feasibility Analysis.md 整合）*
