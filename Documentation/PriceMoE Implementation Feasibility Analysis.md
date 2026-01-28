# PriceMoE 实现可行性分析：从流体模型到 vLLM 代码落地

> 本文档结合你的 Fluid_MoE 模型（version 1.0.0/1.0.1）与 vLLM v0.13.0 代码架构，分析约束选择、策略导出、以及代码实现的可行性和工作量。

---

## 目录

1. [你的模型核心思想回顾](#1-你的模型核心思想回顾)
2. [约束选择：从数学到 vLLM 代码的映射](#2-约束选择从数学到-vllm-代码的映射)
3. [策略导出：影子价格 → 路由规则](#3-策略导出影子价格--路由规则)
4. [代码实现难度评估](#4-代码实现难度评估)
5. [分阶段实施路线图](#5-分阶段实施路线图)
6. [总结与建议](#6-总结与建议)

---

## 1. 你的模型核心思想回顾

### 1.1 PriceMoE 的数学框架

根据 `Fluid_MoE qwen version 1.0.0/1.0.1`，你的模型有以下关键组件：

**状态变量**：
- $Q_e(t)$：专家 $e$ 的队列长度（待处理 token 数）
- $X_e(t) \in \{0,1\}$：专家激活状态
- $F_{kk'}(t)$：GPU 间通信流量

**动力学方程**：
$$\dot{q}_e(t) = \lambda_e(t) - \mu_e \cdot \psi(|\mathcal{A}_k|) \cdot u_e(t)$$

**约束系统**：
1. 计算约束：$\sum_{e \in \mathcal{E}_k} a_e \cdot u_e(t) \leq C_k$
2. 显存约束：$\sum_{e \in \mathcal{E}_k} X_e(t) \cdot \text{size}(e) \leq M_k$
3. 通信约束：$F_{kk'}(t) \leq B_{kk'}$
4. 质量约束：$\text{Score}(j,e) \geq (1-\epsilon) \cdot \max_{e'} \text{Score}(j,e')$

**导出策略**（PriceMoE 路由规则）：
$$\tilde{S}_{je}(t) = s_{je} - \alpha q_e(t) - \gamma_{L(e),k_j} - \delta \cdot \mathbb{I}_{\{X_e(t)=0\}}$$

选择 Top-k 个最大化 $\tilde{S}_{je}(t)$ 的专家。

### 1.2 关键创新点

1. **影子价格反馈**：用队列长度 $q_e(t)$ 作为拥塞信号
2. **跨 GPU 通信惩罚**：$\gamma_{L(e),k_j}$ 避免不必要的 All-to-All
3. **专家激活成本**：$\delta \cdot \mathbb{I}_{\{X_e(t)=0\}}$ 避免频繁激活/休眠
4. **迟滞机制**：稳定专家激活状态，避免震荡

---

## 2. 约束选择：从数学到 vLLM 代码的映射

> **⚠️ 更新说明 (2026-01-21)**：本节已根据 vLLM v0.13.0 实际代码进行全面核对和更正，修复了之前版本中的幻觉内容（行号错误、函数名错误、代码逻辑简化等）。

### 2.1 约束对照表

| 你的模型约束 | vLLM 代码对应 | 实现难度 | 数据来源 | 更正说明 |
|------------|-------------|---------|---------|---------|
| **计算约束** $\sum a_e u_e \leq C_k$ | `max_num_scheduled_tokens` | ⭐ 低 | `SchedulerConfig` | ✅ 一致 |
| **显存约束** $\sum X_e \cdot \text{size}(e) \leq M_k$ | KV cache 块分配 | ⭐⭐ 中 | `KVCacheManager` | ⚠️ 抢占逻辑比之前描述复杂 |
| **通信约束** $F_{kk'} \leq B_{kk'}$ | EP All-to-All | ⭐⭐⭐ 高 | 需新增监控 | ✅ 一致 |
| **质量约束** $\text{Score} \geq (1-\epsilon)\max$ | Top-k routing 修改 | ⭐⭐ 中 | `FusedMoE._select_experts()` | ⚠️ 函数名和路径已更正 |
| **队列长度** $q_e(t)$ | ❌ 未原生支持 | ⭐⭐⭐ 高 | 需新增采集 | ✅ 一致，但已有 `moe_stats.py` 部分支持 |

### 2.2 约束详细映射

---

#### 约束 1：计算容量（Token Budget）

**数学形式**：
$$\sum_{r \in \mathcal{R}} n_r(t) \leq C$$

其中 $n_r(t)$ 表示请求 $r$ 在时刻 $t$ 调度的 token 数，$C$ 是每步最大调度 token 数。

> **与原模型区别**：原模型按专家 $e$ 累加（$\sum_e a_e u_e$），但 vLLM 是按请求累加。两者等价性需要证明（见下方"数学抽象"部分）。

**vLLM 代码体现**（来自 `vllm/v1/core/sched/scheduler.py`）：

```python
# scheduler.py L256-L270 (schedule 函数核心逻辑)
def schedule(self) -> SchedulerOutput:
    """
    调度算法核心：没有显式的 "prefill phase" 或 "decoding phase"，
    每个请求有 num_computed_tokens 和 num_tokens_with_spec，
    调度器尝试让 num_computed_tokens 追上 num_tokens_with_spec。
    """
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    
    # ========== 关键约束：Token Budget ==========
    token_budget = self.max_num_scheduled_tokens  # ← 这就是 C，来自 scheduler_config
    # ...

# scheduler.py L278-L310 (RUNNING 队列调度)
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        
        # 计算该请求需要处理的新 token 数
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        
        # 长 prefill 截断（可配置阈值）
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        
        # ========== Token Budget 约束 ==========
        num_new_tokens = min(num_new_tokens, token_budget)  # 受 budget 约束
        
        # 确保不超过 max_model_len
        num_new_tokens = min(
            num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
        )
        # ...
        
        # 成功调度后消耗 budget
        token_budget -= num_new_tokens
        req_index += 1
```

**关键配置来源**：
```python
# scheduler.py L93-95 (__init__ 中)
self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # 最大并发请求数
self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens  # Token Budget C
self.max_model_len = vllm_config.model_config.max_model_len  # 单请求最大长度
```

**映射关系**：
| 数学符号 | vLLM 变量 | 说明 |
|---------|----------|------|
| $C$ | `max_num_scheduled_tokens` | 每步最大调度 token 数（硬上限） |
| $n_r(t)$ | `num_scheduled_tokens[req_id]` | 请求 $r$ 在该步调度的 token 数 |
| 约束检查 | `min(num_new_tokens, token_budget)` | 截断到剩余 budget |

**数学抽象（更新 2026-01-23）**：

原模型约束：$\sum_{e \in \mathcal{E}_k} a_e \cdot u_e(t) \leq C_k$

vLLM 实际约束：$\sum_{r \in \mathcal{R}_{\text{scheduled}}} n_r(t) \leq C$

---

**等价性论证：从 Token 上限视角**

我们可以将约束从"专家计算开销"转换为"Token 调度上限"，这样更自然地对应 vLLM 的实现：

1. **vLLM 约束的本质**：系统对每个调度周期可处理的 token 总数设有上限 $C$（即 `max_num_scheduled_tokens`）

2. **与专家工作量的关系**：每个 token 被路由到 Top-$k$ 个专家，因此专家调用总数为：
   $$\sum_e u_e(t) = k \cdot \sum_r n_r(t)$$

3. **约束等价性**：由于 $k$ 是固定常数（如 Qwen-MoE 用 $k=4$，Mixtral 用 $k=2$），有：
   $$\sum_e u_e \leq C' \quad \Leftrightarrow \quad \sum_r n_r \leq \frac{C'}{k} = C$$

**结论**：vLLM 的 token budget 约束 **直接等价于** 对专家工作量总和的间接约束 $\sum_e u_e \leq k \cdot C$。

---

**关于 $a_e = 1$ 假设的合理性**

原模型假设所有专家处理单个 token 的计算开销相同（$a_e = 1$）。基于 vLLM v0.13.0 代码验证，**这一假设是合理的**：

1. **架构一致性**（代码证据来自 `unquantized_fused_moe_method.py` L136-180）：
   ```python
   # 所有专家共享相同的 hidden_size 和 intermediate_size
   w13_weight = torch.empty(num_experts, 2*intermediate_size, hidden_size, ...)
   w2_weight = torch.empty(num_experts, hidden_size, intermediate_size, ...)
   ```
   - `hidden_size` 和 `intermediate_size` 是**标量参数**，不是 per-expert 数组
   - 权重张量形状为 `[num_experts, ...]`，所有专家维度完全相同

2. **FLOPs 对称**：单个 token 通过任意专家的计算量完全相同：
   - Gate proj (W1): $\text{hidden\_size} \times \text{intermediate\_size}$
   - Up proj (W3): $\text{hidden\_size} \times \text{intermediate\_size}$
   - Down proj (W2): $\text{intermediate\_size} \times \text{hidden\_size}$

3. **主流模型验证**：Mixtral、Qwen-MoE、DeepSeek-V3 等均使用相同维度的专家网络

**实际运行中的二阶效应**（不影响一阶建模）：

| 因素 | 影响 | 是否打破 $a_e = 1$ |
|------|------|---------------------|
| 专家负载不均衡 | 不同专家被路由到的 token 数不同 | ❌ 否（这是 $u_e$ 的变化，不是 $a_e$） |
| 批处理效率 | 高负载专家可能获得更好的 GPU 利用率 | ⚠️ 二阶效应，可忽略 |
| 内存访问模式 | 专家权重的缓存命中率可能不同 | ⚠️ 实践中影响有限 |

---

**推荐的模型表述**

将约束从"计算开销"改为"Token 调度上限"更加清晰：

> 系统对每个调度周期可处理的 token 总数设有上限 $C$，即 $\sum_r n_r \leq C$，等价于对专家工作量的间接约束 $\sum_e u_e \leq k \cdot C$。

这样做的好处：
- ✅ 避免了对计算均匀性的显式假设（虽然该假设是合理的）
- ✅ 直接对应 vLLM 的 `max_num_scheduled_tokens` 语义
- ✅ 更符合实际调度器的工作方式（调度器先决定调度多少 token，路由决策在之后发生）

**实现难度**：⭐ 低（已有现成接口，参数可配置）

---

#### 约束 2：显存容量（KV Cache）

**数学形式**：
$$\sum_{r \in \mathcal{R}_{\text{active}}} \text{blocks}(r) \leq B_{\text{total}}$$

其中 $\text{blocks}(r)$ 是请求 $r$ 占用的 KV cache block 数，$B_{\text{total}}$ 是 GPU 总 block 数。

**vLLM 代码体现**（来自 `vllm/v1/core/sched/scheduler.py` + `kv_cache_manager.py`）：

```python
# scheduler.py L350-L403 (KV cache 分配与抢占的完整逻辑)
    # 为请求分配 KV cache blocks
    with record_function_or_nullcontext("schedule: allocate_slots"):
        while True:
            # ========== 尝试分配 ==========
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )

            if new_blocks is not None:
                # 分配成功，跳出循环
                break

            # ========== 分配失败，触发抢占 ==========
            # 抢占策略取决于调度策略
            if self.policy == SchedulingPolicy.PRIORITY:
                # 优先级调度：抢占优先级最低（priority 和 arrival_time 最大）的请求
                preempted_req = max(
                    self.running,
                    key=lambda r: (r.priority, r.arrival_time),
                )
                self.running.remove(preempted_req)
                
                # 如果被抢占请求已经被调度过，需要回滚其资源
                if preempted_req in scheduled_running_reqs:
                    scheduled_running_reqs.remove(preempted_req)
                    token_budget += num_scheduled_tokens[preempted_req.request_id]  # 返还 token budget
                    req_to_new_blocks.pop(preempted_req.request_id)
                    num_scheduled_tokens.pop(preempted_req.request_id)
                    # ... 回滚 encoder inputs 等
                    req_index -= 1
            else:
                # FCFS 调度：抢占队列末尾请求
                preempted_req = self.running.pop()

            # 执行抢占
            self._preempt_request(preempted_req, scheduled_timestamp)
            preempted_reqs.append(preempted_req)
            
            if preempted_req == request:
                # 自己被抢占了，无法调度
                break

    if new_blocks is None:
        # 无法调度该请求
        break

# scheduler.py L798-L815 (_preempt_request 函数)
def _preempt_request(self, request: Request, timestamp: float) -> None:
    """
    抢占请求：释放其资源并放回等待队列。
    注意：调用前需先从 running 队列中移除请求。
    """
    assert request.status == RequestStatus.RUNNING
    
    # ========== 释放资源 ==========
    self.kv_cache_manager.free(request)  # 释放 KV cache blocks
    self.encoder_cache_manager.free(request)  # 释放 encoder cache
    
    # ========== 更新请求状态 ==========
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0  # 重置进度（需重新计算）
    request.spec_token_ids.clear()
    request.num_preemptions += 1
    
    # ========== 放回等待队列头部 ==========
    self.waiting.prepend_request(request)
```

**KVCacheManager.allocate_slots 内部逻辑**（来自 `vllm/v1/core/kv_cache_manager.py`）：

```python
# kv_cache_manager.py L203-L290 (allocate_slots 函数)
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
) -> KVCacheBlocks | None:
    """
    为请求分配 KV cache slots。
    
    Blocks 布局:
    | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
              |            < to be allocated >           |
    
    返回:
        新分配的 blocks，如果空间不足则返回 None。
    """
    # ... 计算需要的总 token 数
    num_tokens_need_slot = min(
        total_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len,
    )
    
    # 释放滑动窗口外的 blocks（优化空间利用）
    self.coordinator.remove_skipped_blocks(
        request.request_id, total_computed_tokens
    )
    
    # ========== 检查是否有足够的空闲 blocks ==========
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=num_local_computed_tokens + num_external_computed_tokens,
    )

    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        # ========== 空间不足，返回 None 触发调度器抢占 ==========
        return None
    
    # ... 分配 blocks 并返回
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id, num_tokens_need_slot, num_encoder_tokens
    )
    return self.create_kv_cache_blocks(new_blocks)
```

**映射关系**：
| 数学符号 | vLLM 变量/函数 | 说明 |
|---------|---------------|------|
| $B_{\text{total}}$ | `block_pool.get_num_free_blocks()` | 当前空闲 block 数 |
| $\text{blocks}(r)$ | `coordinator.get_num_blocks_to_allocate(...)` | 请求需要的新 block 数 |
| 约束检查 | `num_blocks_to_allocate > free_blocks` → `return None` | 触发抢占 |

**数学抽象（更新）**：

原模型约束：$\sum_{e \in \mathcal{E}_k} X_e(t) \cdot \text{size}(e) \leq M_k$

vLLM 实际约束：
1. **分配时检查**：$\text{blocks\_needed}(r) \leq \text{free\_blocks}$
2. **全局约束**：$\sum_{r \in \mathcal{R}_{\text{active}}} \text{blocks}(r) \leq B_{\text{total}}$

**关键区别**：
- 原模型是**专家层面**的显存约束（每个专家权重 + 激活状态）
- vLLM 是**请求层面**的 KV cache 约束（每个请求的 K/V tensors）
- MoE 专家权重是**静态分配**的（在模型加载时确定），不参与动态调度

**实现难度**：⭐⭐ 中（框架完善，但抢占逻辑复杂，需理解 block 管理）

---

#### 关于 EP 模式下 KV Cache 的处理（2026-01-28 更新）

**问题**：在 EP（Expert Parallelism）模式下，KV Cache 是否会被复制到多个 GPU 上？

**结论**：**是的，KV Cache 会被复制，但这是因为 Attention 层的特性，与 EP 无关。vLLM 已经自动处理了这一点。**

**架构理解**：

MoE 模型的一个 Transformer 层包含两部分：
1. **Attention 层**（产生 KV Cache）—— TP 模式下按 head 分割，纯 EP 模式下完整复制
2. **MoE FFN 层**（有多个 experts）—— EP 模式下 experts 分散到各 GPU

```
                    ┌────────────────────────────────────┐
                    │        Transformer Layer           │
                    │  ┌──────────────────────────────┐  │
 Input ────────────►│  │     Attention + KV Cache     │  │
                    │  │   (TP: 在 GPU 间按 head 分割)  │  │
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
| **TP** | 权重矩阵按维度分割 | **所有 Dense 层（包括 Attention）** |
| **EP** | Experts 分到不同 GPU | **仅 MoE 层** |

**这意味着**：

1. **EP 只影响 MoE 层**：All-to-All 通信只发生在 MoE FFN 层，不涉及 KV Cache
2. **Attention 层独立于 EP**：
   - 同时启用 TP：KV Cache 按 head 维度分割
   - 未启用 TP（纯 EP）：每个 GPU 都有完整的 Attention 权重和 KV Cache

**对 PriceMoE 模型的影响**：

✅ **好消息：KV Cache 约束仍然不影响 Router 层的决策**

原因：
1. **KV Cache 约束在 Scheduler 层**：Scheduler 决定哪些 requests 进入 batch
2. **EP 下的 All-to-All 在 MoE 层**：只涉及 hidden states 的交换，不涉及 KV Cache
3. **每个 GPU 的 KV Cache 容量**：仍然由 Scheduler 管理，与 EP 无关

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                            │
│   KV Cache 约束在这里生效                                    │
│   决定 batch 中有哪些 requests                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Model Forward (per GPU)                    │
│                                                             │
│  1. Attention Layer                                         │
│     - 每个 GPU 有自己的 KV Cache                             │
│     - TP 模式: KV Cache 按 head 分割                         │
│     - 纯 EP 模式: KV Cache 完整复制                          │
│                                                             │
│  2. MoE Layer (EP 发生在这里)                                │
│     - All-to-All 交换 hidden states                         │
│     - 各 GPU 计算本地的 experts                              │
│     - All-to-All 收回结果                                    │
│     - ❌ 不涉及 KV Cache                                     │
└─────────────────────────────────────────────────────────────┘
```

**结论**：
- EP 模式下 KV Cache 确实会被"复制"到各 GPU（如果没有 TP）
- vLLM 已经自动处理了这一点
- 这不改变约束分析：KV Cache 约束仍然在 Scheduler 层，与 Router 的 expert 选择决策无关
- PriceMoE 模型假设仍然有效：只需关注 token budget 约束 $\sum_e u_e \leq k \cdot C$

---

#### 约束 3：跨 GPU 通信（EP All-to-All）

**数学形式**：
$$F_{kk'}(t) \leq B_{kk'}$$

其中 $F_{kk'}(t)$ 是 GPU $k$ 到 GPU $k'$ 的通信流量，$B_{kk'}$ 是带宽上限。

**vLLM 代码体现**（来自 `vllm/model_executor/layers/fused_moe/layer.py`）：

```python
# layer.py L1743-L1755 (forward_native 函数)
def forward_native(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # ... padding 处理
    
    if self.shared_experts is None:
        if current_platform.is_tpu() or current_platform.is_cpu():
            fused_output = self.forward_impl(hidden_states, router_logits)
        else:
            # ========== MoE forward（包含隐式 All-to-All）==========
            fused_output = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, self.layer_name
            )
        return reduce_output(fused_output)
    # ...

# layer.py L1954-L2005 (forward_impl 中的 EP dispatch/combine)
def forward_impl(self, hidden_states, router_logits):
    # ...
    do_naive_dispatch_combine = self.dp_size > 1 and not isinstance(
        self.quant_method, FusedMoEModularMethod
    )
    
    if do_naive_dispatch_combine:
        # ========== EP All-to-All dispatch ==========
        dispatch_res = get_ep_group().dispatch(
            hidden_states_to_dispatch,
            router_logits,
            self.is_sequence_parallel,
            extra_tensors=extra_tensors,
        )
        hidden_states_combined, router_logits = dispatch_res
    
    # ... 专家计算
    final_hidden_states = self.quant_method.apply(...)
    
    def combine_output(states):
        if do_naive_dispatch_combine:
            # ========== EP All-to-All combine ==========
            states = get_ep_group().combine(states, self.is_sequence_parallel)
        return states
```

**问题**：vLLM **没有原生的通信流量监控**！

All-to-All 发生在：
1. `get_ep_group().dispatch()` - 将 hidden_states 分发到持有目标专家的 GPU
2. `get_ep_group().combine()` - 收集计算结果

**需要新增的监控**：
```python
# 在 dispatch/combine 前后统计
sent_tokens_per_gpu = torch.zeros(ep_size)
recv_tokens_per_gpu = torch.zeros(ep_size)

# 根据 topk_ids 统计每个 token 要发送到哪个 GPU
for token_idx, expert_ids in enumerate(topk_ids):
    for expert_id in expert_ids:
        target_gpu = expert_id % ep_size  # 简化映射
        sent_tokens_per_gpu[target_gpu] += 1
```

**映射关系**：
| 数学符号 | vLLM 变量/函数 | 说明 |
|---------|---------------|------|
| $F_{kk'}(t)$ | ❌ 未实现 | 需要在 dispatch/combine 中统计 |
| $B_{kk'}$ | ❌ 未暴露 | NCCL 带宽是硬件限制 |
| EP 分组 | `get_ep_group()` | Expert Parallel 通信组 |

**实现难度**：⭐⭐⭐ 高（需要修改 EP 通信层或添加 wrapper）

---

#### 约束 4：输出质量

**数学形式**：
$$\text{Score}(j,e) \geq (1-\epsilon) \cdot \max_{e'} \text{Score}(j,e')$$

**vLLM 代码体现**（来自 `vllm/model_executor/layers/fused_moe/layer.py` + `fused_moe.py`）：

> **⚠️ 更正**：之前文档引用了不存在的 `router/base_router.py L203-L245`。实际的 routing 逻辑在 `FusedMoE._select_experts()` 方法中。

```python
# layer.py L1553-1660 (_select_experts 函数)
def _select_experts(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据 router logits 选择 Top-k 专家。
    
    返回:
        (topk_weights, topk_ids): 权重和专家 ID
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_topk,
        fused_topk_bias,
    )
    
    indices_type = self.quant_method.topk_indices_dtype
    
    # ========== 路由策略选择 ==========
    
    # 1. 路由模拟（用于测试）
    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(...)
    
    # 2. DeepSeek V2/V3 使用 grouped_topk
    elif self.use_grouped_topk and valid_grouping():
        topk_weights, topk_ids = self._grouped_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=self.e_score_correction_bias,
        )
    
    # 3. 带 bias 校正的 topk
    elif self.e_score_correction_bias is not None:
        topk_weights, topk_ids = fused_topk_bias(...)
    
    # 4. 标准 fused_topk（最常用）
    elif self.custom_routing_function is None:
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
        )
    
    # 5. 自定义路由函数
    else:
        topk_weights, topk_ids = self.custom_routing_function(...)
    
    # ========== EPLB 映射（专家负载均衡）==========
    if self.enable_eplb:
        topk_ids = eplb_map_to_physical_and_record(
            topk_ids=topk_ids,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )
    
    # ========== 统计记录（用于 E1.1 实验）==========
    try:
        from vllm.moe_stats import record_topk
        record_topk(
            layer_name=self.layer_name,
            topk_ids=topk_ids,
            num_experts=int(router_logits.shape[-1]),
            k_experts=int(self.top_k),
        )
    except Exception:
        pass
    
    return topk_weights, topk_ids, zero_expert_result

# fused_moe.py L1347-1371 (fused_topk 函数)
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合的 Top-k 选择 + softmax 归一化。
    """
    M, _ = hidden_states.size()
    
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    
    # ========== 调用 CUDA kernel 进行 topk + softmax ==========
    topk_func = dispatch_topk_func(use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled())
    topk_weights, topk_ids = topk_func(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )
    
    return topk_weights, topk_ids, token_expert_indices
```

**映射关系**：
| 数学符号 | vLLM 变量 | 说明 |
|---------|----------|------|
| $\text{Score}(j,e)$ | `router_logits[j, e]` 经 softmax/sigmoid | 原始 gating score |
| $\arg\max_e$ | `torch.topk(..., k=self.top_k)` | Top-k 选择 |
| 质量约束 $(1-\epsilon)$ | ❌ 未原生实现 | 需在 `_select_experts` 后添加过滤 |

**实现难度**：⭐⭐ 中（可在 `_select_experts` 返回前添加过滤逻辑）

---

#### 约束 5：专家队列长度（Virtual Queue）

**数学形式**：
$$q_e(t) = \text{专家 } e \text{ 的当前"负载"（该 batch 被路由到 } e \text{ 的 token 数）}$$

**vLLM 现状**：❌ **没有显式的 per-expert 队列，但已有部分统计支持！**

> **⚠️ 更正**：之前说"完全没有这个概念"不准确。vLLM 有 `moe_stats.py` 模块用于统计 expert routing，且 E1.1 实验已经基于此实现。

```python
# layer.py L1672-1684 (已有的统计记录)
# 在 _select_experts 末尾
try:
    from vllm.moe_stats import record_topk
    record_topk(
        layer_name=self.layer_name,
        topk_ids=topk_ids,
        num_experts=int(router_logits.shape[-1]),
        k_experts=int(self.top_k),
    )
except Exception:
    pass

# moe_stats.py (你们 E1.1 扩展的模块)
def record_topk(layer_name, topk_ids, num_experts, k_experts):
    """
    记录每个专家被路由到的 token 数。
    可用于计算 virtual queue q_e(t)。
    """
    # topk_ids: (T, K) 其中 T 是 token 数，K 是 top_k
    # 统计每个专家被选中的次数
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    # ... 保存到全局统计
```

**vLLM 的调度是 per-request 的关键含义**：
1. Scheduler 调度的是 **请求（request）**，不是专家
2. 每个 token 在一个 forward 中选择 Top-k 专家，**计算立即完成**
3. 没有"token 排队等待某个专家计算"的概念
4. "队列长度"需要定义为 **batch 内被路由到该专家的 token 数**（瞬时负载）

**Virtual Queue 的定义**：
$$q_e(t) = \sum_{j \in \mathcal{B}(t)} \mathbb{I}\{e \in \text{Top-k}(j)\}$$

其中 $\mathcal{B}(t)$ 是当前 batch 中的所有 token。

**实现方案**：
```python
# 在 _select_experts 开始时读取上一步的统计
expert_load = self.get_expert_load_ema()  # 指数移动平均

# 修正 router_logits
adjusted_logits = router_logits - alpha * expert_load.unsqueeze(0)

# 执行 Top-k 选择
topk_weights, topk_ids = fused_topk(hidden_states, adjusted_logits, ...)

# 更新统计
self.update_expert_load_ema(topk_ids)
```

**实现难度**：⭐⭐ 中（基于已有 `moe_stats.py`，需要添加 EMA 和反馈机制）

---

### 2.3 推荐的约束优先级

| 优先级 | 约束 | 理由 | vLLM 支持状态 |
|-------|------|------|--------------|
| **P0** | Token Budget | 已有现成接口，直接可用 | ✅ 完全支持 |
| **P0** | KV Cache 容量 | 已有抢占机制，可观测 | ✅ 完全支持 |
| **P1** | 专家负载统计 | E1.1 已有 `moe_stats.py` 基础 | ⚠️ 需扩展 |
| **P1** | 质量约束 | 在 `_select_experts` 添加过滤 | ❌ 需实现 |
| **P2** | 跨 GPU 通信 | 需修改 EP 层或添加 wrapper | ❌ 需大量开发 |

### 2.4 数学模型与 vLLM 的对齐总结

| 原模型假设 | vLLM 实际情况 | 对齐方案 |
|-----------|-------------|---------|
| 按专家累加计算约束 $\sum_e a_e u_e$ | 按请求累加 $\sum_r n_r$ | 等价：$\sum_e u_e = k \cdot \sum_r n_r$ |
| 专家显存约束 $\sum_e X_e \cdot \text{size}$ | KV cache per-request 约束 | 分离建模：专家权重静态，KV 动态 |
| 专家队列 $q_e(t)$ | 无显式队列 | 定义 virtual queue = batch 内负载 |
| 质量约束 $(1-\epsilon)\max$ | 无原生支持 | 在 `_select_experts` 添加过滤 |
| 通信流量 $F_{kk'}$ | EP dispatch/combine 隐式发生 | 添加监控 wrapper |

---

## 3. 策略导出：影子价格 → 路由规则

### 3.1 你的 PriceMoE 策略

**修正评分函数**：
$$\tilde{S}_{je}(t) = s_{je} - \alpha q_e(t) - \gamma_{L(e),k_j} - \delta \cdot \mathbb{I}_{\{X_e(t)=0\}}$$

**路由规则**：选择 Top-k 个最大化 $\tilde{S}_{je}(t)$ 的专家。

### 3.2 在 vLLM 中实现

**修改点**：`vllm/model_executor/layers/fused_moe/router/base_router.py`

```python
class PriceMoERouter(BaseRouter):
    """
    实现 PriceMoE 的影子价格路由策略。
    """
    
    def __init__(
        self,
        alpha: float = 1.0,      # 拥塞惩罚权重
        gamma: float = 0.1,      # 跨 GPU 通信惩罚
        delta: float = 0.5,      # 专家激活成本
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        
        # 专家负载统计（virtual queue）
        self.expert_load = None  # 在 forward 前初始化
        
        # 专家激活状态（迟滞机制）
        self.expert_active = None
        self.activate_threshold = 10.0  # H_activate
        self.deactivate_threshold = 2.0  # H_deactivate
    
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,  # (T, E) 原始 gating scores
        indices_type: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 PriceMoE 的修正评分并选择 Top-k 专家。
        """
        T, E = router_logits.shape
        
        # 1. 获取专家负载（virtual queue）
        if self.expert_load is None:
            self.expert_load = torch.zeros(E, device=router_logits.device)
        
        # 2. 获取专家激活状态
        if self.expert_active is None:
            self.expert_active = torch.ones(E, dtype=torch.bool, device=router_logits.device)
        
        # 3. 计算修正评分
        # 拥塞惩罚：-alpha * q_e
        congestion_penalty = self.alpha * self.expert_load.unsqueeze(0)  # (1, E)
        
        # 跨 GPU 通信惩罚（简化：假设本地专家无惩罚）
        # 实际需要根据 expert_map 和 ep_rank 计算
        comm_penalty = self._compute_comm_penalty(router_logits.device)  # (E,)
        
        # 激活成本：-delta * (1 - X_e)
        activation_cost = self.delta * (~self.expert_active).float().unsqueeze(0)  # (1, E)
        
        # 修正评分
        adjusted_logits = (
            router_logits 
            - congestion_penalty 
            - comm_penalty.unsqueeze(0) 
            - activation_cost
        )
        
        # 4. 选择 Top-k
        topk_weights, topk_ids = torch.topk(
            adjusted_logits,
            k=self.top_k,
            dim=-1
        )
        
        # 5. 更新专家负载统计
        self._update_expert_load(topk_ids)
        
        # 6. 更新专家激活状态（迟滞机制）
        self._update_expert_active()
        
        # 7. 归一化权重
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_ids.to(indices_type)
    
    def _compute_comm_penalty(self, device: torch.device) -> torch.Tensor:
        """
        计算跨 GPU 通信惩罚。
        本地专家惩罚为 0，远程专家惩罚为 gamma。
        """
        E = self.global_num_experts
        penalty = torch.full((E,), self.gamma, device=device)
        
        # 如果有 expert_map，则本地专家惩罚设为 0
        if hasattr(self, 'expert_map') and self.expert_map is not None:
            local_mask = self.expert_map >= 0  # 本地专家
            penalty[local_mask] = 0.0
        
        return penalty
    
    def _update_expert_load(self, topk_ids: torch.Tensor):
        """
        更新专家负载统计。
        采用指数移动平均（EMA）平滑。
        """
        # 统计每个专家被选中的次数
        counts = torch.bincount(
            topk_ids.flatten(),
            minlength=self.global_num_experts
        ).float()
        
        # EMA 更新
        decay = 0.9
        self.expert_load = decay * self.expert_load + (1 - decay) * counts
    
    def _update_expert_active(self):
        """
        迟滞机制：根据负载决定专家激活/休眠。
        """
        # 激活条件：负载超过阈值
        activate_mask = self.expert_load > self.activate_threshold
        # 休眠条件：负载低于阈值
        deactivate_mask = self.expert_load < self.deactivate_threshold
        
        # 更新状态（迟滞：只在明确条件下切换）
        self.expert_active = self.expert_active | activate_mask
        self.expert_active = self.expert_active & ~deactivate_mask
```

### 3.3 集成到 FusedMoE

**修改点**：`vllm/model_executor/layers/fused_moe/layer.py`

```python
# 在 FusedMoE.__init__ 中添加 router 选择逻辑

if routing_method_type == RoutingMethodType.PRICE_MOE:
    self.router = PriceMoERouter(
        alpha=vllm_config.moe_config.price_moe_alpha,
        gamma=vllm_config.moe_config.price_moe_gamma,
        delta=vllm_config.moe_config.price_moe_delta,
        # ... 其他参数
    )
else:
    self.router = BaseRouter(...)
```

---

## 4. 代码实现难度评估

### 4.1 实现层次划分

| 层次 | 描述 | 涉及文件 | 难度 |
|------|------|---------|------|
| **Layer 1** | 配置系统 | `vllm/config/*.py` | ⭐ 低 |
| **Layer 2** | Router 策略 | `vllm/model_executor/layers/fused_moe/router/*.py` | ⭐⭐ 中 |
| **Layer 3** | 统计采集 | `vllm/moe_stats.py`（已有） | ⭐⭐ 中 |
| **Layer 4** | Scheduler 集成 | `vllm/v1/core/sched/scheduler.py` | ⭐⭐⭐ 高 |
| **Layer 5** | Kernel 修改 | `csrc/moe/*.cu` | ⭐⭐⭐⭐ 极高 |

### 4.2 详细难度评估

#### 4.2.1 Router 修改（Layer 2）

**工作量**：约 2-3 周

**需要修改的文件**：
1. `vllm/model_executor/layers/fused_moe/router/base_router.py` - 添加 PriceMoERouter
2. `vllm/model_executor/layers/fused_moe/router/__init__.py` - 导出新类
3. `vllm/model_executor/layers/fused_moe/layer.py` - 集成到 FusedMoE

**关键代码位置**：
```python
# base_router.py L203-L245
def select_experts(self, hidden_states, router_logits):
    # Step 3: Compute routing (这里需要替换/扩展)
    topk_weights, topk_ids = self._compute_routing(
        hidden_states, router_logits, indices_type
    )
    # ...
```

**风险评估**：
- ✅ 低风险：Router 是相对独立的模块
- ✅ 可测试性好：可以单独测试 routing 逻辑
- ⚠️ 需要保持与 EPLB 的兼容性

---

#### 4.2.2 统计采集扩展（Layer 3）

**工作量**：约 1-2 周

**需要修改的文件**：
1. `vllm/moe_stats.py`（你们 E1.1 已有基础）
2. `vllm/v1/worker/gpu/model_runner.py` - 添加更多采集点

**需要新增的统计**：
1. Per-expert 的实时负载（当前 batch 中被选中的 token 数）
2. Per-expert 的历史负载（EMA 平滑）
3. 跨 GPU 通信流量（按 (src, dst) 聚合）

**关键代码位置**：
```python
# 在 FusedMoE.select_experts() 后添加
from vllm.moe_stats import record_expert_load

record_expert_load(
    layer_name=self.layer_name,
    topk_ids=topk_ids,
    num_experts=self.global_num_experts,
)
```

**风险评估**：
- ✅ 低风险：统计采集不影响主逻辑
- ⚠️ 性能开销：需要控制统计的频率和粒度

---

#### 4.2.3 Scheduler 集成（Layer 4）

**工作量**：约 3-4 周

**需要修改的文件**：
1. `vllm/v1/core/sched/scheduler.py` - 添加专家负载感知
2. `vllm/v1/core/sched/output.py` - 扩展 SchedulerOutput

**关键挑战**：

**问题 1**：Scheduler 是 per-request 的，不是 per-expert 的

当前 Scheduler 的决策粒度：
```python
num_scheduled_tokens = {
    "req_0": 100,  # request 0 调度 100 个 token
    "req_1": 50,   # request 1 调度 50 个 token
    # ...
}
```

而 PriceMoE 需要的信息：
```python
expert_loads = {
    0: 150,  # expert 0 的当前负载
    1: 80,   # expert 1 的当前负载
    # ...
}
```

**解决方案**：
1. 在 Scheduler 和 Worker 之间传递 `expert_loads`
2. Worker 在每个 forward 后更新 `expert_loads`
3. Scheduler 在下一次调度时考虑这些信息

**关键代码位置**：
```python
# scheduler.py L336-L369 (EngineCore.step)
def step(self):
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    model_output = future.result()
    
    # ← 这里需要从 model_output 中提取 expert_loads
    # 并传递给下一次 schedule()
```

**风险评估**：
- ⚠️ 中等风险：需要修改调度器的核心逻辑
- ⚠️ 需要处理异步调度的时序问题
- ⚠️ 需要保持与 PP/DP/EP 的兼容性

---

#### 4.2.4 Kernel 修改（Layer 5）

**工作量**：约 4-8 周（如果需要）

**需要修改的文件**：
1. `csrc/moe/moe_kernels.cu` - 如果需要在 kernel 内部统计
2. `vllm/_custom_ops.py` - 添加新的 Python wrapper

**什么情况下需要修改 Kernel**：
1. 如果需要精确统计每个 token 被路由到哪个 GPU（跨 GPU 通信监控）
2. 如果需要在 All-to-All 过程中动态调整路由

**关键挑战**：
- C++/CUDA 代码复杂度高
- 需要与 NCCL 集成
- 调试困难

**风险评估**：
- ⚠️ 高风险：可能影响推理性能
- ⚠️ 需要深入理解 vLLM 的 kernel 实现
- 建议：尽量在 Python 层面解决，避免修改 kernel

---

### 4.3 总工作量估计

| 阶段 | 工作内容 | 工作量 | 依赖 |
|------|---------|--------|------|
| **Phase 1** | 配置系统 + PriceMoERouter 框架 | 2 周 | 无 |
| **Phase 2** | 专家负载统计（扩展 E1.1） | 1 周 | Phase 1 |
| **Phase 3** | PriceMoE routing 逻辑完整实现 | 2 周 | Phase 2 |
| **Phase 4** | Scheduler 集成（可选） | 3 周 | Phase 3 |
| **Phase 5** | 性能测试 + 调优 | 2 周 | Phase 3/4 |

**总计**：约 **10-12 周**（Phase 1-3 必须，Phase 4-5 可选）

---

## 5. 分阶段实施路线图

### Phase 1: Minimal Viable Product (MVP)

**目标**：验证 PriceMoE 的基本效果

**实现范围**：
1. ✅ 添加 `PriceMoERouter` 类
2. ✅ 实现拥塞惩罚 $-\alpha q_e$（用 batch 内统计）
3. ❌ 暂不实现跨 GPU 通信惩罚
4. ❌ 暂不实现迟滞机制

**修改的文件**：
```
vllm/model_executor/layers/fused_moe/router/
├── __init__.py                 # 导出 PriceMoERouter
├── base_router.py              # 参考
└── price_moe_router.py         # 新增
vllm/model_executor/layers/fused_moe/layer.py  # 集成
vllm/config/moe.py              # 新增配置
```

**代码示例**：
```python
# price_moe_router.py (MVP 版本)
class PriceMoERouter(BaseRouter):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.expert_load = None
    
    def _compute_routing(self, hidden_states, router_logits, indices_type):
        T, E = router_logits.shape
        
        # 初始化负载统计
        if self.expert_load is None:
            self.expert_load = torch.zeros(E, device=router_logits.device)
        
        # 拥塞惩罚
        congestion_penalty = self.alpha * self.expert_load.unsqueeze(0)
        adjusted_logits = router_logits - congestion_penalty
        
        # Top-k 选择
        topk_weights, topk_ids = torch.topk(adjusted_logits, k=self.top_k, dim=-1)
        
        # 更新负载（EMA）
        counts = torch.bincount(topk_ids.flatten(), minlength=E).float()
        self.expert_load = 0.9 * self.expert_load + 0.1 * counts
        
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_ids.to(indices_type)
```

**验证方法**：
1. 运行 E1.1 实验，对比原始 routing 和 PriceMoE routing 的专家负载分布
2. 期望：专家负载更均匀（load skew 降低）

---

### Phase 2: 完整 PriceMoE 实现

**目标**：实现论文中的完整策略

**新增功能**：
1. ✅ 跨 GPU 通信惩罚 $\gamma_{L(e),k_j}$
2. ✅ 专家激活成本 $\delta \cdot \mathbb{I}_{\{X_e=0\}}$
3. ✅ 迟滞机制
4. ✅ 熵正则化（可选）

**新增配置**：
```python
# vllm/config/moe.py
class PriceMoEConfig:
    alpha: float = 1.0           # 拥塞惩罚权重
    gamma: float = 0.1           # 跨 GPU 通信惩罚
    delta: float = 0.5           # 专家激活成本
    activate_threshold: float = 10.0   # 激活阈值
    deactivate_threshold: float = 2.0  # 休眠阈值
    eta: float = 1.0             # 熵正则化温度（可选）
```

---

### Phase 3: Scheduler 集成（可选）

**目标**：让 Scheduler 感知专家负载

**新增功能**：
1. Worker → Scheduler 传递 `expert_loads`
2. Scheduler 根据 `expert_loads` 调整调度策略
3. 可能的优化：减少高负载专家的请求量

**修改的文件**：
```
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/engine/core.py
```

**关键接口**：
```python
# SchedulerOutput 扩展
@dataclass
class SchedulerOutput:
    # ... 现有字段
    expert_loads: Optional[Dict[int, float]] = None  # 新增

# ModelRunnerOutput 扩展
@dataclass
class ModelRunnerOutput:
    # ... 现有字段
    expert_loads: Optional[torch.Tensor] = None  # 新增
```

---

## 6. 总结与建议

### 6.1 可行性结论

| 问题 | 答案 |
|------|-----|
| **策略能否在 vLLM 中实现？** | ✅ **可以**，核心逻辑在 Router 层面 |
| **需要改底层架构吗？** | ❌ **不需要**（MVP 阶段）；⚠️ **可能需要**（完整版 + Scheduler 集成） |
| **Coding 工作量？** | MVP：2-3 周；完整版：6-8 周；Scheduler 集成：10-12 周 |
| **主要难点？** | (1) 专家负载的定义和采集；(2) 与现有 EPLB 的兼容；(3) 性能开销控制 |

### 6.2 建议的约束选择

基于 vLLM 的代码现状，我推荐以下约束组合：

**第一阶段（验证概念）**：
1. **Token Budget 约束**：$\sum \text{tokens} \leq \text{max\_num\_scheduled\_tokens}$ ✅ 已有
2. **专家负载均衡约束**：$\max_e q_e - \min_e q_e \leq \epsilon$ ← 新增

**第二阶段（完善模型）**：
3. **KV Cache 约束**：$\sum \text{blocks} \leq \text{num\_gpu\_blocks}$ ✅ 已有
4. **质量约束**：$\text{Score}(j,e) \geq (1-\epsilon)\max$ ← 在 routing 后验证

**第三阶段（可选扩展）**：
5. **跨 GPU 通信约束**：$F_{kk'} \leq B_{kk'}$ ← 需要较大改动

### 6.3 关键建议

1. **先做 MVP**：用 2-3 周验证 PriceMoE 的核心思想（拥塞惩罚）

2. **利用 E1.1 基础**：你们已经有 `moe_stats.py` 的统计框架，可以扩展

3. **避免修改 Kernel**：尽量在 Python 层面解决，降低风险

4. **保持兼容性**：确保 PriceMoE 可以作为"可选策略"启用，不影响默认行为

5. **性能监控**：添加 routing 开销的 profiling，确保不引入过多 overhead

### 6.4 代码修改清单

**必须修改的文件**（MVP）：
```
vllm/model_executor/layers/fused_moe/router/price_moe_router.py  # 新增
vllm/model_executor/layers/fused_moe/router/__init__.py          # 修改
vllm/model_executor/layers/fused_moe/layer.py                    # 修改（集成 router）
vllm/config/moe.py                                               # 新增配置（可选）
```

**可能需要修改的文件**（完整版）：
```
vllm/moe_stats.py                                                # 扩展统计
vllm/v1/worker/gpu/model_runner.py                               # 添加采集点
vllm/v1/core/sched/scheduler.py                                  # 集成负载感知
vllm/v1/core/sched/output.py                                     # 扩展输出
vllm/v1/engine/core.py                                           # 传递负载信息
```

---

## 附录：关键代码位置速查

| 功能 | 文件 | 行号/函数 |
|------|------|---------|
| Router 基类 | `router/base_router.py` | `BaseRouter.select_experts()` |
| Top-k 选择 | `router/base_router.py` | `_compute_routing()` |
| EPLB 映射 | `router/base_router.py` | `_apply_eplb_mapping()` |
| FusedMoE 入口 | `layer.py` | `FusedMoE.forward_impl()` |
| Expert placement | `layer.py` | `determine_expert_map()` |
| Scheduler 调度 | `scheduler.py` | `Scheduler.schedule()` |
| Token budget | `scheduler.py` | `token_budget = self.max_num_scheduled_tokens` |
| KV cache 分配 | `scheduler.py` | `self.kv_cache_manager.allocate_slots()` |
| 统计采集 | `moe_stats.py` | `record_topk()` |
| GPU forward | `gpu/model_runner.py` | `GPUModelRunner.execute_model()` |
