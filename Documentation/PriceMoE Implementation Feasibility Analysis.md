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

### 2.1 约束对照表

| 你的模型约束 | vLLM 代码对应 | 实现难度 | 数据来源 |
|------------|-------------|---------|---------|
| **计算约束** $\sum a_e u_e \leq C_k$ | `max_num_scheduled_tokens` | ⭐ 低 | `SchedulerConfig` |
| **显存约束** $\sum X_e \cdot \text{size}(e) \leq M_k$ | KV cache 块分配 | ⭐⭐ 中 | `KVCacheManager` |
| **通信约束** $F_{kk'} \leq B_{kk'}$ | EP All-to-All | ⭐⭐⭐ 高 | 需新增监控 |
| **质量约束** $\text{Score} \geq (1-\epsilon)\max$ | Top-k routing 修改 | ⭐⭐ 中 | `FusedMoE.select_experts()` |
| **队列长度** $q_e(t)$ | ❌ 未原生支持 | ⭐⭐⭐ 高 | 需新增采集 |

### 2.2 约束详细映射

#### 约束 1：计算容量（Token Budget）

**数学形式**：
$$\sum_{e \in \mathcal{E}_k} a_e \cdot u_e(t) \leq C_k$$

**vLLM 代码体现**：

```python
# vllm/v1/core/sched/scheduler.py L269-L275
def schedule(self) -> SchedulerOutput:
    # ...
    token_budget = self.max_num_scheduled_tokens  # ← 这就是 C_k
    # ...
    while req_index < len(self.running) and token_budget > 0:
        # ...
        num_new_tokens = min(num_new_tokens, token_budget)  # 受 budget 约束
        token_budget -= num_new_tokens  # 消耗 budget
```

**映射关系**：
- $C_k$ ↔ `max_num_scheduled_tokens`（硬上限）
- $a_e$ ↔ 每个 token 的计算开销（简化为 1）
- $u_e(t)$ ↔ `num_scheduled_tokens[req_id]`

**实现难度**：⭐ 低（已有现成接口）

---

#### 约束 2：显存容量（KV Cache）

**数学形式**：
$$\sum_{e \in \mathcal{E}_k} X_e(t) \cdot \text{size}(e) \leq M_k$$

**vLLM 代码体现**：

```python
# vllm/v1/core/sched/scheduler.py L359-L369
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
if new_blocks is None:
    # 内存不足，触发抢占
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    self._preempt_request(preempted_req, scheduled_timestamp)
```

**映射关系**：
- $M_k$ ↔ `num_gpu_blocks * block_size`（总 KV cache 容量）
- $X_e(t) \cdot \text{size}(e)$ ↔ 每个请求分配的 blocks 数
- 抢占机制 ↔ 当 $\sum X_e \cdot \text{size}(e) > M_k$ 时触发

**实现难度**：⭐⭐ 中（有现成框架，但需理解块管理逻辑）

---

#### 约束 3：跨 GPU 通信（EP All-to-All）

**数学形式**：
$$F_{kk'}(t) \leq B_{kk'}$$

**vLLM 代码体现**：

```python
# vllm/model_executor/layers/fused_moe/layer.py
# 在 FusedMoE.forward_native() 中隐含的 All-to-All
fused_output = torch.ops.vllm.moe_forward(
    hidden_states,
    router_logits,
    self.layer_name,
    # ← All-to-All 在这个 kernel 内部发生
)
```

**问题**：vLLM **没有原生的通信流量监控**！

**需要新增**：
1. 在 `moe_forward` 前后统计发送/接收的 token 数
2. 按 (source_gpu, target_gpu) 聚合流量
3. 用 $\gamma_{L(e),k_j}$ 惩罚跨 GPU 路由

**实现难度**：⭐⭐⭐ 高（需要修改 kernel 或添加 wrapper）

---

#### 约束 4：输出质量

**数学形式**：
$$\text{Score}(j,e) \geq (1-\epsilon) \cdot \max_{e'} \text{Score}(j,e')$$

**vLLM 代码体现**：

```python
# vllm/model_executor/layers/fused_moe/router/base_router.py L203-L245
def select_experts(self, hidden_states, router_logits):
    # Step 3: Compute routing
    topk_weights, topk_ids = self._compute_routing(
        hidden_states, router_logits, indices_type
    )
    # Step 4: Apply EPLB mapping (如果启用)
    topk_ids = self._apply_eplb_mapping(topk_ids)
    return topk_weights, topk_ids
```

**映射关系**：
- 原始 gating score $s_{je}$ ↔ `router_logits`
- Top-k 选择 ↔ `torch.topk(router_logits, k=self.top_k)`
- 质量约束 ↔ 在选择后验证是否满足 $(1-\epsilon)$ 条件

**实现难度**：⭐⭐ 中（可在 `select_experts` 后添加过滤）

---

#### 新增约束：专家队列长度

**数学形式**：
$$q_e(t) = \text{专家 } e \text{ 的当前待处理 token 数}$$

**vLLM 现状**：❌ **完全没有这个概念！**

vLLM 的调度是 **per-request** 的，不是 **per-expert** 的。每个 token 在一个 forward 中选择 Top-k 专家，计算立即完成——没有"排队等待某个专家"的概念。

**需要新增**：

方案 A（统计层面）：
```python
# 在 FusedMoE.select_experts() 后累加
expert_load_count[topk_ids] += 1  # 统计每个专家的负载
```

方案 B（逻辑层面）：
```python
# 定义 "virtual queue"：每个专家在当前 batch 中被选中的次数
q_e = count(token j in batch | e in topk_ids[j])
```

**实现难度**：⭐⭐⭐ 高（需要在每个 forward 前后更新统计）

---

### 2.3 推荐的约束优先级

| 优先级 | 约束 | 理由 |
|-------|------|-----|
| **P0** | Token Budget | 已有现成接口，直接可用 |
| **P0** | KV Cache 容量 | 已有抢占机制，可观测 |
| **P1** | 专家负载统计 | E1.1 已有部分基础，需扩展 |
| **P1** | 质量约束 | 在 routing 层面修改，较清晰 |
| **P2** | 跨 GPU 通信 | 需要深入 kernel，工作量大 |

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
