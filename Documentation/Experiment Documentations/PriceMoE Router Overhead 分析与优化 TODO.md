# PriceMoE Router Overhead 分析与优化 TODO

> 文档创建日期: 2026-02-24
> 基于 E1.3 实验 (260212-qwen3-30b-ep-heavy-sharegpt) 结果和 vLLM v0.14.0 EPLB 架构分析

---

## 1. 现状：~5% 恒定性能开销

### 1.1 实验数据（260212, Qwen3-30B-A3B, EP×4, ShareGPT 2000 reqs）

| Config | tok/s | vs Baseline |
|--------|------:|------------|
| **Baseline** | **2376.1** | — |
| α=0.001 | 2241.5 | **−5.7%** |
| α=0.01 | 2264.5 | −4.7% |
| α=0.02 | 2233.3 | −6.0% |
| α=0.05 | 2271.1 | −4.4% |
| α=0.10 | 2277.6 | −4.1% |

**关键发现**：overhead 与 α 值**几乎无关**（α=0.001 损失 5.7% ≈ α=0.10 损失 4.1%），说明损失主要来自 **代码路径本身**（tensor 分配 + GPU kernel launch），而非路由质量下降。

### 1.2 历史趋势

| 时间 | 版本 | α=0.00 开销 | α>0 开销 |
|------|------|------------|---------|
| 260129 (v1) | 全路径替换 | ~10% | ~10% |
| 260203 (v2-early) | fused_topk_price | ~4% | ~5-8% |
| 260212 (v2-final) | bias-on-logit | N/A (未测α=0) | ~4-6% |

v2 架构已将 overhead 从 ~10% 降至 ~5%，但仍有显著优化空间。

---

## 2. 开销来源深入分析

### 2.1 热路径：每层每 Forward 执行的操作

**`update_queue()` — 约占 60-70% 的 overhead：**

```python
# price_moe_router.py L183-196
def update_queue(self, topk_ids):
    E = self.global_num_experts
    counts = torch.zeros(E, dtype=torch.float32, device=topk_ids.device)    # ← 分配 (1)
    ones = torch.ones(topk_ids.numel(), dtype=torch.float32, device=topk_ids.device)  # ← 分配 (2)
    counts.scatter_add_(0, topk_ids.reshape(-1).long(), ones)               # ← kernel (1)
    self._expert_load.mul_(self.ema_decay)                                  # ← kernel (2)
    self._expert_load.add_(counts, alpha=(1.0 - self.ema_decay))            # ← kernel (3)
```

每层开销：**2 次 CUDA 内存分配 + 3 次 kernel launch**

**`get_expert_load_bias()` — 约占 30-40% 的 overhead：**

```python
# price_moe_router.py L155-162
def get_expert_load_bias(self, device):
    return self._expert_load.mul(-self.alpha)   # ← .mul() 非 in-place，分配新 tensor (3)
```

```python
# layer.py L1636-1637
router_logits = router_logits + bias.unsqueeze(0)  # ← tensor addition，分配新 tensor (4)
```

每层开销：**2 次 CUDA 内存分配 + 2 次 kernel launch**

### 2.2 量化总结

Qwen3-30B-A3B 有 48 个 MoE 层，每个 forward step：

| 操作类型 | 每层次数 | 总次数 (48层) |
|---------|---------|-------------|
| CUDA tensor 分配 | 4 | **192** |
| GPU kernel launch | 5 | **240** |

这 192 次小 tensor 分配和 240 次 kernel launch 是 ~5% overhead 的主要来源。

### 2.3 与 EPLB 的对比

vLLM 官方 EPLB 的 `eplb_map_to_physical_and_record()` 实现了几乎**相同功能**的负载追踪：

```python
# fused_moe.py L1672-1680 (EPLB)
topk_ids_flatten = topk_ids.flatten()
expert_load_view.scatter_add_(
    dim=0,
    index=topk_ids_flatten.long(),
    src=torch.ones_like(topk_ids_flatten).to(expert_load_view),
)
```

EPLB 的优势：
- `expert_load_view` 是**预分配**的 tensor（来自 `eplb_state`），无需每次创建
- 整个 `eplb_map_to_physical_and_record` 函数可以被 `torch.compile` 编译
- 累计计数在 `eplb_state.step()` 中统一处理（滑动窗口 + 归零）

EPLB 的局限：
- 追踪的是 **physical** expert IDs（EPLB 做了 logical→physical 映射后的 ID）
- 只有启用 EPLB 时才会填充 `expert_load_view`
- 计数是原始累计值，不是 EMA 平滑过的
- 每 `step()` 归零一次（窗口大小默认 1000 步），粒度粗于每 forward 的 EMA

---

## 3. 优化方案

### 方案 A：优化自维护队列（独立于 EPLB）⭐ 推荐首选

**核心思路**：消除每次 forward 的 tensor 分配，预分配所有缓冲区。

#### TODO A.1：预分配 `counts` 和 `ones` 缓冲区

```python
class PriceMoERouter:
    def __init__(self, ...):
        ...
        # 预分配缓冲区（延迟到第一次 forward）
        self._counts_buf: Optional[torch.Tensor] = None
        self._ones_buf: Optional[torch.Tensor] = None
        self._max_ones_size: int = 0  # 记录当前 ones 缓冲区大小

    def _ensure_buffers(self, device: torch.device, num_tokens_x_k: int):
        """确保缓冲区已分配且大小足够"""
        if self._counts_buf is None or self._counts_buf.device != device:
            self._counts_buf = torch.zeros(
                self.global_num_experts, dtype=torch.float32, device=device)
        if self._ones_buf is None or self._max_ones_size < num_tokens_x_k:
            # 按 1.5× 增长避免频繁重分配
            new_size = max(num_tokens_x_k, int(self._max_ones_size * 1.5))
            self._ones_buf = torch.ones(new_size, dtype=torch.float32, device=device)
            self._max_ones_size = new_size
```

**预期收益**：每层每 forward 减少 2 次 CUDA alloc → 总减少 **96 次/step**

#### TODO A.2：缓存 bias tensor，避免 `.mul()` 创建新 tensor

```python
def get_expert_load_bias(self, device: torch.device) -> Optional[torch.Tensor]:
    if self.alpha == 0.0:
        return None
    # 懒分配
    if self._expert_load is None or self._expert_load.device != device:
        self._expert_load = torch.zeros(self.global_num_experts, ...)
    if self._bias_buf is None:
        self._bias_buf = torch.empty_like(self._expert_load)
    # In-place: bias = -α * q
    torch.mul(self._expert_load, -self.alpha, out=self._bias_buf)
    return self._bias_buf
```

**预期收益**：每层每 forward 减少 1 次 CUDA alloc → 总减少 **48 次/step**

#### TODO A.3：消除 `update_queue` 中的中间 tensor

```python
@torch.no_grad()
def update_queue(self, topk_ids: torch.Tensor) -> None:
    if self.alpha == 0.0 or self._expert_load is None:
        return

    flat_ids = topk_ids.reshape(-1).long()
    self._ensure_buffers(topk_ids.device, flat_ids.numel())

    # 重用预分配的 counts buffer（就地归零 + scatter_add）
    self._counts_buf.zero_()
    self._counts_buf.scatter_add_(0, flat_ids, self._ones_buf[:flat_ids.numel()])

    # EMA blend（全部 in-place）
    self._expert_load.mul_(self.ema_decay).add_(
        self._counts_buf, alpha=(1.0 - self.ema_decay))
```

**预期收益**：结合 A.1，每 forward 从 192 次 alloc → **~0 次 alloc**（稳态下），240 次 kernel → **~192 次 kernel**（减少 zero + scatter 各 48 次）

#### TODO A.4：使用 `torch.compile` 编译热路径（可选，高风险）

```python
@torch.compile(mode="reduce-overhead")
def _compiled_update(expert_load, counts_buf, ones_buf, flat_ids, ema_decay, numel):
    counts_buf.zero_()
    counts_buf.scatter_add_(0, flat_ids, ones_buf[:numel])
    expert_load.mul_(ema_decay).add_(counts_buf, alpha=(1.0 - ema_decay))
```

**注意**：`torch.compile` 对 dynamic shapes 的支持有限，`flat_ids.numel()` 随 batch size 变化。需要测试 `dynamic=True` 是否能正确编译。EPLB 原生代码中也有类似的 `torch.compile` 用法。

**预期收益**：将多个 kernel 融合为单个，减少 launch overhead。但 dynamic shape 可能限制效果。

#### 方案 A 总预期收益

| 优化项 | alloc 减少 | kernel 减少 | 预计 overhead 下降 |
|-------|-----------|------------|-----------------|
| A.1 预分配 counts/ones | 96/step | 0 | ~2% |
| A.2 缓存 bias | 48/step | 0 | ~1% |
| A.3 就地 update | (含在 A.1) | ~48/step | ~0.5% |
| A.4 torch.compile | 0 | 可观 | ~0.5-1% |
| **合计** | **~144/step** | **~48-96/step** | **~3-4%** |

**预期最终 overhead：~1-2%（从 ~5% 降至 ~1-2%）**

---

### 方案 B：复用 EPLB `expert_load_view`（当 EPLB 共存时）

**核心思路**：当 EPLB 已经启用时，利用 EPLB 已经追踪的 `expert_load_view` 作为负载信号，完全消除 `update_queue()`。

#### 前提条件
- `--enable-eplb` 已开启（ParallelConfig.enable_eplb = True）
- PriceMoE 的 layer 能访问到本层的 `expert_load_view`

#### TODO B.1：为 PriceMoERouter 添加 EPLB 数据源模式

新增环境变量控制：
```
VLLM_PRICE_MOE_LOAD_SOURCE=self|eplb   # 默认 "self"
```

```python
class PriceMoERouter:
    def __init__(self, ...):
        ...
        self.load_source = os.environ.get("VLLM_PRICE_MOE_LOAD_SOURCE", "self")
        self._eplb_load_view: Optional[torch.Tensor] = None  # EPLB 提供的 view
        self._phy2log_map: Optional[torch.Tensor] = None
```

#### TODO B.2：在 layer.py 中传递 EPLB state 到 PriceMoERouter

```python
# layer.py — 在 set_eplb_state() 或 __init__ 中
if self._price_moe_router is not None and hasattr(self._price_moe_router, 'load_source'):
    if self._price_moe_router.load_source == "eplb" and self.enable_eplb:
        self._price_moe_router.set_eplb_data(
            expert_load_view=self.expert_load_view,             # (num_physical_experts,)
            physical_to_logical_map=self.physical_to_logical_map # 用于反映射
        )
```

#### TODO B.3：实现 EPLB 模式下的 `get_expert_load_bias()`

```python
def get_expert_load_bias(self, device):
    if self.alpha == 0.0:
        return None

    if self.load_source == "eplb" and self._eplb_load_view is not None:
        # 从 EPLB physical counts → logical expert loads
        # expert_load_view 是 (num_physical_experts,) int32 累计值
        phy_load = self._eplb_load_view.float()

        # Physical → Logical 聚合（多个物理副本求和回逻辑专家）
        log_load = torch.zeros(self.global_num_experts, device=device, dtype=torch.float32)
        log_load.scatter_add_(0, self._phy2log_map.long(), phy_load)

        return log_load.mul(-self.alpha)

    # 默认：使用自维护队列（方案 A 优化后）
    ...
```

#### TODO B.4：EPLB 模式下跳过 `update_queue()`

```python
# layer.py — 仅在 self 模式下调用 update_queue
if self._price_moe_router is not None:
    if getattr(self._price_moe_router, 'load_source', 'self') == 'self':
        self._price_moe_router.update_queue(topk_ids)
```

#### 方案 B 的复杂性与风险

| 问题 | 详细说明 | 缓解方案 |
|------|---------|---------|
| **Physical vs Logical** | EPLB 的 `expert_load_view` 追踪 physical IDs；`physical_to_logical_map` 可做反映射 | 用 `scatter_add_` 从 physical 聚合回 logical |
| **累计 vs EMA** | EPLB 的 `expert_load_pass` 在 `step()` 时才归零（默认 1000 步），不是 EMA | 可直接使用累计值（步间的总负载）作为拥塞信号；或做差分估计 |
| **归零时刻不对齐** | `eplb_state.step()` 归零不与 PriceMoE forward 对齐 | 可读取 window sum 而非 pass；或接受粗粒度 |
| **EPLB 未启用** | 若 EPLB 关闭，`expert_load_view` 全零 | 自动 fallback 到 `self` 模式 |
| **EPLB 重排导致映射变化** | EPLB 动态重排后 `phy2log_map` 会变 | 每次 `get_expert_load_bias` 重新读取映射 |

#### 方案 B 适用场景

方案 B 最适合：
- 已经需要 EPLB 的场景（EP 多节点，需要动态重排）
- 此时 EPLB 的 `scatter_add_` 是**免费已有**的，PriceMoE 读取不增加任何 kernel

方案 B 不适合：
- 当前 E1.3 实验设置（EP 单节点，不需要 EPLB）
- 小规模部署（EPLB 开销 > 收益）

---

## 4. 实施优先级与计划

### Phase 1：方案 A 实施（预计 1-2 天）⭐

| # | 任务 | 文件 | 难度 | 状态 |
|---|------|------|------|------|
| A.1 | 预分配 counts/ones 缓冲区 | `price_moe_router.py` | 低 | [ ] |
| A.2 | 缓存 bias tensor（`torch.mul` → `out=`） | `price_moe_router.py` | 低 | [ ] |
| A.3 | 就地 update_queue 重写 | `price_moe_router.py` | 低 | [ ] |
| A.5 | 重新运行 260212 实验对比 | `run_experiment.py` | 低 | [ ] |
| A.6 | 验证 α=0.00 的 overhead 是否 ≈0 | benchmark | 低 | [ ] |

### Phase 2：方案 A.4 torch.compile（预计 1 天，可选）

| # | 任务 | 难度 | 状态 |
|---|------|------|------|
| A.4a | 编写 compiled update kernel | 中 | [ ] |
| A.4b | 测试 dynamic shape 兼容性 | 中 | [ ] |
| A.4c | 性能对比（compile vs non-compile） | 低 | [ ] |

### Phase 3：方案 B 设计（预计 2-3 天，EPLB 共存场景）

| # | 任务 | 难度 | 状态 |
|---|------|------|------|
| B.1 | 添加 `VLLM_PRICE_MOE_LOAD_SOURCE` 环境变量 | 低 | [ ] |
| B.2 | layer.py 传递 EPLB state 给 router | 中 | [ ] |
| B.3 | 实现 physical→logical 反映射读取 | 中 | [ ] |
| B.4 | 跳过 update_queue 逻辑 | 低 | [ ] |
| B.5 | 启用 EPLB + PriceMoE 联合实验 | 高 | [ ] |

---

## 5. 验证标准

### 方案 A 验收标准

- [ ] α=0.001~0.10 的 overhead **从 ~5% 降至 ≤2%**
- [ ] α=0.00 的 overhead **≈ 0%**（无统计显著差异）
- [ ] 路由统计结果（expert_load 分布）与当前版本**一致**（正确性不变）
- [ ] 吞吐数值本身不低于 baseline（在 noise range 内）

### 方案 B 验收标准

- [ ] EPLB + PriceMoE 联合运行无 crash
- [ ] EPLB 模式下 overhead **≤ 1%**（原有 EPLB overhead 之上的额外开销）
- [ ] 路由统计与 self 模式行为一致（负载均衡改善方向一致）

---

## 6. 结论

当前 ~5% overhead 主要来自 `update_queue()` 和 `get_expert_load_bias()` 中的 **192 次/step CUDA tensor 分配**，而非路由质量下降。

**方案 A（优化自维护队列）是首选**，原因：
1. 改动最小（仅修改 `price_moe_router.py` 一个文件）
2. 不依赖 EPLB（兼容所有场景）
3. 预期收益显著（overhead 从 ~5% → ~1-2%）
4. 实施简单（预分配 + in-place 操作）

**方案 B（复用 EPLB data）是补充方案**，适合后续 EPLB 联合部署的场景，可完全消除 `update_queue` 的开销。

优先实施 Phase 1（方案 A.1-A.3），一天内即可完成并重新实验验证。

---

*相关文档：*
- *实验 1.3 文档：`Documentation/Experiment Documentations/Verification Experiments/实验 1.3：PriceMoE MVP 实现与验证（Queue-aware Routing Policy）.md`*
- *PriceMoE Router 源码：`external/vllm/vllm/model_executor/layers/fused_moe/price_moe_router.py`*
- *EPLB State：`external/vllm/vllm/distributed/eplb/eplb_state.py`*
- *EPLB 负载记录：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` L1613-1680*
