# vLLM Expert Parallel & Tensor Parallel 代码体现指南

> 本文档梳理 vLLM 中 Expert Parallel (EP) 和 Tensor Parallel (TP) 的代码位置、设计意图和交互方式。
> 基于 vLLM v0.13.0（v1 engine）。

---

## 目录

1. [快速导航](#快速导航)
2. [核心概念区分](#核心概念区分)
3. [代码位置地图](#代码位置地图)
4. [关键代码片段深入](#关键代码片段深入)
5. [EP 与 TP 的交互](#ep-与-tp-的交互)
6. [在你的 vLLM 本地副本中查找](#在你的-vllm-本地副本中查找)

---

## 快速导航

如果你只想快速找到代码：

| 功能 | 文件位置 | 关键类/函数 |
|------|---------|-----------|
| **EP 配置声明** | `vllm/config.py` | `ParallelConfig.expert_parallel_size` |
| **TP 配置声明** | `vllm/config.py` | `ParallelConfig.tensor_parallel_size` |
| **初始化分布式** | `vllm/distributed/initialize.py` | `initialize_model_parallel()` |
| **MoE 中的 EP** | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE.forward_native()` 或 `forward_cuda()` |
| **TP 中的通信** | `vllm/distributed/parallel_state.py` | `get_tensor_parallel_group()` |
| **All-to-All 调用** | `vllm/model_executor/layers/fused_moe/layer.py` | `torch.ops.vllm.moe_forward()` 包含 all-to-all |
| **分布式初始化** | `vllm/executor/gpu_executor.py` (v0) 或 `vllm/v1/worker/gpu_worker.py` (v1) | Worker 初始化时的 `torch.distributed` setup |

---

## 核心概念区分

### Tensor Parallel (TP)

**定义**：同一层的权重矩阵在 GPU 之间按维度分割。

**例子**：Linear layer 权重矩阵 $W \in \mathbb{R}^{d_{in} \times d_{out}}$
- **列并行**（输出分割）：`[GPU0 | GPU1]`，各持有 $d_{out}/2$ 列
- **行并行**（输入分割）：垂直分割

**通信**：
- Forward：AllGather（收集各 GPU 的部分输出）
- Backward：ReduceScatter（分散梯度）

### Expert Parallel (EP)

**定义**：在 MoE 层中，不同的 experts 放在不同的 GPU 上。

**例子**：128 个 experts，4 个 GPU → 每个 GPU 上 32 个 experts

**通信**：
- Dispatch phase：All-to-All（根据 routing 重新分配 tokens）
- Compute phase：本地计算
- Combine phase：All-to-All（汇聚 tokens 回原序）

---

## 代码位置地图

### 第一层：配置声明

**文件**：`vllm/config.py` （或 `vllm/v1/configs/*.py` in v1）

```python
class ParallelConfig:
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,  # ← EP 配置
        max_parallel_loading_workers: int = 1,
        disable_custom_all_reduce: bool = False,
        ...
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        # TP size 可能会自动调整以匹配模型的 TP 需求
```

### 第二层：分布式初始化

**文件**：`vllm/distributed/initialize.py`

```python
def initialize_model_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    backend: str = "nccl",
):
    """
    初始化 TP、PP、EP 的进程组。
    每个进程组管理一个通信域，用于 collective ops (AllGather, ReduceScatter 等)。
    """
    # 典型流程：
    # 1. 根据 TP、PP、EP size 计算进程排列（rank assignment）
    # 2. 为每个通信维度创建独立的进程组
    # 3. 设置环境变量（RANK, WORLD_SIZE 等）
```

关键函数签名：
```python
def initialize_model_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    """
    创建 TP group, PP group, EP group, ...
    例如：8 个 GPU，TP=2，PP=2，EP=2 时：
    
    TP 通信组：(GPU0,GPU1), (GPU2,GPU3), (GPU4,GPU5), (GPU6,GPU7)
    PP 通信组：(GPU0,GPU2,GPU4,GPU6), (GPU1,GPU3,GPU5,GPU7)
    EP 通信组：(GPU0,GPU4), (GPU1,GPU5), (GPU2,GPU6), (GPU3,GPU7)
    """
```

### 第三层：分布式状态管理

**文件**：`vllm/distributed/parallel_state.py`

```python
# 全局进程组对象存储
_TENSOR_PARALLEL_GROUP = None
_EXPERT_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_GROUP = None
# ... 等等

def get_tensor_parallel_group():
    """获取 TP 通信组，用于 TP 的 collective ops"""
    return _TENSOR_PARALLEL_GROUP

def get_expert_parallel_group():
    """获取 EP 通信组，用于 MoE all-to-all 等"""
    return _EXPERT_PARALLEL_GROUP

def get_tensor_parallel_rank():
    """当前进程在 TP 组内的 rank"""
    return torch.distributed.get_rank(_TENSOR_PARALLEL_GROUP)

def get_expert_parallel_rank():
    """当前进程在 EP 组内的 rank"""
    return torch.distributed.get_rank(_EXPERT_PARALLEL_GROUP)
```

---

## 关键代码片段深入

### 1) FusedMoE 中的 EP 通信

**文件**：`vllm/model_executor/layers/fused_moe/layer.py`

这是 **MoE 计算的核心**，也是 **EP 最直观体现** 的地方。

#### 关键方法 1：选择 experts（Routing）

```python
class FusedMoE(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype = None,
        use_grouped_topk: bool = False,
        enable_eplb: bool = False,  # ← EP-specific tuning
        use_fp8: bool = False,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.enable_eplb = enable_eplb
        
    def select_experts(self, hidden_states, router_logits):
        """
        计算 Top-k routing。
        output:
          - topk_ids: shape (T, K) 选中的 expert IDs
          - topk_weights: shape (T, K) 权重
        """
        # 1. 从 router_logits (T, E) 中取 Top-K
        topk_weights, topk_ids = torch.topk(
            router_logits,
            k=self.top_k,
            dim=-1
        )
        # topk_ids 的 value 范围是 [0, num_experts)
        
        # 2. 如果启用 EPLB（Expert Placement Load Balancing）：
        if self.enable_eplb:
            # EPLB 会重新映射 expert IDs
            # 使得 EP 各进程的负载更均衡
            topk_ids = self._apply_eplb(topk_ids)  # → 修改映射
        
        # 3. 我们在这里插入 E1.1 的统计采集 hook
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
        
        return topk_weights, topk_ids
```

#### 关键方法 2：MoE forward（分发+计算+合并）

```python
class FusedMoE(torch.nn.Module):
    def forward_native(self, hidden_states, router_logits):
        """
        Native（非 CUDA graph）forward path.
        这里调用 fused MoE kernel，包括 All-to-All 通信。
        """
        topk_weights, topk_ids = self.select_experts(
            hidden_states, 
            router_logits
        )
        
        # ← ★ ALL-TO-ALL 发生在这里 ★
        fused_output = torch.ops.vllm.moe_forward(
            hidden_states,           # (T, H)
            router_logits,           # (T, E)
            self.layer_name,         # for stats / debugging
            # 内部会调用 torch.distributed 的 all-to-all 操作
            # 仅当 EP > 1 时才触发真正的通信
        )
        # fused_output shape: (T, H)
        
        return fused_output
    
    def forward_cuda(self, hidden_states, router_logits):
        """
        CUDA graph path（性能优化路径）。
        CUDA graph 可能会预先录制 all-to-all 的执行模式。
        """
        topk_weights, topk_ids = self.select_experts(
            hidden_states, 
            router_logits
        )
        
        # 同样调用 fused kernel，但可能走不同的代码路径
        fused_output = torch.ops.vllm.moe_forward(
            hidden_states,
            router_logits,
            self.layer_name,
        )
        
        return fused_output
```

### 2) 自定义 Kernel：moe_forward 的 All-to-All 调用

**文件**：`vllm/_custom_ops.py` 和 C++ backend（`csrc/moe/*.cu`）

Python 侧包装器：

```python
# vllm/_custom_ops.py

def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str = "",
    # ... 其他参数
) -> torch.Tensor:
    """
    Python wrapper for torch.ops.vllm.moe_forward custom op.
    
    这个自定义 op 的实现在 C++/CUDA 中，会在以下场景调用 All-to-All：
    
    1. **Dispatch 阶段**：
       - 根据 routing 结果 (topk_ids)，将每个 token 发送到对应的 expert GPU
       - all_to_all_group = get_expert_parallel_group()
       - torch.distributed.all_to_all_single(..., group=all_to_all_group)
    
    2. **Compute 阶段**：
       - 各 GPU 本地计算其分配到的 experts
       - 无通信（假设每个 GPU 上有一个完整的 experts 副本子集）
    
    3. **Combine 阶段**：
       - 再次 All-to-All，将输出汇聚回原始 token 位置
       - 然后加权求和（weighted combine）
    """
    output = torch.ops.vllm.moe_forward(
        hidden_states,
        router_logits,
        layer_name,
    )
    return output
```

**C++ 实现中的通信逻辑**（概念性）：

```cpp
// csrc/moe/moe_kernels.cu (伪代码)

torch::Tensor moe_forward(
    torch::Tensor hidden_states,    // (T, H)
    torch::Tensor router_logits,    // (T, E)
    const std::string& layer_name
) {
    // 1. Routing (Top-K 已在 Python 侧算好)
    auto topk_ids = compute_topk(...);
    
    // 2. Dispatch All-to-All
    if (expert_parallel_size > 1) {
        // ↓ 这里调用 NCCL all_to_all_single
        all_to_all_for_moe_dispatch(
            hidden_states,     // 输入
            topk_ids,          // routing 信息
            expert_parallel_group
        );
    }
    
    // 3. Local GEMM (各 GPU 计算自己的 experts)
    auto expert_outputs = local_moe_compute(...);
    
    // 4. Combine All-to-All
    if (expert_parallel_size > 1) {
        // ↓ 再次 all_to_all 把输出送回
        all_to_all_for_moe_combine(
            expert_outputs,
            topk_weights,
            expert_parallel_group
        );
    }
    
    return combined_output;
}
```

### 3) Tensor Parallel 的 Layer 包装

**文件**：`vllm/model_executor/layers/linear.py` (or similar)

TP 通过在常规层（Linear, Attention）外包一层做分割和通信：

```python
class ColumnParallelLinear(torch.nn.Module):
    """
    Linear 层的 TP 版本：输出按列分割。
    
    权重分割：W (d_in, d_out) → [W_0 (d_in, d_out/TP) | W_1 | ... | W_{TP-1}]
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // tensor_parallel_size
        
        # 各 GPU 只持有部分权重
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.out_features_per_partition,
                in_features,
                dtype=dtype,
            )
        )
    
    def forward(self, input_):
        # 局部 GEMM：output shape (seq_len, out_features_per_partition)
        output = torch.nn.functional.linear(
            input_,
            self.weight,
            self.bias,
        )
        # 输出保持分割状态（用于下一层的 RowParallelLinear 消费）
        return output


class RowParallelLinear(torch.nn.Module):
    """
    Linear 层的 TP 版本：输入按行分割，需要 AllGather。
    
    权重分割：W (d_in, d_out) 按行分割 → 
        [W_0 (d_in/TP, d_out)]
        [W_1 (d_in/TP, d_out)]
        ...
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features_per_partition = in_features // tensor_parallel_size
        self.out_features = out_features
        
        self.weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                self.in_features_per_partition,
                dtype=dtype,
            )
        )
    
    def forward(self, input_):
        # input_ shape: (seq_len, in_features_per_partition)
        # 需要先 AllGather 才能做完整 GEMM
        
        # AllGather 将各 GPU 的部分拼接
        all_gathered = torch.distributed.all_gather_into_tensor(
            input_,  # 各 GPU 各自的部分
            group=get_tensor_parallel_group()
        )
        # all_gathered shape: (seq_len, in_features)
        
        # 本地 GEMM：output shape (seq_len, out_features)
        output = torch.nn.functional.linear(
            all_gathered,
            self.weight,
            self.bias,
        )
        
        return output
```

### 4) TP 和 EP 的交互：MoE 层中的双重并行

**关键洞察**：当同时启用 TP 和 EP 时，MoE 层需要特殊处理。

**文件**：`vllm/model_executor/layers/fused_moe/layer.py`

```python
class FusedMoE(torch.nn.Module):
    def __init__(self, ..., tensor_parallel_size=1, expert_parallel_size=1):
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.num_experts = num_experts
        
        # 实际 expert 数目（在本 GPU 上）
        self.num_local_experts = (
            num_experts // expert_parallel_size
        )
        
        # 如果同时启用 TP：各 expert 的权重也会按 TP 分割
        if self.tensor_parallel_size > 1:
            self.intermediate_size_per_partition = (
                intermediate_size // tensor_parallel_size
            )
        else:
            self.intermediate_size_per_partition = intermediate_size
```

在 All-to-All 时，需要同时处理 TP 和 EP：

```python
def forward(self, hidden_states, router_logits):
    # 1. Routing
    topk_weights, topk_ids = self.select_experts(
        hidden_states,
        router_logits
    )
    
    # 2. 如果启用了 EPLB，可能需要映射 expert IDs
    #    （重新负载均衡以匹配 EP 进程分割）
    if self.enable_eplb:
        topk_ids = self._apply_eplb_mapping(topk_ids)
    
    # 3. All-to-All in EP dimension
    #    ← 这里用的是 get_expert_parallel_group()
    
    # 4. 本地计算（可能包含 TP 的 AllGather）
    #    ← 这里用的是 get_tensor_parallel_group()
    
    # 5. All-to-All in EP dimension 把结果收回
    
    return fused_output
```

---

## EP 与 TP 的交互

### 进程网格布局示例

假设有 **8 个 GPU**，配置 `TP=2, PP=2, EP=2`：

```
        TP dimension (共 2 个 GPU)
        ←─────────────────────→
        GPU0    GPU1
PP  GPU2    GPU3
↓   GPU4    GPU5
    GPU6    GPU7

        ↑ EP 维度（共 2 个 GPU）

各通信组分配：
- TP group 0: GPU0, GPU1
- TP group 1: GPU2, GPU3
- TP group 2: GPU4, GPU5
- TP group 3: GPU6, GPU7

- PP group 0: GPU0, GPU2, GPU4, GPU6
- PP group 1: GPU1, GPU3, GPU5, GPU7

- EP group 0: GPU0, GPU4
- EP group 1: GPU1, GPU5
- EP group 2: GPU2, GPU6
- EP group 3: GPU3, GPU7
```

### 执行流程

```
假设某个 MoE 层有 128 个 experts，TP=2，EP=2：
- 总共 4 个 GPU（TP 2×2 = 4）
- GPU 0,1 处理 64 个 experts（EP group 0）
- GPU 2,3 处理另外 64 个 experts（EP group 1）

同时，在 GPU 内部（比如 GPU 0）：
- 权重矩阵按 TP 分割：expert MLP 的中间层按列分割

执行：
1. Routing 在各 GPU 本地计算（所有 GPU 都有 router)
   → topk_ids (T, K)

2. All-to-All in EP dimension (group 0/1/2/3)
   → 根据 topk_ids 将 tokens 重新分配

3. 本地计算各 GPU 负责的 experts
   - 在 GPU 0 上：计算 experts 0-31
   - 但每个 expert 的权重在 TP 维度分割了
   - 所以还需要 AllGather in TP group

4. All-to-All in EP dimension 把输出收回
```

---

## 在你的 vLLM 本地副本中查找

### 快速查找命令

```bash
# 1. 查找所有 TP 相关配置
cd ~/sthmax/git/vllm
grep -r "tensor_parallel_size" --include="*.py" | head -20

# 2. 查找所有 EP 相关配置
grep -r "expert_parallel_size" --include="*.py" | head -20

# 3. 查找分布式初始化
grep -r "initialize_model_parallel" --include="*.py"

# 4. 查找 MoE forward 的实现
find . -path "*/fused_moe/layer.py" -o -path "*/fused_moe/*moe*.py"

# 5. 查找 all_to_all 调用
grep -r "all_to_all" --include="*.py" | grep -i "moe\|expert"

# 6. 查找 parallel_state 的定义
find . -name "parallel_state.py"
```

### 推荐的阅读顺序

1. **Start**: `vllm/config.py` → 看 `ParallelConfig` 的定义
2. **然后**: `vllm/distributed/parallel_state.py` → 了解全局状态
3. **然后**: `vllm/distributed/initialize.py` → 初始化逻辑
4. **核心**: `vllm/model_executor/layers/fused_moe/layer.py` → MoE 的 All-to-All
5. **细节**: `vllm/_custom_ops.py` → Python wrapper
6. **C++ 端**: `csrc/moe/` 目录 → 真正的 NCCL 调用

---

## 关键代码位置速查表

| 需求 | 文件 | 具体位置 |
|------|------|--------|
| 声明 TP/EP 大小 | `vllm/config.py` | `ParallelConfig.__init__()` |
| 初始化进程组 | `vllm/distributed/initialize.py` | `initialize_model_parallel()` |
| 获取 EP group | `vllm/distributed/parallel_state.py` | `get_expert_parallel_group()` |
| 获取 TP group | `vllm/distributed/parallel_state.py` | `get_tensor_parallel_group()` |
| MoE All-to-All 入口 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE.forward_native()` / `forward_cuda()` |
| Top-K Routing | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE.select_experts()` |
| EPLB 映射 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE._apply_eplb()` 或类似 |
| TP Linear 包装 | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear` |
| 自定义 OP 包装 | `vllm/_custom_ops.py` | `moe_forward()` wrapper |
| C++ 实现 | `csrc/moe/` | `*.cu` 文件 |

---

## 对你建模工作的启示

从 EP/TP 的代码体现来看，有几个关键观测点：

1. **EP All-to-All 是瓶颈**：
   - 代码中 `torch.ops.vllm.moe_forward()` 包含的 All-to-All 是网络密集型操作
   - 在 EP 进程数多、tokens 多时可能成为瓶颈
   - 你的建模中可以加"跨 GPU 通信开销"约束

2. **TP 与 EP 是正交的**：
   - 进程组独立，通信不重叠（通常）
   - 可以分别建模，然后加总

3. **EPLB 做负载均衡**：
   - `enable_eplb` 参数做专家映射优化
   - 这提示你可以考虑"routing 优化"作为一个可控变量

4. **Per-token phase mask 的设置点**：
   - 在 `GPUModelRunner.execute_model()` 准备 batch 时
   - 所有 tokens 经过统一的 mask 在 forward 前
   - 这是 E1.1 的关键采集点

---

## 补充：如何追踪 All-to-All 的调用

### 方式 1：在 Python 侧设置 NCCL 日志

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
cd ~/sthmax/git/vllm
python your_inference_script.py 2>&1 | grep -i "all.to.all\|alltoall"
```

输出示例：
```
[NCCL] All-to-All launch on group size 2
```

### 方式 2：在代码中插入 hook

在 `fused_moe/layer.py` 中的 `forward_native()` 前后加 print：

```python
def forward_native(self, hidden_states, router_logits):
    print(f"[MoE forward] TP rank {get_tensor_parallel_rank()}, "
          f"EP rank {get_expert_parallel_rank()}")
    
    topk_weights, topk_ids = self.select_experts(...)
    
    # 这里会调用 All-to-All
    print(f"[MoE] About to call moe_forward, topk_ids shape: {topk_ids.shape}")
    
    fused_output = torch.ops.vllm.moe_forward(
        hidden_states,
        router_logits,
        self.layer_name,
    )
    
    print(f"[MoE] moe_forward done, output shape: {fused_output.shape}")
    
    return fused_output
```

---

## 总结

| 维度 | Tensor Parallel | Expert Parallel |
|------|-----------------|-----------------|
| **分割对象** | 单个层的权重矩阵 | MoE 层的 experts |
| **进程组** | `get_tensor_parallel_group()` | `get_expert_parallel_group()` |
| **通信操作** | AllGather (forward), ReduceScatter (backward) | All-to-All (dispatch + combine) |
| **通信位置** | 在 ColumnParallelLinear/RowParallelLinear 等包装层 | 在 `torch.ops.vllm.moe_forward()` 内 |
| **代码关键位置** | `vllm/model_executor/layers/linear.py` | `vllm/model_executor/layers/fused_moe/layer.py` |
| **影响对象** | 所有 Dense 层 | 仅 MoE 层 |
| **通信开销** | 与层数/模型大小成正比 | 与 token 数、expert 数、EP 进程数成正比 |

**建模建议**：
- TP 导致的延迟基本固定（取决于模型大小）
- EP 导致的延迟取决于"这一步有多少 tokens 需要重新分配"
- 当 token 数多、expert 数多、EP 进程数多时，All-to-All 可能导致显著延迟
