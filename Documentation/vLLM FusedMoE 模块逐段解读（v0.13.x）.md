# vLLM FusedMoE 模块逐段解读（v0.13.x）

> Scope：聚焦 `external/vllm/vllm/model_executor/layers/fused_moe/` 下的 **fused MoE** 实现：它的前向链路如何从 router logits 走到 kernel 级 fused GEMM / reduce，以及 vLLM 为了吞吐/延迟在 kernel 层做了哪些关键优化。
>
> 写作风格对齐你仓库里两篇参考文档：

> - `Documentation/vLLM MoE Inference Walkthrough (v0.13.0).md`
> - `Documentation/vLLM Scheduler schedule() 逐段解读.md`
>
> 注意：本笔记以“模块内部链路”为主，不重复解释 v1 engine / scheduler 的整体调用栈。

---

## 0) 快速地图：MoE 这次到底 fused 了什么？

把 MoE 的一次前向分解成 6 件事：

1. **routing**：从 `router_logits` 选出每个 token 的 Top-$k$ experts（输出 `topk_ids/topk_weights`）。
2. **dispatch prepare**：把 token 按 expert 分桶（permute / sort / pad），为后续 GEMM 做连续访问。
3. **expert GEMM #1（w1/w3 或 gate+up）**：对每个 expert 的 token block 做矩阵乘。
4. **activation / fusion**：例如 SiLU / GELU，以及 gate*up 的 fused（`is_act_and_mul`）。
5. **expert GEMM #2（w2 或 down）**：再做一次专家线性层。
6. **combine / finalize**：按 `topk_weights` 把各 expert 结果加权合并并 unpermute 回 token 原顺序；如果开启并行，还要做 all2all / reduce。

vLLM 的 “fused MoE” 主要是在 **(2)(3)(4)(5)(6)** 上做 kernel 级融合与布局优化：

- 典型 fused kernel 形态：
  - “**permute+GEMM+epilogue**” 合起来做，减少中间 tensor materialize。
  - “**topk weight + reduce**” 在 finalize 阶段用专用 kernel，一次性完成加权与回写。
- 针对不同硬件/量化/并行策略，vLLM 会选择不同 backend：Triton / CUTLASS / FlashInfer / TRTLLM / DeepGEMM / Marlin / ROCm AITER 等。

---

## 1) 入口：`FusedMoE` 层的 forward 分流（custom op vs Python 实现）

### 1.1 `FusedMoE` 是一个 `CustomOp`

`FusedMoE` 在 Python 侧注册为一个 custom op wrapper，核心 forward 会走 `forward_native()`。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    ...
```

### 1.2 `forward_native()`：CPU/TPU 走 Python；CUDA 走 `torch.ops.vllm.moe_forward(*)`

- CUDA 路径：把输入 `hidden_states/router_logits` 与 `layer_name` 交给 `torch.ops.vllm.moe_forward`。
- 非 CUDA（或 TPU/CPU）：fallback 到 `self.forward_impl(...)`（也就是 Python 侧实现）。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
        if self.shared_experts is None:
            if current_platform.is_tpu() or current_platform.is_cpu():
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            return reduce_output(fused_output)[..., :og_hidden_states]
        else:
            ...
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            return (
                reduce_output(shared_output)[..., :og_hidden_states],
                reduce_output(fused_output)[..., :og_hidden_states],
            )
```

这里的 `layer_name` 很关键：它用于从 `CompilationConfig.static_forward_context` 里“找回”这层的 Python 对象（包括权重、量化方法、并行配置等），让 C++/CUDA 或 Triton 的 custom op 能在运行时拿到必要上下文。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
        compilation_config = vllm_config.compilation_config
        ...
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
```

---

## 2) Router：Top-k 选择在 vLLM 里怎么做（以及为什么要 fused）

### 2.1 Router 抽象：`FusedMoERouter` 与 `FusedMoERouterImpl`

vLLM 把“算 topk”这件事抽象成 `FusedMoERouter`，并在 `FusedMoE` 里用一个 thin wrapper `FusedMoERouterImpl` 转发到 `_select_experts()`。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe_router.py`

```python
class FusedMoERouter(ABC):
    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
class FusedMoERouterImpl(FusedMoERouter):
    def select_experts(self, hidden_states, router_logits):
        return self.layer._select_experts(hidden_states, router_logits)
```

这样做的意义：后面的 kernel backend（modular kernel / quant_method）只依赖 router 接口，而不关心这层具体的 routing 策略细节。

### 2.2 `_select_experts()`：多种 routing 策略与实验钩子

`_select_experts()` 做的事：根据配置/环境变量/实验路由器，决定用哪种 topk 实现。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
    def _select_experts(self, hidden_states, router_logits):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        indices_type = self.quant_method.topk_indices_dtype

        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(...)

        elif self._price_moe_router is not None:
            topk_weights, topk_ids = self._price_moe_router.select_experts(...)

        elif self.use_grouped_topk and valid_grouping():
            topk_weights, topk_ids = self._grouped_topk_impl(...)

        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(...)
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor

        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(...)

        else:
            topk_weights, topk_ids = self.custom_routing_function(...)

        if self.enable_eplb:
            topk_ids = eplb_map_to_physical_and_record(...)

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        ...
        return topk_weights, topk_ids
```

#### 2.2.1 为什么 topk 也要 fused？

MoE 的瓶颈经常不是 router MLP 本身，而是：

- 选 topk 时的 **softmax/归一化 + topk**，以及后续为了 dispatch 做的 **sort/group**。
- 如果 topk 输出布局不好，会导致后面 token permute / expert bucketing 的开销显著增加。

所以 vLLM 里会用 `fused_topk` 来直接输出更适配后续 kernel 的结构（比如 token_expert_indices 这类辅助信息用于后续排序/分桶）。

---

## 3) 从 Python 层到 kernel：`quant_method.apply()` 是真正的“MoE 执行器”

当不走 `torch.ops.vllm.moe_forward` 时（或在 custom op 内部），最终都会落到这句：

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
final_hidden_states = self.quant_method.apply(
    layer=self,
    router=self.router,
    x=staged_hidden_states,
    router_logits=staged_router_logits,
)
```

这里的关键抽象：`quant_method`。

- Unquantized：`UnquantizedFusedMoEMethod`
- 量化：FP8 / GPTQ / AWQ / NVFP4 / MXFP4 / ...
- Modular：当满足条件、切换到 modular kernel 后，`quant_method` 会被替换为 `FusedMoEModularMethod`，它会把“prepare/dispatch/GEMM/finalize”拼成一个 backend pipeline。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
    def maybe_init_modular_kernel(self) -> None:
        ...
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            self.quant_method = FusedMoEModularMethod.make(
                self, self.quant_method, prepare_finalize, self.shared_experts
            )
```

---

## 4) Prepare/Finalize：Non-EP 的最小形态（理解 modular kernel 的基石）

先从最简单的“无 EP（不做 all2all）”看起：`MoEPrepareAndFinalizeNoEP`。

文件：`external/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize.py`

### 4.1 prepare：可选把 router weight 乘到 input；以及输入量化

```python
class MoEPrepareAndFinalizeNoEP(mk.FusedMoEPrepareAndFinalize):
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ):
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1
            a1 = a1 * topk_weights.to(a1.dtype)

        if self.defer_input_quant:
            return a1, None, None, None, None

        a1q, a1q_scale = moe_kernel_quantize_input(...)

        return a1q, a1q_scale, None, None, None
```

这一段透露了一个常见融合点：

- 当 $k=1$ 且选择 `apply_router_weight_on_input=True`，可以把 `topk_weight` 提前乘到输入上，从而在 finalize 阶段少一次“按权重 reduce”。
- `defer_input_quant` 允许某些 backend 把“输入量化 + experts 计算”融合成一个 kernel（比如 ROCm AITER / FlashInfer）。

### 4.2 finalize：把 fused expert 输出按 topk 权重 reduce 回 output

```python
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
```

这对应前面分解的第 (6) 步：combine/unpermute/reduce。

---

## 5) Triton kernel 示例：`fused_moe.py` 里的“按 expert 分块 GEMM 并写回”

`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` 很大，这里先抓住一个典型 kernel（GPTQ/AWQ 的 fused GEMM 版本）来理解 vLLM 的 kernel 级优化套路：

### 5.1 关键输入：`sorted_token_ids` + `expert_ids` + `num_tokens_post_padded`

在 MoE kernel 里，token 会被重复 topk 次并按 expert 分桶。

- `sorted_token_ids`: token 索引（重复 topk 次）按 expert 排序后的“访问序列”。
- `expert_ids`: 每个 token-block 对应的 expert id。
- `num_tokens_post_padded`: 为了让每个 expert bucket 的 token 数对齐到 `BLOCK_SIZE_M` 而做的 padding 后 token 总数。

下面这段代码体现了：**先判断这个 block 是否越界 / 是否属于当前 EP rank，然后决定计算或写 0**。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
@triton.jit
def fused_moe_kernel_gptq_awq(
    ...
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    ...
):
    ...
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(...)
        return
```

#### 5.1.1 这里的 kernel 优化点是什么？

- **按 expert 分桶**：让同一个 program instance 里用同一套 `B`（expert 权重），提高 L2 reuse。
- **padding 到 BLOCK_SIZE_M**：让 token 维度是整 block，避免尾块分支导致 warp divergence。
- **EP rank mask（expert_id == -1）**：在 expert-parallel 下，不属于本 rank 的 expert 直接写 0；后续再做 reduce（或 all2all 合并）。

---

## 6) Chunked DP：为什么要把 hidden_states/router_logits 分块？

当 DP 维度>1 且使用某些 MoE kernel（PPLX / DeepEP-LL / FlashInfer NVFP4 + DP）时，vLLM 会启用 **DP chunking**。

入口：`FusedMoE.ensure_dp_chunking_init()` + `forward_impl_chunked()`。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return
        ...
        self.batched_hidden_states = torch.zeros(states_shape, ...)
        self.batched_router_logits = torch.zeros(logits_shape, ...)
```

`forward_impl_chunked()` 会把大 batch 拆分成若干段，每段把数据 copy 到 staging buffer，再调用 `quant_method.apply()`；这样可以：

- 控制峰值显存（中间 buffer 在一个 chunk 内复用）。
- 在某些通信/流水 backend 下更容易 overlap。

代码片段（节选）：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            final_hidden_states = self.quant_method.apply(
                layer=self,
                router=self.router,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
            )
```

---

## 7) Shared experts overlap：`SharedFusedMoE` 为什么要单独一条 CUDA stream

`SharedFusedMoE` 继承 `FusedMoE`，目标是把 shared experts（普通 MLP）与 routed experts（MoE）并行 overlap。

文件：`external/vllm/vllm/model_executor/layers/fused_moe/shared_fused_moe.py`

### 7.1 不 overlap：先算 shared，再算 fused

```python
    def forward(self, hidden_states, router_logits):
        if not self.use_overlapped:
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)
                ...
            fused_out = super().forward(hidden_states, router_logits)
```

### 7.2 overlap：让 shared 与 MoE kernel 并行

当 overlap 开启时，`FusedMoE` 内部会通过 `_maybe_setup_shared_experts_stream()` 把 `hidden_states.clone()` 放到另一个 stream 上算 shared experts，同时默认 stream 上继续做 router + MoE kernel。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
    def _maybe_setup_shared_experts_stream(...):
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (hidden_states.shape[0] <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD)
        )

        if use_shared_experts_stream:
            hidden_states_clone = hidden_states.clone()
            hidden_states_clone.record_stream(self.shared_experts_stream)
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone
```

这是一种典型“吞吐优化”：在 token 数不大（decode 小 batch）的场景下，尽量把 shared MLP 的时间藏在 MoE 的时间里。

---

## 8) 下一步：把 backend 选择逻辑和完整执行链路补齐

本文件目前已经把：

- `FusedMoE.forward_native` 的分流
- `_select_experts` 的 routing 决策
- `quant_method.apply` 作为统一执行入口
- `prepare/finalize` 的最小形态
- 一个典型 Triton fused kernel 形态
- DP chunking 与 shared experts overlap

串起来了。

下一步需要继续精读并补齐 3 块（我会继续直接更新本文档）：

1. `fused_moe.py` 中 `fused_topk/fused_topk_bias/GroupedTopk` 的实现细节（它们具体如何在 kernel 输出适配 dispatch 的布局）。
2. `FusedMoEModularMethod` + `modular_kernel.py`：backend 选择（CUTLASS/DeepGEMM/FlashInfer/TRTLLM/Marlin/AITER）与 prepare/finalize 的拼装方式。
3. EP/DP 下 all2all 的 prepare/finalize（例如 `all2all_utils.py`, `deepep_*_prepare_finalize.py`, `flashinfer_*_prepare_finalize.py`）。

---

## 9) `fused_topk / GroupedTopk`：routing 输出为什么“更适配后续 dispatch”

这一节的目标是把 `_select_experts()` 里提到的 `fused_topk / fused_topk_bias / GroupedTopk` 对应到代码层的“输出契约（contract）”，回答两个问题：

1. router 的输出到底是什么 shape/dtype？
2. 它为什么能降低后续 `permute/sort/pad`（也就是 dispatch）的成本？

> 说明：`_select_experts()` 的调用点在 `layer.py`，而具体实现集中在 `fused_moe.py`（以及少量 grouped-topk 的实现）。我这里先以“后续 dispatch 真正依赖什么”为主线，把关键变量和数据流讲清楚。

### 9.1 `_select_experts()` 的输出契约

从 `FusedMoEModularMethod.apply()` 能直接看到 router 的输出被立刻喂给 fused experts（无论 backend 是 Triton/DeepGEMM/CUTLASS/…）：

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`

```python
    def apply(self, layer, router, x, router_logits):
        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        result = self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            ...
        )
        return result
```

因此，router 的 contract 实际上就是：

- `topk_ids`: shape = `(num_tokens, top_k)`，表示每个 token 选中的 expert id。
- `topk_weights`: shape = `(num_tokens, top_k)`，表示每个 token 对应 expert 的权重（通常来自 softmax / renormalize / sigmoid 等）。

此外还有一个“隐藏 contract”：`topk_ids.dtype` 可能会被 all2all backend 限制，所以 `_select_experts()` 里会根据 `quant_method.topk_indices_dtype` 做强制 cast。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
        indices_type = self.quant_method.topk_indices_dtype
        ...
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)
```

### 9.2 为什么需要 `token_expert_indices`（以及它跟 dispatch 的关系）

你会注意到 `_select_experts()` 里 `fused_topk()` 返回了 3 个值：

```python
topk_weights, topk_ids, token_expert_indices = fused_topk(...)
```

虽然这一层最终只返回 `(topk_weights, topk_ids)`，但 `token_expert_indices` 的存在暗示：topk 的实现会在计算阶段就构造一些“便于后续分桶/排序”的中间结构，避免后面再做一次昂贵的 scatter/gather。

这类结构在 vLLM 的 fused experts 里最典型的消费者就是 `moe_align_block_size()`：它会把 token 重复 topk 次后按 expert 排序并 padding 到 `BLOCK_SIZE_M`。

代码（使用点）：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
```

这里的关键是：**dispatch 的最小充分信息** 是：

- `sorted_token_ids`：告诉 kernel 以“按 expert 排序后的 token 访问顺序”读取 A。
- `expert_ids`：告诉 kernel 每个 token-block 对应使用哪个 expert 的 B。
- `num_tokens_post_padded`：告诉 kernel padding 后 token 总数，让 grid 可以安全越界返回。

而 `fused_topk` 的“更适配”体现在：它能以更低的 overhead 生成 `topk_ids/weights`，让 `moe_align_block_size` 的输入更规整（例如 dtype、contiguous、以及某些布局假设），从而减少 `moe_align_block_size` 的额外整理成本。

### 9.3 `GroupedTopk`（DeepSeek 风格）为什么会影响后续 dispatch

`FusedMoE.__init__` 里在 `use_grouped_topk=True` 时会创建 `GroupedTopk`：

---

### 9.4 `fused_topk` 的本体实现：先分配输出，再把 topk+softmax 下沉到 custom op

`fused_topk` 的核心目的：把“Top-k 选择 +（可选）Top-k 内归一化”变成一个更少 kernel launch、更少 Python 端张量拼装的过程；并额外产出 `token_expert_indices`，让后续 dispatch 更容易做到“token-expert pair 的扁平化索引”。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`（excerpt）

```python
def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_func = dispatch_topk_func(use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled())
    topk_weights, topk_ids = topk_func(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )

    return topk_weights, topk_ids, token_expert_indices
```

逐段解释：

- `topk_weights/topk_ids/token_expert_indices = torch.empty(...)`：
  - 先在 Python 端分配好输出（包括 `token_expert_indices`），再交给后端填充。
  - 这让后端实现（`ops.topk_softmax` / aiter）可以在 CUDA/C++ 里直接写结果，避免 Python 端再做 `topk + gather + normalize` 的一堆张量操作。
- `indices_type`：允许上层把 `topk_ids` 的 dtype 调整到 all2all/dispatch 期望的类型（例如某些路径需要 `uint32`）。
- `dispatch_topk_func(...)`：把“选择 vLLM 自家 topk softmax kernel”与“选择 ROCm aiter 实现”统一起来。
- `ops.topk_softmax(...)`：这是 performance 的关键点：它在一个 fused 实现里同时产出
  - `topk_weights`
  - `topk_indices`
  - `token_expert_indices`

### 9.5 `fused_topk_bias`：bias 只影响“选谁”，权重仍来自原始 softmax 分布

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`（excerpt）

```python
def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    n_routed_experts = gating_output.shape[-1]
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores.view(
        -1, n_routed_experts
    ) + e_score_correction_bias.unsqueeze(0)

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[1]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)
```

逐段解释：

- `scores = softmax(gating_output)`：把 router logits 变成概率。
- `scores_for_choice = scores + bias`：bias 只用于“排序/选择”，不会直接改变输出权重。
- `use_sorted = vllm_is_batch_invariant()`：如果需要 batch invariance，则 `torch.topk(..., sorted=True)` 保证 deterministic。
- `topk_weights = scores.gather(...)`：权重来自 unbiased 的 `scores`。
- `renormalize`：对 top-k 权重重新归一化。

### 9.6 `grouped_topk`（函数）与 `GroupedTopk`（CustomOp wrapper）：DeepSeek 的“先选组再组内 top-k”

DeepSeek-V2/V3 的 grouped routing 要点：

1. 把专家分成 `num_expert_group` 组。
2. 每个 token 先选 top-`topk_group` 个组。
3. 只在这些组内做 top-`topk` experts。

此外在满足一些约束条件（CUDA + group/topk <= 32 + 有 correction bias）时，会走 `ops.grouped_topk` 的 fused kernel。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`（excerpt）

```python
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and current_platform.is_cuda()
        and num_expert_group <= 32
        and topk <= 32
        and e_score_correction_bias is not None
    ):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )

    use_sorted = vllm_is_batch_invariant()
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
```

逐段解释（聚焦 dispatch 相关的结构性影响）：

- grouped routing 会显著减少“一个 token 可能被路由到的候选专家集合”，从而让后续 `moe_align_block_size` 在统计上更容易得到更集中、更规整的 expert buckets。
- `e_score_correction_bias` 的存在会把“选择逻辑”和“权重”分离：选择走 biased scores，但权重仍用 unbiased scores，避免强行扭曲 routing weights。

而 `GroupedTopk` 则是把它包装成 vLLM CustomOp。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`（excerpt）

```python
@CustomOp.register("grouped_topk")
class GroupedTopk(CustomOp):
    """GroupedTopk used by the Deepseek-V2 and Deepseek-V3 model."""

    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        num_fused_shared_experts: int = 0,
    ) -> None:
        super().__init__()
        self.native_impl = grouped_topk
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.native_impl(
            hidden_states,
            gating_output,
            self.topk,
            self.renormalize,
            self.num_expert_group,
            self.topk_group,
            self.scoring_func,
            self.routed_scaling_factor,
            e_score_correction_bias,
        )
```

逐段解释：

- `@CustomOp.register("grouped_topk")`：接入 vLLM 的 CustomOp 调度体系（native/cuda/hip 三套 forward）。
- `self.native_impl = grouped_topk`：GroupedTopk 本身不实现算法，而是固化参数并转发到函数实现。
- `forward_native(...)`：把实例化时的超参（group 数、组内 topk 等）一次性传进去，形成“可复用的 router op”。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
        if self.use_grouped_topk:
            self._grouped_topk_impl = GroupedTopk(
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
```

Grouped-topk 的要点不是“shape 不同”，而是 **topk 的候选集合先被分组裁剪**。这会改变 token→expert 的稀疏模式：

- 更强的组内局部性：同一个 token 的 topk 更倾向落在某个 expert 子集。
- dispatch 的分桶更集中：`moe_align_block_size` 排序后每个 expert bucket 更可能得到更长的连续 token 段，提升 kernel 的 L2 reuse（同一个 expert 的 B 会被连续使用）。

---

## 10) Modular Kernel：`FusedMoEModularKernel.forward()` 的完整执行链（Prepare → Experts → Finalize）

这一节把 modular kernel 的“执行骨架”按代码逐段拆开，回答：

- modular kernel 到底把哪些步骤标准化了？
- 异步 all2all（prepare_async/finalize_async）是如何插进 pipeline 的？
- chunking/workspace 这类 kernel 级优化怎么落到代码？

### 10.1 modular kernel 的入口：`FusedMoEModularMethod.make()`

它将 3 件事组合成一个可调用模块：

1. `prepare_finalize`：负责 quantize+dispatch（可含 all2all）和 finalize 的 combine/reduce
2. `gemm_impl`（即 `FusedMoEPermuteExpertsUnpermute` 的实现）：负责 permute+experts+unpermute
3. `shared_experts`：可选 overlap

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`

```python
    @staticmethod
    def make(moe_layer, old_quant_method, prepare_finalize, shared_experts):
        return FusedMoEModularMethod(
            old_quant_method,
            FusedMoEModularKernel(
                prepare_finalize,
                old_quant_method.select_gemm_impl(prepare_finalize, moe_layer),
                shared_experts,
                moe_parallel_config=moe_layer.moe_parallel_config,
            ),
        )
```

### 10.2 `FusedMoEModularKernel.forward()` 的总体结构

`forward()` 是最清晰的“骨架”：

1. `output` buffer 选择（inplace vs outplace）
2. `_prepare(...)`：可能触发 all2all + 量化
3. `_fused_experts(...)`：做两次 GEMM + activation（支持 chunking）
4. `_finalize(...)`：apply topk weight + reduce/unpermute（可能 finalize_async）

代码：`external/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`

```python
    def forward(self, hidden_states, w1, w2, topk_weights, topk_ids, inplace=False, ...):
        if inplace and self.shared_experts is None and not disable_inplace():
            output = hidden_states
        else:
            output = torch.zeros_like(hidden_states)

        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )

        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
        )

        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )
```

### 10.3 `_prepare()`：同步 vs 异步，以及 DBO hook

`_prepare()` 的关键点：

- 如果 prepare/finalize 不支持 async：直接调用 `prepare()`。
- 如果支持 async：会走 `prepare_async()`，并引入 hook（用于 DBO 或非 DBO 的 await）。
- 另外，prepare 阶段可能会输出“从其他 EP rank gather 后”的 `_expert_topk_ids/_expert_topk_weights`，用于后续统一使用。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`

```python
        if not self.prepare_finalize.supports_async():
            (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids, _expert_topk_weights) = (
                self.prepare_finalize.prepare(...)
            )
        else:
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(...)

            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )
            if hook is not None:
                if dbo_enabled():
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids, _expert_topk_weights) = receiver()

        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = topk_weights if _expert_topk_weights is None else _expert_topk_weights
```

这段代码本质上把“通信等待”从纯同步变成：

$$\text{dispatch launch} \rightarrow \text{compute overlap} \rightarrow \text{recv hook / receiver barrier}$$

为 shared experts overlap / DBO 提供了插槽。

### 10.4 `_fused_experts()`：chunking、workspace 分配、以及 tokens-per-expert metadata

`_fused_experts()` 解决了两类现实问题：

1. 有的 kernel 支持 activation chunking（减少峰值 workspace），但需要对 chunk 的 `topk_ids/topk_weights` slicing 以及 per-expert token count 重算。
2. DP+EP 下 profiling run 可能“看不到最坏情况路由”，所以要强制 worst-case workspace allocation。

#### 10.4.1 chunk 大小计算

代码：`external/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`

```python
    def _chunk_info(self, M: int) -> tuple[int, int]:
        CHUNK_SIZE = max(
            1,
            (
                M
                if not self.fused_experts.enable_chunking()
                else min(M, envs.VLLM_FUSED_MOE_CHUNK_SIZE)
            ),
        )
        num_chunks = cdiv(M, CHUNK_SIZE)
        return num_chunks, CHUNK_SIZE
```

#### 10.4.2 重新计算 chunk 的 expert token 计数（避免用整批统计）

代码：`external/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`

```python
        c_expert_num_tokens = count_expert_num_tokens(
            chunk_topk_ids, local_num_experts, expert_map
        )
```

这个细节很重要：某些 batched-experts 格式 kernel（例如需要 `[E, max_tokens_per_expert, ...]` 的那类）必须知道每个 expert 当前 chunk 分到多少 token，才能做紧凑 buffer 分配和高效 kernel launch。

### 10.5 `_finalize()`：finalize_async + shared experts overlap

finalize 的逻辑与 prepare 对称：支持 async 的实现会返回 hook+receiver（或 receiver），并允许 shared experts 在等待通信期间执行。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`

```python
        if not self.prepare_finalize.supports_async():
            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            hook, receiver = (
                finalize_ret if isinstance(finalize_ret, tuple) else (None, finalize_ret)
            )
            ...
            if hook is not None:
                ... hook() / dbo_register_recv_hook ...
            receiver()
```

这里的核心是：finalize 阶段需要确保 output 已经完成 combine/reduce 才能被上层使用，所以 receiver() 是最终的“有效性屏障”。

---

## 11) EP/DP 下的 All2All：`maybe_make_prepare_finalize()` 如何选 PPLX/DeepEP（以及它们分别做了什么）

这一节把“EP/DP 相关的 prepare/finalize”落到具体代码路径上：它不是散落在 kernel 里，而是由 `all2all_utils.maybe_make_prepare_finalize()` 统一选择并创建。

文件：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

### 11.1 入口：什么时候才会创建 all2all prepare/finalize？

只有当 `moe_parallel_config.use_all2all_kernels=True` 时，才会让 quant_method 初始化 modular kernel（也就是前面 `maybe_init_modular_kernel()`）去拿一个 `prepare_finalize`。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
def maybe_make_prepare_finalize(moe, quant_config, routing_tables=None):
    if not moe.moe_parallel_config.use_all2all_kernels:
        return None

    all2all_manager = get_ep_group().device_communicator.all2all_manager
    assert all2all_manager is not None
```

> 这里的 `all2all_manager` 是“通信后端的统一入口”，内部能根据 dp/tp/ep 组织拿到 handle。

### 11.2 总分支：PPLX vs DeepEP HT vs DeepEP LL

`maybe_make_prepare_finalize()` 的主体就是三选一：

代码：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
    if moe.use_pplx_kernels:
        ...
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = PplxPrepareAndFinalize(handle, ...)

    elif moe.use_deepep_ht_kernels:
        handle = all2all_manager.get_handle({})
        prepare_finalize = DeepEPHTPrepareAndFinalize(handle, ...)

    elif moe.use_deepep_ll_kernels:
        ...
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPLLPrepareAndFinalize(handle, ...)

    return prepare_finalize
```

这三种实现的共同点是：

- **prepare**：负责把本 rank 的 tokens 发出去（发往拥有对应 experts 的 rank），并把属于本 rank experts 的 tokens 收回来。
- **finalize**：把本 rank 算完的 outputs 再 all2all 回去，让每个 rank 拿到它“原始 token 顺序”的输出，然后进行最后的 combine/reduce（或者 finalize 本身就完成 reduce）。

区别主要在：

- 目标：HT（吞吐） vs LL（延迟）、以及是否要求 batched/graph-friendly。
- 数据格式：是否支持 fp8 dispatch 来降低通信量。
- 对 expert placement 的要求：LL + round-robin placement 需要 routing_tables 来做 global↔physical 映射。

### 11.3 PPLX：为什么需要计算 `hidden_dim_bytes` / `hidden_dim_scale_bytes`

PPLX 的 all2all handle 初始化时会显式传入“每 token hidden 的字节数”和“scale 的字节数”。这说明：PPLX 的通信层面是按“扁平字节 buffer”搬运，而不是按高层 tensor 语义。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
        hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
            moe.max_num_tokens,
            moe.hidden_dim,
            moe.in_dtype,
            quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

        all_to_all_args = dict(
            max_num_tokens=moe.max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,
            ...
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=hidden_dim_bytes,
            hidden_dim_scale_bytes=hidden_scale_bytes,
        )
        handle = all2all_manager.get_handle(all_to_all_args)
```

直觉：

- 量化/scale 的存在会改变通信 payload 的布局。
- 把“payload bytes”显式化能让通信 kernel 在 C++ 层做更紧凑的 pack/unpack。

### 11.4 DeepEP-LL：FP8 dispatch 与 routing_tables（round-robin placement）

DeepEP-LL 分支里有两个关键细节：

#### 11.4.1 FP8 dispatch：为的是减小 all2all 带宽

代码：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            ...
            use_fp8_dispatch=use_fp8_dispatch,
            ...
        )
```

含义：即使最终 experts GEMM 不一定是 FP8，dispatch 阶段也可能用 FP8 来搬运 activation，从而降低通信成本（当然需要 block shape 等条件满足 kernel 要求）。

#### 11.4.2 routing_tables：round-robin placement 需要 global↔physical 映射

代码：`external/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (global_to_physical, physical_to_global, local_expert_global_ids) = routing_tables
```

这解释了为什么 `FusedMoE.maybe_init_modular_kernel()` 里会在特定条件下先构建 routing tables（round-robin + deepep-ll），再把它传给 `maybe_make_prepare_finalize()`。

---

## 12) 对照：`fused_experts_impl()`（旧路径）如何做“topk→align→2xGEMM→sum”

这一节的目的：用 `fused_moe.py` 里更“直接”的实现（非 modular）作为对照，让你看清 modular kernel 其实就是把这些步骤拆成组件，并在 prepare/finalize 里插入 EP/DP 通信。

### 12.1 关键流程（chunked）

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    ...
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        ...
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(...)

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(...)
        else:
            expert_ids = curr_topk_ids.view(-1)
            sorted_token_ids = None

        dispatch_fused_moe_kernel(..., w1, intermediate_cache1, ..., sorted_token_ids, expert_ids, ...)
        apply_moe_activation(activation, intermediate_cache2, intermediate_cache1.view(-1, N))
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(...)
        dispatch_fused_moe_kernel(..., w2, intermediate_cache3, ..., sorted_token_ids, expert_ids, ...)

        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()), out_hidden_states[...])
```

### 12.2 `naive_block_assignment`：小 batch + 大 expert 数时的特殊优化

这段 heuristics 很典型：当 `tokens_in_chunk * top_k` 只激活很少一部分 experts，且没有 expert_map（即非 EP）以及某些量化路径限制时，可以不做全量 `moe_align_block_size`，直接把 `sorted_token_ids=None` 交给 kernel，让 kernel 走 `naive_block_assignment` 的路径（每个 block 对应一个 token）。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
        SPARSITY_FACTOR = 4
        naive_block_assignment = (
            expert_map is None
            and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (use_int8_w8a16 or use_int4_w4a16)
                and block_shape is not None
                and block_shape[1] > 0
            )
        )
```

进一步看 kernel 内部就能看到对应分支：当 `naive_block_assignment=True`，`sorted_token_ids` 不再用于加载 token 序列，而是把 `offs_token` 构造为“一个 block 只算一个 token”。

代码：`external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,
            num_valid_tokens,
        )
```

这个优化的效果：在 decode 小 batch 时，减少一次 `moe_align_block_size` 的排序/分桶开销。
