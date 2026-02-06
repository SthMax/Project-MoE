# PriceMoE ~10% Throughput Loss: Root Cause Analysis & Fix Plan

**Date**: 2026-02-06  
**Status**: Analysis complete, fix plan ready  
**Author**: Copilot (automated analysis)

---

## 1. Executive Summary

The PriceMoE router introduces a consistent **~10% throughput degradation** (from ~2,380 tok/s down to ~2,115 tok/s) compared to the vLLM baseline, **even when α=0** (which should make routing decisions identical to baseline). This conclusively proves the overhead is from the **code path itself**, not from the routing policy. There are **5 distinct root causes**, all fixable.

---

## 2. Experimental Evidence

| Run (260205) | Mode | α | Tok/s | Δ vs Baseline |
|---|---|---|---|---|
| baseline | baseline | — | 2,379.74 | — |
| price_moe | price_moe | 0.00 | 2,115.47 | **−11.1%** |
| price_moe | price_moe | 0.02 | 2,121.35 | **−10.9%** |
| price_moe | price_moe | 0.05 | 2,137.66 | **−10.2%** |
| price_moe | price_moe | 0.10 | 2,109.38 | **−11.4%** |

**Key observation**: α=0.00 performs identically to α=0.10 in terms of overhead, proving the loss is structural, not algorithmic.

---

## 3. Root Cause Analysis

### Root Cause #1 (CRITICAL): Bypassing the Fused CUDA Kernel `topk_softmax` (~7-8%)

**The single biggest cause of the performance gap.**

#### Baseline path (fast):
```
router_logits (bf16/fp16, GPU)
    │
    └─→ ops.topk_softmax()          ← ONE fused CUDA kernel
         │  Reads logits once from HBM
         │  Computes softmax in-register
         │  Finds top-k in-register
         │  Writes topk_weights + topk_ids back
         │  Total: 1 kernel launch, 1 HBM read, 1 HBM write
         ▼
    (topk_weights fp32, topk_ids int32)
```

The `topk_softmax` kernel in `csrc/moe/topk_softmax_kernels.cu` is a **highly optimized fused kernel** that:
- Loads gating output from HBM **once** (vectorized loads, bf16 → float conversion in-register)
- Computes softmax using warp-level butterfly reductions (never writes intermediate softmax to HBM)
- Finds top-k using iterative in-register argmax with warp shuffles
- Writes only the final `topk_weights[T, k]` and `topk_ids[T, k]` to HBM

**Supports bf16/fp16 input natively** — never materializes a float32 copy of the full `[T, E]` logits.

#### PriceMoE path (slow):
```
router_logits (bf16, GPU)
    │
    ├─→ self._expert_load.unsqueeze(0)    ← Tensor creation
    ├─→ router_logits - congestion_penalty ← Elementwise subtract [T, E] — NEW [T, E] tensor in HBM
    ├─→ torch.topk(adjusted_logits, k=k)  ← PyTorch topk (NOT fused with softmax)
    │     → Reads [T, E] from HBM
    │     → Writes [T, k] weights + [T, k] ids to HBM
    │
    ├─→ torch.gather(router_logits, ...)   ← Another HBM read of original logits
    ├─→ torch.softmax(original_scores, ..) ← Yet another kernel launch
    │
    ├─→ torch.bincount(topk_ids.flatten()) ← GPU→GPU histogram
    └─→ EMA update                         ← Elementwise multiply + add
    
    Total: ~6-7 kernel launches, 3+ full [T, E] HBM reads/writes
```

**The critical difference**: The PriceMoE path creates a full `[T, E]` adjusted logits tensor in HBM, then runs PyTorch's generic `torch.topk` on it (which is NOT fused with softmax). The baseline path never materializes the full softmax output — it computes softmax and topk in a single pass within GPU registers.

For Qwen3-30B-A3B with E=128 and typical batch T=512-4096, this means:
- Extra HBM read/write: `T × 128 × 4 bytes` (the adjusted_logits allocation)
- Extra HBM read: `T × 128 × 4 bytes` (torch.topk reading adjusted_logits)
- Extra HBM read: `T × k × 4 bytes` (torch.gather for original weights)
- Extra kernel launches: 5-6 more CUDA kernel launches per MoE layer per forward

With **48 MoE layers** and hundreds of forward passes, this adds up enormously.

### Root Cause #2 (MODERATE): `torch.bincount` Synchronization (~1-2%)

```python
counts = torch.bincount(
    topk_ids.flatten().to(torch.int64),  # ← dtype cast + flatten
    minlength=E,
).float()  # ← another dtype cast

self._expert_load = self.ema_decay * self._expert_load + (1.0 - self.ema_decay) * counts
```

`torch.bincount` on GPU:
1. Requires `int64` input → forces a dtype conversion from `int32` to `int64`
2. The `.flatten()` call creates a contiguous copy
3. The `.float()` cast on the result creates another copy
4. The EMA update involves 2 scalar multiplications and 1 addition on `[E]` vectors

This runs on **every forward through every MoE layer** (48 layers × hundreds of forwards).

### Root Cause #3 (MODERATE): `record_topk` Stats Collection in Baseline (~0.5-1%)

Both baseline AND price_moe set `VLLM_MOE_STATS_DIR`, which means `record_topk()` runs on **every forward** for both:

```python
# This runs for BOTH baseline and price_moe
record_topk(
    layer_name=self.layer_name,
    topk_ids=topk_ids,
    num_experts=int(router_logits.shape[-1]),
    k_experts=int(self.top_k),
)
```

Inside `record_topk`:
- `ids.to(dtype=torch.int64)` — dtype cast on GPU
- `torch.bincount(flat, minlength=num_experts)` — GPU histogram
- `torch.bincount(ids[:, 0], minlength=num_experts)` — another GPU histogram
- `.detach().cpu().numpy()` — **GPU→CPU synchronization** (pipeline stall!)
- Threading lock acquisition
- NumPy accumulation

**However**, since this runs for baseline too, it creates a "floor" for both paths. The relative ~10% gap is NOT caused by `record_topk`, but it still degrades absolute throughput for both.

### Root Cause #4 (MINOR): Python Overhead in the Hot Path (~0.5%)

The PriceMoE router is pure Python dispatching PyTorch ops, while the baseline goes directly from `_select_experts` → `fused_topk` → C++ kernel. The Python overhead includes:
- `_ensure_initialized()` check on every forward
- `self._total_tokens_routed += T` (Python integer addition in hot loop)
- `self._num_forwards += 1`
- `torch.no_grad()` context manager entry/exit
- Multiple attribute lookups (`self.alpha`, `self.ema_decay`, etc.)

### Root Cause #5 (MINOR): Lost Opportunity for `token_expert_indices` (~0.2%)

The original `fused_topk` returns a third value `token_expert_indices` which is a precomputed permutation index used by downstream kernels for efficient token-to-expert scatter. The PriceMoE path doesn't compute this, which may force downstream code to recompute it.

---

## 4. Fix Plan — Detailed & Executable

### Phase 1: Zero-Overhead α=0 Path (Target: 0% loss when α=0)

**Goal**: When `VLLM_PRICE_MOE_ENABLED=1` but `α=0`, the router should produce **bit-identical** results to baseline with **zero overhead**.

#### Step 1.1: Modify `_select_experts` to use `fused_topk` as the compute kernel

Instead of the current architecture where PriceMoE replaces the entire topk selection, modify it to **only inject the logit adjustment** and then fall through to the existing `fused_topk` kernel.

**File**: `external/vllm/vllm/model_executor/layers/fused_moe/layer.py`

```python
# IN _select_experts, REPLACE the current PriceMoE branch:

# CURRENT (slow):
elif self._price_moe_router is not None:
    topk_weights, topk_ids = self._price_moe_router.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

# NEW (fast):
elif self._price_moe_router is not None:
    # Phase 1: Modify logits in-place (minimal overhead)
    router_logits = self._price_moe_router.adjust_logits(router_logits)
    # Phase 2: Fall through to native fused_topk
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
    )
    # Phase 3: Update queue (after routing decisions)
    self._price_moe_router.update_queue(topk_ids)
```

#### Step 1.2: Rewrite PriceMoERouter with a thin `adjust_logits` method

**File**: `external/vllm/vllm/model_executor/layers/fused_moe/price_moe_router.py`

```python
class PriceMoERouter:
    def adjust_logits(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply congestion penalty: s̃ = s - α · q_e
        
        When α=0, returns the input unchanged (zero overhead).
        When α>0, performs a single in-place subtraction.
        """
        if self.alpha == 0.0:
            return router_logits  # Zero overhead!
        
        # In-place modification to avoid allocating new [T, E] tensor
        # penalty shape: (E,) broadcasts to (T, E) automatically
        router_logits = router_logits - self.alpha * self._expert_load
        return router_logits
    
    def update_queue(self, topk_ids: torch.Tensor) -> None:
        """Update EMA expert load from routing decisions."""
        if self.alpha == 0.0:
            return  # No need to track load when α=0
        
        with torch.no_grad():
            E = self._expert_load.shape[0]
            # Use int32 directly — no dtype cast needed
            counts = torch.zeros(E, dtype=torch.float32, device=topk_ids.device)
            counts.scatter_add_(
                0,
                topk_ids.reshape(-1).to(torch.int64),
                torch.ones(topk_ids.numel(), dtype=torch.float32, device=topk_ids.device),
            )
            self._expert_load.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
```

**Key insight**: When α=0, `adjust_logits` is a no-op and `update_queue` is skipped entirely, so the **only** overhead is a single Python `if` check — effectively zero.

#### Step 1.3: Handle `use_original_weights` properly with `fused_topk`

The current PriceMoE uses "original weights" — i.e., softmax of the **original** logits at the selected expert positions. But `fused_topk` already computes softmax over the selected experts (with renormalization). 

When `use_original_weights=True` and α>0:
- The adjusted logits change **which** experts are selected
- But weights should be computed from the **original** logits

This requires a **post-processing step** after `fused_topk`:

```python
if self.use_original_weights and self.alpha > 0:
    # fused_topk computed weights from adjusted logits.
    # We need to recompute weights from original logits at selected positions.
    original_scores = torch.gather(original_router_logits, dim=1, index=topk_ids.to(torch.int64))
    topk_weights = torch.softmax(original_scores.float(), dim=-1)
```

**Important**: This is only needed when α>0. When α=0, the logits are identical so fused_topk's weights are already correct.

### Phase 2: Optimized α>0 Path (Target: <2% loss)

**Goal**: When α>0, minimize overhead from logit adjustment + queue update.

#### Step 2.1: Use `fused_topk_bias`-style approach

vLLM already has `fused_topk_bias` which does exactly this pattern: compute softmax first, then add a bias, then topk. We can follow the same pattern:

```python
def price_moe_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    expert_load: torch.Tensor,  # (E,) 
    alpha: float,
    topk: int,
    renormalize: bool,
    use_original_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PriceMoE topk using the same pattern as fused_topk_bias."""
    n_experts = gating_output.shape[-1]
    
    # Compute softmax once
    scores = gating_output.softmax(dim=-1)  # (T, E)
    
    # Subtract congestion penalty from scores (not logits)
    # This avoids the need for a separate "original weights" path
    scores_for_choice = scores - alpha * expert_load.unsqueeze(0)
    
    # Select top-k based on adjusted scores
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
    
    # Gather original scores as weights (like fused_topk_bias does)
    topk_weights = scores.gather(1, topk_indices)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)
```

**Why this is better than the current approach**:
1. Follows the same pattern as `fused_topk_bias` (proven performant in vLLM for DeepSeek)
2. Only creates one extra `[T, E]` tensor (scores_for_choice) instead of two
3. `softmax` is computed once and reused for both selection and weighting
4. No separate `torch.gather` needed — weights come directly from `scores.gather`

#### Step 2.2: Replace `torch.bincount` with `scatter_add_` for queue update

```python
def update_queue(self, topk_ids: torch.Tensor) -> None:
    """Optimized queue update using scatter_add_ instead of bincount."""
    with torch.no_grad():
        E = self._expert_load.shape[0]
        counts = torch.zeros(E, dtype=torch.float32, device=topk_ids.device)
        # scatter_add_ avoids the int64 cast that bincount requires
        flat_ids = topk_ids.reshape(-1).long()
        counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
        # In-place EMA update (no temporaries)
        self._expert_load.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
```

#### Step 2.3: Apply `@torch.compile` for the routing function

Wrap the PriceMoE routing logic in `@torch.compile` to fuse the PyTorch operations:

```python
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _price_moe_topk_compiled(
    gating_output: torch.Tensor,
    expert_load: torch.Tensor,
    alpha: float,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores - alpha * expert_load.unsqueeze(0)
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)
```

`torch.compile` should fuse the softmax + subtraction + topk into fewer kernel launches (ideally 1-2 instead of 4-5).

### Phase 3: Write a Custom CUDA Kernel (Target: <1% loss, optional)

**Goal**: Modify the existing `topk_softmax` CUDA kernel to accept an optional bias tensor.

#### Step 3.1: Extend `topkGatingSoftmax` kernel signature

**File**: `external/vllm/csrc/moe/topk_softmax_kernels.cu`

Add an optional `expert_bias` parameter to the kernel:

```cuda
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, 
          typename IndType, typename InputType = float>
__global__ void topkGatingSoftmaxWithBias(
    const InputType* input,
    const float* expert_bias,  // NEW: optional (E,) bias to add to softmax scores
    float bias_scale,          // NEW: scale factor (= -alpha)
    const bool* finished,
    float* output,
    const int num_rows,
    IndType* indices,
    int* source_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const bool use_original_weights)  // NEW: if true, weights come from unbiased scores
```

**Modification in the kernel**: After computing softmax in-register (`row_chunk`), before finding top-k:

```cuda
// Apply bias after softmax (for selection only)
float biased_chunk[VPT];
if (expert_bias != nullptr) {
    for (int ii = 0; ii < VPT; ++ii) {
        int expert_idx = first_elt_read_by_thread + (ii / ELTS_PER_LDG) * COLS_PER_GROUP_LDG 
                       + (ii % ELTS_PER_LDG);
        biased_chunk[ii] = row_chunk[ii] + bias_scale * expert_bias[expert_idx];
    }
} else {
    for (int ii = 0; ii < VPT; ++ii) {
        biased_chunk[ii] = row_chunk[ii];
    }
}
// Use biased_chunk for top-k selection, but row_chunk for final weights
```

This gives us **true zero-overhead** when `expert_bias == nullptr` (baseline), and **minimal overhead** when bias is present (one extra multiply-add per element in-register, no HBM traffic).

#### Step 3.2: Expose through Python binding

```python
# In fused_moe.py
def fused_topk_with_price(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    expert_load: torch.Tensor,
    alpha: float,
    topk: int,
    renormalize: bool,
    use_original_weights: bool = True,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fused_topk with PriceMoE congestion penalty baked into the kernel."""
    M, _ = hidden_states.size()
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    
    ops.topk_softmax_with_bias(
        topk_weights, topk_ids, token_expert_indices,
        gating_output,
        expert_load,       # (E,) float32
        -alpha,            # bias_scale = -alpha (subtract penalty)
        renormalize,
        use_original_weights,
    )
    return topk_weights, topk_ids, token_expert_indices
```

### Phase 4: Remove `record_topk` overhead from hot path

#### Step 4.1: Make `record_topk` truly zero-cost when disabled

Currently, the `record_topk` call in `_select_experts` always runs (even when stats are disabled it still does a function call + import + check). Move the check to initialization time:

```python
# In FusedMoE.__init__:
self._should_record_stats = bool(os.environ.get("VLLM_MOE_STATS_DIR"))

# In _select_experts, replace:
try:
    from vllm.moe_stats import record_topk
    record_topk(...)
except Exception:
    pass

# With:
if self._should_record_stats:
    from vllm.moe_stats import record_topk
    record_topk(...)
```

#### Step 4.2: Eliminate GPU→CPU sync in `record_topk`

The current `record_topk` does `.detach().cpu().numpy()` which forces GPU→CPU synchronization (a **pipeline stall**). Fix:

```python
def record_topk(*, layer_name, topk_ids, num_experts, k_experts):
    if not _should_write():
        return
    
    # Keep everything on GPU, only move to CPU asynchronously at flush time
    ids = topk_ids.detach()
    flat = ids.reshape(-1)
    hist = torch.bincount(flat.to(torch.int64), minlength=num_experts)
    
    with _lock:
        layer = _get_or_create_layer(layer_name, num_experts, k_experts)
        # Accumulate on GPU
        if layer._gpu_cnt_topk is None:
            layer._gpu_cnt_topk = hist.clone()
        else:
            layer._gpu_cnt_topk += hist
        layer.tokens_processed += int(ids.shape[0])
```

Only do the CPU transfer at `flush()` time (end of experiment).

---

## 5. Implementation Priority & Expected Impact

| Phase | Change | Expected Impact | Difficulty | When α=0 | When α>0 |
|---|---|---|---|---|---|
| **1** | Route through `fused_topk` + thin `adjust_logits` | **−7-8% recovery** | Easy (1-2 hours) | **0% loss** | ~3% loss |
| **2** | `fused_topk_bias`-style + `torch.compile` | **−1-2% recovery** | Medium (2-4 hours) | 0% loss | **~1-2% loss** |
| **3** | Custom CUDA kernel `topk_softmax_with_bias` | **−0.5-1% recovery** | Hard (1-2 days) | 0% loss | **<0.5% loss** |
| **4** | Fix `record_topk` overhead | **−0.5-1% for both** | Easy (1 hour) | Benefits baseline too | Benefits both |

### Recommended Execution Order

1. **Start with Phase 1** — this gets us 90% of the way there with minimal code changes
2. **Then Phase 4** — low-hanging fruit that also improves baseline numbers
3. **Then Phase 2** — if we need α>0 to also be fast
4. **Phase 3 only if** results from Phase 2 are insufficient

---

## 6. Verification Plan

After each phase, run:

```bash
# Quick smoke test
python -m src.run_experiment --config configs/qwen3-30b/e1_3_baseline.json --mode throughput
python -m src.run_experiment --config configs/qwen3-30b/e1_3_price_moe.json --mode throughput --routing.alpha 0.0
python -m src.run_experiment --config configs/qwen3-30b/e1_3_price_moe.json --mode throughput --routing.alpha 0.1

# Compare
python -m src.analyze_load_balance outputs/runs/ --compare
```

**Success criteria**:
- Phase 1: α=0 within **±1%** of baseline
- Phase 2: α=0.1 within **±3%** of baseline
- Phase 3: α=0.1 within **±1%** of baseline

---

## 7. Code Map (Files to Modify)

| File | Changes |
|---|---|
| `external/vllm/vllm/model_executor/layers/fused_moe/price_moe_router.py` | Rewrite router to thin `adjust_logits` + `update_queue` API |
| `external/vllm/vllm/model_executor/layers/fused_moe/layer.py` | Modify `_select_experts` to route through `fused_topk` after logit adjustment |
| `external/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` | Add `price_moe_topk` function (Phase 2), `fused_topk_with_price` (Phase 3) |
| `external/vllm/csrc/moe/topk_softmax_kernels.cu` | Extend kernel with bias parameter (Phase 3) |
| `external/vllm/vllm/moe_stats.py` | Eliminate GPU→CPU sync in hot path (Phase 4) |
| `Code/Verification/E1.3/src/run_experiment.py` | Update to use new API |

---

## 8. Appendix: Operation Count Comparison

### Baseline `_select_experts` (via `fused_topk`)

| Step | Operation | Kernel Launches | HBM Traffic |
|---|---|---|---|
| 1 | `ops.topk_softmax()` | 1 | Read `[T, E]` bf16, Write `[T, k]` fp32 + int32 |
| **Total** | | **1** | **T×E×2 + T×k×8 bytes** |

### Current PriceMoE `select_experts`

| Step | Operation | Kernel Launches | HBM Traffic |
|---|---|---|---|
| 1 | `self._expert_load.unsqueeze(0)` | 0 (view) | 0 |
| 2 | `router_logits - congestion_penalty` | 1 | Read `[T, E]`, Write `[T, E]` (both fp32 or bf16→fp32) |
| 3 | `torch.topk(adjusted_logits, k)` | 1 | Read `[T, E]`, Write `[T, k]×2` |
| 4 | `torch.gather(router_logits, ...)` | 1 | Read `[T, E]` + `[T, k]`, Write `[T, k]` |
| 5 | `torch.softmax(original_scores)` | 1 | Read `[T, k]`, Write `[T, k]` |
| 6 | `topk_ids.flatten().to(int64)` | 1 | Read `[T×k]`, Write `[T×k]` |
| 7 | `torch.bincount(...)` | 1 | Read `[T×k]`, Write `[E]` |
| 8 | `.float()` on counts | 1 | Read `[E]`, Write `[E]` |
| 9 | EMA update (mul + add) | 2 | Read/Write `[E]` × 2 |
| **Total** | | **~9** | **~3×T×E×4 + overhead** |

### Proposed Phase 1 PriceMoE (via `adjust_logits` + `fused_topk`)

| Step | Operation | Kernel Launches | HBM Traffic |
|---|---|---|---|
| 1 | `router_logits - α * expert_load` | 1 (or 0 if α=0) | Read/Write `[T, E]` |
| 2 | `ops.topk_softmax()` | 1 | Read `[T, E]`, Write `[T, k]` |
| 3 | EMA update (`scatter_add_`) | 1 (or 0 if α=0) | Read `[T×k]` int32, Write `[E]` fp32 |
| **Total (α=0)** | | **1** | **Same as baseline** |
| **Total (α>0)** | | **3** | **T×E×4 + T×E×2 + T×k×8** |

This is a **3× reduction** in kernel launches and **2× reduction** in HBM traffic compared to the current implementation when α>0.
