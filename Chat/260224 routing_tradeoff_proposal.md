# Research Proposal: Bias-Augmented MoE Top-K Routing for Explicit Throughput–Quality Tradeoff

> **Status**: Active — bias designs proposed, awaiting selection  
> **Created**: 2026-02-24  
> **Location**: `llm_moe/docs/routing_tradeoff_proposal.md`

---

## 1. Background & Motivation

Mixture-of-Experts (MoE) models scale parameter capacity by routing each token to a learned subset of *k* experts via a gate function. While this sparse activation reduces per-token compute, it introduces a persistent challenge during inference: **load imbalance** — some experts receive many tokens while others sit idle.

Two recent works characterize the state of the art and its limitations:

- **LASER** — *From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing*  
  Shahout et al., arXiv 2025. [[arXiv:2510.03293]](https://arxiv.org/abs/2510.03293)  
  Proposes an inference-time routing algorithm that adapts to the shape of the gate score distribution: when scores are near-uniform, it expands the candidate pool and routes tokens to the **least-loaded** experts to balance token counts per expert.

- **METRO** — *Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens*  
  Yu et al., arXiv 2025. [[arXiv:2512.09277]](https://arxiv.org/abs/2512.09277)  
  Reveals a fundamental problem with token-balancing approaches: in the **memory-bound** regime (typical during the decode phase), GPU runtime is dominated by *loading expert weights from memory*, not by the number of tokens processed. Consequently, spreading tokens across more experts — as LASER and EPLB do — **inflates the number of activated experts per GPU and degrades throughput**. METRO instead minimizes the number of activated expert replicas per GPU, achieving 11–22% lower decode latency.

### The Gap

LASER's key insight — *the gate score distribution shape tells us when we have routing flexibility* — is genuinely useful. However, LASER exploits that flexibility in the **wrong direction** for memory-bound serving: selecting least-loaded experts increases activated expert count, worsening the very bottleneck METRO identifies.

Meanwhile, METRO operates strictly **after** top-K selection (at the replica-dispatch level) and does not change *which* experts are selected. Neither work provides an **explicit, continuously adjustable tradeoff** between output quality and system throughput at the expert-selection level.

---

## 2. Problem Formulation

### 2.1 Standard MoE Routing (Baseline)

Let $E = \{1, \dots, N\}$ be the set of experts in a layer. For an input token with hidden state $x$, the gate computes a logit vector $z \in \mathbb{R}^N$, from which softmax scores are derived:

$$
s_e = \frac{\exp(z_e)}{\sum_{j=1}^N \exp(z_j)}, \quad e \in E
$$

Standard top-K routing selects the index set $\mathcal{A} = \text{top-K}(\{s_e\})$ and computes the output as:

$$
y = \sum_{e \in \mathcal{A}} \tilde{s}_e \cdot f_e(x), \quad \tilde{s}_e = \frac{s_e}{\sum_{j \in \mathcal{A}} s_j}
$$

where $f_e(\cdot)$ is the $e$-th expert's feedforward computation and $\tilde{s}_e$ is the renormalized weight.

### 2.2 Bias-Augmented Top-K Routing

We propose modifying the **selection step only** by adding a throughput-aware bias to the pre-softmax **logits** before computing adjusted scores for the top-K decision:

$$
z'_e = z_e + \gamma \cdot b_e
$$

$$
s'_e = \frac{\exp(z'_e)}{\sum_{j=1}^N \exp(z'_j)}, \quad \mathcal{A} = \text{top-K}(\{s'_e\})
$$

Critically, the **output aggregation still uses the original scores** $s_e$ (renormalized over $\mathcal{A}$), not $s'_e$:

$$
y = \sum_{e \in \mathcal{A}} \tilde{s}_e \cdot f_e(x), \quad \tilde{s}_e = \frac{s_e}{\sum_{j \in \mathcal{A}} s_j}
$$

This design choice ensures that once the set of selected experts $\mathcal{A}$ is determined, the model output is computed **exactly as the original model would** for those experts — the bias only influences *which* experts are chosen, not *how* their outputs are weighted.

| Symbol | Definition |
|--------|-----------|
| $z_e \in \mathbb{R}$ | Pre-softmax gate logit for expert $e$ |
| $b_e \in \mathbb{R}$ | Throughput-aware bias (TBD; computed from batch state) |
| $\gamma \geq 0$ | Tradeoff coefficient — the primary experimental knob |
| $z'_e$ | Bias-adjusted logit used for top-K selection |
| $s'_e$ | Softmax of adjusted logits; used only for selecting $\mathcal{A}$ |
| $\tilde{s}_e$ | Renormalized original score; used for output aggregation |

### 2.3 Scope and Boundary

This proposal targets the **expert-selection step** (which $k$ experts are activated), which is upstream of and orthogonal to replica-level dispatch (which physical GPU/replica handles a token). Therefore:

- **Relation to METRO**: This method and METRO operate at different levels and are **composable**. After our bias-augmented top-K selects $\mathcal{A}$, METRO's replica-dispatch algorithm can subsequently decide how to route each token to replicas of the chosen experts. Evaluating whether combining both yields additive gains is a secondary research question.
- **Scope exclusion**: We do not modify expert placement, replication strategy, or the all-to-all communication protocol.

### 2.4 Constraints on $b_e$ (Design Principles, TBD)

The specific form of $b_e$ is not fixed in this proposal. However, any valid design must satisfy the following constraints:

1. **Direction**: $b_e$ should favor experts that are already active / receiving tokens in the current batch, so that token load **concentrates** rather than spreads — reducing the total number of activated experts per step (aligned with METRO's insight).
2. **Scale**: $b_e$ must be on a comparable scale to gate logits $z_e$, so that γ has a consistent, interpretable magnitude across layers and models. Normalization (e.g., dividing by the standard deviation of $z$) is recommended.
3. **Overhead**: $b_e$ must be computable with negligible latency overhead relative to a single MoE layer's forward pass. It should rely only on information already available at routing time (e.g., running token counts per expert in the current batch).
4. **Non-interference at γ = 0**: When $\gamma = 0$, the method must reduce exactly to standard top-K routing.

---

## 3. Research Questions

> **RQ1 (Primary)**: What is the achievable throughput–quality Pareto frontier as a function of γ, and where is its "knee" — i.e., the region where throughput gains are largest relative to quality loss?

> **RQ2 (Bias Sensitivity)**: How sensitive are the throughput and quality outcomes to the specific functional form of $b_e$? Is there a robust class of bias designs that consistently yields favorable tradeoffs across models and workloads?

> **RQ3 (Regime Dependence)**: Does the method behave differently in the compute-bound (prefill) vs. memory-bound (decode) regimes? Specifically, does the bias-induced concentration of expert activations provide throughput gains primarily during decode, as METRO's analysis would predict?

> **RQ4 (Composability)**: Can bias-augmented top-K selection be combined with METRO's replica-level routing to yield additive throughput improvements, or do the two methods interact in unexpected ways?

---

## 4. Hypotheses

- **H1**: In the decode phase, a moderate γ (e.g., 0.1–0.5 in normalized logit scale) will reduce the number of activated experts per step by 10–25% with a quality degradation of less than 2% on standard benchmarks (MMLU, GSM8K).
- **H2**: In the prefill phase, the same γ range will yield little or no throughput improvement (since prefill is compute-bound and runtime does not depend on activated expert count), but will also cause minimal quality loss — making the method safe to apply uniformly across both phases.
- **H3**: The tradeoff curve will exhibit a clear "knee": a relatively small γ captures most of the throughput gain, while further increases yield diminishing throughput returns at the cost of accelerating quality degradation.
- **H4**: Combining bias-augmented routing with METRO's replica dispatch provides additive decode latency reduction, as the two mechanisms operate at different levels of the routing hierarchy.

---

## 5. Experiment Design

### 5.1 Models

| Model | Architecture | Experts (total / active) |
|-------|-------------|--------------------------|
| Mixtral-8×7B | Decoder-only MoE | 8 / 2 per layer |
| DeepSeek-MoE-16B | Fine-grained MoE | 64 / 6 per layer |
| Qwen3-30B-A3B | Large-scale MoE | ~128 / varies |

Mixtral and DeepSeek are chosen to match prior baselines (LASER, METRO); Qwen3 is included for a more recent large-scale setting.

### 5.2 Baselines

| Baseline | Description |
|----------|-------------|
| **Top-K** | Standard top-K routing, $\gamma = 0$ |
| **LASER** | Score-distribution-aware routing with least-loaded expert selection [[arXiv:2510.03293]](https://arxiv.org/abs/2510.03293) |
| **METRO** | Replica-level min-activated-expert routing [[arXiv:2512.09277]](https://arxiv.org/abs/2512.09277) |
| **Ours (γ sweep)** | Bias-augmented top-K at γ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} |
| **Ours + METRO** | Bias-augmented top-K (best γ) combined with METRO replica dispatch |

### 5.3 Datasets

| Dataset | Purpose |
|---------|---------|
| GSM8K | Reasoning quality (accuracy) |
| MMLU | Factual knowledge quality (accuracy) |
| ARC-Easy / ARC-Challenge | Science QA quality (accuracy) |
| WikiText-2 / -103 | Perplexity measurement |
| InstructCoder | Throughput benchmark (coding workload, following METRO) |

### 5.4 Metrics

**Quality**
- Accuracy on GSM8K, MMLU, ARC-Easy, ARC-Challenge (following LASER's eval protocol)
- Perplexity on WikiText-2 / -103

**Throughput / System**
- Number of activated experts per layer per step (primary efficiency proxy, hardware-independent)
- Time-Per-Output-Token (TPOT) — decode latency
- Time-To-First-Token (TTFT) — prefill latency
- Total token throughput (tokens/s)

**Tradeoff Summary**
- Pareto curve: throughput gain (%) vs. quality loss (%) relative to standard top-K, parametrized by γ
- "Knee" identification: the γ* that maximizes $\Delta \text{throughput} / \Delta \text{quality}$

### 5.5 Hardware & Software

- **Hardware**: 8× NVIDIA A100 80GB GPUs (to match METRO's real-system evaluation setup)
- **Software**: vLLM (to match METRO), PyTorch + HuggingFace Transformers (to match LASER's eval protocol)
- **EP configuration**: Expert Parallelism across 8 GPUs; replication ratios {1.0×, 1.5×} following METRO

---

## 6. Distinction from Prior Work

| | LASER | METRO | **This Work** |
|---|---|---|---|
| Routing stage | Top-K expert selection | Token-to-replica dispatch (post top-K) | Top-K expert selection |
| Objective | Balance token count per expert | Minimize activated replicas per GPU | Minimize activated experts via bias; explicit γ tradeoff |
| Mechanism | Expand candidate pool; pick least-loaded | Greedy min-activated-expert assignment | Bias-adjusted logits; top-K on $z + \gamma b$ |
| Aggregation weights | Modified (uses re-routed expert scores) | Unchanged | Unchanged (always uses original $s_e$) |
| Memory-bound correctness | ✗ (inflates activated experts) | ✓ | ✓ |
| Requires retraining | No | No | No |
| Explicit tradeoff control | No | No | **Yes (via γ)** |
| Composable with the other | — | — | **Yes (can layer on top of METRO)** |

---

## 7. Expected Contributions

1. **A new routing formulation** with a single, interpretable tradeoff coefficient γ that continuously interpolates between quality-optimal (γ = 0) and throughput-optimal routing.
2. **Empirical tradeoff curves** across models, phases (prefill/decode), and workloads — enabling practitioners to choose γ based on their SLO constraints.
3. **Identification of the "safe operating region"** for γ: where throughput gains are substantial and quality degradation is negligible.
4. **Composability analysis** with METRO, clarifying whether expert-selection-level and replica-dispatch-level optimizations interact additively.
5. **A plug-and-play implementation** applicable to any pre-trained MoE model without retraining.

---

## 8. Limitations & Risks

| Risk | Mitigation |
|------|-----------|
| $b_e$ computation overhead may be non-negligible in very low-latency settings | Benchmark overhead separately; constrain bias design to O(N) per token |
| γ may need per-model or per-layer tuning, reducing plug-and-play convenience | Evaluate global vs. layer-wise γ; report sensitivity |
| Concentration effect may be weak for small batch sizes (few tokens → little information to concentrate) | Report results across batch sizes; identify minimum effective batch size |
| Bias-augmented routing may interfere with load balancing at the replica level when combined with METRO | Evaluate METRO + ours jointly; verify no regression on prefill |
| Quality degradation may exceed acceptable bounds for certain domains (e.g., code generation) | Report per-dataset results; identify domain-specific γ thresholds |

---

## 9. Bias Design

### 9.1 Overview

The bias $b_e$ must encode "how much does routing token $t$ to expert $e$ help reduce the total number of activated experts." This section proposes four concrete designs of increasing sophistication. They share a common input: a **count vector** $n \in \mathbb{Z}_{\geq 0}^N$, where $n_e$ is the number of tokens already routed to expert $e$ in the current context.

**Source of $n_e$** is a separate design axis:

| Source | Description | Overhead |
|--------|-------------|----------|
| **Cross-step** | $n_e$ from the previous decode step | None; maintain a running counter |
| **Cross-layer** | $n_e$ from the previous MoE layer in the current step | None; routing info already in memory |
| **Intra-batch** | $n_e$ from tokens already routed in the current batch/layer (sequential) | Breaks batch-level parallelism; small but nonzero |

All four bias designs below treat $n_e$ as a given; the source is a separate hyperparameter to ablate.

---

### 9.2 Design 1 — Binary Activation Bias (B-bin)

$$
b_e = \mathbb{1}[n_e > 0]
$$

**Intuition**: The activated-expert count increases by 1 whenever a *new* expert is activated. This bias gives every already-activated expert a fixed positive bonus, directly discouraging the activation of new experts.

**Properties**:
- Values: $b_e \in \{0, 1\}$. With logit scale ~1–5, $\gamma \in [0, 2]$ is a natural sweep range.
- At $\gamma = 0$: standard top-K.
- Threshold effect: once γ exceeds the logit gap between the top and the $k$-th expert, routing will almost always stay within already-activated experts (for tokens with near-uniform gate scores).

**Pros**: Simplest; most directly encodes the "minimize activated expert count" objective; highly interpretable.

**Cons**: Binary — makes no distinction between an expert with 1 token and one with 100 tokens. Can create a sharp phase transition as γ increases.

**Theoretical connection**: Directly penalizes the Hamming distance from the current activation set. If all tokens in a batch adopted this bias, the expected number of newly activated experts decreases monotonically with γ.

---

### 9.3 Design 2 — Normalized Count Bias (B-cnt)

$$
b_e = \frac{n_e}{\sum_{j=1}^{N} n_j + \epsilon}
$$

where $\epsilon > 0$ is a small constant to avoid division by zero at the start of a batch.

**Intuition**: Proportional reward — experts that have already absorbed more tokens receive a stronger nudge. This is the "rich get richer" mechanism applied at the routing level.

**Properties**:
- Values: $b_e \in [0, 1]$, with $\sum_e b_e = 1$ (it is a proper distribution over experts). This makes the scale of $b_e$ comparable to softmax gate scores $s_e$, so γ is directly interpretable as "how much weight to put on throughput vs. gate quality."
- At $n = 0$ (start of batch or first step): $b_e = 1/N$ for all $e$ → uniform, no bias → reduces to top-K.
- As the batch progresses, $b_e$ becomes increasingly concentrated on high-load experts.

**Pros**: Smooth and differentiable; automatically normalized; non-zero only for activated experts (like B-bin) but with graded strength.

**Cons**: The sum $\sum_j n_j$ grows as more tokens are routed, so the absolute magnitude of $b_e$ for a specific expert can decrease even as $n_e$ increases (if other experts also accumulate tokens). This can reduce the effective bias over time within a long batch.

---

### 9.4 Design 3 — Log Count Bias (B-log)

$$
b_e = \frac{\log(1 + n_e)}{\log\!\left(1 + \max_{j} n_j\right) + \epsilon}
$$

**Intuition**: Sublinear (diminishing returns) reward for already-loaded experts. Concentrating 50 tokens on one expert is encouraged relative to spreading across two experts, but the 51st token does not get a much stronger pull than the 50th.

**Properties**:
- Values: $b_e \in [0, 1]$ after normalization by the maximum log-count.
- Shape: the bias curve is concave in $n_e$, preventing any single expert from accumulating a disproportionately large bias and dominating routing entirely.
- At $n = 0$: $b_e = 0$ for all $e$ → no bias → reduces to top-K.

**Pros**: More robust than B-cnt when $n_e$ values are highly skewed; avoids extreme concentration on a single "mega-expert"; numerically stable.

**Cons**: Slightly more complex to compute; the normalization by $\max_j n_j$ requires an additional pass over the count vector.

**Recommended starting point** for experiments due to its robustness and smooth tradeoff behavior.

---

### 9.5 Design 4 — Standardized Count Bias (B-std)

$$
b_e = \frac{n_e - \bar{n}}{\mathrm{std}(n) + \epsilon}, \quad \bar{n} = \frac{1}{N}\sum_{j} n_j, \quad \mathrm{std}(n) = \sqrt{\frac{1}{N}\sum_j (n_j - \bar{n})^2}
$$

**Intuition**: Zero-mean, unit-variance normalization makes $b_e$ automatically scaled to the same order of magnitude as gate logits $z_e$ (which also typically have std ~1–3). This gives γ the most consistent, model-agnostic meaning: $\gamma = 1$ means the bias has the same effective magnitude as the gate's own logit variability.

**Properties**:
- Values: $b_e \in \mathbb{R}$, zero mean by construction. Experts with above-average load get positive bias; below-average get negative bias (i.e., a penalty for lightly-used experts).
- The negative values for $n_e < \bar{n}$ are a feature: they actively discourage routing to cold experts, not just passively reward hot ones.
- At $n = 0$ or uniform $n$: $\mathrm{std}(n) = 0$, so $b_e = 0$ for all $e$ → reduces to top-K.

**Pros**: Best scale calibration; γ is interpretable and portable across models/layers; two-sided (reward hot + penalize cold).

**Cons**: Requires computing mean and std over all $N$ experts per routing step (O(N) compute); negative values can cause cold experts to be systematically underused even when they are the model-preferred experts, which may hurt quality more aggressively at large γ.

---

### 9.6 Summary Comparison

| | B-bin | B-cnt | B-log | B-std |
|---|---|---|---|---|
| Formula | $\mathbb{1}[n_e>0]$ | $n_e / \sum n_j$ | $\log(1+n_e)/\log(1+\max n_j)$ | $(n_e - \bar{n})/\mathrm{std}(n)$ |
| Range | $\{0,1\}$ | $[0,1]$ | $[0,1]$ | $\mathbb{R}$ (zero-mean) |
| Cold expert penalty | No (0, neutral) | No (0, neutral) | No (0, neutral) | **Yes** (negative) |
| Scale calibration | Manual | ~Gate softmax | ~Gate softmax | **Auto (logit scale)** |
| Smoothness | Step function | Linear | Sublinear (concave) | Linear (standardized) |
| Robustness to skew | Low | Medium | **High** | Medium |
| Complexity | O(1) | O(N) | O(N) | O(N) |
| Recommended γ range | [0, 3] | [0, 5] | [0, 5] | [0, 2] |
| **Recommended for** | Ablation baseline | Initial experiments | **Primary design** | Scale-sensitive settings |

### 9.7 Proposed Experimental Order

1. **Establish baseline**: B-bin across γ ∈ {0, 0.5, 1.0, 2.0} — confirms the binary activation signal works.
2. **Primary sweep**: B-log across γ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} — generates the main tradeoff curves.
3. **Scale ablation**: B-std at the same γ values — tests whether auto-scaling improves portability across models.
4. **Source ablation**: repeat B-log with cross-step / cross-layer / intra-batch $n_e$ — isolates the contribution of information freshness.
5. **Composition**: B-log (best γ) + METRO replica dispatch — tests RQ4.

---

## 10. References

1. Rana Shahout, Colin Cai, Yilun Du, Minlan Yu, Michael Mitzenmacher.  
   **"From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing."**  
   arXiv:2510.03293, 2025.  
   <https://arxiv.org/abs/2510.03293>

2. Yanpeng Yu, Haiyue Ma, Krish Agarwal, et al.  
   **"Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens."**  
   arXiv:2512.09277, 2025.  
   <https://arxiv.org/abs/2512.09277>
