# Constraint Survey：MoE 推理系统的约束建模（学界与业界共识）

> 目的：用“老师可快速审阅”的方式，归纳大规模 MoE 推理系统在 **Prefill / Decode** 两阶段下的关键物理约束、典型数学形式、以及主流系统（vLLM / DeepSpeed-MoE / Metro 等）的建模侧重点。

---

## 1. 结论速览（给评审的 30 秒版）

- MoE 推理的主要约束不再是单纯的“算力 TFLOPS”，而是 **计算吞吐（Token/s）**、**显存容量（KV Cache）**、**显存带宽（Expert 权重加载）**、**通信带宽（All-to-All）**、以及 **质量约束（Top-k/可容忍降级）** 的组合。
- **Prefill 阶段**更偏 compute-bound：调度核心是把 GPU 计算管道填满（continuous batching）。
- **Decode 阶段**常见转为 memory-bound：系统瓶颈从“算多少”转为“搬多少”，尤其是 **每步激活的专家集合大小（Activated Expert Set）** 带来的 HBM 读放大。
- 约束建模正在从“线性 token 均衡”转向“组合型的激活集合控制 + 带宽预算”，对应调度目标也从“balance tokens”逐渐转为“balance activated experts”。

---

## 2. 约束维度一：计算约束（Compute Constraints）

### 2.1 Prefill：Token 预算/吞吐约束（vLLM：Continuous Batching 视角）

Prefill 主要是大 GEMM，常见策略把计算抽象成一个在调度周期 $\tau$ 内可处理的 token 预算：

$$
\sum_{r \in \mathcal{R}} \mathrm{len}(r_{\tau}) \le C_{\mathrm{GPU}} \cdot \tau
$$

- $C_{\mathrm{GPU}}$：GPU 在该模型/精度/并行策略下的 token/s 吞吐。
- 该约束是 **系统调度**（而不只是单请求）层面的可行域：避免 iteration latency 激增与 TTFT 违约。

### 2.2 MoE 特有：专家负载不均衡与“短板专家”

即使总 token 数满足预算，若大量 token 路由到同一专家，会出现 straggler（专家队列排长）。常见抽象为对每个专家的容量上界：

$$
 x_e \le C_{\mathrm{expert}}^{\max},\quad \forall e \in \mathcal{E}
$$

- 体现 MoE 的核心困难：**gating 导致的输入分布偏斜**。
- 工程上常通过容量因子（capacity factor）、EPLB、重排/重分配等缓解。

---

## 3. 约束维度二：显存与激活（Memory / Activation Constraints）

### 3.1 显存容量：KV Cache 与分页分配（vLLM：PagedAttention）

在大模型服务中，显存往往由“权重 + KV Cache + 系统开销”三部分组成。将 KV Cache 以 page/block 方式管理后，容量约束可抽象为：

$$
N_{\mathrm{used\_blocks}} \cdot S_{\mathrm{block}} \le M_{\mathrm{total}} - M_{\mathrm{model}} - M_{\mathrm{overhead}}
$$

- 违反时需要触发抢占、换出（swap）、或限制并发。
- 这类约束决定了可并发的请求数与上下文长度上限。

### 3.2 Decode 的关键：显存带宽与“激活专家集合”约束（Metro：memory-bound 视角）

在 decode 的每一步，激活更多不同专家意味着需要从 HBM 读更多专家权重，导致带宽压力上升。Metro 的核心洞察是：

> 在 decode 阶段，**控制每 GPU 的 Activated Expert Set** 往往比均衡 token 更有效。

可用二元变量刻画专家是否被激活：$y_{e,g}\in\{0,1\}$，并施加带宽预算：

$$
\sum_{e\in\mathcal{E}} y_{e,g}\cdot \mathrm{Size}(e) \le \mathrm{Bandwidth\_Budget}_g
$$

- 这类建模本质是 **组合优化**（激活集合大小/权重体积/共享复用），比线性 token 计数更贴近 decode 的物理瓶颈。
- 直接解释了 token 均衡策略的潜在副作用：token 被均匀打散到所有 GPU → 每个 GPU 都要加载很多专家 → 触发带宽墙。

---

## 4. 约束维度三：通信（Communication Constraints）

在 Expert Parallel（EP）中，token/激活在设备间交换，常建模为启动延迟 + 传输量线性项：

$$
T_{\mathrm{comm}} = \alpha + \beta \cdot \sum_{g\ne g'} \mathrm{Data}(g\to g')
$$

- $\alpha$：启动延迟；$\beta$：单位数据传输耗时（带宽倒数）。
- All-to-All 在规模增大时通信代价可能呈 $O(n)$ 或 $O(n^2)$（视拓扑/实现）。

工程共识：需要 **分层通信建模**（如节点内 NVLink 与节点间 IB 分开）：

$$
\sum \mathrm{Data}_{\mathrm{intra}} \le BW_{\mathrm{NVLink}},\qquad
\sum \mathrm{Data}_{\mathrm{inter}} \le BW_{\mathrm{IB}}
$$

路由/调度应倾向于把 token 尽量留在高带宽域内，减少跨域 All-to-All。

---

## 5. 约束维度四：质量（Quality Constraints）

系统优化不能突破模型质量红线。常见形式包括：

### 5.1 Token Choice：Top-k 硬约束

$$
x_{i,e}=1 \iff e\in\mathrm{Top}_k(\mathrm{Gating}(i))
$$

- 优点：质量稳定、实现简单。
- 代价：降低负载均衡自由度，热门专家更易过载。

### 5.2 Expert Choice：Top-C（反向约束）

$$
\sum_i x_{i,e}=C,\quad x_{i,e}=1 \iff i\in\mathrm{Top}_C(\mathrm{Gating}(\cdot,e))
$$

- 优点：专家侧负载更可控。
- 风险：可能出现 token 被“服务不足/过度服务”，需结合质量评估。

### 5.3 有界降级（Bounded Degradation）：软约束化

为缓解热点可允许次优专家，但限制亲和度损失：

$$
S_{i,\mathrm{routed}} \ge (1-\epsilon)\cdot S_{i,\mathrm{best}}
$$

这相当于把“硬 Top-k”改写成“可控质量损失”的可行域扩张，是系统侧优化空间的重要来源。

---

## 6. 主流系统：约束视角对比（用于课堂展示）

| 维度 | vLLM / SGLang | DeepSpeed-MoE | Metro（EuroSys’25） | Expert Choice（Google） |
| --- | --- | --- | --- | --- |
| 核心瓶颈视角 | 显存容量（KV Cache）与调度吞吐 | 通信（All-to-All）+ 专家负载偏斜 | 显存带宽（HBM IO） | 计算均衡/专家侧容量 |
| 路由决策主体 | Token（Top-k） | Token（Top-k + capacity） | 全局调度（控制激活集合） | Expert（Top-C） |
| 关键建模粒度 | token budget + page/block | 分层 A2A + capacity factor | Activated Expert Set 计数/体积 | 专家选择 token 的“配额” |
| 负载目标 | 填充计算管线/降低 prefill 排队 | 抑制 straggler + 通信优化 | **最小化激活专家数**（decode） | 严格负载均衡 |
| 超载策略 | 不丢包（排队/抢占） | 超过 capacity 可能丢弃/溢出 | 不丢包（用调度解决） | 超过 buffer 丢弃 |

---

## 7. 对本项目的启示（可作为后续章节引言）

1. **分阶段建模**：Prefill 用 token budget（compute-centric）更合适；Decode 必须显式引入带宽/激活集合约束（memory-centric）。
2. **不要只 balance tokens**：在 decode，应该把“每 GPU 激活的专家集合大小/权重体积”作为一等公民约束或目标。
3. **通信与拓扑要进入模型**：EP 的 All-to-All 成本不是常数；需要分层约束或把跨域流量作为惩罚项。
4. **质量约束要软硬结合**：Top-k 是底线；在热点与稳定性之间，可用“有界降级”获得工程可行解空间。

---

## 参考线索（来自原报告的材料来源）

- vLLM：Continuous Batching、PagedAttention（KV Cache/page 管理）
- DeepSpeed-MoE：专家负载偏斜、EPLB、分层 All-to-All 通信
- Metro：Decode 阶段 memory-bound，主张 balance activated experts not tokens
- Expert Choice：专家侧 Top-C 选择与严格容量控制
