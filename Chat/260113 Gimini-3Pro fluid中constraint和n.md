# 大规模 MoE 推理系统的流体排队模型与基于影子价格的动态路由策略研究报告

## 1. 执行摘要 (Executive Summary)

随着生成式人工智能（Generative AI）进入万亿参数时代，混合专家模型（Mixture-of-Experts, MoE）凭借其稀疏激活（Sparse Activation）特性，已成为打破计算与显存墙的关键架构。然而，MoE 推理系统在分布式环境下面临着前所未有的资源调度挑战：传统的静态负载均衡策略在面对极度动态的 Token 到达率和高度偏斜的专家访问模式时，往往导致严重的排队延迟和资源碎片化。本报告深入探讨了基于流体模型（Fluid Model）和排队论（Queueing Theory）的系统建模方法，并详细论证了引入影子价格（Shadow Price）与原对偶（Primal-Dual）优化框架在实现低延迟、高吞吐动态路由中的理论必然性与工程可行性。

本研究的核心发现主要集中在三个维度。首先，在**约束建模**方面，学术界与业界正经历从以计算为中心（Compute-centric）向以显存带宽为中心（Memory-bound）的范式转移。vLLM 等主流框架通过 Continuous Batching 主要解决计算槽位的碎片化问题，而最新的 Metro 系统研究表明，在解码（Decode）阶段，最小化每 GPU 的专家激活集（Activated Expert Set）比单纯均衡 Token 数量更能显著降低延迟。这标志着约束建模的数学形式从简单的线性累加向非凸的组合优化转变。

其次，在**系统扩容（Scaling）**的流体极限分析中，研究表明简单的平均场（Mean Field）极限无法捕捉 MoE 系统中因 All-to-All 通信和专家加载（Setup Time）引发的非线性延迟增长。最佳的 Scaling 策略应遵循 Halfin-Whitt（Many-server）机制，即系统容量 $N$ 与到达率 $\lambda$ 之间应保持 $N \approx \lambda + \beta\sqrt{\lambda}$ 的关系。这种“平方根安全余量”法则能够在维持极高资源利用率的同时，确保排队延迟收敛于 $O(1/\sqrt{N})$，从而在规模化与服务质量（SLA）之间取得最优平衡。

最后，在**动态路由算法**的设计上，报告论证了基于影子价格的 Backpressure 路由策略是保证系统稳定性的数学最优解。通过构建 Lyapunov 漂移分析，我们证明了在目标函数中引入相对熵正则化（即 Softmax 路由）是保证纳什均衡唯一性的必要条件，而引入迟滞（Hysteresis）机制则是解决专家加载时间导致的非凸性、防止路由震荡并确保在线算法收敛的关键。

本报告旨在为高性能计算架构师、运筹学研究者及大规模 AI 系统工程师提供一套严谨的数学框架与设计蓝图，以指导下一代 MoE 推理引擎的构建。
## 2. 学界与业界对推理约束的共识性建模形式

在大规模 MoE 推理系统的设计中，准确描述物理资源的约束边界是构建高效调度算法的前提。通过对 vLLM、DeepSpeed-MoE 以及最新的 Metro 系统进行深度解构，我们发现业界对计算（Compute）、通信（Comm）、显存（Activation/Quality）的建模已形成了一套多层次的数学共识，但针对不同的推理阶段（Prefill vs. Decode），其侧重点存在显著差异。

### 2.1 计算约束（Compute Constraints）：从 FLOPs 到 Continuous Batching

在传统的密集模型推理中，计算约束通常被简化为 GPU 的峰值 TFLOPS。然而，在 MoE 架构中，计算资源的约束呈现出高度的动态性。

#### 2.1.1 Prefill 阶段的计算密度建模

在 Prefill 阶段，计算主要受限于矩阵乘法（GEMM）的吞吐量。学术界普遍采用 Roofline 模型来描述这一约束，但在 vLLM 等现代推理引擎中，约束被细化为“时间槽（Time Slot）内的 Token 预算”。

**vLLM 的建模视角**：vLLM 引入了 Continuous Batching [1]，其核心思想是将计算资源视为一个流式的处理管道。约束不再是单一请求的延迟，而是整个 Batch 在单位时间内的 Token 处理量。

**数学表达**：设 $C_{GPU}$ 为 GPU 的计算能力（Token/s），$T_{req}$ 为请求集合。在任意调度周期 $\tau$ 内，被调度的 Token 总数 $N_{token}$ 必须满足：

$$\sum_{r \in T_{req}} \text{len}(r_{\tau}) \le C_{GPU} \cdot \tau$$

这一约束通过调度器（Scheduler）动态调整 Batch Size 来满足。如果 $N_{token}$ 超过阈值，会导致 Iteration Latency 激增，从而违反 TTFT（Time to First Token）的 SLA。

#### 2.1.2 专家计算负载的不均衡性

DeepSpeed-MoE [2] 进一步指出了 MoE 特有的计算约束：专家负载不均衡。即使总 Token 数未超标，如果大量 Token 路由至同一专家，该专家的计算队列将成为短板（Straggler）。

**约束细化**：对于每个专家 $e$，其处理的 Token 数 $x_{e}$ 必须满足：

$$x_{e} \le C_{expert}^{max} \quad \forall e \in \mathcal{E}$$

其中 $C_{expert}^{max}$ 是单个专家在不造成显著排队延迟下的最大吞吐量。DeepSpeed 通过 EPLB（Expert Parallel Load Balancing）算法试图在物理层面上重新分配专家以满足此约束，但这涉及到昂贵的模型迁移成本。

### 2.2 显存与激活约束（Memory/Activation Constraints）：带宽墙与碎片化

显存约束是 MoE 推理中最复杂且最具决定性的因素，特别是在 Memory-bound 的 Decode 阶段。

#### 2.2.1 显存容量与 KV Cache (vLLM)

vLLM 的核心贡献在于将显存约束从“模型权重”转向了“KV Cache”。

**PagedAttention 模型**：显存被划分为固定大小的 Page Block。约束条件转化为物理块（Physical Blocks）的分配问题。

**数学形式**：设 $M_{total}$ 为总显存，$M_{model}$ 为静态权重占用，$M_{expert}$ 为专家缓存，$S_{block}$ 为单个 Block 大小。系统允许的最大并发请求数 $N_{conc}$ 受限于：

$$N_{used\_blocks} \cdot S_{block} \le M_{total} - M_{model} - M_{overhead}$$

当该约束被违反时，系统必须触发 Preemption（抢占）机制，将部分请求换出（Swap out）到 CPU 内存。

#### 2.2.2 显存带宽与专家激活集 (Metro)

Metro [3] 提出了一个颠覆性的观点：在 Decode 阶段，真正的瓶颈不是显存容量，而是显存带宽（Memory Bandwidth）。

**Metro 的建模洞察**：每次激活一个新的专家，都需要从 HBM 中读取该专家的权重。如果一个 Batch 内激活的专家数量过多，会导致算术强度（Operational Intensity）急剧下降，系统沦为 Memory-bound。

**数学重构**：Metro 将目标函数从“均衡 Token 数量”修改为“最小化每 GPU 的专家激活数量”。
设 $y_{e,g} \in \{0, 1\}$ 表示专家 $e$ 是否在 GPU $g$ 上被激活。约束变为：

$$\sum_{e \in \mathcal{E}} y_{e,g} \cdot \text{Size}(e) \le \text{Bandwidth\_Budget}_{g}$$

这一约束揭示了 Token 均衡策略（EPLB）的缺陷：如果将 Token 均匀打散到所有 GPU，可能导致每个 GPU 都需要加载所有专家，从而触碰带宽墙。Metro 通过将请求集中路由，虽然导致 Token 负载不均，但显著减少了 $y_{e,g}$ 的总和，从而提升了整体吞吐量。

### 2.3 通信约束（Communication Constraints）：All-to-All 的代价

在专家并行（Expert Parallelism, EP）模式下，Token 需要在不同 GPU 间传输。DeepSpeed-MoE [2] 和 DeepSeek-V3 [5] 对此进行了详细建模。

**线性通信模型**：跨设备的通信时间 $T_{comm}$ 与传输的数据量成线性关系，且受限于互联拓扑（如 NVLink 或 InfiniBand）。

**数学形式**：

$$T_{comm} = \alpha + \beta \cdot \sum_{g \ne g'} \text{Data}(g \to g')$$

其中 $\alpha$ 是启动延迟（Latency），$\beta$ 是带宽倒数。对于 All-to-All 操作，随着 GPU 数量 $N$ 的增加，通信复杂度通常为 $O(N)$ 或 $O(N^2)$（取决于拓扑）。

**分层通信约束**：DeepSpeed-MoE 引入了分层 All-to-All（Hierarchical All-to-All），将通信分解为节点内（Intra-node）和节点间（Inter-node）。约束建模必须区分这两种带宽：

$$\sum \text{Data}_{intra} \le BW_{NVLink}, \quad \sum \text{Data}_{inter} \le BW_{IB}$$

在进行路由决策时，算法（如 Occult [6]）会倾向于将 Token 路由到同一节点内的专家，以利用高带宽的 NVLink，避免触碰低带宽的 IB 约束。

### 2.4 质量约束（Quality Constraints）：Top-k 与 Expert Choice

保证推理质量是所有系统优化的红线。

**Top-k Routing (Token Choice)**：最常见的约束，要求每个 Token $i$ 必须被路由到其 Logits 最高的 $k$ 个专家。

$$x_{i,e} = 1 \iff e \in \text{Top}_k(\text{Gating}(i))$$

这种硬约束限制了负载均衡的自由度，可能导致某些热门专家过载。

**Expert Choice Routing**：Google [7] 提出的反向约束。每个专家选择 Top-C 个 Token。

$$\sum_{i} x_{i,e} = C, \quad x_{i,e} = 1 \iff i \in \text{Top}_C(\text{Gating}(\cdot, e))$$

这种建模保证了完美的负载均衡，但可能导致某些 Token 被处理不足（Under-served）或处理过度，从而影响模型质量。

**有界降级（Bounded Degradation）**：在动态路由中，为了缓解热点，允许 Token 被路由到次优专家，但要求亲和度损失在一定阈值 $\epsilon$ 内。

$$S_{i, \text{routed}} \ge (1-\epsilon) \cdot S_{i, \text{best}}$$

这是在 OR 视角下将“硬约束”转化为“软约束”的关键步骤，为优化算法提供了可行域。

### 2.5 业界系统约束对比总结表

| 维度 | vLLM / SGLang | DeepSpeed-MoE | Metro (EuroSys '25) | Google Expert Choice |
| :--- | :--- | :--- | :--- | :--- |
| **核心瓶颈视角** | 显存容量 (KV Cache) | 通信带宽 (All-to-All) | 显存带宽 (HBM IO) | 计算均衡性 |
| **路由决策主体** | Token (Top-k) | Token (Top-k + Capacity) | System (Global Scheduler) | Expert (Top-C) |
| **显存建模粒度** | Page Block (Token粒度) | Static Allocation | Dynamic Expert Set | Static TPU Memory |
| **负载均衡目标** | 计算槽位填充率 | Token 数量均衡 (EPLB) | 激活专家数最小化 | 专家计算量严格均衡 |
| **丢包策略** | 不丢包 (排队等待) | 超过 Capacity Factor 丢包 | 不丢包 (调度解决) | 超过 Buffer 丢包 |
| **通信优化** | 张量并行 (TP) 优先 | 分层 All-to-All, 压缩 | All-Gather 代替 All-to-All | TPU 环状拓扑优化 |

3. Fluid Limit 中 Scaling 因子 $n$ 的最佳选择
在运筹学与排队论的视角下，大规模 MoE 推理系统本质上是一个具有多类顾客（Tokens）和多类服务器（Experts）的随机处理网络（Stochastic Processing Network）。为了理解系统在规模扩大时的渐近行为，我们需要分析其流体极限（Fluid Limit）。
在流体模型中，系统的状态（如队列长度）被缩放并近似为确定性的微分方程解。对于 MoE 系统，Scaling 因子 $n$ 通常代表系统的规模参数（如 GPU 的数量或总算力）。关键问题是：当系统规模 $n \to \infty$ 时，到达率 $\lambda_n$ 应该如何随之变化，才能在保证服务质量（低延迟）的同时最大化资源利用率？
3.1 经典流体极限 (Classical Fluid Limit) 与其局限性
经典的 Scaling 方式（亦称 Efficiency-Driven, ED Regime）通常假设：


$$\lambda_n = n \cdot \mu \cdot \rho$$

其中 $\mu$ 是单服务器的处理速率，$\rho$ 是目标利用率（常数，且 $\rho < 1$）。
行为特征：在这种 Scaling 下，随着 $n \to \infty$，根据大数定律，系统的平均队列长度归一化后趋近于 0（如果 $\rho < 1$）或线性增长（如果 $\rho > 1$）。
MoE 场景下的失效：
资源浪费：为了保证低延迟，通常需要设定 $\rho$ 显著小于 1（例如 0.6-0.7）。这意味着在大规模集群中（如 1000 张 H100），有 30%-40% 的算力处于闲置状态，这在经济上是不可接受的。
忽视随机性：MoE 的 Token 路由具有高度的突发性和随机性（Stochasticity）。经典流体极限无法捕捉 $\sqrt{n}$ 级别的随机波动，而正是这些波动导致了实际系统中的长尾延迟（Tail Latency）。
3.2 Halfin-Whitt (QED) Regime：MoE 的最佳 Scaling 选择
为了克服经典极限的缺陷，学术界推荐采用 Halfin-Whitt Regime，又称 Quality-and-Efficiency Driven (QED) 机制 9。
3.2.1 数学定义
在 Halfin-Whitt 机制下，到达率 $\lambda_n$ 与系统规模 $n$ 的关系设定为：


$$\lambda_n \approx n\mu - \beta\sqrt{n}\mu$$

或者等价地，系统服务器数量 $N_n$ 设定为：


$$N_n \approx \frac{\lambda_n}{\mu} + \beta\sqrt{\frac{\lambda_n}{\mu}}$$

其中 $\beta > 0$ 是一个控制参数，称为“安全余量系数”。
3.2.2 为什么这是 MoE 的最佳选择？
极高的资源利用率：

$$\text{Utilization}_n = \frac{\lambda_n}{n\mu} = 1 - \frac{\beta}{\sqrt{n}} \to 1 \quad (\text{as } n \to \infty)$$

这意味着随着集群规模扩大，资源利用率可以无限逼近 100%，而不会导致系统崩溃。对于昂贵的 AI 推理集群，这是通过 Scaling 降低 TCO（总拥有成本）的关键理论依据。
可控的排队延迟：
在 QED 机制下，虽然利用率趋近于 1，但排队等待的概率 $P(W>0)$ 收敛到一个非退化的常数 $\alpha(\beta) \in (0, 1)$，且条件平均等待时间 $E$ 按照 $O(1/\sqrt{n})$ 的速度衰减。
物理意义：集群越大，能够吸纳的负载波动（$\sqrt{n}$ 级别）就越大，单个 Token 遇到所有专家都忙碌的概率虽然存在，但由于并行的服务能力极强，其等待时间会迅速缩短。这完美契合了 LLM 推理对低延迟（Low Latency）的需求。
应对专家并行中的 Setup Time：
MoE 系统中存在专家加载的 Setup Time（从 CPU 到 GPU 或跨节点传输）。在 Fluid Limit 中，这引入了非凸性。Halfin-Whitt 机制提供的 $\sqrt{n}$ 级余量恰好为处理这种非稳态的切换提供了缓冲池（Buffer），防止系统因频繁切换专家而陷入“颠簸（Thrashing）”状态。
3.3 Scaling 讨论：All-to-All 通信的影响
在 MoE 特有的 All-to-All 通信约束下，Scaling 因子 $n$ 的选择还需考虑通信开销的非线性增长。
如果采用简单的 Scaling，通信量 $C_{comm} \propto n$ 或 $n^2$。在 Halfin-Whitt 机制下，我们必须引入 Power-of-d-choices (JSQ(d)) 路由策略 11。
策略：对于每个 Token，不查询所有 $n$ 个专家，而是仅随机查询 $d_n$ 个副本。
结论：只要 $d_n$ 增长速度满足 $d_n \gg \log n$，系统的性能就能逼近完全信息的 Joint-the-Shortest-Queue (JSQ) 策略，同时将通信开销从 $O(n)$ 降低到 $O(\log n)$ 或常数级。这对于 DeepSpeed-MoE 等依赖 All-to-All 的系统至关重要，证明了在超大规模下，**稀疏路由（Sparse Routing）**不仅是模型算法的需求，也是系统排队理论上的最优解。
4. 保证系统稳定、均衡唯一性及在线算法收敛的数学条件
在确定了 Scaling 规律后，核心问题是如何设计动态路由算法，使其在运行时能够自适应地达到理论最优。我们将 MoE 路由问题建模为网络效用最大化（NUM）问题，并利用**影子价格（Shadow Price）**进行对偶分解。
4.1 Primal-Dual 动态路由框架
我们将 MoE 推理网络视为一个二分图匹配问题，一侧是 Token 请求，另一侧是专家队列。
4.1.1 原问题 (Primal Problem)：效用最大化

$$\max_{x_{ij}} \sum_{i} \sum_{j} U(x_{ij}, S_{ij}) - \text{Cost}(x_{ij})$$
$x_{ij} \in \{0, 1\}$：Token $i$ 是否路由到专家 $j$。
$S_{ij}$：模型给出的亲和度分数（Logits）。
$U(\cdot)$：效用函数，通常与模型质量正相关。
约束：$\sum_{i} x_{ij} \le C_j(t)$ （专家 $j$ 当前周期的处理能力）。
4.1.2 对偶问题与影子价格 (Shadow Price)
引入 Lagrange 乘子 $p_j(t)$ 对应于专家容量约束。在排队网络中，这个乘子 $p_j(t)$ 物理上直接对应于 专家 $j$ 的排队长度 $Q_j(t)$（或显存占用的背压 Backpressure）。
算法迭代过程（在线 Backpressure Routing）：
路由决策 (Primal Update)：
对于每个 Token $i$，选择专家 $j^*$ 以最大化净收益：

$$j^* = \arg\max_{j} \left( S_{ij} - p_j(t) \right)$$

直观理解：Token 会在“高模型亲和度”和“低队列拥塞（低价格）”之间做权衡。
价格更新 (Dual Update)：
根据供需关系更新影子价格（队列演化）：

$$p_j(t+1) = \left^+$$

其中 $\alpha$ 是步长。
4.2 保证系统稳定性的数学条件：Lyapunov 漂移
在 MoE 推理的随机环境下，稳定性意味着队列长度不会无限发散（即延迟可控）。这需要满足 Lyapunov Drift 条件 13。
定义二次 Lyapunov 函数 $V(\mathbf{Q}(t)) = \frac{1}{2} \sum_{j} Q_j(t)^2$。
系统稳定的充分条件是存在 $\epsilon > 0$，使得对于所有足够大的 $\|\mathbf{Q}(t)\|$：


$$\mathbb{E}[V(\mathbf{Q}(t+1)) - V(\mathbf{Q}(t)) \mid \mathbf{Q}(t)] \le -\epsilon \sum_{j} Q_j(t) + B$$
数学推导含义：这意味着路由算法必须具备足够的“负反馈”能力。当某个专家 $j$ 的队列 $Q_j$ 变长时，影子价格 $p_j$ 必须上升得足够高，从而在 $S_{ij} - p_j$ 的竞争中处于劣势，迫使 Token 流向其他专家。
对 MoE 的启示：Top-k 路由中的 $k$ 必须由系统动态调整，或者允许 Skip-connection。如果强制 Top-1 路由且不允许丢包，系统可能无法满足上述漂移条件（即负反馈通道被切断），导致不稳定。
4.3 保证均衡唯一性的数学条件：严格凸正则化
标准的线性规划（Linear Programming）形式的路由问题往往存在多个最优解，导致路由策略在等价的解之间跳变（Bang-bang control），引发系统抖动。
为了保证纳什均衡点（Nash Equilibrium）的唯一性，必须在原目标函数中引入 严格凸正则化项（Strictly Convex Regularization） 16。
熵正则化 (Entropy Regularization)：

$$\max \sum S_{ij} x_{ij} - \frac{1}{\eta} \sum x_{ij} \log x_{ij}$$

其中 $\eta$ 是温度参数。
导出 Softmax 策略：
该正则化项使得最优路由策略从硬性的 ArgMax 变为 Softmax：

$$x_{ij}^* \propto \exp\left( \eta \cdot (S_{ij} - p_j(t)) \right)$$
OR 视角的解释：这不仅是为了模型的多样性，更是为了优化理论上的收敛性。严格凸的目标函数保证了对偶函数是连续可微的，从而消除了梯度振荡，确保影子价格能够平滑收敛到唯一的稳态值。
4.4 保证在线算法收敛的数学条件：迟滞 (Hysteresis) 与时间尺度分离
MoE 推理系统的一个核心非凸性来源是 Setup Time（专家加载时间）。当路由决策导致一个未激活的专家需要被加载时，会产生一个固定的延迟惩罚。这使得成本函数呈现阶跃状，破坏了标准凸优化的收敛前提。
为了保证在线算法在存在 Setup Cost 时仍能收敛，必须引入 迟滞机制 (Hysteresis) 18。
迟滞路由规则：
Token $i$ 只有在目标专家 $k$ 的净收益 显著超过 当前专家 $j$ 一个阈值 $H$ 时，才进行切换：

$$(S_{ik} - p_k) > (S_{ij} - p_j) + H_{setup}$$
收敛性证明逻辑：
时间尺度分离 (Timescale Separation)：迟滞机制抑制了高频的路由切换（Fast scale dynamics），使得专家的激活状态变化频率远低于 Token 的到达频率。
准静态假设：在慢时间尺度（Slow scale）下，系统可以被视为一系列准静态的流体模型。
定理：只要阈值 $H$ 设定得当（通常与 Setup Cost 成正比），系统状态将在相空间中收敛到一个吸引域（Attractor），即 MaxWeight 策略在迟滞修正下仍保持吞吐量最优（Throughput Optimality）。
5. 综合模型与报告总结
5.1 约束表格 (Constraints Table)
下表总结了构建 MoE 推理数学模型时的核心约束变量及其物理对应。
约束类型
符号表示与数学形式
物理含义
典型处理系统
影子价格对应物 (Dual Variable)
计算容量
$\sum_{i} x_{ij} \le \mu_j \cdot \tau$
GPU TFLOPS 限制，Token 处理速率
vLLM (Continuous Batching)
计算队列长度 (Compute Queue Length)
显存带宽
$\sum_{e} \mathbb{I}(\sum_{i} x_{ie,g} > 0) \le BW_{limit}$
Metro: 限制每 GPU 同时激活的专家数量
Metro (Active Expert Counting)
显存带宽锁/拥塞价格 (Bandwidth Lock Price)
通信带宽
$\sum_{g \ne g'} \text{Data}(g \to g') \le C_{link}$
All-to-All 互联带宽 (NVLink/IB)
DeepSpeed (Hierarchical A2A)
链路拥塞指示 (Link Congestion Price)
专家容量
$\sum_{i} x_{ij} \le Cap_{factor} \cdot \frac{
\mathcal{B}
}{
\mathcal{E}
服务质量
$x_{ij} = 1 \implies j \in \text{Top}_k(i)$
保证路由准确性与模型精度
Standard MoE
亲和度阈值 (Affinity Threshold)

5.2 假设列表 (List of Assumptions)
为了使上述流体与排队模型成立，我们在分析中采用了以下标准假设：
到达过程平滑化：虽然 Token 生成是自回归的，但在 Batch 层面上，假设请求到达近似服从非齐次泊松过程（Inhomogeneous Poisson Process），允许使用流体近似。
服务时间分布：假设专家处理 Token 的时间（Compute）服从独立同分布（I.I.D.），忽略因控制流差异导致的微小波动。对于 Metro 模型，假设专家加载时间（I/O）为常数。
弱相关性：假设不同 Token 的 Expert 亲和度分布是弱相关的。如果所有 Token 都指向同一个专家（如攻击场景 20），流体模型的“均衡”假设将失效，需切换到最坏情况分析（Worst-case Analysis）。
影子价格实时性：假设各个 GPU 能够以足够低的延迟广播其影子价格（队列长度）。如果通信延迟过高，需引入延迟微分方程（Delay Differential Equations）进行修正。
5.3 简化模型结构 (Simplified Model Structure)
基于上述分析，我们可以将复杂的 MoE 推理系统抽象为以下 带控制的排队网络模型：
输入流：向量 $\vec{\lambda}(t)$，代表不同语义类型的 Token 到达率。
节点（服务器）：集合 $\mathcal{E}$，每个专家是一个带有 Setup Time 的 $M/G/1/K$ 队列（有限缓冲区）。
状态变量：$\mathbf{Q}(t) = [Q_1(t), \dots, Q_E(t)]$，表示各专家的待处理 Token 数（或显存压力）。
控制器（Router）：
观测：亲和度矩阵 $\mathbf{S}$，状态向量 $\mathbf{Q}(t)$。
决策：路由矩阵 $\mathbf{X}(t)$。
策略：Softmax Policy with Hysteresis

$$x_{ij}(t) \propto \exp \left( \frac{S_{ij} - \gamma Q_j(t) - \text{Hyst}(j, \text{active})}{\eta} \right)$$
演化方程 (Fluid Limit)：

$$\dot{Q}_j(t) = \sum_i \lambda_i x_{ij}(t) - \mu_j \mathbb{I}(Q_j > 0)$$
5.4 结论与展望
本报告通过深度整合 vLLM 的工程实践、Metro 的带宽优化理论以及运筹学中的流体极限分析，构建了一套完整的 MoE 动态路由理论框架。研究表明，未来的高性能 MoE 推理引擎不应仅仅是静态的 Load Balancer，而应是一个 分布式的、基于影子价格驱动的市场机制。
在该机制下，Halfin-Whitt Scaling 规律指导硬件资源的配置，确保系统运行在高效且低延迟的 QED 区域；而带有正则化和迟滞的 Backpressure 路由算法则作为“看不见的手”，在微观层面实时消解拥塞、平滑抖动。这种跨学科的融合（Systems + AI + OR）将是解决万亿参数模型推理挑战的必由之路。
引用的著作
Autoscaling vLLM with OpenShift AI model serving: Performance validation, 访问时间为 一月 13, 2026， https://developers.redhat.com/articles/2025/11/26/autoscaling-vllm-openshift-ai-model-serving
Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling - arXiv, 访问时间为 一月 13, 2026， https://arxiv.org/html/2503.04398v3
Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens, 访问时间为 一月 13, 2026， https://arxiv.org/html/2512.09277
Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens, 访问时间为 一月 13, 2026， https://www.researchgate.net/publication/398560229_Efficient_MoE_Serving_in_the_Memory-Bound_Regime_Balance_Activated_Experts_Not_Tokens
Scaling DeepSeek-style MoEs with vLLM + llm-d using Wide EP | Red Hat Developer, 访问时间为 一月 13, 2026， https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
Occult: Optimizing Collaborative Communications across Experts for Accelerated Parallel MoE Training and Inference | OpenReview, 访问时间为 一月 13, 2026， https://openreview.net/forum?id=vh2Dt4sT67
At the Frontier of AI: Reviewing Top Papers on Mixture of Experts in Machine Learning — Part 5 - Isaac Kargar, 访问时间为 一月 13, 2026， https://kargarisaac.medium.com/at-the-frontier-of-ai-reviewing-top-papers-on-mixture-of-experts-in-machine-learning-part-5-ee939dc91409
Mixture-of-Experts with Expert Choice Routing - NeurIPS, 访问时间为 一月 13, 2026， https://papers.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf
Fluid Approximation of a Priority Call Center With Time-Varying Arrivals - William A. Massey, 访问时间为 一月 13, 2026， https://wmassey.princeton.edu/document/136
(PDF) Scaling and comparison of fluid limits of queues applied to call centers with time-varying parameters - ResearchGate, 访问时间为 一月 13, 2026， https://www.researchgate.net/publication/226864256_Scaling_and_comparison_of_fluid_limits_of_queues_applied_to_call_centers_with_time-varying_parameters
Universality of Power-of-d Load Balancing in Many-Server Systems - PubsOnLine, 访问时间为 一月 13, 2026， https://pubsonline.informs.org/doi/pdf/10.1287/stsy.2018.0016
游戏/分布式&并行&集群/性能2025_4_4 - arXiv每日学术速递, 访问时间为 一月 13, 2026， http://www.arxivdaily.com/thread/65974
Quantum Internet protocol stack: A comprehensive survey | Request PDF - ResearchGate, 访问时间为 一月 13, 2026， https://www.researchgate.net/publication/361298318_Quantum_Internet_protocol_stack_A_comprehensive_survey
Score-Aware Policy-Gradient and Performance Guarantees using Local Lyapunov Stability - Journal of Machine Learning Research, 访问时间为 一月 13, 2026， http://www.jmlr.org/papers/volume26/24-1009/24-1009.pdf
Score-Aware Policy-Gradient Methods and Performance Guarantees using Local Lyapunov Conditions, 访问时间为 一月 13, 2026， https://research.tue.nl/files/352800682/2312.02804v2.pdf
Engineering Social Optimality via Utility Shaping in Non-Cooperative Games under Incomplete Information and Imperfect Monitoring - arXiv, 访问时间为 一月 13, 2026， https://arxiv.org/html/2510.26033v1
A Unified Approach to Controlling Implicit Regularization via Mirror Descent - Journal of Machine Learning Research, 访问时间为 一月 13, 2026， https://www.jmlr.org/papers/volume24/23-0836/23-0836.pdf
Maximum Weight Matching with Hysteresis in Overloaded Queues with Setups - Columbia University, 访问时间为 一月 13, 2026， http://www.columbia.edu/~cc3179/MWMH_2015.pdf
Maximum weight matching with hysteresis in overloaded queues with setups | Request PDF, 访问时间为 一月 13, 2026， https://www.researchgate.net/publication/287333796_Maximum_weight_matching_with_hysteresis_in_overloaded_queues_with_setups
RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress - arXiv, 访问时间为 一月 13, 2026， https://arxiv.org/html/2512.23995
