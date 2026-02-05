# Fluid-SemMoE: 基于流体模型与影子价格的 MoE 动态路由策略  
## Fluid Model, Limit Theorems, and Lyapunov Control for Distributed Inference

**Version:** 2.1 (Rigorous OR / SPN Formulation)  

**Abstract.**  
本文提出一种基于流体缩放（fluid scaling）与影子价格（shadow pricing）的在线调度框架，用以刻画并解决分布式 MoE 推理系统中由**通信瓶颈与负载突发性**共同导致的动态资源失衡问题。不同于现有仅关注 Expert 级别计算排队的工作，我们从随机处理网络（Stochastic Processing Networks, SPN）的视角出发，将系统瓶颈重构为 **Device 级别的准入与通信耦合约束**，并在 Markov Modulated Poisson Process (MMPP) 驱动的非平稳流量下，推导出一个具有最大稳定性与渐近最优性的 Lyapunov 型动态路由策略。

---

## 1. 背景与建模层级的重构 (Introduction)

### 1.1 从微观 Expert 到宏观 Device 的转变

在 **PriceMoE (v1.0)** 及相关工作中，MoE 推理系统通常被建模为一个多队列网络，其中每个 Expert 对应一个独立的服务节点。然而，在 vLLM、DeepSpeed-MoE 等现代推理框架中，这一建模在结构上存在根本性的 **层级错位（modeling mismatch）**：

- **物理事实（System Reality）**：  
  在 MoE 层执行之前，Token 必须经过跨设备（cross-device）的 All-to-All 集合通信；随后，各设备在本地以 batch 形式执行专家计算。因此，**Token 的准入与等待主要发生在 Device 层，而非 Expert 层**。

- **瓶颈迁移（Bottleneck Shift）**：  
  实际系统的性能瓶颈主要来源于  
  (i) 通信同步带来的等待延迟，  
  (ii) 显存（尤其是 KV Cache）约束下的准入压力，  
  而非单个 Expert 的算力排队。

因此，继续在 Expert 层定义队列并引入影子价格，无法正确刻画系统的稳定性边界与吞吐瓶颈。

---

### 1.2 核心问题定义

我们考虑如下问题：

> 在 **MMPP 驱动的非平稳语义流量** 下，如何设计一种 **在线路由策略**，在满足系统最大稳定性（maximum stability）的前提下，最小化长期平均通信成本？

该问题同时涉及：
- 随机到达（stochastic arrivals），
- 通信受限的服务能力（communication-constrained service），
- 以及在线、无先验到达率知识的控制决策。

---

## 2. 随机处理网络模型 (Stochastic Processing Network)

### 2.1 系统参数与基本集合

- **设备集合（Devices）**：  
  \[
  \mathcal{D} = \{1, \dots, D\},
  \]
  每个设备 $d$ 具有等效处理能力 $\mu_d$（以 requests/sec 计），该能力已隐式包含通信与计算的综合瓶颈。

- **语义类别（Semantic Classes）**：  
  \[
  \mathcal{C} = \{1, \dots, C\},
  \]
  每个到达请求在进入系统时被分类为某一语义类别 $c \in \mathcal{C}$。

- **通信惩罚矩阵（Communication Cost Matrix）**：  
  \[
  \mathbf{G} = [g_{c,d}] \in \mathbb{R}_+^{C \times D},
  \]
  其中 $g_{c,d}$ 表示将类别 $c$ 的请求路由至设备 $d$ 时，所诱导的**期望跨设备通信负载**。该量由 Sem-MoE 式的离线语义–Expert–Device 画像得到，视为已知常数。

---

### 2.2 流量动力学：MMPP 到达过程

到达过程由一个有限状态连续时间马尔可夫链 $\{Z(t)\}_{t \ge 0}$ 调制：

- 当 $Z(t) = z$ 时，类别 $c$ 的请求到达率为 $\lambda_c^{(z)}$；
- $Z(t)$ 的状态切换刻画了生成式 AI 业务中常见的 **语义场景切换与突发性（burstiness）**。

该建模显式捕捉了到达过程的时间相关性与非平稳性，是挑战静态路由策略（如 Sem-MoE）的关键来源。

---

### 2.3 队列演化（Admission Backlog）

定义设备 $d$ 的准入队列长度 $Q_d(t)$，其演化满足：

\[Q_d(t+1)
=\left[ Q_d(t) - S_d(t) \right]^+
+
\sum_{c \in \mathcal{C}} x_{c,d}(t),
\]

其中：
- $x_{c,d}(t) \in \mathbb{Z}_+$ 表示在时刻 $t$ 被路由至设备 $d$ 的类别 $c$ 请求数；
- $S_d(t)$ 为设备 $d$ 在时刻 $t$ 能够完成的服务量，其期望上界为 $\mu_d$。

该队列并非物理 FIFO 队列，而是一个 **等效（effective）队列**，用于刻画通信与显存约束共同诱导的准入压力。

---

## 3. 流体极限与极限定理 (Fluid Limits)

### 3.1 流体缩放（Fluid Scaling）

引入缩放参数 $n \to \infty$，定义：

\[
\bar{Q}^n(t) = \frac{Q^n(nt)}{n}, \quad
\bar{A}^n(t) = \frac{A^n(nt)}{n}.
\]

在标准正则性条件下（有界二阶矩、不可约马尔可夫调制等），可应用 **泛函强大数定律（FSLLN）**。

---

### 3.2 流体极限定理

**定理 3.1 (Fluid Limit Convergence).**  
在任意有限时间区间 $[0,T]$ 上，序列 $\{\bar{Q}^n(t)\}$ 几乎处处一致收敛于确定性函数 $q(t)$，该极限满足以下常微分方程（ODE）：

\[\dot{q}_d(t)
=\left(
\sum_{c \in \mathcal{C}} \phi_{c,d}(t) - \mu_d
\right)^+_{q_d(t)},
\quad \forall d \in \mathcal{D},
\]

其中 $\phi_{c,d}(t)$ 为类别 $c$ 流向设备 $d$ 的瞬时流率。

该流体模型刻画了系统在大规模与长时间尺度下的平均行为。

---

## 4. 稳态流体优化与拉格朗日对偶 (Steady-State Fluid Optimization)

### 4.1 原始优化问题（Primal Problem）

在稳态下，考虑如下通信最小化问题：

\[
\begin{aligned}
\min_{\{\phi_{c,d}\}} \quad
& \sum_{c \in \mathcal{C}} \sum_{d \in \mathcal{D}} \phi_{c,d} \, g_{c,d} \\
\text{s.t.} \quad
& \sum_{c \in \mathcal{C}} \phi_{c,d} \le \mu_d,
\quad \forall d \in \mathcal{D} \quad (\text{Stability}) \\
& \sum_{d \in \mathcal{D}} \phi_{c,d} = \lambda_c,
\quad \forall c \in \mathcal{C} \quad (\text{Flow Conservation}) \\
& \phi_{c,d} \ge 0.
\end{aligned}
\]

---

### 4.2 拉格朗日对偶与影子价格

引入对偶变量 $\nu_d \ge 0$ 对应稳定性约束，部分拉格朗日函数为：

\[\mathcal{L}(\phi, \nu)=
\sum_{c,d} \phi_{c,d}(g_{c,d} + \nu_d)-
\sum_d \nu_d \mu_d.
\]

**经济学解释：**
- $\nu_d$ 是设备 $d$ 的 **影子价格（shadow price）**，表示其单位服务能力的边际稀缺性；
- 最优解满足：若 $\phi_{c,d}^* > 0$，则
  \[
  d \in \arg\min_j \{ g_{c,j} + \nu_j^* \}.
  \]

---

## 5. Lyapunov 动态控制律 (Online Routing via Drift Minimization)

### 5.1 Lyapunov 函数与漂移

定义 Lyapunov 函数：
\[
V(\mathbf{Q}(t)) = \frac{1}{2} \sum_{d \in \mathcal{D}} Q_d(t)^2.
\]

考虑 **Drift-Plus-Penalty**：
\[
\Delta V(t)
+
\eta \cdot
\mathbb{E}[\text{CommCost}(t) \mid \mathbf{Q}(t)],
\]
其中 $\eta > 0$ 控制通信成本与排队稳定性的权衡。

---

### 5.2 Fluid-SemMoE 路由策略（最终控制律）

通过最小化上述上界，可得逐请求最优决策：

> **Fluid-SemMoE Routing Rule**
>
> 对于到达的语义类别为 $c$ 的请求，在时刻 $t$ 选择设备
> \[d^*(t)=
> \arg\min_{d \in \mathcal{D}}
> \left\{
> Q_d(t) + \eta \cdot g_{c,d}
> \right\}.
> \]

---

## 6. 理论性质 (Theoretical Guarantees)

1. **最大稳定性（Throughput Optimality）**  
   在存在任何稳定策略的前提下，Fluid-SemMoE 能稳定所有可行到达率向量 $\lambda$。

2. **渐近最优性（Asymptotic Optimality）**  
   长期平均通信成本与流体最优值之间的差距为 $O(1/\eta)$。

3. **自适应分流（Adaptive Spillover）**  
   - 低负载时：$g_{c,d}$ 主导，策略退化为 Sem-MoE；
   - 高负载或突发时：$Q_d(t)$ 主导，自动触发跨设备分流。

---

## 7. 结论 (Conclusion)

Fluid-SemMoE 通过一次关键的建模层级重构，将 PriceMoE 的影子价格思想从 Expert 计算队列推广至 **通信诱导的 Device 准入队列**。在流体极限与 Lyapunov 优化框架下，我们得到了一个同时具备 **OR 理论严谨性** 与 **系统工程可落地性** 的动态 MoE 路由策略，为大规模分布式推理提供了新的调度范式。
```python
class FluidSemMoERouter:
    def __init__(self, devices, comm_cost_table, V=1.0):
        self.devices = devices
        self.G = comm_cost_table # 离线 Sem-MoE 表
        self.V = V               # 权衡参数

    def route(self, req):
        c = self.get_semantic_category(req)
        best_dev = None
        min_price = float('inf')

        for d in self.devices:
            # 1. 拥塞价格 (归一化负载)
            # 使用 KV Cache 使用率作为 "积压" 的代理
            q_price = d.get_kv_cache_usage() * 100 
            
            # 2. 通信价格 (期望开销)
            comm_price = self.G[c][d]
            
            # 3. 广义影子价格
            total_price = q_price + (self.V * comm_price)
            
            if total_price < min_price:
                min_price = total_price
                best_dev = d
                
        return best_dev
```



---

# Fluid-SemMoE: 基于流体模型与影子价格的 MoE 动态路由策略  
## Fluid Model, Semantic Redundancy, and Lyapunov Control for Distributed Inference

**Version:** 2.3 (Addressing Physical Feasibility & Dynamic Service Rate)  

**Abstract.**  
本文提出一种基于流体缩放与影子价格的在线调度框架 **Fluid-SemMoE**，旨在解决分布式 MoE 推理中由通信瓶颈、语义突发性及动态服务能力共同引发的资源失衡问题。我们指出当前工作在**语义映射物理基础缺失**与**设备服务率动态性**方面的不足，并引入基于语义冗余的可行域约束与状态感知的队列模型。在 Markov 调制泊松过程（MMPP）驱动的流量下，我们证明了静态路由策略在突发场景下的不稳定性，并通过**队列强度原理**建立拉格朗日对偶变量与队列背压的联系，推导出一种 Lyapunov 优化的动态控制律。该策略在严格满足物理可行性的前提下，实现了最大稳定性与渐近最优的通信效率。

---

## 1. 背景与核心挑战 (Introduction & Challenges)

### 1.1 瓶颈迁移：从 Expert 到 Device

现代推理框架（如 vLLM, DeepSpeed-MoE）的物理现实表明，MoE 推理的瓶颈已从单个 Expert 的计算能力转移至跨设备的 **All-to-All 通信同步** 与 **Device 级别的 KV Cache 显存准入**。

- **Expert 计算**：高度并行的 Batched MatMul，在 Device 内部通常非饱和。
- **Device 准入**：受限于 PCIe/NVLink 带宽和显存，Token 在进入 Device 前形成逻辑排队。

因此，我们将影子价格的载体从 Expert 队列重构为 Device 准入队列 $Q_d(t)$，以准确刻画系统瓶颈。

### 1.2 现有工作的潜在挑战与提升空间

尽管现有的静态优化（如 MoETuner）和简单的动态路由取得了一定进展，但在面对实际的大规模分布式推理时，仍存在两个关键的物理建模缺失：

**1. 语义映射的物理基础缺失**
在许多理论公式中，虽然出现了语义类别 $c$ 和通信成本 $g_{c,d}$，但并未明确回答一个根本问题：**“为什么类别 $c$ 的请求允许被路由到设备 $d$？”**
- *缺失点*：缺乏对 **专家冗余/复制假设** 的显式建模。
- *后果*：算法可能将请求路由至根本不具备处理该语义能力的设备上，导致严重的模型质量下降。
- *本方案*：通过引入可行设备集 $\mathcal{D}_c$ 和放置矩阵 $\mathbf{P}$，严格定义路由的物理边界。

**2. 设备服务能力 $\mu_d$ 的动态性**
传统排队论模型常假设服务率 $\mu_d$ 为常数。但在 LLM 推理中，$\mu_d$ 具有高度动态性：
- **Batch Size 效应**：实际吞吐取决于当前 Batch 的大小；
- **KV Cache 约束**：当显存接近上限时，服务速率因 Swap 操作或请求拒绝而急剧下降。
- *本方案*：我们将 $\mu_d$ 建模为与 KV Cache 占用率相关的变量，并通过队列 $Q_d(t)$ 的累积来隐式捕捉这种服务能力的衰退。

### 1.3 静态路由在突发流量下的局限性

为了证明动态控制的必要性，我们首先分析静态路由策略在非平稳流量下的表现。

**Lemma 1 (Static Routing Instability under MMPP).**
*考虑由 MMPP $\{Z(t)\}$ 调制的到达过程。设 $\boldsymbol{\Phi}^{\text{static}} \in \mathbb{R}^{C \times D}$ 为静态路由矩阵，满足 $\sum_d \phi^{\text{static}}_{c,d} = 1$。若存在调制状态 $z$ 和设备 $d$，使得该状态下的有效到达率超过设备服务能力，即：*
$$
\sum_{c \in \mathcal{C}} \phi^{\text{static}}_{c,d} \cdot \lambda_c^{(z)} > \mu_d,
$$
*则该流体模型不稳定，即存在 $\epsilon > 0$ 使得 $\lim_{t \to \infty} q_d(t) = \infty$。*

**证明.**
在流体极限下，设备 $d$ 的队列长度演化满足 ODE：
$$
\dot{q}_d(t) = \sum_{c \in \mathcal{C}} \phi^{\text{static}}_{c,d} \cdot \lambda_c^{(Z(t))} - \mu_d.
$$
由于 $Z(t)$ 是遍历的马尔可夫链，若存在状态 $z$ 使得 $\dot{q}_d(t) > 0$，则当系统停留在状态 $z$ 时，$q_d(t)$ 将以正斜率线性增长。即便系统离开状态 $z$，队列长度也不会减少（除非 $\dot{q}_d(t) < 0$ 且持续足够长时间）。在 MMPP 的突发性假设下，高强度的状态 $z$ 会频繁出现，导致净积累量随时间趋于无穷。因此，静态路由无法处理超过静态容量预留的突发流量。 $\square$

---

## 2. 随机处理网络模型 (Stochastic Processing Network)

### 2.1 系统参数与符号定义

我们定义如下集合与矩阵，以精确描述语义冗余与物理映射：

- **设备集合**：$\mathcal{D} = \{1, \dots, D\}$。
- **专家集合**：$\mathcal{E} = \{1, \dots, E\}$。
- **语义类别**：$\mathcal{C} = \{1, \dots, C\}$。

**Definition 1 (Placement Matrix $\mathbf{P}$).**
定义 $\mathbf{P} \in \{0, 1\}^{E \times D}$ 为专家放置矩阵，其中：
$$
P_{e,d} = \begin{cases} 
1, & \text{if Expert } e \text{ is placed on Device } d \\
0, & \text{otherwise}
\end{cases}
$$

**Definition 2 (Semantic Redundancy Set $\mathcal{E}_c^{\text{sim}}$).**
对于语义类别 $c$，定义 $\mathcal{E}_c^{\text{sim}} \subseteq \mathcal{E}$ 为能够有效处理该类别请求的专家集合（由离线语义相似度分析得到，如 REAP）。该集合体现了 MoE 模型的功能冗余特性。

**Definition 3 (Feasible Device Set $\mathcal{D}_c$).**
基于物理放置与语义冗余，定义语义类别 $c$ 的可行设备域：
$$
\mathcal{D}_c = \left\{ d \in \mathcal{D} \mid \exists e \in \mathcal{E}_c^{\text{sim}}, \text{ s.t. } P_{e,d} = 1 \right\}.
$$
*注：只有 $d \in \mathcal{D}_c$ 时，类别 $c$ 的请求才允许被路由至设备 $d$，否则将导致模型崩坏。*

### 2.2 流量动力学与通信成本

- **到达过程**：MMPP 调制，状态 $Z(t)=z$ 时，类别 $c$ 的到达率为 $\lambda_c^{(z)}$。
- **通信惩罚矩阵** $\mathbf{G} = [g_{c,d}]$：
    - 若设备 $d$ 与请求源节点（如 Prefill 节点）处于同一物理机，$g_{c,d} \approx 0$。
    - 若需跨 InfiniBand 传输，$g_{c,d} \gg 0$。

### 2.3 队列演化（Device-Level Backlog）

定义设备 $d$ 的准入队列长度 $Q_d(t)$，其动力学方程为：
$$
Q_d(t+1) = \left[ Q_d(t) - S_d(t) \right]^+ + \sum_{c \in \mathcal{C}} x_{c,d}(t),
$$
其中 $x_{c,d}(t)$ 为时刻 $t$ 路由至设备 $d$ 的类别 $c$ 请求数。$S_d(t)$ 为服务过程，其期望受限于动态服务能力 $\mu_d(t)$（受 KV Cache 影响）。**关键约束**：若 $d \notin \mathcal{D}_c$，则 $x_{c,d}(t) = 0$。

---

## 3. 流体极限与优化目标

### 3.1 流体模型

引入缩放因子 $r \to \infty$，定义流体标度变量 $\bar{Q}^r(t) = Q^r(rt)/r$。根据泛函强大数定律（FSLLN），$\bar{Q}^r(t)$ 几乎处处收敛于确定性极限 $q(t)$，满足：
$$
\dot{q}_d(t) = \left( \sum_{c \in \mathcal{C}} \phi_{c,d}(t) - \mu_d \right)^+_{q_d(t)}, \quad \forall d \in \mathcal{D}.
$$

### 3.2 稳态优化问题

我们的目标是最小化长期平均通信成本，同时保证队列稳定。

$$
\begin{aligned}
\min_{\{\phi_{c,d}\}} \quad & \sum_{c \in \mathcal{C}} \sum_{d \in \mathcal{D}} \phi_{c,d} \, g_{c,d} \\
\text{s.t.} \quad & \sum_{c \in \mathcal{C}} \phi_{c,d} \le \mu_d, & \forall d \in \mathcal{D} \quad &\text{(Capacity)} \\
& \sum_{d \in \mathcal{D}} \phi_{c,d} = \bar{\lambda}_c, & \forall c \in \mathcal{C} \quad &\text{(Flow Conservation)} \\
& \phi_{c,d} = 0, & \forall d \notin \mathcal{D}_c \quad &\text{(Semantic Feasibility)} \\
& \phi_{c,d} \ge 0.
\end{aligned}
$$

其中 $\bar{\lambda}_c$ 为类别 $c$ 的长期平均到达率。第三个约束显式保证了路由决策必须符合物理上的专家分布和语义冗余要求。

---

## 4. Lyapunov 动态控制律

### 4.1 拉格朗日松弛与影子价格

对容量约束引入对偶变量 $\nu_d \ge 0$，拉格朗日函数为：
$$
L(\phi, \nu) = \sum_{c,d} \phi_{c,d}(g_{c,d} + \nu_d) - \sum_d \nu_d \mu_d.
$$
根据对偶理论，最优解倾向于将流量分配到 $g_{c,d} + \nu_d$ 最小的可行设备上。$\nu_d$ 即为设备 $d$ 的**影子价格**，反映其拥塞程度。

### 4.2 从对偶变量到队列背压

为了将稳态优化理论转化为在线算法，我们需要建立拉格朗日乘子 $\nu_d$ 与系统状态 $Q_d(t)$ 之间的联系。

**Theorem (Queueing Intensity Principle).**  
在强稳定的随机处理网络中，若系统处于稳态，则长期平均队列长度 $\lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[Q_d(t)]$ 收敛于容量约束的对偶变量 $\nu_d^*$。

**桥接推导：**
由拉格朗日对偶理论可知，稳态最优解中，影子价格 $\nu_d^*$ 对应稳定性约束 $\sum_c \phi_{c,d} \le \mu_d$。在动态系统中，根据 **队列强度原理**，长期平均队列长度 $\mathbb{E}[Q_d]$ 恰好收敛于影子价格 $\nu_d^*$。

因此，我们直接使用**瞬时队列积压 $Q_d(t)$** 作为影子价格 $\nu_d$ 的无偏估计，代入瞬时决策，从而获得 $\eta$-optimal 的在线策略。这一步转换不仅赋予了 $Q_d(t)$ 明确的经济学含义（拥塞定价），也避免了在线求解复杂的对偶问题。

### 4.3 Drift-Plus-Penalty 算法

定义 Lyapunov 函数 $V(\mathbf{Q}(t)) = \frac{1}{2} \sum_{d} Q_d(t)^2$。我们最小化每一步的漂移加惩罚：
$$
\Delta V(t) + \eta \cdot \mathbb{E} \left[ \sum_{c,d} x_{c,d}(t) g_{c,d} \mid \mathbf{Q}(t) \right].
$$

### 4.4 Fluid-SemMoE 最终控制律

通过最小化上述上界，并代入 $Q_d(t)$ 作为影子价格 $\nu_d$ 的即时估计，我们得到 **Fluid-SemMoE Routing Rule**：

> **Theorem (Fluid-SemMoE Control Law).**  
> 对于在时刻 $t$ 到达的、语义类别为 $c$ 的请求，选择设备：
> $$
> d^*(t) = \arg\min_{\substack{d \in \mathcal{D} \\ d \in \mathcal{D}_c}} \left\{ Q_d(t) + \eta \cdot g_{c,d} \right\}.
> $$

**解读：**
- **$\min_{d \in \mathcal{D}_c}$**：确保仅路由至拥有语义冗余专家的设备，保障模型质量（回答挑战 1）。
- **$Q_d(t)$**：**队列背压**。若某设备队列积压严重（KV Cache 满或计算拥堵），影子价格升高，自动排斥新流量，实现**负载均衡**（回答挑战 2）。
- **$g_{c,d}$**：**通信成本**。在负载相近时，优先选择通信开销小的设备。

---

## 5. 理论性质

1.  **最大稳定性**：在满足 $\sum_c \bar{\lambda}_c < \sum_d \mu_d$ 的前提下，该策略能保证所有队列 $Q_d(t)$ 有限。
2.  **渐近最优性**：长期平均通信成本与最优流体解的差距为 $O(1/\eta)$。
3.  **自适应拥塞控制**：由 Lemma 1 可知，当 MMPP 进入高负荷状态 $z$ 时，静态策略会发散，而 Fluid-SemMoE 通过 $Q_d(t)$ 的上升动态调整路由路径，维持系统稳定。

---

## 6. 结论

Fluid-SemMoE 通过引入语义可行域 $\mathcal{D}_c$ 和放置矩阵 $\mathbf{P}$，在数学上严格界定了动态路由的物理空间，解决了语义映射的物理基础缺失问题。同时，利用队列强度原理将影子价格与瞬时队列长度挂钩，有效捕捉了设备服务能力的动态性。我们证明了静态路由在突发流量下的不稳定性，并提出基于 Lyapunov 优化的在线算法。该方案不仅在理论上满足吞吐最优性，更在工程上通过语义冗余约束确保了模型推理的准确性，为大规模分布式 MoE 推理提供了鲁棒的调度方案。
