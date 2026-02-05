# PriceMoE：通信感知的 Shadow-Price 动态路由（MS/OR 风格模型，v1.1.2）

> 本版本是对 `Fluid_MoE version 1.1.md` 的**审稿人级别重审与重写**：更贴近 OR/MS 的传统建模风格（明确决策变量、容量域、稳态 LP、影子价格/KKT、以及在线原对偶算法与可落地实现）。
>
> 我们的核心 idea B 不变：**用影子价格（shadow prices）把“通信 + 计算 + 内存”等资源稀缺性编码进 routing score**，从而得到可解释、可证明、可部署的 token 路由策略。

---

## 0. 从 OR/MS 审稿人的角度：v1.1 的问题清单与修正原则

站在 OR/MS 顶刊审稿人的直觉里，v1.1 有三类风险：

1. **“模型很大但不够落地”**：容量域 $\mathcal U(X)$ 写得过抽象，容易被质疑“你到底在约束什么、影子价格对应什么”。  
2. **“流体极限 OK，但与 pricing mechanism 的主线衔接不够强”**：缺少一个清晰的稳态 LP（或凸规划）→ KKT → shadow price → 在线策略的闭环叙事。
3. **“工程可部署性”**：需要更明确地说明：系统实际只需要维护哪些统计量（队列/链路拥塞/激活集），并如何更新价格。

因此 v1.1.2 的修正原则：

- 用 **单层、同构 GPU、单轮路由** 的简化模型作为主模型（符合 OR 传统：先把结构讲清楚）。
- 把约束写成**可写成 LP 的线性容量约束**（通信/计算），把非凸激活集问题单独作为扩展（避免主线被非凸拖累）。
- 明确给出：
  1) 稳态流体 LP；  
  2) KKT 与影子价格的经济解释；  
  3) 在线 primal–dual 更新；  
  4) 工程实现需要的观测量与开销。

**重要的主线修正（针对你这条反馈）**：PriceMoE 的原始目标是缓解 MoE routing 的 **congestion / load skew / tail latency**，因此 shadow price 必须包含“拥塞的边际代价”。在排队网络里这天然对应 **队列长度（或其单调变换）**。所以本版本明确把

- $q_e(t)$（流体队列）/ $Q_e(t)$（随机队列）作为系统状态；
- 计算拥塞影子价格设为 $p_e^{\text{queue}}(t):=\alpha\,q_e(t)$（或更一般 $p_e^{\text{queue}}=\phi'(q_e)$）；
- 资源稀缺影子价格仍由对偶变量给出（专家/链路容量）。

最终路由的“减分项”是两类价格之和：**拥塞价（queue/backpressure） + 资源价（capacity shadow price）**。

---

## 1. 系统与决策对象（单层 MoE 推理的一次 routing 周期）

### 1.1 系统结构

- GPU 集合：$\mathcal K=\{1,\dots,K\}$（同构）。
- 专家集合：$\mathcal E=\{1,\dots,E\}$。
- 专家放置：$L(e)\in\mathcal K$。
- token 类别集合：$\mathcal C=\{1,\dots,C\}$（可取 $C=1$ 作为最简）。

### 1.2 到达率（prefill/decode 均可视为“token 到达”）

在连续时间 $t$，类别 $c$ 的到达是更新过程 $A_c(t)$，满足（流体尺度）
\[
\bar A_c^{(n)}(t):=\frac{1}{n}A_c^{(n)}(nt)\Rightarrow \lambda_c t,\qquad t\in[0,T].
\]
令总到达率 $\Lambda:=\sum_c \lambda_c$。

### 1.3 路由决策变量（OR/MS 核心：变量要清晰）

我们用“类别—专家”的流体分配比例表示路由：
\[
x_{c,e}\in[0,1],\qquad \sum_{e\in\mathcal E} x_{c,e}=1,\qquad \forall c.
\]
并有质量可行域约束（Top-$k$ / bounded degradation）：
\[
 x_{c,e}=0\quad\text{若 }e\notin\mathcal A_c,
\]
其中 $\mathcal A_c$ 是类别 $c$ 的允许专家集合（由 gating 决定）。

这个 $x$ 是 OR 模型里最标准的“routing matrix”。

补充：在动态系统中，我们将使用 $x(t)$ 表示“时刻 $t$ 的瞬时分配”，并把其与队列状态 $Q(t)$/ $q(t)$ 形成反馈闭环。

---

## 2. 资源消耗与容量约束（给出可写 LP 的约束）

OR/MS 审稿人通常希望看到：每个决策 $x_{c,e}$ 对资源的线性消耗，然后容量约束线性约束。

### 2.1 计算（专家服务）容量

令专家 $e$ 的有效服务能力为 $\mu_e$（token/s）。稳态下到专家 $e$ 的到达率为
\[
\lambda_e(x):=\sum_{c\in\mathcal C} \lambda_c x_{c,e}.
\]
稳定性要求（流体稳态下的“no growth”条件）：
\[
\lambda_e(x)\le \mu_e,\qquad \forall e\in\mathcal E.
\]

> 说明：若考虑“GPU 级 compute budget”而不是“专家级”，可在 2.3 里用矩阵把专家汇总到 GPU。

补充（与拥塞相关）：容量约束刻画的是“可行服务域”，但 congestion/tail latency 的直接状态变量是队列 $Q_e(t)$。本模型稍后将通过 **Drift-plus-Penalty** 把队列状态进入目标，从而得到“队列长度就是影子价格”的 backpressure 结构。

### 2.2 通信容量（核心：通信感知 routing）

关键点：当 token 路由到不在本 GPU 的专家时，会产生跨 GPU 通信负载。

为保持 MS 主线简洁，我们采用一个经典可线性化的通信模型：

- 类别 $c$ 的 token “源 GPU” 为 $s(c)\in\mathcal K$（可由请求粘滞/前一层所在 GPU 确定）。
- 若 token 分配到专家 $e$，其目的 GPU 是 $L(e)$。
- 单位 token 的通信开销为 $d_{s(c),L(e)}\ge 0$（例如是否跨节点、NVLink/IB 分层等，可折算成等价“带宽占用单位”）。

令链路集合 $\mathcal L$，每条链路 $\ell$ 有容量 $B_\ell$（带宽单位/秒）。令 $a_{c,e}^{(\ell)}\ge 0$ 表示“类别 $c$ 的 token 若路由到专家 $e$，在链路 $\ell$ 上产生的单位负载”。此时链路负载为
\[
\mathrm{Load}_\ell(x)=\sum_{c,e} \lambda_c\,a_{c,e}^{(\ell)}\,x_{c,e}.
\]
通信容量约束：
\[
\mathrm{Load}_\ell(x)\le B_\ell,\qquad \forall \ell\in\mathcal L.
\]

> 这是 OR 最经典的“多资源容量约束”写法：$Ax\le b$。

### 2.3 统一矩阵形式（便于写对偶/影子价格）

把所有容量约束写成
\[
A x \le b,
\]
其中 $x$ 表示所有 $x_{c,e}$ 的向量化，$A$ 聚合了专家容量与链路容量的系数，$b$ 聚合了 $\mu$ 与 $B$。

---

## 3. 稳态流体 LP（主模型）与影子价格（KKT/对偶）

这是 v1.1 缺的“OR 主线”。我们在这里补齐：稳态 LP → 对偶 → shadow prices。

### 3.1 目标：效用–拥塞权衡（而非仅 gating 效用）

仅最大化 gated utility 的 LP 虽然能通过容量惩罚“间接”减少超载，但它不直接编码 queue length，因此无法在 OR 语境下正面回应 congestion/tail latency。

为此我们采用排队网络中经典的 **Drift-plus-Penalty**（或等价的“效用-拥塞”加权）框架：

- 效用：$u_{c,e}$（gating 期望效用或可保证的质量下界）；
- 拥塞代价：用队列长度 $q_e(t)$（流体）或 $Q_e(t)$（随机）表征；
- 价格映射：$p_e^{\text{queue}}(t)=\phi'_e(q_e(t))$，最常用取 $\phi_e(q)=\tfrac12 w_e q^2$，则 $p_e^{\text{queue}}(t)=w_e q_e(t)$。

在离散时间窗/控制周期 $t$ 内，我们考虑瞬时优化：
\[
\max_{x(t)}\ \sum_{c,e} \lambda_c\,u_{c,e}\,x_{c,e}(t)\; -\; \sum_e p_e^{\text{queue}}(t)\,\lambda_e(x(t))
\quad\text{s.t.}\quad Ax(t)\le b,\; x(t)\in\Delta(\mathcal A).
  ag{DPP}
\]
其中 $\lambda_e(x(t))=\sum_c \lambda_c x_{c,e}(t)$。

直观上：当某专家队列很长时，$p_e^{\text{queue}}$ 上升，系统会把 token 从该专家“价格挤出”，这正是 backpressure 稳定性的负反馈来源。

> 说明：若你希望保留一个“纯稳态”的静态规划，也可以将 $q$ 看作稳态在边际处的内生变量，并把 $\phi(q)$ 写入目标；但 OR 社区在在线控制问题里更常用 DPP/MaxWeight 来直接把 $Q(t)$ 进入决策。

（参考基准）若忽略拥塞项（即 $p_e^{\text{queue}}\equiv 0$），则退化为之前的纯效用 LP：
\[
\begin{aligned}
\max_{x} \quad & \sum_{c\in\mathcal C}\sum_{e\in\mathcal E} \lambda_c\,u_{c,e}\,x_{c,e}\\
\text{s.t.}\quad
& \sum_{e\in\mathcal E} x_{c,e}=1,\quad \forall c,\\
& x_{c,e}\ge 0,\quad x_{c,e}=0\ \text{if }e\notin\mathcal A_c,\\
& A x\le b.
\end{aligned}
\tag{LP}
\]

这是标准的“带容量约束的静态 routing LP”。

### 3.2 对偶问题与 shadow prices

令 $\pi\ge 0$ 为容量约束 $Ax\le b$ 的对偶变量（向量）；它可分解为（资源稀缺影子价格）：

- 专家容量影子价格：$p_e$（对应 $\lambda_e\le\mu_e$）；
- 链路容量影子价格：$\gamma_\ell$（对应 $\mathrm{Load}_\ell\le B_\ell$）。

KKT/互补松弛告诉我们：

- 若某资源约束紧（瓶颈），其影子价格 $>0$；
- 若约束松弛，则影子价格为 0。

更重要的是**最优路由的结构**：对任意类别 $c$，只有在“调整后收益”最大的专家才会被分配质量：
\[
 e\in \arg\max_{e\in\mathcal A_c}\Big\{u_{c,e}-\underbrace{p_e^{\text{queue}}(t)}_{\text{拥塞影子价格}}-\underbrace{\langle \pi, a_{c,e}\rangle}_{\text{资源机会成本}}\Big\},
\]
其中 $a_{c,e}$ 是“选择 $e$ 的资源消耗向量”（含 compute/comm）。

这正是我们要的 **pricing mechanism/index policy 形式**：

> **AdjustedScore**$(c,e)=u_{c,e}-p_e^{\text{queue}}(t)-\sum_{\ell}\gamma_\ell a^{(\ell)}_{c,e}$。

这比 v1.1 的“抽象 $\mathcal U(X)$”更符合 OR 审稿人的口味：价格与约束一一对应。

### 3.3 唯一性：熵正则化（可选，但建议保留）

LP 的最优解可能不唯一，导致路由在等价最优解间跳变（工程上表现为抖动）。

因此我们采用 OR 中非常常见的做法：对（DPP）或其拉格朗日松弛加入一个小的熵正则（或二次正则）得到严格凹目标：
\[
\max_x \ \sum_{c,e} \lambda_c u_{c,e}x_{c,e} - \frac{1}{\eta}\sum_c\lambda_c\sum_e x_{c,e}\log x_{c,e}
\quad\text{s.t. 同 }(LP).
\tag{RLP}
\]
此时 primal 最优解 $x^*$ 唯一，对偶最优解 $\pi^*$ 在标准 Slater 条件下也唯一，因而得到唯一的平滑路由（softmax 结构）。

---

## 4. 随机网络与流体极限：写得“足够可证明”且不喧宾夺主

OR/MS 顶刊通常不要求你在正文给出所有技术细节，但需要一个可引用、假设清楚的“定理式陈述”。

### 4.1 随机网络定义（与 LP 一致）

定义队列 $Q_e(t)$ 满足
\[
Q_e(t)=Q_e(0)+A_e(t)-S_e(t),
\]
其中 $A_e(t)=\sum_c A_{c,e}(t)$，$A_{c,e}(t)$ 与策略 $\pi(t)$ 一致。

服务过程使用时间变更：
\[
S_e(t)=N_e\Big(\mu_e\int_0^t \mathbf 1\{Q_e(s)>0\}\,ds\Big).
\]

### 4.2 流体缩放与极限

令 $\bar Q^{(n)}(t)=Q^{(n)}(nt)/n$。在 FSLLN（到达/服务）与工作守恒下，存在流体极限 $q(t)$ 满足（a.e.）
\[
\dot q_e(t)=\lambda_e(x(t)) - \mu_e\mathbf 1\{q_e(t)>0\},
\]
其中 $x(t)$ 是策略诱导的瞬时分配。

在采用（DPP）/MaxWeight 类策略时，$x(t)$ 由 $q(t)$ 反馈决定，因此整个系统是一个闭环控制的流体模型。

### 4.3 反射形式与存在唯一性

为避免 $\mathbf 1\{q>0\}$ 的边界不光滑争议，我们采用 Skorokhod 反射映射：存在非减过程 $y_e(t)$ 使
\[
q_e(t)=q_e(0)+\int_0^t (\lambda_e(x(s)) - \mu_e)\,ds + y_e(t),
\]
且 $q_e(t)\ge 0$、$\int_0^t q_e(s)\,dy_e(s)=0$。在 $x(t)$ 可测且分段 Lipschitz 的条件下，解存在且唯一。

> 这部分在 v1.1 里有，但 v1.1.2 将其与 LP 主线对齐：$\lambda_e(x)$ 是同一个对象。

---

## 5. 在线可落地策略：Primal–Dual 价格更新 + Price-aware routing

### 5.1 在线 primal–dual（经典 OR 范式）

我们目标是在线逼近 $\pi^*$（对偶最优影子价格）。使用投影次梯度更新：
\[
\pi(t+1)=\big[\pi(t)+\alpha\,(A x(t)-b)\big]_+.
\tag{PD}
\]
其中 $Ax(t)-b$ 是“资源超载量”的观测/估计。

分解后得到可解释的更新（资源影子价格）：

- 专家价格：$p_e(t+1)=[p_e(t)+\alpha(\widehat{\lambda}_e(t)-\mu_e)]_+$  
- 链路价格：$\gamma_\ell(t+1)=[\gamma_\ell(t)+\alpha(\widehat{\mathrm{Load}}_\ell(t)-B_\ell)]_+$

这些量在工程上都能从 runtime 统计直接获得：专家队列长度/吞吐、链路实际带宽、A2A 时间等。

此外，拥塞影子价格来自队列状态（backpressure）：
\[
p_e^{\text{queue}}(t)=\phi'_e(\widehat q_e(t))\approx \phi'_e\Big(\frac{\widehat Q_e(t)}{n}\Big),
\]
在工程上最常见的实现就是直接用队列长度/等待时间的单调函数作为 price（例如线性 $p_e^{\text{queue}}\propto Q_e$ 或 $\propto$ moving-average latency）。

### 5.2 实时路由（只需改 score，工程非常轻量）

当一个 token i（类别 c）到达且 gating 给出 logits $S_{i,e}$ 时，路由只需做：
\[
 e^*(i)\in\arg\max_{e\in\mathcal A_c}\Big\{S_{i,e} - p_e^{\text{queue}}(t) - \sum_{\ell\in\mathcal L}\gamma_\ell(t)\,a^{(\ell)}_{c,e}\Big\}.
\tag{Route}
\]
这就是你最初想要的“在 gating score 中减去资源成本”，其中**资源成本**必须包含最关键的拥塞项 $p^{\text{queue}}$，它直接缓解 congestion/load skew，并对 tail latency 给出负反馈控制。

### 5.3 通信可扩展性：JSQ($d$) / 稀疏观测

在 all-to-all 成本高时，不必查询所有 $e$：只在 $\mathcal A_c$ 内抽样 $d$ 个候选专家估计其价格/队列（power-of-d）。

---

## 6. 工程可落地性检查（审稿人会问的那几句）

**Q1：你需要哪些在线观测量？**  

- $\widehat{\lambda}_e(t)$：专家到达率（可以用滑动窗口计数）；
- $\widehat{\mathrm{Load}}_\ell(t)$：链路负载/通信时间；
- $\mu_e, B_\ell$：可离线 profile 或用控制器估计；
- $S_{i,e}$：gating logits（模型已有）。

**Q2：策略计算开销？**  

- 路由层：对每个 token，只做 Top-$k$ 内的加减和 argmax（或 softmax），复杂度 $O(k)$。
- 价格更新：每个时间窗更新一次 $p_e$ 与 $\gamma_\ell$，复杂度 $O(E+|\mathcal L|)$，可在 control plane 完成。

**Q3：与 vLLM/DeepSpeed 兼容吗？**  
是的：

- vLLM/continuous batching 只影响“到达聚合与时间窗”；
- DeepSpeed EP 的 A2A 统计可以直接产生 $\widehat{\mathrm{Load}}_\ell$；
- routing score 的修改是框架层最小侵入。

---

## 7. 结论：为何这更符合 OR/MS 的传统

- 主线是 **稳态 LP / KKT / shadow prices**（OR 标志性叙事）。
- 在线算法是 **primal–dual / dual decomposition**（经典且可证明）。
- 工程落地是 **“只改 score + 维护价格”**（非常轻量）。
- 通信感知通过链路影子价格 $\gamma_\ell$ 自然进入（与 idea B 高度一致）。

---

## 参考（建议论文中规范化）

- Dai, J. G. (1995). _On positive Harris recurrence of multiclass queueing networks: A unified approach via fluid limit models._
- Chen & Yao. _Fundamentals of Queueing Networks._
- 以及 vLLM / DeepSpeed-MoE / Metro 等系统论文（动机与参数量级来自系统事实）。
