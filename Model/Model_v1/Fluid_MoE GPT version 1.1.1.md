# PriceMoE：面向大规模 MoE 推理的随机处理网络—流体极限—影子价格路由（MS 版本，v1.1）

> 本文档是 **Management Science（MS）风格**的数学模型版本：强调可验证的随机模型定义、流体极限（fluid limit）收敛、ODE 的存在唯一性、稳态（equilibrium）刻画与影子价格/原对偶动态的可证明性质。  
> CS 版本（ICML/NeurIPS）可在此基础上弱化证明、强化系统实现与实验；但本文件聚焦论文级数学严谨性。

---

## 0. 记号与建模目标（Contract）

### 0.1 系统对象

- 专家集合：$\mathcal E = \{1,\dots,E\}$。
- GPU/节点集合：$\mathcal K = \{1,\dots,K\}$。
- 专家放置：$L(e)\in\mathcal K$ 表示专家 $e$ 所在 GPU（或节点/设备组）。
- token 类型/类别（可选）：$\mathcal C=\{1,\dots,C\}$。用于建模不同语义类到达率与 gating 分布差异；如不需要，可令 $C=1$。

### 0.2 关键输出

我们希望构建如下三层结果：

1. **随机模型（stochastic processing network）**：给出到达、路由、服务、队列演化与资源约束的严格随机过程定义。
2. **流体缩放 + 收敛**：证明在规模参数 $n\to\infty$ 下，缩放后的队列过程收敛到确定性轨迹 $q(t)$，满足一组 ODE/微分包含（differential inclusion）。
3. **均衡与影子价格**：给出稳态（equilibrium）定义；在适当正则化下证明均衡存在唯一；并给出影子价格（对偶变量）驱动的路由与其稳定/收敛性质。

---

## 1. 系统约束的“可发表”建模：Compute / Memory / Comm / Quality

MoE 推理的真实瓶颈随阶段变化（Prefill vs Decode），因此我们采用“多约束资源系统”的统一抽象，并在后续流体模型中用 **容量域（capacity region）** 统一处理。

### 1.1 计算容量约束（Compute）

对每个专家 $e$，假设单位时间可完成 token 的服务速率为 $\mu_e>0$。在连续时间模型下，累计服务过程满足
$$
S_e(t)\le \mu_e\,T_e(t),\qquad T_e(t)=\int_0^t \mathbf 1\{Q_e(s)>0\}\,ds
$$
其中 $T_e(t)$ 是忙期累计时间。

> 说明：这等价于传统排队系统的工作守恒 (work-conserving) 约束。随机服务时间分布可一般化到 $M/G/1$ 类，但流体极限只需要 FSLLN 条件。

### 1.2 显存容量约束（KV cache / blocks）

令 $M_k$ 为 GPU $k$ 的可用 KV/显存容量（扣除权重、框架开销），单位为 block 或字节。令 $b(r)$ 为请求 $r$ 每个 token 追加的 KV 占用。
可用抽象为：并发请求集合 $\mathcal R(t)$ 的内存占用不得超过
$$
\sum_{r\in\mathcal R(t)} b(r)\le M_k,\qquad \forall k.
$$
该约束影响可服务请求集合与预占策略；在本文中，它通过“到达可受理域”和“可用服务能力”进入容量域 $\mathcal U$（见 2.3）。

### 1.3 显存带宽/激活专家集约束（Decode 阶段，Metro 视角）

Decode 期间的关键瓶颈常来自 HBM 带宽：每激活一个专家需要读取权重或产生额外 IO。令

- $y_{e,k}(t)\in\{0,1\}$：时刻 $t$ GPU $k$ 是否“激活”专家 $e$（该 GPU 本轮 decode 批次内至少有一个 token 需要该专家）。
- $w_e$：专家 $e$ 的权重/带宽足迹。

则带宽预算可抽象为
$$
\sum_{e\in\mathcal E} w_e\,y_{e,k}(t)\le B_k,\qquad \forall k.
$$
这是非凸/组合约束（因 $y$ 是指示变量）。MS 版本处理方式：将该项建模为 **切换成本/稀疏激活惩罚**（见 4.4），并在流体层面引入迟滞（hysteresis）以获得可证明的稳定与收敛。

### 1.4 通信约束（All-to-All / 跨节点）

将跨 GPU/节点的 token 交换抽象为线性链路容量：对任意链路/层级 $\ell$，其单位时间可传输的数据量有限。
在模型层面，我们不直接刻画拓扑细节，而是将通信成本写入路由的即时成本（penalty）或写入可行域 $\mathcal U$：
$$
\text{CommCost}(e\mid \text{token})=\gamma\,\mathbf 1\{L(e)\neq \text{local}\}\quad\text{或}\quad \gamma\,d(L(e),\text{local}).
$$

### 1.5 质量约束（Top-$k$ / bounded degradation）

对每个到达 token $i$，gating 给出分数向量 $S_i=(S_{i,e})_{e\in\mathcal E}$。

- Top-$k$ 可行集：$\mathcal A_i=\text{Top}_k(S_i)$。
- 有界降级：允许扩展为
$$
\mathcal A_i(\varepsilon)=\big\{e: S_{i,e}\ge (1-\varepsilon)\max_{e'} S_{i,e'}\big\}.
$$
路由决策必须满足 $e\in\mathcal A_i$ 或 $\mathcal A_i(\varepsilon)$。

---

## 2. 随机网络模型（Stochastic MoE Processing Network）

我们采用连续时间、右连续左极限（càdlàg）过程，并用“计数过程 + 时间变更”表达，便于应用 FSLLN 得到流体极限。

### 2.1 到达过程

对每个类别 $c\in\mathcal C$，令外部到达 $A_c(t)$ 是更新过程（可取泊松或更一般的更新过程），满足强大数定律（FSLLN）：存在 $\lambda_c(t)$（可为常数或分段连续函数），使得
$$
\frac{1}{n}A_c^{(n)}(nt)\Rightarrow \int_0^t \lambda_c(s)\,ds\qquad \text{u.o.c.}
$$
其中 $n$ 是规模参数（含义见 3.1），$\Rightarrow$ 表示一致收敛于紧集（uniform on compact sets）。

> MS 写作习惯：这里直接把到达过程置于“更新过程族”并假设 FSLLN/功能强大数定律，后续引用 Dai (1995) / Chen & Yao / Robert 等 SPN 流体极限理论即可。

### 2.2 路由与累计分配过程

每个到达 token（类别 $c$）携带随机 gating 分数 $S$（向量）。给定系统状态 $X(t)$（含队列、激活集等），路由策略 $\pi$ 产生分配概率
$$
P_\pi\big(e\mid c,S,X(t)\big)\in \Delta(\mathcal A(S))
$$
并诱导“到达—专家”累计计数过程 $A_{c,e}(t)$，满足
$$
\sum_{e\in\mathcal E} A_{c,e}(t)=A_c(t),\quad A_{c,e}(t)\text{ 为非减整数值过程}.
$$
定义专家 $e$ 的总到达为 $A_e(t)=\sum_c A_{c,e}(t)$。

### 2.3 服务与资源可行域

为统一处理 compute/memory/comm 等约束，我们定义每个时刻的**可行服务率集合**（capacity set）
$$
\mathcal U(X)=\Big\{u\in\mathbb R_+^E: \; 0\le u_e\le 1,\; \text{以及 }u\text{ 满足 GPU/带宽/通信等资源约束}\Big\}.
$$
给定 $u(t)\in\mathcal U(X(t))$，服务累计过程用“潜在服务过程 + 时间变更”表示：
$$
S_e(t)=N_e\Big(\int_0^t \mu_e u_e(s)\,\mathbf 1\{Q_e(s)>0\}\,ds\Big),
$$
其中 $N_e(\cdot)$ 可取为单位速率泊松过程或更一般的更新过程，只需满足 FSLLN。

### 2.4 队列动态

对每个专家 $e$：
$$
Q_e(t)=Q_e(0)+A_e(t)-S_e(t),\qquad Q_e(t)\in\mathbb Z_{\ge 0}.
$$
令状态向量 $X(t)$ 包含 $Q(t)$ 以及必要的附加状态（如激活集、setup 状态等）。

---

## 3. 规模化与流体缩放（Fluid Scaling）

### 3.1 规模参数 $n$ 的含义与两种渐近机制

在 MoE 推理系统中，$n$ 可对应：GPU 数、专家副本数、或“总可服务能力”的线性缩放。

- **流体极限（first-order）**：研究 $n\to\infty$ 下平均行为，得到确定性 ODE。
- **QED/Halfin–Whitt（二阶）**：为刻画尾延迟与高利用率并存，设到达率与容量满足
$$
\lambda_n \approx n\mu - \beta\mu\sqrt n,
$$
或等价的服务器数 $N_n\approx \lambda_n/\mu + \beta\sqrt{\lambda_n/\mu}$。

本文 MS 版本的主线是 **流体极限 + 均衡与最优化**；QED 作为“部署建议/扩容法则”给出，并可在扩展部分引入扩散极限（diffusion limit）。

### 3.2 流体缩放定义

对任意计数过程 $Z(t)$（如 $Q,A,S$），定义缩放过程
$$
\bar Z^{(n)}(t)=\frac{1}{n}Z^{(n)}(nt).
$$
在常规正则条件（FSLLN、紧性、工作守恒）下，可证明序列 $\{\bar X^{(n)}\}$ 相对紧，并存在子序列收敛到 $x(\cdot)$。

### 3.3 流体极限方程（ODE / 微分包含）

令极限轨迹为 $q(t)$、$a(t)$、$s(t)$。则对几乎处处 (a.e.) 的 $t$：
$$
q_e(t)=q_e(0)+a_e(t)-s_e(t),\qquad \dot q_e(t)=\dot a_e(t)-\dot s_e(t).
$$
并且
$$
\dot s_e(t)=\mu_e u_e(t)\,\mathbf 1\{q_e(t)>0\},\qquad u(t)\in\mathcal U(x(t)).
$$
到达率由路由策略诱导：存在可测函数 $r_{c,e}(t)$ 表示类别 $c$ 到专家 $e$ 的流体分配比例，使
$$
\dot a_{c,e}(t)=\lambda_c(t)\,r_{c,e}(t),\qquad \sum_e r_{c,e}(t)=1,\quad r_{c,e}(t)=0\ \text{若 }e\notin\mathcal A(\cdot).
$$
因此
$$
\dot q_e(t)=\sum_{c\in\mathcal C}\lambda_c(t)\,r_{c,e}(t)-\mu_e u_e(t)\,\mathbf 1\{q_e(t)>0\}.
$$

> 重要：当引入激活集/切换（setup）状态时，右端可能变为非光滑/非凸，严格形式往往是 **微分包含**。本文通过“熵正则化 + 迟滞”使右端变为分段 Lipschitz，从而可恢复经典 ODE 存在唯一性（见 5）。

### 3.4 收敛定理（可引用的标准陈述）

**定理（流体极限存在）**：在 2.1–2.4 的 FSLLN/工作守恒/可行域闭紧等假设下，序列 $\{\bar X^{(n)}\}$ 相对紧；任意收敛子序列的极限 $x(\cdot)$ 满足 3.3 的流体方程（在 a.e. 意义下）。

**进一步（稳定性传递）**：若对所有初值 $q(0)$，流体模型满足 $\|q(t)\|\to 0$，则原随机网络正再生（positive Harris recurrent）并稳定。（Dai 1995 型结果）

---

## 4. 影子价格、NUM 与正则化：从静态最优化到在线路由

### 4.1 稳态的容量域与“社会最优”NUM

在稳态/常到达率 $\lambda_c$ 下，令 $\Lambda_e=\sum_c \lambda_c r_{c,e}$ 为流体给专家 $e$ 的总到达。
若系统可稳定，则存在 $u\in\mathcal U$ 使
$$
\Lambda_e\le \mu_e u_e,\qquad \forall e.
$$
定义稳定容量域
$$
\mathcal C:=\Big\{\Lambda\in\mathbb R_+^E: \exists u\in\mathcal U\ \text{s.t.}\ \Lambda_e\le \mu_e u_e\ \forall e\Big\}.
$$
将“质量”抽象为效用 $U_{c,e}$（可由 gating 分数的期望或分位数定义），得到 NUM：
$$
\max_{r}\ \sum_{c}\sum_{e} \lambda_c\,U_{c,e}\,r_{c,e}\quad\text{s.t.}\quad \Lambda(r)\in \mathcal C,\ \ r_c\in\Delta(\mathcal A_c).
$$

### 4.2 对偶与影子价格

对容量约束引入对偶变量（影子价格）$p=(p_e)\ge 0$ 或更一般的 GPU 价格 $\nu_k$。
形式上，可将约束写为 $G\Lambda\le C$（$G$ 汇总 compute/memory/comm 资源消耗），其拉格朗日函数
$$
\mathcal L(r,\nu)=\sum_{c,e}\lambda_c U_{c,e}r_{c,e}-\nu^\top(G\Lambda(r)-C).
$$

在该结构下，给定价格 $\nu$，最优路由（对每个类别/每个 token 的可行集）表现为
$$
\arg\max_{e\in\mathcal A} \big(U_{c,e}-\langle\nu, g_e\rangle\big),
$$
其中 $g_e$ 为选择专家 $e$ 所消耗的资源向量。

### 4.3 严格凸正则化与唯一性（Softmax / Logit 选择）

为避免线性规划多解导致的 bang-bang 抖动，并获得可微对偶函数，我们对路由加入熵正则：
$$
\max_{r}\ \sum_{c,e}\lambda_c U_{c,e}r_{c,e}-\frac{1}{\eta}\sum_c\lambda_c\sum_e r_{c,e}\log r_{c,e}
\quad\text{s.t. }\Lambda(r)\in\mathcal C,\ r_c\in\Delta(\mathcal A_c).
$$
该目标对 $r$ **严格凹**（在单纯形内部），从而最优 $r^*$ 唯一；并导出 logit/softmax 形式：
$$
 r^*_{c,e}(\nu) = \frac{\exp\big(\eta(U_{c,e}-\langle\nu,g_e\rangle)\big)}{\sum_{e'\in\mathcal A_c}\exp\big(\eta(U_{c,e'}-\langle\nu,g_{e'}\rangle)\big)}.
$$

### 4.4 Setup time / 激活集的非凸性与迟滞（Hysteresis）

激活一个“当前未激活”专家会产生固定 setup/IO 成本，导致即时成本是阶跃型（非凸）。MS 处理方式：

- 引入二态激活变量 $z_{e,k}(t)\in\{0,1\}$.
- 激活切换惩罚 $H_{e,k}>0$.
- 路由采用迟滞规则：仅当“切换收益”超过阈值才切换。

一个可操作且可分析的形式：对 token $i$（其可行集 $\mathcal A_i$），在时刻 $t$ 选择专家
$$
 e^*(t)\in \arg\max_{e\in\mathcal A_i}\Big\{S_{i,e}-p_e(t)-H\cdot \mathbf 1\{z_{e,L(e)}(t^-)=0\}\Big\}.
$$
其中 $H$ 与 setup 成本同阶。迟滞使激活状态变化处于“慢时间尺度”，从而可证明系统避免 thrashing，并在流体层面形成分段光滑动力学。

---

## 5. 流体系统的存在唯一性（Existence & Uniqueness）

### 5.1 采用“正则化路由”确保 Lipschitz

若路由使用 4.3 的 softmax 形式（并将迟滞写成分段常数惩罚），则右端函数
$$
F(q,t)=\lambda(t)R(q,t)-\mu\,U(q,t)\odot \mathbf 1\{q>0\}
$$
在任意紧集上对 $q$ 是 **分段 Lipschitz** 的：

- $q\mapsto p(q)$（价格映射）若取 $p_e=\alpha q_e$ 或更一般的单调 Lipschitz 映射，即可；
- softmax 对输入是光滑 Lipschitz；
- 指示项 $\mathbf 1\{q>0\}$ 造成边界不光滑，但这是标准排队流体模型结构，可用“Skorokhod 反射”或 a.e. 微分形式处理。

### 5.2 ODE/反射微分方程的唯一解

将队列非负性写为反射形式：存在非减过程 $y_e(t)$（推力项），使
$$
q_e(t)=q_e(0)+\int_0^t \Big(\sum_c \lambda_c(s)r_{c,e}(s)-\mu_e u_e(s)\Big)ds + y_e(t),
$$
并满足互补条件
$$
q_e(t)\ge 0,\quad y_e(0)=0,\quad \int_0^t q_e(s)\,dy_e(s)=0.
$$
在 $r,u$ 为可测且对 $q$ 局部 Lipschitz 的条件下，反射常微分方程在正交象限上存在唯一解（标准 Skorokhod 映射性质）。

> 这一步是 MS 写作里“把存在唯一性说清楚”的关键：不直接靠直觉写 $\mathbf 1(q>0)$，而用反射项 $y(t)$ 处理边界。

---

## 6. 稳态（Equilibrium）定义、存在与唯一性

### 6.1 流体均衡的定义

在常到达率 $\lambda$ 下，称 $q^*$ 为流体均衡，若存在常数控制 $r^*,u^*$ 使对所有 $e$：
$$
0=\dot q_e(t)=\sum_c \lambda_c r^*_{c,e}-\mu_e u^*_e\cdot \mathbf 1\{q^*_e>0\},
$$
且 $u^*\in\mathcal U(x^*)$、$r^*$ 满足质量可行集约束。

### 6.2 均衡存在性

若 $\lambda$ 位于容量域内部：$\Lambda\in\mathrm{int}(\mathcal C)$，则存在稳定策略使随机系统稳定；因此流体模型存在不动点/吸引集。
在正则化 NUM（4.3）下，最优解 $r^*$ 存在（紧可行域 + 连续目标）。

### 6.3 均衡唯一性（严格凹/凸保证）

在熵正则化下：

- 原始目标对 $r$ 严格凹 $\Rightarrow$ $r^*$ 唯一；
- 对偶目标严格凸/可微 $\Rightarrow$ 价格 $\nu^*$ 唯一（在常见 Slater 条件下）；
- 因此诱导的稳态分配与“等价的均衡”唯一。

> 解释：若不加正则，线性目标导致多最优点，路由可能在等价最优点间跳变；这在 MoE 中对应“路由震荡/专家 thrashing”。熵正则化提供 MS 论文里非常常用的“唯一性修复”。

---

## 7. 稳定性与影子价格在线更新（Primal–Dual / Backpressure）

### 7.1 Lyapunov 漂移与吞吐量最优（Throughput optimality）

令 Lyapunov 函数 $V(q)=\tfrac12\sum_e q_e^2$。
对 MaxWeight/Backpressure 类策略（价格与队列单调相关，如 $p_e=\alpha q_e$），在容量域内部可证明存在 $\epsilon,B>0$ 使
$$
\mathbb E[V(Q(t+1))-V(Q(t))\mid Q(t)]\le -\epsilon\sum_e Q_e(t)+B.
$$
从而得到正再生与稳态尾界。

### 7.2 原对偶更新的连续时间近似

将对偶变量视为影子价格，采用投影梯度/镜像下降：
$$
\dot\nu(t)=\big[\nabla d(\nu(t))\big]_+\quad\text{或}\quad \dot\nu(t)=\big[G\Lambda(r^*(\nu(t)))-C\big]_+.
$$
在 4.3 的正则化下，$r^*(\nu)$ 光滑，使得该动力系统在标准条件下收敛到唯一 $\nu^*$。

### 7.3 稀疏信息与 JSQ($d$) 近似（通信可扩展性）

为避免 All-to-All 查询所有队列，采用 power-of-$d$ choices：每个 token 只抽样 $d_n$ 个候选专家估计价格。
经典结果表明若 $d_n\gg\log n$，性能逼近 JSQ（全信息最短队列），而通信成本从 $O(n)$ 降至 $O(\log n)$ 甚至常数级。

---

## 8. 假设清单（MS 写作版）

为保证“可证明”，我们明确列出足以支撑 3–7 节结论的假设。

**A1（FSLLN 到达）**：$A_c^{(n)}(t)$ 满足功能强大数定律，极限到达率 $\lambda_c(t)$ 分段连续且有界。  
**A2（FSLLN 服务）**：潜在服务过程 $N_e^{(n)}(t)$ 满足 FSLLN，且独立性/弱相关性足以保证工作守恒下的时间变更极限。  
**A3（容量域闭紧）**：对每个状态 $X$，$\mathcal U(X)$ 非空、凸、紧；且 $X\mapsto \mathcal U(X)$ 上半连续，保证选择可测性。  
**A4（路由正则性）**：采用熵正则化路由（softmax）与 Lipschitz 价格映射 $p(q)$；迟滞惩罚为分段常数，从而右端分段 Lipschitz。  
**A5（Slater 条件）**：存在严格可行点，使对偶最优存在且（在严格凸/凹下）唯一。  

> 备注：若希望进一步写到“随机模型到流体模型收敛的完整证明”，可在论文附录补充紧性 + 识别极限点（martingale decomposition + FSLLN）。本文件给出 MS 风格的可引用定理形式与假设集合。

---

## 9. 与工程系统的接口（给 CS 版本留口子）

虽然本文件是 MS 版本，但模型刻画已经为工程落地提供直接接口：

- **价格信号**：$p_e(t)=\alpha Q_e(t)$ 或 GPU 级价格 $\nu_k(t)$（可由队列/显存/带宽计数器估计）。
- **可行集**：$\mathcal A_i$ 对应 Top-$k$/bounded degradation gating；工程里可直接取 top-k logits。
- **带宽/激活惩罚**：$H\cdot\mathbf 1\{z_{e,k}=0\}$ 对应 Metro 的“少激活专家集”；工程里可通过 cache 命中/专家驻留控制实现。
- **稀疏查询**：JSQ($d$) 对应仅采样少量候选专家获取队列/价格。

---

## 参考（建议在论文中规范化）

- Dai, J. G. (1995). _On positive Harris recurrence of multiclass queueing networks: A unified approach via fluid limit models._  
- Chen & Yao. _Fundamentals of Queueing Networks._  
- Halfin & Whitt (1981). Many-server heavy-traffic (QED) regime.  
- 以及你在 `Chat/260113 ...` 中列出的 Metro / DeepSpeed-MoE / vLLM 等系统论文（用于约束建模动机）。
