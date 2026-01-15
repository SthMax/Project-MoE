# Fluid MoE（version 1.1.3）：Queue-based Shadow Price（Backpressure）+ Resource Dual Pricing

> 这版是对 v1.1.2 的关键修正：把“拥塞/负载偏斜/尾延迟”作为一等公民。核心机制是 **队列长度 $Q_e(t)$（或流体 $q_e(t)$）就是拥塞影子价格**，并与计算/通信等资源的对偶价格一起进入 AdjustedScore。

## 0. 修正叙事：为什么必须把队列引入 price

如果目标是缓解 congestion、load skew、tail latency，那么仅做

- maximize gating utility
- capacity penalty（来自 $Ax\le b$ 的对偶）

是不够的：它对“瞬时拥塞状态”不敏感，无法形成强负反馈去压制队列增长。

排队网络/随机处理网络的经典结论是：

- 想要稳定（stability）并控制延迟，需要策略显式依赖队列状态；
- **MaxWeight / Backpressure / Drift-plus-Penalty** 体系中，队列长度就是“拥塞的边际代价”（shadow price）。

因此本版明确把价格拆成两部分：

- 拥塞价（queue price）：$p_e^{\text{queue}}(t)=\phi'_e(q_e(t))$，最常用选 $\phi_e(q)=\tfrac12 w_e q^2$，则 $p_e^{\text{queue}}=w_e q_e$；
- 资源价（dual price）：$\pi(t)$（可分解为专家容量价 $p_e^{\text{cap}}$、链路容量价 $\gamma_\ell$ 等）。

最终路由使用

$$
\text{AdjustedScore}(c,e;t)=S_{c,e}(t)-p_e^{\text{queue}}(t)-\langle \pi(t),a_{c,e}\rangle.
$$

## 1. 随机队列网络（stochastic processing network）与流体缩放

### 1.1 系统元素

- 专家集合 $\mathcal E$；类别集合 $\mathcal C$。
- 到达：每类 $c$ 的到达过程 $A_c(t)$，满足 FLLN：$A_c(t)/t\to \lambda_c$。
- 路由决策：对每个到达 token（或批内 token）选择专家 $e\in\mathcal A_c$（允许 top-$k$ 分流）。
- 服务：专家 $e$ 的服务过程 $S_e(\cdot)$，速率参数 $\mu_e$。

### 1.2 队列动力学（随机）

令 $Q_e(t)$ 为专家待处理 token 数（或等价的微批计数）。则

$$
Q_e(t+1)=\big[Q_e(t)+R_e(t)-D_e(t)\big]^+,
$$

其中

- $R_e(t)=\sum_c R_{c,e}(t)$ 为周期 $t$ 分配到专家 $e$ 的 token 数；
- $D_e(t)$ 为周期 $t$ 完成服务的 token 数。

### 1.3 流体缩放与极限（要点式可证明表述）

设置缩放参数 $n\to\infty$（如并行度或系统规模），定义

$$
q_e^n(t)=\frac{1}{n}Q_e^n(nt),\quad r_{c,e}^n(t)=\frac{1}{n}R_{c,e}^n(nt).
$$

在常见假设（到达 FLLN、服务可表示为 time-change 的更新过程并满足 FSLLN、初值有界）下，可提取子序列使得

$$
(q^n(\cdot),r^n(\cdot))\Rightarrow (q(\cdot),r(\cdot)),
$$

并满足（几乎处处）

$$
\dot q_e(t)=\sum_c \lambda_c x_{c,e}(t)-\mu_e\mathbf 1\{q_e(t)>0\},\quad q_e(t)\ge 0.
$$

其中 $x(t)$ 为策略诱导的瞬时分配比例。

## 2. 资源约束与对偶价格（capacity shadow price）

我们仍保留 OR/MS 可落地的资源约束矩阵形式：

- 决策变量：$x_{c,e}(t)\in[0,1]$，且 $\sum_{e\in\mathcal A_c}x_{c,e}(t)=1$。
- 资源消耗向量：$a_{c,e}$（计算、KV-cache、链路等）。
- 容量：$Ax(t)\le b$。

其对偶变量 $\pi(t)\ge 0$ 给出资源稀缺影子价格。

## 3. Drift-plus-Penalty：queue price 的来源（核心）

令 Lyapunov 函数 $L(Q)=\tfrac12\sum_e w_e Q_e^2$。在边界条件满足下，标准漂移分析给出

$$
\Delta(Q(t)):=\mathbb E[L(Q(t+1))-L(Q(t))\mid Q(t)]
$$

可被以下“线性化”项控制：

$$
\sum_e w_e Q_e(t)\,\mathbb E[R_e(t)-D_e(t)\mid Q(t)].
$$

因此，在每个周期选择 $x(t)$ 去最小化漂移（并同时追求效用）得到 Drift-plus-Penalty：

$$
\max_{x(t)}\ \sum_{c,e}\lambda_c u_{c,e}x_{c,e}(t)
-\sum_e w_e Q_e(t)\,\lambda_e(x(t))
\quad\text{s.t.}\quad Ax(t)\le b.
$$

这直接导出：**$w_e Q_e(t)$ 就是专家 $e$ 的拥塞影子价格**。

> 与 v1.1.2 的衔接：资源对偶价格 $\pi$ 控制“可行域/稀缺资源”；而 $Q$ 控制“系统状态/拥塞”。两者一起进入 AdjustedScore 才能同时解决 tail latency 与通信/算力约束。

## 4. 路由规则（部署时真正要改的那一行）

对一个 token（类别 $c$）及其候选专家 $e\in\mathcal A_c$，令 $S_{c,e}(t)$ 为 gating score（logit 或可校准分数）。在线路由：

$$
 e^*(t)\in\arg\max_{e\in\mathcal A_c}\Big\{S_{c,e}(t)
-\underbrace{w_e Q_e(t)}_{\text{queue/backpressure}}
-\underbrace{\langle \pi(t),a_{c,e}\rangle}_{\text{resource/comm price}}\Big\}.
$$

若需要 soft routing（top-$k$ 或保持可微），用 softmax / entropy regularization：

$$
 x_{c,e}(t)\propto \exp\Big(\eta\,[S_{c,e}(t)-w_e Q_e(t)-\langle\pi(t),a_{c,e}\rangle]\Big).
$$

## 5. 在线更新：资源价格用 primal–dual，拥塞价格用队列

- 队列价：直接由观测得到 $Q_e(t)$（或 latency/排队时间的单调函数）。
- 资源价：对偶变量做投影次梯度：

$$
\pi(t+1)=\big[\pi(t)+\beta\,(A\bar x(t)-b)\big]_+.
$$

其中 $\bar x(t)$ 为周期内实际分配的样本平均。

## 6. 你最关心的“可证明性”该怎么写（建议写法）

论文中通常分两层写：

1) **存在性/唯一性（流体方程）**：给定一个可测策略映射 $x(t)=\Psi(q(t),\text{obs})$，在熵正则使得 $\Psi$ Lipschitz 的条件下，反射 ODE/Skorokhod 问题给出解的存在唯一。
2) **随机→流体收敛**：在 FSLLN 下，用紧性 + 连续映射定理/Skorokhod 表示，得到任何子序列都有流体极限，且极限满足上述闭环 ODE。
3) **稳定性/最优性（DPP）**：引用 MaxWeight/Drift-plus-Penalty 的标准结论：若到达率在容量区域内部，则队列稳定；并且在权衡参数 $V$ 下得到 $O(1/V)$ 的效用差距与 $O(V)$ 的平均队列界。

## 7. 与原始 idea B 的对齐检查（requirements coverage）

- queue length shadow price：已在第 3–5 节作为核心机制写清楚。
- 通信感知 token routing：通过 $a_{c,e}$ 与链路对偶 $\gamma_\ell\subset\pi$ 保留。
- 简化流体模型→稳态 LP：保留了资源约束的 LP/KKT 视角，但在线控制以 DPP 为主线。
- 可落地：实现只需（1）专家队列/延迟观测，（2）链路/算力统计，（3）改 AdjustedScore 一行。
