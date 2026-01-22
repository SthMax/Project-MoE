# 260115 Weekly Task: Idea Alignment

我们在一周内需要做的事情：
互相align知识结构，对llm system level物理（或者技术上）进行精确理解。
从而进一步正确导出数学模型上的constraint，从而精准确定objective function以完成对基于vLLM的MoE system存在的如load skew，congestion等问题的优化模型制定。
从而为未来我们下一步想利用这个模型借助fluid model还有shadow price做优化打基础。

**我们之前存在的问题：** 我们之间由于学术背景一个是CS，一个偏OR导致模型设置和物理系统（vLLM）无法对应。之前hongfan的思路是调研全部现有各种系统架构的constraints来制定自己的模型构建思路，但这样类似拿着锄头找地，导致数学模型缺乏实际指导。现在我们的思路是，先双方理解好互相的工作，接着确定物理模型（vLLM），基于物理模型得到对应数学模型及具体constraint设定。

## Hongfan's Task

## 本周目标（Hongfan）

把 vLLM（尤其是 MoE 推理路径）从“代码结构/模块”层面梳理成**可映射到 OR/排队/流体模型的物理过程图**，并产出一份“可用于建模的约束清单 + 可观测量清单”。

### 1) 在 Ampere 上阅读 vLLM 代码并形成系统级理解（优先 MoE 路径）

- 地点/路径：`/home/zhhuang/sthmax/git/vllm`
- 方法：

    - 先用目录级阅读建立模块地图（engine / scheduler / worker / attention backend / kv cache / distributed / model runner / sampling 等）。
    - 再围绕一次 request 的生命周期做“跟踪式阅读”（从请求进入到输出 token 返回）。

- 产出（交付物）：

    1. `vLLM MoE 推理路径时序图`（建议用 Mermaid / 文字也可）：
         - request arrival → batching → prefill → decode steps → routing → expert compute → allreduce/通信（若有）→ token sampling → 返回。
    2. `模块-职责对照表`：列出关键模块文件/类、职责、关键状态（队列/缓存/调度决策点）。

### 2) 抽取“物理系统可观测量”和“可控决策变量”（用于后续建模/优化）

- 可观测量（示例方向）：

    - 每 step 的 batch size、prefill/decode 占比
    - expert activation / routing 分布、top-k 命中、专家负载（tokens/sec 或 FLOPs）
    - queue length / waiting time（scheduler 队列）、prefill 队列 vs decode 队列
    - KV cache 使用量、block/页分配与回收、cache miss/evict
    - 通信时间（all-to-all / allreduce）与 compute 时间分解

- 可控变量（示例方向）：

    - batching 策略（最大 batch、混合 prefill/decode、max tokens per step 等）
    - routing 策略/阈值/重定向（若系统支持）
    - expert 并行/映射（expert → GPU/worker）
    - admission control / priority（QA/interactive vs offline）

- 产出（交付物）：

    - `Observation & Control List`：每个量注明“在哪里拿（代码点/日志点）”“单位/频率”“与模型变量对应关系”。

### 3) 从 vLLM 代码中提炼“建模约束候选集”（约束必须能对上物理来源）

- 把约束分类写清楚（每条都要写：物理来源、数学形式草案、需要的数据/参数）：

    1) 资源约束：GPU compute、HBM、通信带宽/拓扑
    2) 调度/并发约束：一个 step 内能容纳的 token 数、并行 worker 数、pipeline/同步点
    3) 路由/专家容量约束：每个 expert 的 token capacity、top-k/归一化、溢出/丢弃/回退规则
    4) SLA/时延约束：P50/P95 latency、TTFT、TPOT 等指标与队列关系

- 产出（交付物）：

    - `Constraint Candidate Sheet (vLLM-grounded)`：至少 10 条候选约束（可粗略，但要“可追溯到代码/机制”）。

### 4) 和 Zhenghong 的对齐与收敛（本周必须可执行的“对齐事件”）

- 约定 2 次对齐会议（建议 30-45min/次）：

    1) 周中：对齐 vLLM 物理流程图，确认“系统里到底有哪些队列/哪些同步点”。
    2) 周末：把约束候选集与 OR 数学符号对齐，确定下一周要验证/实验的 3-5 条关键约束。

- 会议输入：你产出的时序图 + 观测/控制清单 + 约束候选集。
- 会议输出：共同确认的“版本 0.1 物理模型与变量表”。

### 5) （可选但低风险高收益）把关键观测量落到代码/日志点

- 如果 vLLM/你们的运行脚本里已有 logging hook：补充“打印/导出”指定指标。
- 产出（交付物）：

    - 一个最小可复现实验记录（一次跑的 output + 指标说明），证明上述观测量拿得到。


## Zhenghong's Task

## 本周目标（Zhenghong）

把 OR/流体/排队的建模语言“对齐到 vLLM 的真实机制”，产出一份**建模草案 v0.1**：清晰定义状态变量、控制变量、约束集合、目标函数候选，以及它们在 vLLM 中的物理含义与可测量性。

### 1) 建立统一符号体系与“物理→数学”映射表

- 先立一个最小符号集（后续统一用这套符号交流）：

    - arrival：每域/每类请求到达率 $\lambda_d(t)$ 或 $\lambda_c(t)$
    - service：每资源/每 expert 的服务率 $\mu_e(t)$ / 单 token 成本 $s_e$
    - queue：队列长度 $Q(t)$（可拆 prefill/decode 或按优先级拆）
    - routing：$p_{e|d}(t)$（域 $d$ 到 expert $e$ 的路由概率/占比）
    - capacity：expert 容量/步内 token 预算 $C_e(t)$
    - state：KV cache / memory 状态 $M(t)$（必要时）

- 产出（交付物）：

    - `Symbol & Mapping Table`：每个符号写清“vLLM 对应物理量/日志点/可观测性”。

### 2) 给出 2-3 个“最小可用”的物理模型抽象（并说明取舍）

为了避免一次数学化过深，建议本周先定 2-3 个候选抽象层级：

1) **单队列 + 多 expert 负载模型**：聚合 scheduler 队列，核心看 routing → load skew → congestion。
2) **双队列（prefill/decode）+ step 服务模型**：更贴近 vLLM 的迭代 decode 过程。
3) **按域/按优先级分流的多类队列模型**：用于解释不同 workload domain 的 arrival process。

- 每个抽象都要写：状态、控制、约束、可观测量、适用范围/会失真哪里。
- 产出（交付物）：

    - `Model Candidates v0.1`（1-2 页即可）。

### 3) 明确 objective function 候选，并与系统指标对齐

- 需要明确：优化目标到底对齐哪个系统 KPI（至少选 2 个 principal + 1 个 regularizer）：

    - 主目标候选：最小化平均延迟 / P95 延迟 / queue cost $\int Q(t)dt$ / TTFT/TPOT
    - 正则项候选：expert 负载方差（load skew）、丢弃/溢出惩罚、routing 变化总变差（stability）

- 产出（交付物）：

    - `Objective Options + KPI Mapping`：每个 objective 写清对哪个 KPI，有无可计算 proxy。

### 4) 把 Hongfan 的约束候选集收敛成“可验证的关键约束”

- 目标：从候选约束里选出 3-5 条最关键、最好验证、且最影响优化结果的约束。
- 对每条关键约束定义“验证方法”：

    - 需要哪些日志/数据
    - 怎么画图/怎么做 sanity check
    - 如果不成立，模型怎么降阶/怎么修正

- 产出（交付物）：

    - `Key Constraints (top 3-5) + Verification Plan`。

### 5) 安排本周对齐节奏与最终共同交付

- 周中与 Hongfan 对齐：确认 vLLM 物理流程 + 变量表。
- 周末与 Hongfan 对齐：冻结 v0.1 的“物理模型叙述 + 数学符号 + 约束 + 目标”。
- 最终共同交付（本周结束时）：

    - `vLLM→Math Model v0.1`：一份 2-4 页的短文档/笔记，包含：

        - 物理流程图（来自 Hongfan）
        - 符号与映射表（来自 Zhenghong）
        - 关键约束集合（共同确认）
        - 目标函数候选（共同确认）
        - 下一周实验/数据采集清单（3-5 条）


