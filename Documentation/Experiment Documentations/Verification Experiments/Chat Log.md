你们这个设定（**serving-time routing：router preference + 可变 cost → 在质量/速度间做权衡**）是合理的，而且很“系统论文”范式：先证明 **(i) routing 确实呈现可建模的负载形态**，再证明 **(ii) top-k/路由策略会带来可控的质量差距**，最后证明 **(iii) 引入 queue-aware policy 能在高并发下提升吞吐/延迟，同时质量损失可控**。下面我按你要的 3 个小实验给出一个工程可落地的方案，尽量以 vLLM 为主。

---

## 共同前提：你要先解决两件“工程底座”

### A) vLLM 里怎么拿到 routing/activation？

vLLM **默认不会**把每层的 router logits / expert assignment 返回给用户（社区也有人问过，通常需要改代码做日志/统计）。([vLLM Forums](https://discuss.vllm.ai/t/can-vllm-return-expert-selection-info-to-support-routing-replay/1742?utm_source=chatgpt.com "Can vLLM return expert selection info to support routing replay?"))  
好消息是 vLLM 的 MoE 路由是模块化的：有 `FusedMoERouter` 抽象类、`select_experts` 入口，非常适合你插桩/改策略。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/fused_moe_router/?utm_source=chatgpt.com "fused_moe_router - vLLM")) 另外还有专门的 routing simulator 可以先离线验证路由策略逻辑。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/routing_simulator/?utm_source=chatgpt.com "routing_simulator - vLLM"))

**建议做法（最小侵入、不会爆盘）：**

- 不要存“每 token 每层”的 assignment（量会炸）。
    
- 只做 **在线聚合统计**：每步 forward 只累计
    
    - `count[layer, expert] += routed_tokens`
        
    - `count_top1[layer, expert] += top1_tokens`
        
    - 可选：`sum_gate_weight[layer, expert] += gate_weight`
        
- 每 N 个 step flush 一次到磁盘（npz/parquet），避免 worker 崩了数据丢。
    

### B) perplexity / 跑分怎么用 vLLM 做？

- vLLM 支持 `prompt_logprobs`：可以返回每个 prompt token 的 logprob，用于算 perplexity/打分。([vLLM](https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html?utm_source=chatgpt.com "Sampling Parameters - vLLM"))（论坛也明确说它常用于 perplexity 计算/MCQ scoring）。([vLLM Forums](https://discuss.vllm.ai/t/what-is-the-purpose-of-prompt-logprobs/1714?utm_source=chatgpt.com "What is the purpose of prompt logprobs? - vLLM Forums"))
    
- 跑基准建议用 **lm-evaluation-harness**，它直接支持 vLLM backend（`pip install lm_eval[vllm]`）。([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness: A framework for few- ..."))
    
- 吞吐/延迟用 vLLM 自带 benchmark/bench 工具链（throughput API、latency CLI 都有文档）。([vLLM](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/throughput/?utm_source=chatgpt.com "vllm.benchmarks.throughput"))
    

---

# 实验 1.1：多数据集观察 expert activation 分布（“arrival process 长什么样”）

### 目标

给 fluid approximation 提供最关键的输入：**每层每 expert 的到达率/负载分布**，以及它如何随数据域变化。

### 数据集建议（选 4 类就够写 paper）

- **General text**：WikiText-103 / C4 子集
    
- **Code**：MBPP / HumanEval prompts（或你们内部 code corpus）
    
- **Math/Reasoning**：GSM8K（只用题目部分）/ MATH 子集
    
- **Multilingual**：Flores-200 子集 / 中英混合语料
    

每类先取一个“轻量版”：比如 1k～5k 段，每段截到固定 token 长度（例如 512 或 1024），保证统计可比。

### 运行模式（推荐“teacher-forcing 的 prefill-only”）

- 对每段文本，喂给模型做 **prompt logprobs**（不生成），拿到 token logprob 只是为了后续 ppl；routing 统计来自你在 MoE router 里插桩得到的 top-k indices。
    

### 你要记录的统计量（paper 图表直接用）

对每个 MoE layer（48 层）：

1. **Load heatmap**：`load[l,e] = count[l,e] / sum_e count[l,e]`
    
2. **不均衡度**：Gini / CV / max-to-mean ratio / entropy
    
3. **top-p expert 占比**：比如 top-1%、top-5% experts 吃掉多少 token
    
4. **域间相似度**：同一 layer 的 `load` 在不同数据集之间做 cosine/JSD，证明“路由对 domain 有响应”
    
5. （可选）**co-activation**：top-k 里 expert pair 的共现矩阵（看是否存在“专家簇”）
    

### 工程实现建议（vLLM）

- 在 `FusedMoERouter.select_experts`（或等价位置）里拿到 `topk_indices` 后直接 `bincount` 聚合。
    
- 注意：你们的模型是 **E=512、k=10(+shared)**，聚合数组规模是 `48×512`，非常小。
    
- **不要**把 logits 全存下来；如果你确实需要 margin/entropy，只抽样 0.1% token 存 logits 统计分位数即可。
    

---

# 实验 1.2：改变 top-k，观察 perplexity 与 benchmark（“质量-稀疏度曲线”）

### 目标

证明：**路由稀疏度（top-k）确实带来可量化的性能差距**，并给你后面“加 cost”提供对照基线。

### 关键点：Qwen3-Next 训练时是 top-10(+shared)，你只能安全地“减 k”

建议测试：`k ∈ {1,2,4,8,10}`（shared expert 始终保留）。

### 路由实现的两个版本（强烈建议都做）

- **Version A（更保守）**：只用原 router logits 选 top-k，然后对选中的 k 个 expert **重新 softmax 归一化权重**（shared 单独加回）。
    
- **Version B（更“贴近训练”）**：先按原本 top-10 得到权重，再把其中非前 k 的权重置 0，然后对剩余权重 renormalize。（有时更稳）
    

两版都测一下，你能在 paper 里说“我们避免把 weight 分配搞得太离谱”。

### 指标与流程

1. **Perplexity**
    

- 用 vLLM 的 `prompt_logprobs` 算 ppl：`ppl = exp(mean(-log p(x_t | x_<t)))`。([vLLM](https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html?utm_source=chatgpt.com "Sampling Parameters - vLLM"))
    
- 在实验 1.1 的同一批数据上算（节省数据准备）。
    

2. **Benchmark / 跑分（代表 performance gap）**
    

- 用 lm-eval-harness（vLLM backend）：([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness: A framework for few- ..."))
    
    - 轻量组合：HellaSwag、ARC-Challenge、GSM8K（或 MMLU subset）
        
- 注意固定 decoding（temperature=0），避免 sampling 噪声影响。
    

3. **Throughput**
    

- 用 vLLM bench/throughput 工具统计 tokens/s、requests/s；并在不同并发下测（比如 concurrency=1/4/16/32）。([vLLM](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/throughput/?utm_source=chatgpt.com "vllm.benchmarks.throughput"))
    

### 你预期要看到的“可发表现象”

- k 越小：吞吐上升（尤其高并发），ppl/accuracy 下降
    
- 并发越高：k 的吞吐收益越明显（因为 MoE 层更容易出现 load hotspot）
    

---

# 实验 1.3：Queue-aware 的新 Top-K policy（不考虑通信），看 ppl/跑分/吞吐

你这里最关键的问题是：**queue length 怎么定义、怎么统计**。我建议你把它定义成 **“专家侧工作量导致的排队延迟 proxy”**，工程上可观测、可控、可用于 OR/fluid。

## 1) Queue length 的可操作定义（两种都能做）

### 定义 Q1（最简单、强烈推荐作为 1.3 主实验）

把“queue length”定义成 **上一调度周期该 expert 的 routed token 数**（或其 EMA）：

- 每个 decode step（vLLM 的一个 iteration）你能得到 `a_{l,e}(t)`：这一轮在 layer l 路由到 expert e 的 token 数
    
- 用 EMA 平滑成“队列状态”：  
    $$ 
    q_{l,e}(t) \leftarrow (1-\beta),q_{l,e}(t-1)+\beta,a_{l,e}(t-1)  
    $$
    
- 它不是严格的“FIFO backlog”，但在 GPU 上它是非常好的 **延迟 proxy**：谁吃到的 token 多，谁就是热点，MoE kernel 的时间往往被热点拖慢。
    

### 定义 Q2（更像真正队列，适合你后续 fluid）

把每个 expert 看成服务率为 (\mu_{l,e}) 的 server：  
$$ 
q_{l,e}(t+1) = \max{0, q_{l,e}(t) + a_{l,e}(t) - \mu_{l,e}\Delta t}  
$$
其中 (\mu_{l,e}) 可以用一个 microbenchmark 估出来（固定 hidden size，扫不同 token count，拟合“每 token 处理速率”）。  
这版更“运筹/流体”，但工程要多一点；你可以先用 Q1 做 feasibility，再升级到 Q2。

## 2) Queue-aware routing policy（你说的 “preference - cost”）

对每层 l，每个 token 的 router logits 记为 $z_{l,e}$。你引入 cost：

- cost（忽略通信）：$c_{l,e}(t)=\lambda\cdot \tilde q_{l,e}(t)$
    
- 调整后的 logits：  
    $$  
    z'_{l,e}(t)=z_{l,e}(t)-c_{l,e}(t)  
    $$
    
- 再用 (z') 做 top-k 选择
    

**两个安全实现选项：**

- **Policy S（更稳）**：用 $z'$ 只做“选谁”，但 gate weight 仍用原始 $z$ 在选中集合内归一化（尽量保质量）
    
- **Policy A（更激进）**：选谁和权重都用 $z'$
    

建议主实验用 Policy S，paper 里再补一个消融：Policy A 更快但掉分更多。

## 3) 怎么让“queue”在实验里真的出现？

你需要“多 prompt 并发 + continuous batching”让路由热点持续存在。

**负载设计（强烈建议分两档）：**

- **Low contention**：并发=1/2（看质量几乎不变、吞吐差异小）
    
- **High contention**：并发=16/32（看 queue-aware 是否显著改善 tokens/s / tail latency）
    

在 vLLM 上你可以通过 bench 工具设置请求数、并发、输入输出长度分布来制造 hotspot（bench/throughput/latency 都能做到）。([vLLM](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/throughput/?utm_source=chatgpt.com "vllm.benchmarks.throughput"))

## 4) 评估指标（你这实验的“赢法”）

1. **吞吐/延迟**
    

- tokens/s、requests/s
    
- p50/p95 latency（高并发下尤其看 p95）
    

2. **质量**
    

- ppl（用 prompt_logprobs）([vLLM](https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html?utm_source=chatgpt.com "Sampling Parameters - vLLM"))
    
- lm-eval accuracy（同 1.2 的 task 组合）([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness: A framework for few- ..."))
    

3. **系统态势（写 paper 很关键）**
    

- **负载不均衡改善**：max-load / mean-load、Gini、top experts share
    
- **routing drift**：与 baseline（无 cost）相比，expert 分配变化幅度（Jaccard / KL）
    

你最终要呈现的图非常清晰：

- x 轴：(\lambda)（或者你 OR 里算出来的 cost scale）
    
- y 轴：吞吐提升 vs accuracy/ppl 下降  
    这就是你们“balance”的核心结果。
    

---

## 框架选择：尽量 vLLM（你想要的“最后能变系统”）

### 必选：vLLM（主干）

- 你会在 vLLM 里：
    
    1. 插桩收集 activation 分布（1.1）
        
    2. 改 top-k（1.2）
        
    3. 实现 queue-aware router（1.3，改 `select_experts` 最自然）([vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/fused_moe_router/?utm_source=chatgpt.com "fused_moe_router - vLLM"))
        
- vLLM 还有 routing simulator 模块，适合你先离线把策略写对。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/routing_simulator/?utm_source=chatgpt.com "routing_simulator - vLLM"))
    
- 后续如果你们要考虑“通信/专家跨 GPU”，vLLM 也支持 MoE 的 Expert Parallelism/组合并行（EP/DP/TP）。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/?utm_source=chatgpt.com "Expert Parallel Deployment - vLLM"))
    

### 可选：Transformers（只做 sanity check，不抢你“系统路线”）

- 用 HF Transformers 做少量对照（比如 k=10 baseline 的 ppl 和 vLLM 对齐），防止你在 vLLM patch 里引入 bug。
    
- 但主体实验完全可以留在 vLLM（你想要的）。
    

---

## 额外 comments：你们 idea 的风险点 & 我建议你提前规避

1. **inference-time 改路由会掉分是必然的**  
    所以你们 paper 要强调的是“**可控的 trade-off**”，而不是“免费提速”。用 (\lambda) 扫一条 Pareto 曲线会更有说服力。
    
2. **queue length 术语可能会被 reviewer 挑**  
    因为 GPU 上更多是“负载/热点/imbalance”，不一定是严格队列。你可以在 paper 里写成：
    

- “expert load / backlog proxy”
    
- “queue length surrogate”  
    并用 Q2（流体那套）做理论化定义，Q1 做工程可观测实现。
    

3. **最好加一个“只在 near-tie 时重排”的 safety 机制**  
    比如只允许从原 top-(k+m) 里换（m=2/4），避免把一个很不适合的 expert 硬塞进来。这会显著减少质量损失，工程也简单。
    

---

如果你愿意，我可以把这 3 个实验再进一步“落到工程任务列表”级别：

- vLLM 里要改/加哪些类（router 插桩、router policy、统计 flush）
    
- 每个实验需要的脚本入口（ppl 计算、lm-eval、bench 并发配置）
    
- 以及你们最终论文里应该怎么组织 3 个实验的图表与消融结构。