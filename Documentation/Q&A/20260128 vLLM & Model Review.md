# 1. Model

### 我们针对 Request 建模还是针对 Token 建模？

参考 [[final_model]]，我们是针对 Request 建模的。但是实际的路由策略应用在 Token-level。

### 我们的 Constraint 有什么

Reference: 

#### 1. Token Budget

由 Scheduler output 来传递这个信息给到 GPU Worker -> GPU Model Runner 进行执行

#### 2. 



# 2.  vLLM

### MoE 在哪里起作用？

1. 在scheduling的时候，实际上在vllm系统里面，scheduler是不知道模型是什么样的
2. MoE是在整个系统的最底层执行的，即Transformer + Layer执行层
3. 我们要改动的系统实际上也在这里
4. 所以这里会有个Token Budget信息同步的问题 -- 在多线程（Expert Parallel）的时候我们如何去维护各个worker对于全局状态的感知 [[260123 Implementation Problems]]



