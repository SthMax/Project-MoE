# 1. Model

### 我们针对 Request 建模还是针对 Token 建模？

我们针对Token流进行建模，因为我们assume整个系统是Heavy Traffic的，所以会有非常多的request / token到来，然后模型里面我们会监测各个Expert的Queue Length，而在vLLM里面这个 scheduling 是 not depend on request 的

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



