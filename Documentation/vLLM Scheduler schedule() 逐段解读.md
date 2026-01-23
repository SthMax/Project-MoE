# vLLM V1 Scheduler 的 `schedule()`：逐段解读（带行号）

本文基于 vLLM 源码：
- `external/vllm/vllm/v1/core/sched/scheduler.py`
- `external/vllm/vllm/v1/core/kv_cache_manager.py`

目标：把 `Scheduler.schedule()` 的控制流、关键数据结构、以及它如何与 KV cache / speculative decoding / encoder cache / KVConnector 协作，按“片段”解释清楚，并在关键处标注行号。

---

## 0. 这段函数在做什么？（高层概念）

下面是 `schedule()` 的函数头与作者对“调度抽象”的核心注释（原代码，来源：`scheduler.py` L252-L262）：

```python
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
```

`schedule()` 每一步（一次 engine step）会：
1. 决定哪些请求（Request）在本 step 要被执行（prefill 或 decode 都统一成“追赶 token”）。
2. 为这些请求分配 KV cache block（必要时抢占/驱逐低优先级请求）。
3. 处理 prefix cache 命中、KVConnector 外部 KV 命中/加载、spec decode 的 lookahead、以及 multimodal/encoder 输入的预算与缓存。
4. 产出 `SchedulerOutput`（包含：本步每个请求要算多少 token、需要哪些新 block、可能的 spec token、encoder 输入等），供 model runner 执行。

作者的核心注释（L253-L262）非常关键（建议先读一遍）：

> L253-L262：scheduler 里没有严格“prefill phase / decode phase”，只有每个 request 的 `num_computed_tokens` 与 `num_tokens_with_spec`。
> scheduler 的目标是在每个 step 分配 token，让 `num_computed_tokens` 追上 `num_tokens_with_spec`。
> 该抽象覆盖 chunked prefill、prefix caching、speculative decoding 等。

这句话基本决定了你要怎么理解后面的 `num_new_tokens` 计算逻辑。

---

## 1. 关键状态与变量速查表

### 1.1 Scheduler 成员变量（与本函数强相关）

- `self.running`：RUNNING 队列（list[Request]），当前已在运行态、持有 KV cache 的请求列表。
- `self.waiting`：WAITING 队列（RequestQueue），待调度的请求（包括新请求、被抢占的请求、等待远端 KV 的请求等）。
- `self.requests`：`req_id -> Request` 的总表。
- `self.policy`：调度策略（`SchedulingPolicy`），例如 FCFS 或 PRIORITY。
- `self.max_num_scheduled_tokens`：本 step 的 token 总预算（token_budget 初值），来自 scheduler config。
- `self.max_num_running_reqs`：并发运行请求数上限。
- `self.kv_cache_manager`：KV cache 管理器，负责 prefix cache、block 分配/释放、统计等。
- `self.encoder_cache_manager`：encoder/multimodal 的 cache 管理器。
- `self.connector`：KVConnector（可选），用于 P/D disaggregation、offload、远端 KV load。
- `self.ec_connector`：ECConnector（可选），用于远端 encoder cache。
- `self.use_eagle / self.num_lookahead_tokens`：spec decode（eagle）相关。

### 1.2 schedule() 内的核心局部变量

这些变量贯穿全函数，理解它们能把代码“压缩成几条关键不变量”。

- `scheduled_new_reqs`（L264）：本 step 新进入 running 的请求（原 status=WAITING）。
- `scheduled_resumed_reqs`（L265）：本 step 恢复运行的请求（原 status=PREEMPTED）。
- `scheduled_running_reqs`（L266）：本 step 在 running 队列中、且被真正分配了 token 的请求。
- `preempted_reqs`（L267）：本 step 被抢占（preempt）出去的请求列表。

- `req_to_new_blocks`（L269）：`req_id -> KVCacheBlocks`，记录本 step 为该请求“本次”分配/使用的 block 视图（供输出给 model runner）。
- `num_scheduled_tokens`（L270）：`req_id -> int`，本 step 每个请求要算多少 token。
- `token_budget`（L271）：剩余 token 预算，初值为 `self.max_num_scheduled_tokens`。

- `scheduled_encoder_inputs`（L273）：`req_id -> list[int]`，本 step 需要计算/加载的 encoder 输入 index（对应 `request.mm_features[i]`）。
- `encoder_compute_budget`（L274）：encoder token/embed 预算（对 multimodal encoder 是 embed 数预算）。

- `scheduled_spec_decode_tokens`（L276）：`req_id -> list[int]`，本 step 对应的 spec token（draft token）占位，用于 speculative decoding。

- `scheduled_timestamp`（L279）：用于 event logging 的时间戳。

原代码（这些变量都在 `schedule()` 开头初始化，来源：`scheduler.py` L264-L279）：

```python
  scheduled_new_reqs: list[Request] = []
  scheduled_resumed_reqs: list[Request] = []
  scheduled_running_reqs: list[Request] = []
  preempted_reqs: list[Request] = []

  req_to_new_blocks: dict[str, KVCacheBlocks] = {}
  num_scheduled_tokens: dict[str, int] = {}
  token_budget = self.max_num_scheduled_tokens
  # Encoder-related.
  scheduled_encoder_inputs: dict[str, list[int]] = {}
  encoder_compute_budget = self.max_num_encoder_input_tokens
  # Spec decode-related.
  scheduled_spec_decode_tokens: dict[str, list[int]] = {}

  # For logging.
  scheduled_timestamp = time.monotonic()
```

---

## 2. 先调度 RUNNING 请求（L281-L446）

这一大段的原代码（从 RUNNING 主循环开始到 RUNNING 部分结束，来源：`scheduler.py` L281-L446）：

```python
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
```

主循环（L282-L446）：

- 条件：`req_index < len(self.running)` 且 `token_budget > 0`。
- 逐个取 `request = self.running[req_index]`。

### 2.1 “输出占位符”导致的跳过（L287-L301）

这一段的目标：在 **async scheduling + spec decode** 情况下，避免多 schedule 一步。

关键字段：
- `request.num_output_placeholders`：输出占位符 token 数（常见于 speculative decoding：draft token 先占位）。
- `request.num_prompt_tokens`：prompt token 数。
- `request.max_tokens`：允许生成的最大输出 token。
- `request.num_computed_tokens`：到目前为止已经“计算过”的 token 数（注意：包括 placeholder）。

逻辑（L287-L301）大意：
- 如果我们已经可以确定上一 step 已经达到 `max_tokens`，则不再 schedule 额外一步（并且不支持“部分 draft tokens”调度，避免破坏 uniform decode 优化）。

### 2.2 计算“本 step 需要补齐的新 token 数” num_new_tokens（L303-L319）

核心公式（L303-L306）：

- `num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens`

含义：
- `request.num_tokens_with_spec` 代表“目标 token 进度”（prompt + 已确认 output + spec token 等），再加上 `num_output_placeholders`（占位）减掉当前 `num_computed_tokens`（已经算到哪里），得到还欠多少 token。

随后有三个 clamp：
- 长 prefill 阈值（L307-L308）：如果本次欠太多 token，就把每步上限压到 `long_prefill_token_threshold`。
- 不能超过全局预算（L309）：`min(num_new_tokens, token_budget)`。
- 不能超过 max model len（L313-L316）：确保 speculative decoding 时 position 不溢出。

### 2.3 处理 encoder/mm 输入（L318-L346）

如果 `request.has_encoder_inputs` 为真，则调用 `_try_schedule_encoder_inputs(...)`（L325-L333）。

这一步会返回四元组（见 helper 的 docstring，L906-L944）：
- `encoder_inputs_to_schedule`：本 step 需要“计算”的 encoder 输入 index。
- `num_new_tokens`：可能被缩减（如果 encoder cache / encoder budget 不够，会把 decode 范围缩回到 encoder 输入开始之前）。
- `new_encoder_compute_budget`：更新后的 encoder 预算（扣除本 step encoder embed）。
- `external_load_encoder_input`：本 step 需要从 ECConnector 远端加载的 encoder 输入 index。

如果结果使 `num_new_tokens == 0`（L339-L346），表示本 request 本步不可调度（原因包含：没 token 可算、encoder budget/cache 用尽等）；这里作者用 `continue` 而不是 `break`，说明允许“跳过高优先级/前面的请求，让后面的请求先跑”（见 L342-L345 的注释）。

### 2.4 KV block 分配 + 必要时抢占（L348-L406）

调用点（L352）：
- `new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)`

这里要理解 `allocate_slots` 的语义：
- 定义在 `kv_cache_manager.py` 的 L206 起。
- 它会为“本步要算的 token”（以及 spec lookahead token）分配 KV cache blocks；如果空间不足返回 `None`。
- 它的 docstring 还画了 blocks layout（`comp/new_comp/ext_comp/new/lookahead`）解释哪些 token 对应哪些区域（`kv_cache_manager.py` L206-L260 附近）。

当 `allocate_slots` 返回 `None`（L358-L405）：
- 说明 KV cache 不够，scheduler 需要抢占（preempt）某个 request 来释放块。

抢占策略分两种：
- PRIORITY（L365-L389）：抢占 `max(self.running, key=(priority, arrival_time))`，即优先级最低/最晚到的那个。
  - 如果被抢占的 request 已经在本 step 的 `scheduled_running_reqs` 里（L373-L386），还需要把它占用的预算/记录回滚：
    - `token_budget += num_scheduled_tokens[preempted_req_id]`
    - 从 `req_to_new_blocks / num_scheduled_tokens / scheduled_spec_decode_tokens / scheduled_encoder_inputs` 里移除
    - 如果移除了 encoder inputs，还要把 encoder budget 加回去（L384-L389）
    - `req_index -= 1`（因为 running 列表被改了，当前 index 需要回退）
- 其他策略（典型 FCFS）（L390-L391）：`preempted_req = self.running.pop()`（末尾最低优先级/最“后”的那个）。

最终执行抢占动作（L393-L404）：
- `self._preempt_request(preempted_req, scheduled_timestamp)`：释放 KV cache、释放 encoder cache，把 request 状态置为 PREEMPTED 并放回 waiting 队首（详见 L801-L819）。
- 如果抢占到的就是当前 request（`preempted_req == request`），说明已经“无 request 可抢”，本 request 本步无法调度（break）。

### 2.5 成功调度：记录本步对该 request 的决策（L408-L446）

当 `new_blocks is not None`：
- `scheduled_running_reqs.append(request)`（L409）
- `req_to_new_blocks[req_id] = new_blocks`（L410）
- `num_scheduled_tokens[req_id] = num_new_tokens`（L411）
- `token_budget -= num_new_tokens`（L412）

#### Spec decode（L414-L431）

- 条件：`if request.spec_token_ids:`（L415）
- 计算 `num_scheduled_spec_tokens`（L416-L421）：在本步的 `num_new_tokens` 里有多少属于 spec token（draft）。
- 如果 >0：截断 `request.spec_token_ids` 并放入 `scheduled_spec_decode_tokens[req_id]`（L422-L427）。
- 最后把 `request.spec_token_ids = []`（L431）：表示“本步把这些 spec token 作为占位交给 runner 了；下一步新的 draft 会在 `update_draft_token_ids` 里再写回”。

#### Encoder cache 分配（L432-L446）

- 如果 `encoder_inputs_to_schedule` 非空：
  - 记录到 `scheduled_encoder_inputs[req_id]`（L434-L436）
  - 对每个 encoder input `i` 调 `self.encoder_cache_manager.allocate(request, i)`（L438-L439）
  - 更新 `encoder_compute_budget = new_encoder_compute_budget`（L440）
- 如果 `external_load_encoder_input` 非空：
  - 也会 allocate（为“远端加载的 encoder 结果”预留本地 cache entry），并通知 ECConnector 更新状态（L441-L446）。

---

## 3. LoRA 上限约束的 bookkeeping（L447-L459）

原代码（来源：`scheduler.py` L447-L459）：

```python
        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras
```

如果启用 LoRA（`self.lora_config`），scheduler 需要保证本 step 进入运行的请求涉及的 LoRA 数量不超过 `max_loras`。

- `scheduled_loras`（L447-L457）：收集本 step 已经 scheduled 的 running 请求中涉及的 LoRA id。
- `assert len(scheduled_loras) <= max_loras`（L458）。

注意：这只是本 step 的“被调度集合”的约束检查；在后面调度 WAITING 请求时，还会用 `scheduled_loras` 来决定是否跳过某些请求（L501-L513）。

---

## 4. 再调度 WAITING 请求（L461-L699）

这一大段的原代码（WAITING 主循环整体，来源：`scheduler.py` L456-L699）：

```python
        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self._update_connector_prefix_cache_stats(request)

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
```

另外，本节里经常提到 prefix cache 命中，这个命中的计算来自下面这个函数（原代码，来源：`kv_cache_manager.py` L164-L203）：

```python
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

只有当本 step 没有发生抢占（`if not preempted_reqs:`，L462）才会调度 WAITING。

直觉解释：
- 如果 RUNNING 阶段都需要抢占才能塞下 token，说明 KV cache 很紧张；这时再把 WAITING 拉进来大概率更糟，所以直接不做。

### 4.1 “跳过队列” skipped_waiting_requests（L458-L460, L700-L702）

- `skipped_waiting_requests = create_request_queue(self.policy)`（L458-L460）：临时队列。
- WAITING 调度循环里遇到暂时不能跑的 request，会从 waiting pop 出来并 prepend 到 skipped。
- 循环结束后再把 skipped 整体 prepend 回 `self.waiting`（L700-L702），保证这些 request 仍在队首附近。

这是一种避免“卡住队首”的技巧。

### 4.2 循环入口与基本上限（L463-L468）

- 退出条件：waiting 为空或 token_budget 用尽。
- 如果 `len(self.running) == self.max_num_running_reqs` 直接 break（L464-L465）。
- 取队首：`request = self.waiting.peek_request()`（L467）。

### 4.3 KVTransfer：等待远端 KV 的 request（L470-L488）

当 `request.status == WAITING_FOR_REMOTE_KVS`（L470）：
- 调 `_update_waiting_for_remote_kv(request)`（L471）。
- 如果 ready：
  - 如果 `request.num_preemptions > 0`，说明这是“恢复的 preempted request”在加载远端 KV，设回 PREEMPTED（L474-L477）。
  - 否则设回 WAITING（L478-L479）。
- 如果不 ready：
  - 从 waiting pop 掉（L484），放到 skipped（L485），并 continue。

变量含义：
- `WAITING_FOR_REMOTE_KVS`：请求已经申请了加载远端 KV，但 KV 尚未到达，不能分配新计算。

### 4.4 Structured output：FSM 编译等待（L490-L499）

当 `request.status == WAITING_FOR_FSM`（L490）：
- 如果 grammar 已经就绪，则变回 WAITING（L492-L494）。
- 否则 pop 并放到 skipped，continue（L495-L499）。

### 4.5 LoRA 约束导致的跳过（L501-L513）

如果引入该 request 会让本 step 同时活跃的 LoRA 超过 `max_loras`，就跳过：
- pop + 放 skipped + continue。

### 4.6 prefix cache / 外部 KV 命中：先算“已经算过多少 token”（L514-L547）

初始化（L514-L515）：
- `num_external_computed_tokens = 0`
- `load_kv_async = False`

如果这是第一次调度（`request.num_computed_tokens == 0`，L518）：
1. 先取本地 prefix cache 命中：
   - `new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)`（L520-L522）
   - `get_computed_blocks` 定义在 `kv_cache_manager.py` L164 起：它会找“最长前缀命中”，并且当全命中时会强制最后一个 token 重新算（max_cache_hit_length = num_tokens-1），避免没有 logits（见 `kv_cache_manager.py` L183-L191）。
2. 再取 KVConnector 的外部命中（如果配置了 connector）：
   - `ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(...)`（L526-L528）
   - 若 ext_tokens 为 None：connector 无法判定，跳过该 request（L531-L536）。
   - 否则把 `request.num_external_computed_tokens` 和 `num_external_computed_tokens` 置为 ext_tokens（L540-L541）。

最后：
- `num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens`（L544-L546）。

如果 `request.num_computed_tokens > 0`（L547-L556）：
- 说明这是 KVTransfer 完成后的 WAITING request（注释 L548-L549），其 computed token 已由其他流程写入。

### 4.7 决定本步要算多少 token（L557-L606）

两条路径：

1) 如果 `load_kv_async` 为真（L558-L563）：
- 代表远端 KV 需要异步加载，这一轮 **不做新计算**。
- `num_new_tokens = 0`。

2) 否则：
- 计算欠多少 token：`num_new_tokens = request.num_tokens - num_computed_tokens`（L567）。
  - 注意这里用的是 `request.num_tokens`，而不是 `num_prompt_tokens`，注释解释了为了支持 resumed requests（L565-L567）。
- long prefill 阈值 clamp（L568-L571）。
- chunked prefill 约束（L573-L581）：
  - 如果没开 chunked prefill 且 `num_new_tokens > token_budget`，直接 break（L575-L580）。
- `num_new_tokens = min(num_new_tokens, token_budget)`（L582），并 assert >0。

若有 encoder inputs，则调用 `_try_schedule_encoder_inputs`（L588-L600），可能把 `num_new_tokens` 缩成更小（甚至 0 -> break）。

### 4.8 lookahead/encoder-decoder 的特殊参数（L603-L619）

- `effective_lookahead_tokens`（L608-L611）：
  - 只有在 `request.num_computed_tokens > 0` 时才分配 lookahead。
  - 注释（L603-L607）说明：这是为 P/D disaggregation + spec decode 的一个 block mismatch edge case。

- `num_encoder_tokens`（L613-L618）：
  - 仅在 encoder-decoder 且 request 有 encoder inputs 时，分配 cross-attn blocks（Whisper 等可能需要固定最大输入长度）。

### 4.9 统一调用 allocate_slots（WAITING 版本参数更复杂）（L620-L637）

调用（L620-L637）：
- `self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
  )`

对照 `kv_cache_manager.py` 的 docstring（L206+）：
- `num_new_computed_tokens/new_computed_blocks`：本地 prefix cache 命中的 token 与 blocks。
- `num_external_computed_tokens`：connector 命中的 token（vLLM 不缓存，但 connector 缓存）。
- `delay_cache_blocks=True`：KVTransfer 场景下，先分配 block 但暂时不 cache（等 transfer 完成）。

若返回 None（L639-L648）：
- request 无法调度。
- 如果有 encoder inputs，还要把 encoder cache manager 对该 request 的占用释放掉（L643-L646），避免泄露。
- break。

### 4.10 通知 connector、更新队列与状态（L638-L699）

若配置了 connector，则更新 connector 侧状态（L638-L646）：
- `self.connector.update_state_after_alloc(request, blocks, num_external_computed_tokens)`

然后把 request 从 waiting 弹出（L651）。

若 `load_kv_async`：
- 把 request 放入 skipped（L655），将状态置为 WAITING_FOR_REMOTE_KVS（L656），continue（L657）。

否则（真正进入 RUNNING）：
- `self._update_connector_prefix_cache_stats(request)`（L659）：统计。
- `self.running.append(request)`（L661）。
- 记录 event（L662-L666）。
- 按原 status 分类：
  - WAITING -> `scheduled_new_reqs`（L667-L668）
  - PREEMPTED -> `scheduled_resumed_reqs`（L669-L670）
  - 否则报错（L671-L672）

把本 step 的分配写入几个核心表（L675-L679）：
- `req_to_new_blocks[req_id] = kv_cache_manager.get_blocks(req_id)`
- `num_scheduled_tokens[req_id] = num_new_tokens`
- `token_budget -= num_new_tokens`

并把 request 的运行态字段更新到“正确的起点”（L680-L689）：
- `request.status = RUNNING`
- `request.num_computed_tokens = num_computed_tokens`（注意：这里把 prefix cache 命中 + 外部命中算进来了）
- `request.num_cached_tokens` 若未设置，则初始化为 `num_computed_tokens`（L686-L688）

最后同 RUNNING 部分一样处理 encoder：
- `scheduled_encoder_inputs` 记录 + `encoder_cache_manager.allocate`（L690-L699）。

---

## 5. 把 skipped 请求放回 waiting 队首（L700-L702）

原代码（来源：`scheduler.py` L700-L702）：

```python
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)
```

这一步保证：
- 由于 KVTransfer/FSM/LoRA 等原因暂时跳过的请求不会丢；
- 同时它们会回到队首，等待下一 step 再尝试。

---

## 6. 一堆不变量断言（L704-L716）

原代码（来源：`scheduler.py` L704-L716）：

```python
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)
```

这些 assert 是理解 scheduler 正确性的“契约”：
- `total_num_scheduled_tokens <= max_num_scheduled_tokens`（L705-L707）
- `token_budget >= 0`（L709）
- `len(self.running) <= max_num_running_reqs`（L710）
- scheduled 的请求数不超过 running 数（L714-L716）

如果你在改 scheduler 或加新策略，这些 assert 很容易告诉你哪里破坏了 invariants。

---

## 7. 计算 common prefix blocks（cascade attention 线索）（L717-L727）

原代码（schedule() 内调用处，来源：`scheduler.py` L717-L727）：

```python
        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )
```

它最终会走到 `KVCacheManager.get_num_common_prefix_blocks`（原代码，来源：`kv_cache_manager.py` L420-L460）：

```python
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)
```

- `num_common_prefix_blocks = [0] * num_kv_cache_groups`（L719）
- 如果有 running：选任意一个 running request（L722-L723），调用：
  - `self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)`（L724-L726）

这个方法在 `kv_cache_manager.py` L420 起，它的 docstring 解释了一个重要边界条件：
- “拥有已分配 KV cache 的请求集合”可能大于“本步被 schedule 的请求集合”，因此 common prefix 可能因为某些未 schedule 的请求而变成 0（`kv_cache_manager.py` L430-L452）。

---

## 8. 构造 SchedulerOutput（L729-L800）

原代码（从构造 new/cached 数据到返回 `scheduler_output`，来源：`scheduler.py` L729-L800）：

```python
        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
```

补充：`_make_cached_request_data` 是构造 cached_reqs_data 的关键 helper（原代码，来源：`scheduler.py` L849-L904）：

```python
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )
```

### 8.1 new requests 数据（L730-L748）

分两种 model runner：
- 如果 `self.use_v2_model_runner`：
  - 把 `scheduled_resumed_reqs` 合并进 new（L731-L733）
  - `NewRequestData.from_request(req, block_ids, req._all_token_ids)`（L734-L741）
- 否则：
  - `NewRequestData.from_request(req, block_ids)`（L743-L747）

变量含义：
- `new_reqs_data` 是给 model runner 的“新加入 batch 的请求”的描述。

### 8.2 cached requests 数据（L749-L756）

- 通过 `_make_cached_request_data(...)` 构造（L750-L756）。
- helper 逻辑见 L849-L904：
  - 它会把 running + resumed 的 req_id 列出来。
  - 在 pipeline parallel（`self.use_pp`）时需要把“本步 new token ids”回传（L868-L880），因为 PP 下首尾 stage 不直接通信。
  - 如果某 request 没有在上一步被 schedule，则需要把 `all_token_ids` 全量回传（L883-L887），让 model runner 重建状态。

### 8.3 记录“上一步 schedule 过哪些请求”（L758-L761）

- `self.prev_step_scheduled_req_ids` 用于下一步 `_make_cached_request_data` 判断是否需要回传全量 token ids。

### 8.4 组装 SchedulerOutput 对象（L762-L785）

关键字段：
- `num_scheduled_tokens`：每个 request 本步要算多少 token（核心）。
- `scheduled_spec_decode_tokens`：spec decode 的占位 token。
- `scheduled_encoder_inputs`：encoder/mm 的输入 index。
- `preempted_req_ids`：本步被抢占出去的 req_id 集合。
- `finished_req_ids`：上一轮到这一轮之间完成的请求 id（注意：这是 scheduler 内部已有状态，不是本步新 schedule 出来的）。

### 8.5 connector metadata（L787-L795）

如果启用了 KVConnector 或 ECConnector：
- 分别构建 connector meta 塞进 `scheduler_output`（L787-L795）。
- 这些 meta 用于后续 worker-side 的 KV/EC transfer 计划与状态机。

### 8.6 “schedule 之后”的状态推进（L796-L800）

- 调 `_update_after_schedule(scheduler_output)`（L796-L797）
- return `scheduler_output`（L798-L800）

`_update_after_schedule` 的关键点（L821-L848）：
- 对每个被 schedule 的 request：`request.num_computed_tokens += num_scheduled_token`（L833-L834）。
- 如果 request 有 encoder inputs，则调用 `_free_encoder_inputs(request)`（L842-L843），将已经用完的 encoder cache 释放。
- 最后把 `self.finished_req_ids` 重置为新的 set（L848），避免影响已经返回给上层的 output（注释 L845-L847）。

---

## 9. 你可以带着哪些“心智模型”再读一遍代码？

我建议带着下面三个不变量回看 `schedule()`，会明显更顺：

1) **token_budget 是全局硬预算**（L271, L412, L679）：
- 谁被 schedule，谁就会消耗预算；被抢占回滚时会归还。

2) **`num_computed_tokens` 是“进度条”**：
- RUNNING 阶段它已经是最新进度；
- WAITING 阶段要先用 prefix cache + external cache 估一个 `num_computed_tokens` 起点（L520-L546），再在最后写回到 request（L681-L683）。

3) **KV cache 分配失败 == 需要抢占（或停止引入新请求）**：
- RUNNING 部分会通过 preempt 来强行腾空间（L358-L405）；
- WAITING 部分则更保守，失败就 break（L639-L648），并且只在本 step 无抢占时才尝试引入 WAITING（L462）。

---

## 10. 如果你想继续深挖：建议顺着哪些函数读？

- `KVCacheManager.allocate_slots`：决定“为什么会返回 None、何时会 free blocks、prefix/ext/new/lookahead 如何布局”（`kv_cache_manager.py` L206+）。
- `KVCacheManager.get_computed_blocks`：prefix cache 的最长命中逻辑，以及“全命中也要重算最后 token”这个设计细节（`kv_cache_manager.py` L164+）。
- `Scheduler._update_waiting_for_remote_kv`：WAITING_FOR_REMOTE_KVS 状态如何转回 WAITING / PREEMPTED（同文件后部）。
- `Scheduler._try_schedule_encoder_inputs`：encoder budget/cache 不够时如何把 `num_new_tokens` 回退（L906+）。

