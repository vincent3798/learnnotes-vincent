# `rollout_engine.py` 详解：把「采样」抽成可替换的引擎

> 对象：`minimind-master/trainer/rollout_engine.py`（224 行）
> 使用者：`train_ppo.py`、`train_grpo.py`、`train_agent.py`
> 说明：本文按源码讲解，不跑训练。

---

## 0. 为什么值得单独抽一个文件

在 RL 训练里，**采样（rollout）通常占 80% 以上的时间**：
一条 prompt 要生成 1024 个 token，而一次参数更新只要几百毫秒。

所以工程上会把两件事解耦：

```
算法侧（PPO/GRPO）  ──调用──►  采样侧（engine）
  只关心 logp/奖励            可以是本地 generate，
                              也可以是远程的 SGLang / vLLM
```

**收益**：算法代码一行不改，采样吞吐就能翻几倍；训练和推理还能分到不同机器上。

> 口诀：**采样最贵，抽成接口；本地能跑，远程能快。**

---

## 1. 数据契约：`RolloutResult`

```python
@dataclass
class RolloutResult:
    output_ids: Tensor        # (B*G, P+R)  prompt + 回答
    completion_ids: Tensor    # (B*G, R)    只有回答
    per_token_logps: Tensor   # (B*G, R)    采样时的 log 概率
    completions: List[str]    # 解码后的文本
    prompt_lens: Tensor       # (B*G,)      每行 prompt 的有效长度
    completion_mask: Tensor   # (B*G, R)    1=真实 token，0=padding
```

**六个字段就是两个引擎必须共同遵守的契约**：

| 字段 | 谁用 | 用途 |
| --- | --- | --- |
| `output_ids` | 训练脚本 | 重新前向算 logp |
| `completion_ids` | 训练脚本 | 定位回答 token |
| `per_token_logps` | PPO/GRPO | ratio 的分母（旧策略） |
| `completions` | 奖励函数 | 文本级打分 |
| `prompt_lens` | 训练脚本 | 算 `logp_pos` |
| `completion_mask` | 训练脚本 | 屏蔽 padding |

**关键不变量**：`per_token_logps[i]` 必须与 `completion_ids[i]` 的 token 一一对应。
两个引擎用完全不同的方式实现，但都必须满足这一条——否则 ratio 就是错的。

---

## 2. `compute_per_token_logps`：只算需要的 logits

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    if n_keep <= 0: return input_ids.new_empty((B, 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    logits = unwrapped(input_ids, attention_mask=attention_mask,
                       logits_to_keep=n_keep + 1).logits[:, :-1, :]     # ① 只算尾部
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):     # ② 逐行 gather
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(-1), 1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token_logps)
```

**① `logits_to_keep=n_keep+1` 是全文件最聪明的一行**：

- 只为最后 `n_keep+1` 个位置算 logits，前面的 prompt 位置全跳过；
- 生成 1024 个 token、prompt 只有 10 个时，**省掉 99% 的输出层计算**；
- 多算 1 个再 `[:, :-1]`，是为了对齐「位置 t 预测 t+1」的错位。

**② 为什么要逐行 gather**：每一行的 `n_keep` 长度相同（padding 补齐），
但目标 id 各不相同，所以按行取。

**③ `is_inference()` 判断**：如果输入来自 `torch.inference_mode()` 的张量，
在后面做 autograd 相关操作前需要 clone，避免「inference tensor 泄漏进计算图」。

---

## 3. 抽象基类

```python
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8) -> RolloutResult: ...

    @abstractmethod
    def update_policy(self, model): ...
```

只有两个方法：

| 方法 | 语义 |
| --- | --- |
| `rollout` | 给一批 prompt，返回 RolloutResult |
| `update_policy` | 把最新的策略权重同步给引擎 |

**这就是「可替换」的全部接口**。接 vLLM 只需再实现一个类。

---

## 4. `TorchRolloutEngine`：本地直接生成

### 4.1 流程

```python
with torch.no_grad(), ctx:                       # ① 不建图 + 混合精度
    output_ids = model.generate(
        input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
        attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
        max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        num_return_sequences=1, pad_token_id=..., eos_token_id=...
    ).clone()                                    # ② clone 脱离 inference 上下文
    prompt_len = prompt_ids.size(1)              # ③ 左填充后的统一长度
    completion_ids = output_ids[:, prompt_len:]  # ④ 切掉 prompt
    full_mask = torch.cat([attn_mask.repeat_interleave(G), ones(R)], dim=1)
    per_token_logps = compute_per_token_logps(self.policy_model, output_ids,
                                              completion_ids.size(1),
                                              attention_mask=full_mask)
```

| 步骤 | 为什么 |
| --- | --- |
| `repeat_interleave(G)` | 每个 prompt 连续复制 G 份：`[p0,p0,p0,p1,p1,p1]`——**顺序必须是这样**，训练脚本靠它 `view(B,G)` |
| `num_return_sequences=1` | 复制已经在外面做了，避免 generate 内部再复制一次 |
| `.clone()` | `MiniMind.generate` 标了 `@torch.inference_mode()`，返回的是 inference tensor，后续要参与 `torch.cat` 等操作，先 clone 更安全 |
| `no_grad + autocast` | 采样不需要梯度；精度与训练一致，logp 才对得上 |
| `prompt_len = prompt_ids.size(1)` | **左填充**后所有 prompt 等长，所以是一个常数 |

### 4.2 这个引擎的「形状约定」

```
prompt_ids:      (B, P)          ← 调用方已左填充
output_ids:      (B*G, P+R)
completion_ids:  (B*G, R)
prompt_lens:     (B*G,) 全部等于 P
completion_mask: (B*G, R) 全 1
```

**注意**：本地 generate 不会在回答中途补 pad（EOS 之后是重复的 EOS），
所以 `completion_mask` 全 1；真正有效的长度由训练脚本按 EOS 再算一遍。

### 4.3 `update_policy`

```python
def update_policy(self, model):
    self.policy_model = model        # 只是换个引用
```

因为本地引擎直接持有模型对象，DDP 包装后换一下引用即可。

---

## 5. `SGLangRolloutEngine`：HTTP 调用远程推理

### 5.1 使用前提

文件顶部写了启动命令：

```bash
python -m sglang.launch_server --model-path ./minimind-3 \
       --attention-backend triton --host 0.0.0.0 --port 8998
```

### 5.2 rollout 流程

```python
# ① 去掉左侧 padding，只保留有效 token
input_ids_list = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask)]
all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

# ② 发请求
payload = {"input_ids": all_input_ids,
           "sampling_params": {"temperature": temperature,
                               "max_new_tokens": max_new_tokens,
                               "stop_token_ids": [eos_token_id]},
           "return_logprob": True}
resp = requests.post(f"{base_url}/generate", json=payload)

# ③ 解析
for result in results:
    completion_ids = meta["output_ids"]
    raw_logprobs   = meta["output_token_logprobs"]   # 形如 [[logp, token_id], ...]
    logprobs = [item[0] if isinstance(item, (list, tuple)) else item for item in raw_logprobs]

# ④ 补齐 → 转张量
output_ids = pad_to_tensor(all_output_ids, max_out_len, pad_id)
prompt_lens = [len(ids) for ids in all_input_ids]     # ← 每行不同，真实长度
```

| 与本地引擎的差异 | 影响 |
| --- | --- |
| prompt **去掉**左填充 | `prompt_lens` 是**每行不同的真实长度** |
| `output_ids` 是 prompt+completion **右补齐** | 形状仍是矩形，但填充在后面 |
| logp 来自服务端 | 与本地重算的 logp 可能有数值差异 |

**契约仍然成立**：训练脚本用 `prompt_lens` 算 `logp_pos = prompt_len - 1 + t`，
两种填充方式下都指向正确的 logits 位置——这是抽象设计的关键。

### 5.3 一个需要警惕的处理

```python
if len(logprobs) < len(completion_ids):
    logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
```

服务端返回的 logp 个数比 token 少时，**在开头补 0.0**。
`0.0` 在 log 空间里等价于概率 1——如果这些位置没被 `completion_mask` 屏蔽，
ratio 会立刻失真。

**排查手段**：训练时用 `--debug_log_ratio` 打印首轮 `ratio`，
正常应在 1.0 附近；如果偏离明显，先怀疑 logp 对齐。

### 5.4 `update_policy`：把权重推给服务

```python
if not dist.is_initialized() or dist.get_rank() == 0:
    state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
    unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
    self.tokenizer.save_pretrained(abs_path)
    resp = requests.post(f"{base_url}/update_weights_from_disk", json={"model_path": abs_path})

# 所有 rank 同步成功标志
if dist.is_initialized():
    dist.broadcast(ok_t, src=0); dist.barrier()
if not ok: raise RuntimeError("SGLang update_policy failed")
```

| 细节 | 原因 |
| --- | --- |
| 只有 rank0 存盘 | 共享存储只写一次 |
| 存成完整 HF 目录 | SGLang 只认磁盘路径加载 |
| 广播 `ok` | 防止某卡失败、其它卡继续训练 |
| 失败直接抛错 | 宁可中断，也不要拿旧策略继续更新 |

**代价**：每同步一次要写盘 128MB+（64M 模型 fp16），训练的 `save_interval` 越密越伤 IO。

### 5.5 两个运维辅助方法

```python
def flush_cache(self):  # POST /flush_cache，清空 KV cache
def health(self):       # GET /health，探活
```

训练脚本里没调用，是留给调试/运维的。

---

## 6. 两个引擎怎么做到「接口一致」

| 项 | Torch 引擎 | SGLang 引擎 |
| --- | --- | --- |
| prompt 填充 | 左填充，统一 P | 去填充，逐行真实长度 |
| output 填充 | 无（generate 统一长度） | 右侧补 pad |
| `prompt_lens` | 常数 P | 每行不同 |
| logp 来源 | 本地重算（同精度） | 服务端返回 |
| `completion_mask` | 全 1 | 按真实长度 |

**为什么训练代码两种都能用**：它从不假设「回答在第 P 位开始」，
而是用 `logp_pos = prompt_lens - 1 + t` 动态定位。
**抽象层把差异吃掉了，这是这个文件最大的价值。**

---

## 7. 工厂函数

```python
def create_rollout_engine(engine_type="torch", ...):
    if engine_type == "torch":  return TorchRolloutEngine(...)
    elif engine_type == "sglang": return SGLangRolloutEngine(...)
    else: raise ValueError(f"不支持的引擎类型: {engine_type}")
```

训练脚本通过 `--rollout_engine torch|sglang` 切换，其余代码不动。

---

## 8. 训练脚本怎么用它

以 GRPO 为例：

```python
prompt_inputs = tokenizer(prompts, padding="left", ...)
rollout_result = rollout_engine.rollout(prompt_inputs["input_ids"],
                                        prompt_inputs["attention_mask"],
                                        num_generations=6,
                                        max_new_tokens=1024, temperature=0.8)

outputs        = rollout_result.output_ids
completion_ids = rollout_result.completion_ids
old_logps      = rollout_result.per_token_logps
prompt_lens    = rollout_result.prompt_lens

logp_pos = prompt_lens.unsqueeze(1) - 1 + arange(R)   # ← 全靠 prompt_lens
...
if step % save_interval == 0:
    rollout_engine.update_policy(model)               # ← 同步权重
```

---

## 9. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| `repeat_interleave` 写成 `repeat` | 分组错乱，奖励与回答对不上 | 必须 `[p0,p0,p0,p1,p1,p1]` |
| logp 对齐错误 | ratio 远离 1，训练崩 | 用 `--debug_log_ratio` 检查；SGLang 补齐 0.0 时尤其要查 |
| 训练时 dropout > 0 | ratio 不≈1 | MiniMind 默认 dropout=0，改配置时要留意 |
| SGLang 请求超时 | 训练卡住 | `timeout=120` 默认；长序列要调大 |
| `update_policy` 频繁调用 | IO 打满、训练变慢 | 与 `save_interval` 对齐，别每步同步 |
| tokenizer 不一致 | 远程端解码/对齐错误 | `sglang_model_path` 必须指向同一份词表 |
| 本地引擎显存 | generate 时 KV cache 很大 | 长度 1024 时尤其明显 |
| `is_inference()` 张量 | 在需要梯度的路径上报错 | 函数内已 clone；自己扩展时注意 |
| 引擎不支持 logprob | 无法算 ratio | 接新引擎时先确认 API 支持返回 logp |

---

## 10. 如果要接 vLLM，改什么

只需新增一个类：

```python
class VLLMRolloutEngine(RolloutEngine):
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature):
        # 1. 发请求（vLLM 的 /generate 或离线 LLM.generate）
        # 2. 拿到 token ids 与 prompt_logprobs / 或本地重算
        # 3. 组装 RolloutResult（六个字段，注意 completion_mask 与 prompt_lens）
        ...
    def update_policy(self, model):
        # 要么重建引擎（重加载权重），要么用 vLLM 的权重更新接口
        ...
```

**必须满足的三条**：

1. `per_token_logps` 与 `completion_ids` 逐 token 对齐；
2. `prompt_lens` 是每行的有效长度；
3. `completion_mask` 标出真实 token（含 EOS，不含 padding）。

---

## 11. 自测题

1. 为什么要把 rollout 抽成接口？
2. `RolloutResult` 的六个字段分别给谁用？
3. `logits_to_keep=n_keep+1` 省了什么？为什么要 +1？
4. 为什么本地引擎先 `repeat_interleave` 再让 `num_return_sequences=1`？
5. 两个引擎的 `prompt_lens` 语义有什么不同？训练代码为什么都能用？
6. SGLang 返回的 logp 不够长时怎么处理？风险是什么？
7. `update_policy` 为什么要广播成功标志？
8. `.clone()` 在本地引擎里防什么？
9. 接新引擎必须满足哪三条契约？
10. 为什么 rollout 阶段要开 autocast？

### 答案要点

1. 采样占大头，解耦后可以换高性能引擎、训练推理分离。
2. 重新前向、定位回答、ratio 分母、文本打分、算位置、屏蔽 padding。
3. 只为尾部位置算输出层；+1 是为了错位后正好覆盖 n_keep 个预测。
4. 分组顺序可控（`[p0,p0,p1,p1]`），且避免 generate 内部再做一次复制。
5. 本地是统一常数 P（左填充），远程是逐行真实长度；训练用 `prompt_lens` 动态定位。
6. 开头补 0.0（log 概率为 1）；若 mask 没挡住会让 ratio 失真。
7. 防止某卡失败、其它卡继续训练导致策略不一致或死锁。
8. `MiniMind.generate` 返回 inference tensor，后续 cat 等操作前需要脱离。
9. logp 对齐、prompt_lens 正确、completion_mask 正确。
10. 与训练精度一致，保证「采样 logp」与「重算 logp」可比（ratio≈1）。

---

## 12. 一表速查

| 部分 | 关键点 |
| --- | --- |
| RolloutResult | 六个字段构成引擎契约 |
| compute_per_token_logps | `logits_to_keep` 只算尾部 |
| ABC | `rollout` + `update_policy` |
| Torch 引擎 | 左填充、repeat_interleave、本地 generate |
| SGLang 引擎 | 去填充、HTTP、logp 服务端返回、磁盘同步权重 |
| 工厂 | `--rollout_engine torch|sglang` |
| 一致性 | 靠 `prompt_lens` 抹平填充差异 |
