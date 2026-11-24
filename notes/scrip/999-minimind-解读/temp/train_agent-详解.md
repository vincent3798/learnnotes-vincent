# `train_agent.py` 详解：多轮工具调用的强化学习

> 对象：`minimind-master/trainer/train_agent.py`（484 行，全项目最长的训练脚本）
> 配套：`AgentRLDataset`、`rollout_engine`、`safe_math_eval`、`LMForRewardModel`
> 说明：本文按源码讲解，不跑训练。

---

## 0. 定位：GRPO + 一个「环境」

前面所有 RL 脚本都是一轮问答：prompt 进、回答出、打分、更新。
Agent RL 多了一个东西——**模型可以在回答里调用工具，拿到结果后继续回答**。

```
用户问题 → 模型输出 <tool_call> → 执行工具 → 结果塞回对话 → 模型继续 → …（最多 3 轮）
```

| 项 | GRPO | Agent RL |
| --- | --- | --- |
| 交互轮数 | 1 | 最多 3 |
| 奖励来源 | 规则 + RM | **几乎纯规则**（工具 + GT 校验） |
| 每条 prompt 采样 | 6 | 4（但每条约 3 轮 × 768 token） |
| 训练数据 | prompt | prompt + tools + **gt** |
| 复杂度 | 中 | **最高** |

> 口诀：**模型提请求，环境给结果；奖励靠验证，多轮再结算。**

---

## 1. 工具层：六个假工具，真接口

### 1.1 工具 schema（:34）

```python
TOOLS = [
  {"type": "function", "function": {"name": "calculate_math",
      "description": "计算数学表达式",
      "parameters": {"type": "object",
                     "properties": {"expression": {"type": "string"}},
                     "required": ["expression"]}}},
  ...
]
```

六个工具：`calculate_math`、`unit_converter`、`get_current_weather`、`get_current_time`、
`get_exchange_rate`、`translate_text`。

**为什么用 OpenAI function-calling 格式**：这是当前最通用的工具描述协议，
能直接塞进 chat template 的 tools 分支。

### 1.2 三层实现（:44-68）

| 层 | 内容 | 作用 |
| --- | --- | --- |
| `WEATHER_DATA` / `TIME_DATA` / `EXCHANGE_DATA` … | 写死的字典 | 模拟真实 API |
| `MOCK_RESULTS` | 每个工具一个 lambda | 真正「执行」 |
| `CHECK_ARGS` | 每个工具一个校验函数 | 判断参数是否合法 |

**为什么要 mock**：训练环境必须是**确定性、零成本、可重复**的。
真调天气 API 会让同一道题每次得分不同，RL 无法学习。

---

## 2. 解析与执行

### 2.1 `parse_tool_calls`（:71）

```python
for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
    try: call = json.loads(m.strip())
    except: continue
    args = call.get("arguments", {})
    if isinstance(args, str):
        try: args = json.loads(args)
        except (ValueError, RecursionError): args = {}
```

| 处理 | 原因 |
| --- | --- |
| 正则抓 `<tool_call>...</tool_call>` | 模板规定的格式 |
| json 解析失败就跳过 | 模型可能输出半截 JSON |
| `arguments` 是字符串再解析一次 | 模型有时会双重编码 |
| `RecursionError` | 极端嵌套 JSON |

**注意**：解析失败不会报错，只是「没有调用」——奖励里会用「工具数量不符」来惩罚。

### 2.2 `execute_tool`（:85）

```python
fn = MOCK_RESULTS.get(name)
if not fn: return {"error": f"未知工具: {name}"}
if valid_names is not None and name not in valid_names: return {"error": ...}
if not CHECK_ARGS[name](args): return {"error": ...}
try: return fn(args)
except Exception as e: return {"error": f"工具执行失败: {str(e)[:80]}"}
```

**四级防御**：工具存在 → 在允许列表 → 参数合法 → 执行不抛异常。
**所有失败都变成 `{"error": ...}` 字符串**塞回对话——模型能学会「看到错误就改」。

---

## 3. 多轮 rollout：本文件最烧脑的一段

### 3.1 `rollout_single`（:96）：一个状态机

```python
open_thinking = random.random() < thinking_ratio      # 整条轨迹共用一个开关
for turn in range(max_turns):                         # 最多 3 轮
    context = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
    if prompt_ids is None:
        prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]   # 只算一次

    input_ids = torch.tensor([prompt_ids + response_ids])      # prompt + 已生成
    rollout_result = rollout_engine.rollout(..., num_generations=1, temperature=0.8)

    new_ids  = completion_ids[0, :valid_len]      # 本轮模型新生成
    new_logps = per_token_logps[0, :valid_len]
    ...
    calls = parse_tool_calls(new_text)
    if not calls: break                            # 没有调用 → 结束

    # 执行工具，把结果作为 tool 消息塞回
    for call in calls:
        result = execute_tool(call["name"], call["arguments"], valid_names)
        messages.append({"role": "tool", "content": json.dumps(result)[:2048]})
```

**关键设计：`prompt_ids` 只算一次，后续用 `prompt_ids + response_ids` 续接。**

为什么可以这样：chat template 是确定性的，同一个 messages 列表渲染结果相同；
所以第二轮的完整输入 = 第一轮 prompt + 第一轮生成 + 观察 + 新的 assistant 头。
`response_ids` 就是「第一轮生成 + 观察 + 后续所有生成」的拼接。

### 3.2 最精妙的一段：把观察结果对齐成 token（:140-154）

工具执行完，结果是文本。要把它拼进 `response_ids`，必须先知道它对应哪些 token：

```python
marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
assistant_message["content"] += marker
marked_context = tokenizer.apply_chat_template(messages, ...)      # 带标记渲染一次
assistant_message["content"] = new_text                            # 还原

_, found, observation = marked_context.partition(marker)           # 标记之后的全部文本
obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
if new_ids and new_ids[-1] == eos and obs_delta[:1] == [eos]:
    obs_delta = obs_delta[1:]                                      # 去掉重复的 EOS

response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))                         # 观察不训练
response_old_logps.extend([0.0] * len(obs_delta))
```

**为什么需要标记**：不能直接 `tokenizer(json.dumps(result))`——模板会在工具结果外面包
`<|im_start|>user\n<tool_response>...`，还有下一个 assistant 头。
标记法一次性拿到「标记之后的所有文本」，再整体 tokenize，**与下一轮真实输入完全对齐**。

**为什么 mask 设 0**：工具返回的内容不是模型生成的，不该被训练。
但它们在上下文里，模型生成后续 token 时要看到它们。

**为什么要处理重复 EOS**：生成可能以 EOS 结尾，而模板紧接着又加了换行/标记，
导致两个 EOS 相邻，去掉一个保持 token 串干净。

### 3.3 `rollout_batch`（:160）

对每个样本复制 `num_gen` 份（浅拷贝 messages —— 因为 `rollout_single` 会往里 append），
逐条跑 `rollout_single`。**串行执行**，所以这一步在 batch 2 × 4 条时就已经很慢。

---

## 4. 打包与对齐（`rl_train_epoch`，:240）

### 4.1 把变长轨迹拼成张量

```python
ids  = prompt_ids + response_ids
mask = [0]*len(prompt_ids) + response_mask
old_logps = [0.0]*max(len(prompt_ids)-1, 0) + response_old_logps   # 注意 -1

if len(ids) > args.max_total_len:          # 截断只保留尾部
    ids = ids[-max_total_len:]; mask = mask[-max_total_len:]
    old_logps = old_logps[-(len(ids)-1):]

prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
```

| 设计 | 原因 |
| --- | --- |
| `old_logps` 前面补 `len(prompt)-1` 个 0 | logits 与 token 错位一位：位置 i 的 logp 对应 token i+1 |
| 提示词部分 mask=0 | 不训练提问 |
| 观察部分 mask=0（rollout 时就设好） | 不训练工具结果 |
| 截断保留尾部 | 保留最近上下文；同时 `old_logps` 同步截到 `len-1` |

### 4.2 填充成矩形

```python
input_ids        = pad(ids)                      # 右补 pad
full_response_masks = pad(mask)
old_per_token_logps = pad(old_logps, to max_len-1)
full_mask        = (arange(max_len) < seq_lens)  # 真实长度
```

**注意**：这里用的是**右填充**（因为整条序列已经确定了，不像生成需要左填充）。

### 4.3 前向与掩码

```python
res = model(input_ids, attention_mask=full_mask)
per_token_logps = log_softmax(res.logits[:, :-1]).gather(2, input_ids[:, 1:]).squeeze(-1)
ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, L-1, attention_mask=full_mask)

completion_mask = full_response_masks[:, 1:]
# 再按第一个 EOS 截断一次
completion_mask = completion_mask * (pos <= eos_idx)
```

**与 PPO/GRPO 的差别**：这里没有 `logp_pos`，而是在**整条序列**上算 logp，
再用 mask 挑出「模型生成的 token」。因为轨迹里生成与观察交错，用位置索引不如用 mask。

---

## 5. 奖励：纯规则化的 RLVR（:189）

奖励分两个分支：

### 5.1 通用扣分

```python
reward -= 0.5 * sum(|count('<tool_call>') - count('</tool_call>')|)   # 标签不成对
```

### 5.2 没有工具调用：格式 + RM

| 项 | 分数 |
| --- | --- |
| 回答长度 5~800 | ±0.5 |
| 思考长度 20~300 | +1.0 / -0.5 |
| `<think>` 恰好一次 | ±0.25 |
| 奖励模型 | [-3, 3] |
| 重复惩罚 | 负 |

### 5.3 有工具调用：执行结果 + GT 校验（核心）

```python
valid_call_count = Σ(name 在允许列表 且 参数合法)
tool_gap = |valid_call_count - len(gt)| + max(0, len(tool_calls) - valid_call_count)
reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap       # 工具对齐分

final_text = "" if unfinished else answer.split('</tool_call>')[-1]
verified = validate_gt_in_text(final_text, gt)             # 从最终回答里找 GT
reward += 2.5 * len(verified) / len(gt)                    # GT 分（最高 +2.5）
if unfinished: reward -= 0.5
```

### 5.4 `validate_gt_in_text`（:184）：怎么「自动判卷」

```python
nums = re.findall(r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_]|\.\d)', text)
# 数字型 gt：与回答里的数字比较，误差 < 1e-6
# 字符串型 gt：忽略大小写做子串匹配
```

两个正则细节值得学：

| 细节 | 作用 |
| --- | --- |
| `(?<![A-Za-z0-9_.])` 与 `(?![A-Za-z0-9_]|\.\d)` | 排除 `v1.2`、`abc123`、`3.14.15` 这类误匹配 |
| 数字误差 1e-6 | 允许不同精度写法（`1.50` vs `1.5`） |

**这就是 RLVR（可验证奖励）的核心**：奖励不需要模型打分，只需要**规则能判定对错**。

---

## 6. 训练步：与 GRPO 同构

拿到优势之后的损失与 GRPO 完全一致：

```python
advantages = (rewards - 组均值) / (组标准差 + 1e-4)

kl_div  = ref - policy
per_token_kl = exp(kl_div) - kl_div - 1          # k3
ratio = exp(policy - old)

# 默认 CISPO
per_token_loss = -(clamp(ratio, max=5.0).detach() * A * logp - beta * kl)
```

**唯一的差别在聚合**：

```python
policy_loss = (per_token_loss * completion_mask).sum(1)[valid_rows] / token_counts[...].mean()
```

只对 `token_counts > 0` 的行求平均，避免空轨迹（全被 mask）导致 nan。

---

## 7. `main` 的差异

| 项 | 说明 |
| --- | --- |
| `AgentRLDataset` | 返回 `{messages, tools, gt}`，**丢掉最后一条消息**（通常是标准答案的 assistant 回复，避免把答案喂给模型） |
| `collate_fn` | 自定义：把 dict 列表原样打包，不转张量 |
| `num_generations` | 4（比 GRPO 的 6 少，因为每条更贵） |
| `max_turns` | 3（写死在 rollout 调用里） |
| `max_gen_len` | 768（单轮） |
| `max_total_len` | 2500（整条轨迹训练时的上限） |
| `thinking_ratio` | **0.1**（比 GRPO 的 0.9 低很多） |
| 奖励模型 | 仍然加载，但**只在无工具分支使用** |

**为什么 thinking_ratio 只有 0.1**：Agent 任务以「调工具 + 给结论」为主，
长思考收益低、还拖慢生成。

---

## 8. 与 GRPO 的逐项对照

| 环节 | GRPO | Agent RL |
| --- | --- | --- |
| 采样 | 一次生成 | 最多 3 轮，中间插工具结果 |
| 输入构造 | prompt + 一次回答 | prompt + 生成/观察交错 |
| 位置对齐 | `logp_pos` 索引 | 全靠 mask |
| 奖励 | 规则 + RM | 规则 + GT 校验（工具分支不用 RM） |
| 数据 | prompt | prompt + tools + gt |
| 成本 | 6 条 × 1024 | 4 条 × 3 轮 × 768 |

---

## 9. 成本与显存

| 项 | 量级 |
| --- | --- |
| 每条样本生成 | 最多 3 × 768 token |
| 每步生成 | batch 2 × 4 条 = 8 条轨迹 |
| 训练序列 | 最多 2500 token × 8 行 |
| 模型 | policy + ref + RM（1.8B ≈ 3.6GB） |

**结论**：与 PPO/GRPO 一样，本机 4GB 因奖励模型而不可行。
即使去掉 RM（工具分支本来就不用它），8 条 2500 token 的轨迹也远超 4GB。

---

## 10. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| 模板不稳定 | 标记找不到，抛 `chat template did not preserve...` | 模板必须原样保留 assistant content |
| 观察 token 对不齐 | ratio 异常、KL 爆炸 | marker 法失败；检查 tokenizer 是否把标记拆开 |
| EOS 重复 | 序列里出现两个 EOS | 已有处理（去掉一个）；换模板要重测 |
| 截断切掉 prompt | `prompt_len=0` | `max_total_len` 太小 |
| 全空轨迹 | `valid_rows` 全 False | 奖励/生成长度异常；代码已保护不报 nan |
| 工具参数非法 | 奖励被扣 | 模型在学习；观察结果里的 error 就是监督信号 |
| 工具名不在允许列表 | `{"error": "当前样本未提供工具"}` | 数据里的 tools 要与调用一致 |
| 串行 rollout 很慢 | 每步几十秒 | 换 SGLang 引擎可显著加速 |
| `use_sglang` 参数 | 传了但未使用 | 与 GRPO 一样的遗留 |
| RM 只在无工具分支 | 工具任务不享语义奖励 | 设计如此；工具任务靠规则 |

---

## 11. 自测题

1. Agent RL 与 GRPO 的核心区别是什么？
2. 工具为什么用 mock 实现？
3. `parse_tool_calls` 遇到坏 JSON 怎么处理？
4. `rollout_single` 里 `prompt_ids` 为什么只算一次？
5. marker 技巧解决什么问题？
6. 观察结果的 mask 为什么是 0？它参与训练吗？
7. `old_logps` 前面为什么补 `len(prompt)-1` 个 0？
8. 为什么这里用 mask 而不是 `logp_pos`？
9. 工具分支的奖励由哪几项组成？
10. `validate_gt_in_text` 的两个正则边界在防什么？
11. 为什么 thinking_ratio 只有 0.1？
12. 为什么空轨迹要单独处理？

### 答案要点

1. 可多轮调用工具，奖励来自工具执行与 GT 校验。
2. 确定性、零成本、可重复；真实 API 会让 RL 无法学习。
3. 静默跳过，视为「没调用」，后续用工具数量差扣分。
4. chat template 确定性，后续输入 = prompt + 已累积的 response。
5. 精确取出「工具结果 + 下一轮头部」的 token，保持与真实输入对齐。
6. 不是模型生成的内容；mask=0 → 不参与损失，但在上下文里。
7. logits 与 token 错位一位。
8. 轨迹里生成与观察交错，位置索引不可靠。
9. 标签成对、工具数量对齐、GT 命中、未完成扣分、重复惩罚。
10. 避免把版本号/标识符里的数字当成答案。
11. Agent 任务以工具调用为主，长思考收益低、成本高。
12. 防止 `sum/0` 产生 nan。

---

## 12. 一表速查

| 段落 | 关键点 |
| --- | --- |
| 工具 | 6 个 mock + schema + 校验 |
| 解析 | 正则 + JSON，失败静默 |
| 多轮 | 最多 3 轮，生成/观察交错 |
| marker | 把观察对齐成 token |
| 打包 | prompt+response，右填充，mask 标记 |
| 奖励 | 规则 + GT（数字容差 1e-6） |
| 优势 | 组内标准化（G=4） |
| 损失 | 与 GRPO 相同（默认 CISPO） |
| 成本 | 4 条 × 3 轮 × 768 token |
