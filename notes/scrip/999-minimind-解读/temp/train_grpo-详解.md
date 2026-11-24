# `train_grpo.py` 详解：没有 critic 的强化学习

> 对象：`minimind-master/trainer/train_grpo.py`（321 行）
> 配套：`rollout_engine.py`、`RLAIFDataset`、`LMForRewardModel`、`rep_penalty`
> 说明：本文按源码讲解，不跑训练（本机显存放不下 1.8B 奖励模型，见第 8 节）。

---

## 0. 定位：把 PPO 的 critic 换成「组内比较」

PPO 需要四个模型：actor、critic、ref、reward。
GRPO（Group Relative Policy Optimization）的核心洞察是：

> **不需要价值网络来估计基线——同一个问题采样多条回答，互相比较就够了。**

| 项 | PPO | GRPO |
| --- | --- | --- |
| 模型数 | 4 | **3**（actor / ref / reward） |
| 基线来源 | critic 估价值 + GAE | **同组回答的均值** |
| 每条 prompt 采样 | 1 条 | **6 条** |
| 一次 rollout 更新 | 2 轮 × mini-batch | **1 次** |
| 损失 | clip + value + KL | clip 或 CISPO + KL |
| 显存 | 最高 | 少一个模型 |

> 口诀：**一题多答，组内排名；均值当基线，方差做归一。**

---

## 1. 核心公式：组内相对优势

```python
grouped_rewards = rewards.view(-1, num_generations)          # (B, G)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G)    # 组内均值
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)
advantages = (rewards - mean_r) / (std_r + 1e-4)             # 标准化
```

**一句话**：同一道题生成 6 个回答，第 i 个回答的优势 = 「它的分数」减去「这 6 个的平均」，
再除以标准差。高于平均为正，低于平均为负。

**为什么这样可以替代 critic**：

| 方法 | 基线估计 | 成本 |
| --- | --- | --- |
| PPO | 训练一个价值网络 V(s) | 多一个模型 + 训练不稳定 |
| GRPO | 同组样本的均值 | 多采 5 条回答 |

对「答案可自动验证」的任务（数学、代码、工具调用），多采样比训练 critic 划算得多。

---

## 2. 数据与采样

### 2.1 数据：只要 prompt

`RLAIFDataset` 返回 `{"prompt": str, "answer": ""}`——只要问题，答案由模型现场生成。
`thinking_ratio=0.9` 表示 90% 的样本在模板里开启思考。

### 2.2 编码：左填充 + 左截断

```python
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          return_token_type_ids=False, padding_side="left",
                          add_special_tokens=False).to(device)
if args.max_seq_len:
    prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
    prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

| 设计 | 原因 |
| --- | --- |
| `padding_side="left"` | 生成从末尾续写 |
| `[:, -max_seq_len:]` | 超长时**保留最近的部分**（保留 system + 最近几轮） |
| `add_special_tokens=False` | 模板里已有特殊标记 |

### 2.3 采样：一次 6 条

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"], attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,        # 6
    max_new_tokens=args.max_gen_len,             # 1024
    temperature=0.8)
```

返回仍然是那套：`output_ids`、`completion_ids`、`completions`、`prompt_lens`、
以及**采样时的旧 log 概率** `per_token_logps`。

**为什么温度是 0.8 而不是 0**：组内需要**有差异**。如果 6 条回答一模一样，
均值等于每条，优势全为 0——这一题就白采了。

---

## 3. 奖励：与 PPO 同一套规则

```python
rewards = calculate_rewards(prompts, completions, reward_model)   # (B*G,)
```

| 项 | 分值 |
| --- | --- |
| 回答长度 20~800 | ±0.5 |
| 思考长度 20~300 | +1.0 / -0.5 |
| `<think>` 恰好一次 | ±0.25 |
| 3-gram 重复 | -0 ~ -0.5 |
| 奖励模型 | [-3, 3] |

**注意形状**：`completions` 长度是 `B × G`（每道题 6 条），所以奖励也是 `B*G`。

---

## 4. `grpo_train_epoch` 逐段

### 4.1 定位回答 token（与 PPO 相同）

```python
full_mask = (outputs != pad_token_id).long()
logp_pos = prompt_lens.unsqueeze(1) - 1 + arange(R)
full_mask.scatter_(1, logp_pos + 1, completion_mask)
```

回答第 t 个 token 在序列里的位置是 `P+t`，它的 logits 在 `P+t-1`。

### 4.2 一次前向：策略与参考

```python
with autocast_ctx:
    res = model(outputs, attention_mask=full_mask)
    per_token_logps = log_softmax(res.logits[:, :-1]).gather(2, outputs[:, 1:]).gather(1, logp_pos)

with torch.no_grad():
    ref_per_token_logps = ...同上，用 ref_model...
```

拿到三份概率：

| 变量 | 来源 | 用途 |
| --- | --- | --- |
| `old_per_token_logps` | rollout 采样时 | ratio 的分母 |
| `per_token_logps` | 当前策略前向 | ratio 的分子 |
| `ref_per_token_logps` | 参考模型 | KL 惩罚 |

### 4.3 组内优势

见第 1 节。两个细节：

- `std(unbiased=False)`：用总体标准差（除以 G 而不是 G-1），G=6 时差别不大但更稳定；
- `+1e-4`：防止组内全同分时除零。

### 4.4 KL 与 ratio

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1     # k3 估计
ratio = torch.exp(per_token_logps - old_per_token_logps)
```

**k3 估计量的好处**：永远非负、无偏、数值稳定。比直接 `-kl_div` 好用。

### 4.5 两种损失：GRPO 与 CISPO

```python
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                       - args.beta * per_token_kl)
else:  # grpo
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss = -(torch.min(ratio * A, clipped_ratio * A)
                       - args.beta * per_token_kl)
```

**两者的本质差异**：

| | GRPO（PPO 式裁剪） | CISPO |
| --- | --- | --- |
| 裁剪 | 双侧 `[1-ε, 1+ε]` | 只截上界 `ε_high=5.0` |
| ratio 是否 detach | 否（但 min 后可能梯度为 0） | **是** |
| 梯度路径 | 超出信任域的 token 梯度被**截断为 0** | 所有权重的梯度都**经过 logp** 回传 |
| 效果 | 保守、稳 | 保留更多学习信号，尤其负优势样本 |

**CISPO 的动机**：PPO 式裁剪会把「ratio 跑太远」的 token 的梯度直接清零，
在 GRPO 这种「组内差异大」的场景里，容易丢掉有价值的样本。
CISPO 把 ratio 当**常数权重**（detach + 只截上界），让每个 token 都能贡献梯度。

`epsilon_high=5.0` 很宽，实际只防极端值。

### 4.6 汇总与更新

```python
policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)).mean()
loss = (policy_loss + aux_loss) / accumulation_steps
loss.backward()

if step % accumulation_steps == 0 or step == iters:
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

**注意**：GRPO **没有** PPO 那样的 `ppo_update_iters` 和 mini-batch 循环——
一次 rollout 只做一次更新。这也是它比 PPO 简单的地方。

### 4.7 保存与权重同步

```python
torch.save({k: v.half().cpu() ...}, f'{save_dir}/{save_weight}_{hidden}.pth')
lm_checkpoint(..., model=model, optimizer=optimizer, scheduler=scheduler)
...
if step % save_interval == 0 or step == iters:
    rollout_engine.update_policy(model)      # 同步给 SGLang
```

保存的 actor 权重前缀是 `grpo`（`grpo_768.pth`），可作为后续 DPO/评测的输入。

---

## 5. `main`：与 PPO 的差异

### 5.1 参数速览

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `num_generations` | **6** | 每题采样条数 |
| `beta` | 0.1 | KL 惩罚系数 |
| `loss_type` | **cispo** | `grpo` / `cispo` |
| `epsilon` | 0.2 | GRPO 裁剪 |
| `epsilon_high` | 5.0 | CISPO 上界 |
| `learning_rate` | 3e-7 | |
| `max_seq_len` / `max_gen_len` | 768 / 1024 | prompt / 生成 |
| `temperature` | 0.8（写死） | 采样温度 |
| `thinking_ratio` | 0.9 | |

### 5.2 三个模型

```python
model, tokenizer = init_model(lm_config, base_weight)       # 策略
ref_model = init_model(...).eval().requires_grad_(False)    # 参考
reward_model = LMForRewardModel(path, dtype=torch.float16)  # 裁判
```

**没有 critic**——这就是和 PPO 最大的区别。

### 5.3 调度器步数

```python
total_optimizer_steps = ceil(iters / accumulation_steps) * epochs
```

比 PPO 简单：没有 `ppo_update_iters`、没有 mini-batch 系数，因为每步只更新一次。

### 5.4 一个容易忽略的配置

```python
lm_config = MiniMindConfig(..., max_seq_len=args.max_seq_len + args.max_gen_len, ...)
```

`max_seq_len=1792` 被传给 `RLAIFDataset` 用作编码上限——**prompt + 生成的总长**。
（`MiniMindConfig` 本身没有这个字段，它通过 `**kwargs` 存在 `PretrainedConfig` 上。）

### 5.5 DDP 与 rollout 同步

```python
model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
rollout_engine.update_policy(model)
```

`broadcast_buffers=False` 的理由同 PPO：RoPE buffer 各卡一致，不需要每步广播。

---

## 6. 超参为什么这么取

| 超参 | 值 | 理由 |
| --- | --- | --- |
| G | 6 | 太少组内统计不稳，太多采样成本翻倍 |
| temperature | 0.8 | 保证组内有差异 |
| beta | 0.1 | 比 PPO 的 0.02 更大：GRPO 一次 rollout 只更新一次，作者选择更紧的 KL 约束 |
| epsilon | 0.2 | 标准 PPO 值 |
| epsilon_high | 5.0 | CISPO 只防极端 |
| lr | 3e-7 | RL 通用的小学习率 |
| epochs | 1 | RL 数据是「每题现场生成」，多轮意义不大 |

---

## 7. 显存账

| 组件 | 占用 |
| --- | --- |
| 策略（fp16 前向 + fp32 主权重 + Adam） | ~1 GB |
| 参考（fp16，冻结） | ~0.13 GB |
| **奖励模型（1.8B fp16）** | **~3.6 GB** |
| 激活（6 条 × 1024 token × batch 2） | 1~2 GB |

**结论**：与 PPO 一样，本机 4GB 跑不了——瓶颈是奖励模型。
想在本机体验 GRPO 的思路，建议用**纯规则奖励**的任务（如数学题，`train_agent.py` 的做法），
把 RM 去掉。

---

## 8. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| **组内奖励全相同** | advantages 全 0，loss 只有 KL 项 | 采样温度太低 / 奖励区分度差；提高温度或换任务 |
| 奖励方差过小 | 学习极慢 | 同上；检查 RM 是否给所有回答打同一个分 |
| ratio 不≈1 | KL 曲线异常 | 采样与更新时的精度/模式不一致；`dropout` 必须为 0 |
| 只保存 actor | 想续训找不到 critic | GRPO 本来就没有 critic；resume 里有 optimizer/scheduler |
| `use_sglang` 参数 | 传了但函数体没用到 | 代码遗留；引擎类型由 `rollout_engine` 对象决定 |
| prompt 超长 | 被左截断，丢了 system | `max_seq_len` 太小；注意 `[:, -max_seq_len:]` 保留尾部 |
| `long` 的 completion 被截断 | 没有 eos，mask 覆盖到末尾 | `max_gen_len` 要够；否则奖励长度项也会异常 |
| 奖励模型路径 | 默认路径不存在 | 需自行下载 internlm2-1.8B-reward |
| 长度偏置 | 模型学会写长 | 奖励里有长度塑形，但只在 20~800 区间 |

**其中第一条最重要**：GRPO 的全部信号来自「组内差异」。如果 6 条回答得分一样，
数学上优势就是 0，训练等于空转。排查顺序：先看日志里的 `Reward` 和 `Adv Std`。

---

## 9. 怎么判断 GRPO 在学

日志每步打印：

```
Reward: 0.42, KL_ref: 0.0012, Adv Std: 0.83, Adv Mean: 0.00,
Actor Loss: -0.02, Avg Response Len: 156.3, Learning Rate: 3.0e-7
```

| 指标 | 健康表现 | 异常 |
| --- | --- | --- |
| `Reward` | 缓慢上升 | 长期不动 → 奖励没区分度 |
| `Adv Std` | ≈1（已标准化） | 接近 0 → 组内无差异 |
| `Adv Mean` | ≈ 0 | 明显偏离 0 → 归一化有问题 |
| `KL_ref` | 缓慢上升但很小 | 快速上升 → 偏离参考太远，调大 beta |
| `Avg Response Len` | 稳定或缓慢增长 | 暴涨 → 模型在「凑长度」骗奖励 |

---

## 10. 与 PPO 的逐项对照

| 环节 | PPO | GRPO |
| --- | --- | --- |
| 采样数量 | 1 条/prompt | 6 条/prompt |
| 优势 | critic + GAE | 组内标准化 |
| 价值损失 | 有 | **无** |
| 更新轮数 | 2 轮 × mini-batch | 1 次 |
| KL 系数 | 0.02 | 0.1 |
| 损失 | clip + vf·value + KL | clip/CISPO + KL |
| 调度器步数 | 含 update_iters 与 mb_factor | 只含 accumulation |
| 显存 | 4 模型 | 3 模型 |

**一句话总结**：GRPO = 用「多采样 + 组内比较」换掉「critic + GAE」，
代码少了 120 行，显存少一个模型，代价是采样量 ×6。

---

## 11. 自测题

1. GRPO 用什么替代了 critic？
2. 为什么每条 prompt 要采样 6 条？
3. 组内奖励全相同时会发生什么？
4. `std` 为什么用 `unbiased=False`？`+1e-4` 防什么？
5. CISPO 与 GRPO 损失的三个差异是什么？
6. CISPO 为什么要把 ratio detach？
7. KL 为什么用 `exp(kl)-kl-1`？
8. 为什么一次 rollout 只更新一次？
9. `max_seq_len` 在 config 里为什么是 prompt+gen？
10. `use_sglang` 参数为什么没被用到？
11. 判断「训练是否在学」看哪三个指标？
12. 本机 4GB 为什么跑不了？

### 答案要点

1. 同一 prompt 多条回答的组内均值/标准差。
2. 组内比较需要样本，且能降低基线估计的方差。
3. 优势全 0，只剩 KL 项，模型不学。
4. 总体标准差；防止除零。
5. 裁剪单侧 vs 双侧；ratio 是否 detach；梯度是否保留。
6. 让 ratio 只作权重，梯度全部经由 logp，避免裁剪清零。
7. 非负、无偏、稳定。
8. 采样贵，且 GRPO 用一次更新换取稳定。
9. 编码上限要覆盖 prompt + 生成。
10. 遗留参数；实际由 `rollout_engine` 对象决定。
11. Reward、Adv Std、KL_ref。
12. 奖励模型 1.8B fp16 ≈ 3.6GB，加上其它模型超 4GB。

---

## 12. 一表速查

| 段落 | 关键点 |
| --- | --- |
| 数据 | 只要 prompt，thinking_ratio=0.9 |
| 采样 | 左填充、6 条、温度 0.8 |
| 奖励 | 规则 + RM，形状 B×G |
| 优势 | `(r - 组均值) / (组标准差)` |
| KL | k3：`exp(kl)-kl-1` |
| 损失 | 默认 CISPO；可选 PPO 式 GRPO |
| 更新 | 每步一次，无 mini-batch 循环 |
| 调度 | `ceil(iters/accum) × epochs` |
| 保存 | `grpo_768.pth` + resume |
| 显存 | 3 模型，RM 占 3.6GB |
