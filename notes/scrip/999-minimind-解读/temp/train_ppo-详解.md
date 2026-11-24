# `train_ppo.py` 详解：四个模型的强化学习闭环

> 对象：`minimind-master/trainer/train_ppo.py`（442 行）
> 配套：`trainer/rollout_engine.py`、`dataset/lm_dataset.py` 的 `RLAIFDataset`、`trainer_utils.py`
> 说明：本文按源码讲解，不跑训练（本机 4GB 显存放不下奖励模型，见第 9 节）。

---

## 0. 定位：整个项目最复杂的脚本

前面几个脚本都是「一个模型 + 一个损失」，PPO 第一次出现**四个模型同时在场**：

| 模型 | 角色 | 是否训练 | 输入 → 输出 |
| --- | --- | --- | --- |
| **Actor** | 策略，生成回答 | ✅ | token → logits |
| **Critic** | 价值，评估「这个状态值多少」 | ✅ | token → 每个位置一个标量 |
| **Ref** | 参考，算 KL 约束 | ❌ 冻结 | token → logits |
| **Reward** | 裁判，给回答打分 | ❌ 冻结 | 对话 → 一个分数（外部模型） |

**两阶段循环**：

```
rollout（采样）: actor 生成回答 → reward 打分 → critic 估价值 → GAE 算优势
      ↓
update（更新）: 用同一批数据，把 actor 和 critic 各更新 ppo_update_iters 轮
```

> 口诀：**生成—打分—估价值—算优势—裁剪更新；四个模型各司其职，actor 与 critic 同时学。**

---

## 1. 文件结构

```
CriticModel(MiniMindForCausalLM)        :30   带 value_head 的价值网络
calculate_rewards(prompts, responses)   :45   规则奖励 + 奖励模型
ppo_train_epoch(...)                    :72   核心：rollout + update
if __name__ == "__main__":              :300  参数 + 九段初始化 + epoch 循环
```

文件顶部还有一段说明（:0-10 附近）列出 SGLang 的启动命令——因为 rollout 可以换成 HTTP 引擎。

---

## 2. `CriticModel`：给语言模型加一个「价值头」

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)   # (B, L)
        return values
```

**三个要点**：

| 要点 | 说明 |
| --- | --- |
| 继承 `MiniMindForCausalLM` | 复用全部主干权重，只需加载后接一个头 |
| `value_head` 是新参数 | 所以加载 SFT 权重时必须 `strict=False` |
| **归一化做了两次** | 主干 `MiniMindModel.forward` 末尾已经 `self.norm(hidden)` 过；这里又 `self.model.norm(outputs[0])` 了一次 |

**关于双重 norm**：这不是笔误式的崩溃，但严格说两次带可学习缩放的 RMSNorm 并不等价于一次。
实际影响有限（RMSNorm 近似幂等），读代码时知道有这回事即可；如果要严格复现论文，建议去掉一次。

**为什么不直接复用 `lm_head`**：价值是一个标量，不是词表分布，所以必须换头。
注释里也解释了：`lm_head` 不参与 critic 的前向，靠 `tie_word_embeddings` 与 embedding 共享权重，
只要保持共享，DDP 就不会报「参数未使用」。

---

## 3. `calculate_rewards`：规则分 + 模型分

```python
def calculate_rewards(prompts, responses, reward_model):
    with torch.no_grad():
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # ① 用正则把渲染好的 prompt 反解回 messages
            matches = re.findall(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", prompt, re.DOTALL)
            messages = [{"role": r, "content": c.strip()} for r, c in matches]

            # ② 长度塑形
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

            # ③ 思考塑形
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer.strip()

            # ④ 重复惩罚
            rewards[i] -= rep_penalty(answer)

            # ⑤ 奖励模型打分
            rewards[i] += reward_model.get_score(messages, answer)
    return rewards
```

| 奖励项 | 分值 | 目的 |
| --- | --- | --- |
| 回答长度 20~800 字 | +0.5 / -0.5 | 防「一个字」和「复读长文」 |
| 思考块 20~300 字 | +1.0 / -0.5 | 鼓励「先想再答」且别想太长 |
| `<think>` 恰好出现一次 | +0.25 / -0.25 | 格式正确 |
| 3-gram 重复率 | -0 ~ -0.5 | 防复读 |
| 奖励模型 | 约 [-3, 3] | 语义质量（`LMForRewardModel` 里裁剪过） |

**为什么要有规则分**：小模型的奖励模型打分很粗糙。规则分提供了「格式与长度」的强信号，
让 RL 至少先学会「像样地回答」，再靠 RM 提升内容。

**注意**：这段代码在 `train_grpo.py` / `train_agent.py` 里各有一份几乎相同的实现——
三份复制粘贴，是这个项目里比较明显的重复（改奖励要改三处）。

---

## 4. rollout 阶段：从 prompt 到优势

### 4.1 编码 prompt

```python
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_seq_len, padding_side="left").to(device)
```

**必须左填充**：生成从最后一个位置续写（见 transformers 教材第 6 章）。

### 4.2 采样

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=enc.input_ids, attention_mask=enc.attention_mask,
    num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.8)
```

拿到四样东西：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `output_ids` | `(B, P+R)` | prompt + 回答 |
| `completion_ids` | `(B, R)` | 只含回答 |
| `per_token_logps` | `(B, R)` | **采样时**每个回答 token 的 log 概率 |
| `prompt_lens` | `(B,)` | 每行 prompt 的有效长度 |

`per_token_logps` 是 PPO 的「旧策略概率」，后面算 ratio 要用。

### 4.3 奖励

一行：`rewards = calculate_rewards(prompts, responses_text, reward_model)`，形状 `(B,)`。

### 4.4 构造 mask 与位置

```python
full_mask = (gen_out != pad_token_id).long()
labels = gen_out[:, 1:]
logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(R)
full_mask.scatter_(1, logp_pos + 1, resp_pad_mask)
```

**`logp_pos` 是全脚本最关键的下标**：回答第 t 个 token 在 `gen_out` 里的位置是 `P + t`，
它的 logits 在 `P + t - 1` 处（错位一位）。所以取 logp 时用 `prompt_len - 1 + t`。

**回答的有效长度**：找第一个 EOS，后面的都算 padding：

```python
eos_mask = resp_labels.eq(eos_token_id) & resp_pad_mask
resp_lengths = where(has_eos, eos_pos + 1, resp_lengths)
resp_policy_mask = (arange(R) < resp_lengths) & resp_pad_mask
```

### 4.5 价值与参考概率（不建图）

```python
with torch.no_grad():
    values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # (B, P+R)
    old_resp_values = values_seq.gather(1, logp_pos)                        # (B, R)
    ref_resp_logp = ref_model(...).logits → gather at logp_pos              # (B, R)
```

### 4.6 GAE：把终局奖励摊到每个 token

```python
token_rewards = zeros_like(old_resp_logp)
token_rewards[arange(B), resp_lengths - 1] += rewards    # 只在最后一个 token 加奖励

for t in reversed(range(R)):
    nv = old_resp_values[:, t+1] if t < R-1 else 0.0
    delta = token_rewards[:, t] + gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advs_rev.append(lastgaelam)

returns = advantages + old_resp_values                   # 价值回归目标
```

**三行公式的含义**：

| 量 | 公式 | 直观解释 |
| --- | --- | --- |
| TD 误差 `delta` | `r_t + γV(s_{t+1}) - V(s_t)` | 这一步比预期好多少 |
| GAE | `delta + γλ·上一步的GAE` | 把未来几步的「超预期」平滑累积 |
| `returns` | `advantages + V` | 回报目标 = 优势 + 基线 |

**为什么只在末尾加奖励**：对话任务没有中间奖励，整段回答只在结束时被打一次分。
GAE 的作用就是把这个「终局分数」通过价值函数往回摊。

### 4.7 优势归一化

```python
adv = (adv - mean) / sqrt(var + 1e-8)     # 按 mask 统计
adv *= resp_policy_mask
```

让优势均值 0、方差 1，梯度尺度稳定。这是 PPO 收敛的必要条件之一。

---

## 5. update 阶段：mini-batch 里反复更新

### 5.1 双层循环

```python
for ppo_epoch in range(args.ppo_update_iters):        # 同一批数据更新 2 轮
    for i in range(0, B, mb_size):                    # 再切成 mini-batch
        ...
```

**为什么要重复用同一批数据**：采样很贵，PPO 允许在「信任域」内多走几步。
`clip_epsilon=0.2` 和 `early_stop_kl=0.25` 都是为了防止走太远。

### 5.2 五个损失项

```python
log_ratio  = mb_resp_logp - old_resp_logp
ratio      = exp(log_ratio)
approx_kl  = 0.5 * mean(log_ratio²)                   # 与旧策略的偏离
kl_ref     = mean(exp(ref - mb) - (ref - mb) - 1)     # 与参考模型的 k3 估计

policy_loss = -mean(max(-adv*ratio, -adv*clamp(ratio, 1-ε, 1+ε))) + kl_coef * kl_ref
value_loss  = 0.5 * mean(max((v-returns)², (clip(v, old±range)-returns)²))
loss = policy_loss + vf_coef * value_loss + aux_loss
```

| 项 | 作用 | 系数 |
| --- | --- | --- |
| 策略损失 | 提高好回答概率、压低坏回答 | — |
| **clip** | 单步更新幅度不超过 ±20% | `clip_epsilon=0.2` |
| KL 惩罚 | 别离参考模型太远（防止语言能力崩） | `kl_coef=0.02` |
| 价值损失 | critic 学会预测回报 | `vf_coef=0.5` |
| 值裁剪 | critic 也别一步跳太远 | `cliprange_value=0.2` |
| aux_loss | MoE 路由均衡 | 仅 MoE |

### 5.3 早停：KL 超阈值就停

```python
approx_kl_val = approx_kl.detach().clone()
if dist.is_initialized():
    dist.all_reduce(approx_kl_val, op=ReduceOp.AVG)   # 各卡取平均
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

**为什么要 `all_reduce`**：如果只有 0 号卡决定停，其它卡继续 forward，DDP 会永久挂起。
所以先同步 KL，再让所有卡同时停。

### 5.4 早停后仍要「假装更新」

```python
if stop_ppo:
    loss = (policy_loss + vf_coef * value_loss + aux_loss) * 0.0
else:
    loss = (...) / args.accumulation_steps
loss.backward()
```

**`* 0.0` 的作用**：乘 0 后梯度全为 0，但**计算图仍然建完、backward 仍然执行**，
DDP 的 reducer 正常收尾，不会死锁。这是分布式 RL 里很实用的一招。

### 5.5 优化器与调度器步进

```python
grad_accum_step += 1
if grad_accum_step % args.accumulation_steps == 0:
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
    actor_optimizer.step();  critic_optimizer.step()
    actor_scheduler.step();  critic_scheduler.step()
    actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
```

循环结束后还有一段「冲尾」：剩余不足一个累积周期的梯度也统一更新一次。

**actor 与 critic 各有一套优化器和调度器**：

| 模型 | lr | 调度 |
| --- | --- | --- |
| Actor | 3e-7 | 余弦，`eta_min = lr/10` |
| Critic | 5e-7 | 余弦，`eta_min = lr/10` |

Critic 的 lr 比 actor 大——它从零学「价值」，需要更快。

---

## 6. 主程序：与 pretrain 相同的九段，多了五件事

### 6.1 参数速览（新增的关键项）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `learning_rate` | 3e-7 | Actor 学习率（极小） |
| `critic_learning_rate` | 5e-7 | Critic 学习率 |
| `batch_size` | 2 | 每步 2 条 prompt |
| `max_seq_len` | 768 | prompt 长度上限 |
| `max_gen_len` | 1024 | 生成长度上限 |
| `clip_epsilon` | 0.2 | PPO 裁剪 |
| `vf_coef` / `cliprange_value` | 0.5 / 0.2 | 价值损失系数与裁剪 |
| `kl_coef` | 0.02 | 参考模型 KL 惩罚 |
| `gamma` / `lam` | 1.0 / 0.95 | GAE 参数 |
| `ppo_update_iters` | 2 | 同一批数据更新轮数 |
| `mini_batch_size` | 2 | 更新时的微批 |
| `early_stop_kl` | 0.25 | 早停阈值 |
| `reward_model_path` | `../../internlm2-1_8b-reward` | 外部奖励模型 |
| `thinking_ratio` | 0.9 | 采样时开启思考的概率 |
| `rollout_engine` | `torch` | 或 `sglang` |

### 6.2 四个模型的初始化

```python
actor_model, tokenizer = init_model(lm_config, base_weight, device=...)     # 可训练
ref_model, _ = init_model(lm_config, base_weight, device=...)               # 冻结
ref_model = ref_model.eval().requires_grad_(False)

state_dict = torch.load(f'{save_dir}/{base_weight}_{hidden}.pth', map_location=...)
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)                      # value_head 是新参数

reward_model = LMForRewardModel(args.reward_model_path, device=..., dtype=torch.float16)
```

**ref 为什么冻结**：它提供「原始偏好」的基线，KL 惩罚以它为参照。

**critic 为什么从 SFT 权重起步**：随机初始化的价值网络要很久才能学会「什么回答好」；
用底座权重初始化相当于给它一个「懂语言」的起点。

### 6.3 调度器的总步数

```python
mb_factor = max(1, ceil(batch_size / mini_batch_size))
total_optimizer_steps = ceil(iters × epochs × ppo_update_iters × mb_factor / accumulation_steps)
```

**这是全脚本最容易算错的地方**：一次 epoch 里优化器到底走多少步？
= 数据批数 × 更新轮数 × 每批切几个 mini-batch ÷ 累积步数。
算错会导致余弦调度提前归零或走不到底。

### 6.4 DDP 的一个优化

```python
actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)
critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)
```

**`broadcast_buffers=False`**：默认每个 top-level buffer 每步从 rank0 广播一次。
这里唯一的 buffer 是 RoPE 的 `freqs_cos/sin`，由 config 确定性算出，**各卡必然相同**，
广播纯属浪费。注释写得很清楚。

**前提**：buffer 必须真的各卡一致；如果以后加了随训练变化的 buffer，这个开关就会导致不同步。

### 6.5 权重同步给 rollout 引擎

```python
rollout_engine.update_policy(actor_model)          # 初始化后、每次保存后
```

- Torch 引擎：只是把引用指向新模型（DDP 包装后的）；
- SGLang 引擎：把当前权重推到推理服务（共享存储 + 通知加载）。

**这就是「训练/推理分离」的接口**：算法代码不动，换引擎就能把采样交给 vLLM/SGLang。

---

## 7. 一轮 step 的完整数据流

```
batch_size=2 条 prompt
   ↓ tokenizer（左填充）
(2, ≤768)
   ↓ rollout（各自生成 1 条，温度 0.8）
output_ids (2, P+1024)  completion_ids (2, 1024)  old_logp (2, 1024)
   ↓ reward（规则 + RM）
rewards (2,)
   ↓ critic + ref（no_grad）
values (2, 1024)  ref_logp (2, 1024)
   ↓ GAE
advantages (2, 1024)  returns (2, 1024)
   ↓ ppo_update_iters=2 × mini-batch=2
actor 更新 + critic 更新（各 2 次优化步）
```

**时间占比**：采样通常占 80% 以上。所以才有 SGLang 引擎、才有 `num_generations`（GRPO）
这类「一次多采、多学几轮」的设计。

---

## 8. 超参为什么这么小

| 超参 | 值 | 理由 |
| --- | --- | --- |
| actor lr | 3e-7 | RL 信号噪声大，步子大了语言能力直接崩 |
| batch | 2 | 每条要生成 1024 token，显存与时间都贵 |
| ppo_update_iters | 2 | 信任域内多走两步，多了会过拟合当前批次 |
| kl_coef | 0.02 | 偏离参考的代价要够大，防止「刷分」 |
| early_stop_kl | 0.25 | 超过就停，宁可少学 |
| gamma | 1.0 | 对话是「终局奖励」，不折扣 |
| lam | 0.95 | GAE 平滑 |
| thinking_ratio | 0.9 | 大部分样本带思考，少量不带，保持两种模式 |

---

## 9. 显存账：为什么本机跑不了

| 组件 | 64M 模型 fp16 | 说明 |
| --- | --- | --- |
| Actor 权重 | ~128 MB | |
| Ref 权重 | ~128 MB | 冻结但仍占显存 |
| Critic 权重 | ~128 MB | 多一个 value_head |
| **Reward 模型** | **~3.6 GB** | 默认 `internlm2-1.8B-reward`，fp16 |
| 激活 + KV cache | 1~2 GB | 生成 1024 token 时更多 |

**结论**：仅奖励模型就超过 4GB。**这台机器无法跑 PPO**，需要 ≥24GB 显存（可跑 7B 底座 + 1.8B RM），
或者换用规则奖励（`train_agent.py` / GRPO 的数学任务）绕开 RM。

---

## 10. 坑

| 坑 | 说明 |
| --- | --- |
| 奖励模型路径不存在 | 默认 `../../internlm2-1_8b-reward` 是相对路径，需自行下载 |
| RM 太大 | 1.8B fp16 ≈ 3.6GB，4GB 卡必爆 |
| ratio 不≈1 | 采样时的 logp 与更新时重算不一致：精度、dropout、引擎差异都会导致；用 `--debug_log_ratio` 检查 |
| 早停导致 DDP 死锁 | 必须 `all_reduce` 后再停，且用 `loss*0` 走完 backward |
| 调度器步数算错 | `total_optimizer_steps` 公式漏乘 `ppo_update_iters` 或 `mb_factor` |
| 只保存 actor | `ppo_actor_768.pth` 只有 actor；critic 在 resume 文件里 |
| `strict=False` 载 critic | `value_head` 是新参数，必须 false；但主干键名要对上 |
| 换卡续训 | 调度器 `T_max` 按当前卡数重算，可能与旧状态不一致 |
| `broadcast_buffers=False` | 依赖 RoPE buffer 各卡一致；改 buffer 逻辑时要重新评估 |
| 生成太长 | `max_gen_len=1024` 让 rollout 非常慢；调试时先调小 |

---

## 11. 与 GRPO 的对比（预告）

| 项 | PPO | GRPO |
| --- | --- | --- |
| 模型数 | 4（actor/critic/ref/RM） | 3（去掉 critic） |
| 优势来源 | critic 估价值 + GAE | 同一 prompt 多条回答的**组内相对分数** |
| 采样量 | 每条 prompt 1 个回答 | 每条 prompt 6 个回答 |
| 显存 | 最高 | 低约 1/4 |
| 调参难度 | 高（价值网络难训） | 低 |

**读懂 PPO 再看 GRPO，会发现它就是把「critic 那部分」换成了「组内标准化」**——
其余 mask、GAE 之外的损失结构、早停逻辑几乎同源。

---

## 12. 自测题

1. PPO 需要哪四个模型？各自是否训练？
2. `CriticModel` 为什么必须 `strict=False` 加载？
3. 奖励里的规则分有哪三项？
4. 为什么采样时 `padding_side="left"`？
5. `logp_pos = prompt_lens - 1 + arange(R)` 在算什么？
6. GAE 的三行公式分别是什么含义？
7. 奖励为什么只加在最后一个 token？
8. 优势归一化的作用？
9. `ratio` 超过 `1±ε` 时损失怎么变？
10. 早停时为什么还要 backward？
11. 为什么各卡要 `all_reduce` 比较 KL？
12. `total_optimizer_steps` 为什么要乘 `ppo_update_iters`？
13. `broadcast_buffers=False` 的前提是什么？
14. `update_policy` 是给谁用的？
15. 本机 4GB 为什么跑不了？

### 答案要点

1. actor（训）、critic（训）、ref（冻结）、reward（冻结）。
2. `value_head` 是新参数，SFT 权重里没有。
3. 回答长度、思考长度、思考标记次数。
4. 生成从最后位置续写，左填充保证最后位置是真实 token。
5. 回答第 t 个 token 的 logits 下标。
6. TD 误差、GAE 递推、回报 = 优势 + 基线。
7. 对话只有终局奖励，没有中间奖励。
8. 稳定梯度尺度。
9. 被 clip 截断，梯度为 0（不再鼓励）。
10. 保持 DDP 的 reducer 闭环，避免死锁。
11. 避免某卡停、其它卡继续导致 DDP 挂起。
12. 同一批数据会更新多轮，每轮都算优化步。
13. 所有 buffer 各卡一致（这里 RoPE 由 config 决定）。
14. 把最新权重同步给 rollout 引擎（尤其 SGLang）。
15. 默认奖励模型 1.8B fp16 ≈ 3.6GB，加上其它模型超 4GB。

---

## 13. 一表速查

| 段落 | 关键点 |
| --- | --- |
| CriticModel | 主干 + value_head，双重 norm |
| 奖励 | 长度 + 思考 + 重复 + RM |
| rollout | 左填充 → 采样 → old_logp → 奖励 |
| 位置 | `logp_pos = prompt_len - 1 + t` |
| GAE | δ = r + γV' − V；A = δ + γλA_prev |
| 归一化 | 优势减均值除标准差 |
| 损失 | clip 策略损失 + KL + vf·价值损失 |
| 早停 | KL > 0.25，同步各卡，`loss×0` 走完 backward |
| 优化 | actor/critic 两套优化器与调度器 |
| 保存 | actor 半精度 + 含 critic 的 resume |
| 显存 | 4 模型，RM 占 3.6GB |
