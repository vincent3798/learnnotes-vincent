# `train_dpo.py` 详解：不用奖励模型的偏好对齐

> 对象：`minimind-master/trainer/train_dpo.py`（220 行）
> 配套：`dataset/lm_dataset.py` 的 `DPODataset`、`trainer_utils.get_lr` / `lm_checkpoint`
> 说明：本文按源码讲解，不跑训练。

---

## 0. 定位：SFT 之后的第二步

| 阶段 | 目标 | 需要的模型 | 数据形态 |
| --- | --- | --- | --- |
| Pretrain | 学会说话 | 1 | 纯文本 |
| SFT | 学会听指令 | 1 | 问答对 |
| **DPO** | **学会「哪种回答更好」** | **2（策略 + 参考）** | **偏好对（chosen / rejected）** |
| PPO / GRPO | 在线试错提升 | 3~4 | prompt + 奖励 |

DPO 的最大卖点：**不需要奖励模型、不需要在线采样**，只用一份静态的偏好数据，
两个模型（策略 + 参考）跑监督式训练。

> 口诀：**同一问题两个答案，好的拉高、坏的压低；参考模型当尺子，β 管偏离多少。**

---

## 1. 一句话数学

DPO 的损失（源码 `dpo_loss` 就是它的直译）：

```
L = -log σ( β · [ log(πθ(y_w|x)/πref(y_w|x)) - log(πθ(y_l|x)/πref(y_l|x)) ] )
```

| 符号 | 含义 |
| --- | --- |
| `y_w` / `y_l` | chosen（好答案）/ rejected（坏答案） |
| `πθ` | 正在训练的策略（policy） |
| `πref` | 冻结的参考模型（就是训练开始前的策略） |
| `β` | 允许偏离参考的程度，越大越保守 |
| `σ` | sigmoid，`-log σ(x)` 在 x 越大时越小 |

**直观解释**：策略要做的，是让「好答案相对坏答案的优势」**比参考模型**更大。
差值越大，损失越小。

**为什么不需要奖励模型**：DPO 从「带 KL 约束的 PPO 目标」出发，
在偏好对数据上推导出了一个**解析解式的损失**——它证明了「奖励」可以被
「策略与参考的对数概率比」隐式表示出来（这就是常说的 implicit reward）。

---

## 2. 数据：`DPODataset` 给你 6 个张量

```python
{
    'x_chosen':  (L,),  'y_chosen':  (L,),  'mask_chosen':  (L,),
    'x_rejected':(L,),  'y_rejected':(L,),  'mask_rejected':(L,),
}
```

构造过程（`dataset/lm_dataset.py:125`）：

```python
chosen_prompt  = tokenizer.apply_chat_template(chosen,  tokenize=False, add_generation_prompt=False)
rejected_prompt= tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
ids  = tokenizer(prompt, truncation=True, max_length=1024, padding='max_length')['input_ids']
mask = generate_loss_mask(ids)              # 只覆盖 assistant 区间（1/0）
x, y, mask = ids[:-1], ids[1:], mask[1:]    # 已经错位
```

| 设计 | 原因 |
| --- | --- |
| 两个 prompt 各渲染一次 | 好答案和坏答案的**问题部分相同**，但回答不同，必须分别渲染 |
| `mask` 只覆盖 assistant | DPO 只比较「回答部分」的概率，问题部分不影响偏好 |
| **已经错位** `[:-1]` / `[1:]` | 模型内部 `logits` 不再错位，直接用 `y` gather |
| `max_length=1024` | DPO 样本通常比 SFT 长（含两条回答） |

**注意 mask 的语义和 SFT 不同**：SFT 的 mask 是 `labels` 里的 `-100`，
DPO 的 mask 是独立的 `0/1` 向量，用来**对 log 概率做加权求和**。

---

## 3. 三个函数逐个拆

### 3.1 `logits_to_log_probs`：取出「真实 token」的 log 概率

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)                      # (B, L, V)
    return torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)   # (B, L)
```

| 步骤 | 形状 |
| --- | --- |
| `log_softmax(dim=2)` | 每个位置对词表的对数概率 |
| `labels.unsqueeze(2)` | `(B, L, 1)` |
| `gather` | 取出「该位置真实 token」的那一项 |
| `squeeze(-1)` | `(B, L)` |

**为什么是 log 而不是概率**：后续要**相加**多个 token 的比值，对数把乘积变加法，
数值上也更稳。

### 3.2 `dpo_loss`：五行核心

```python
ref_log_probs    = (ref_log_probs    * mask).sum(dim=1)      # (B,) 序列级
policy_log_probs = (policy_log_probs * mask).sum(dim=1)

half = batch_size // 2
chosen_ref, reject_ref = ref[:half], ref[half:]              # 前半 chosen
chosen_pol, reject_pol = pol[:half], pol[half:]

pi_logratios  = chosen_pol - reject_pol                      # 策略认为好的比坏的好多少
ref_logratios = chosen_ref - reject_ref                      # 参考模型认为好多少
logits = pi_logratios - ref_logratios                        # 相对参考的改进
loss = -F.logsigmoid(beta * logits).mean()
```

**为什么要 `mask` 加权求和**：把「整段回答」的 log 概率加起来，
得到一个序列级的分数；不加 mask 会把问题部分也算进去，偏好判断被稀释。

**为什么「对半拆」**：`train_epoch` 里先把 chosen 和 rejected 拼成一个 2B 的 batch
一起前向（效率高），这里再按顺序拆回两半。**所以拼接顺序不能乱**：
前半必须是 chosen，后半必须是 rejected。

**`-log σ(β·x)` 的梯度行为**：

| `β·logits` | 损失 | 梯度 |
| --- | --- | --- |
| 很大（策略已经明显偏好 chosen） | ≈0 | ≈0，不再更新 |
| 0（策略和参考没差别） | ln2 ≈ 0.693 | 最大 |
| 很小（策略偏向 rejected） | 很大 | 强烈纠正 |

### 3.3 `train_epoch`：一次前向算两个答案

```python
x = torch.cat([x_chosen, x_rejected], dim=0)     # (2B, L)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)

with autocast_ctx:
    with torch.no_grad():
        ref_logits = ref_model(x).logits          # 参考模型：不建图
    ref_log_probs = logits_to_log_probs(ref_logits, y)

    outputs = model(x)                            # 策略：建图
    policy_log_probs = logits_to_log_probs(outputs.logits, y)

    dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
    loss = (dpo_loss_val + outputs.aux_loss) / args.accumulation_steps

scaler.scale(loss).backward()
...
```

| 设计 | 原因 |
| --- | --- |
| `torch.cat` 后一次前向 | 同样的计算量下，批处理效率更高 |
| 参考模型 `no_grad` | 它不训练，建图纯属浪费显存 |
| `+ aux_loss` | MoE 路由均衡；Dense 时为 0 |
| 只有策略模型的参数在优化器里 | 参考模型 `requires_grad_(False)` |

其余部分（scaler 五步、日志、保存、`del`）与前面的脚本完全一致。

---

## 4. `main`：与 SFT 相同的九段，多了「第二个模型」

### 4.1 参数速览

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `save_weight` | `dpo` | 产出 `dpo_768.pth` |
| `epochs` | **1** | 偏好数据训一轮就够 |
| `batch_size` | 4 | 拼接后变成 8 |
| `learning_rate` | **4e-8** | 全项目最小的学习率 |
| `beta` | **0.15**（CLI）/ 0.1（函数默认） | 偏离参考的约束强度 |
| `max_seq_len` | 1024 | 两条回答拼起来更长 |
| `data_path` | `dpo.jsonl` | 偏好对 |
| `from_weight` | `full_sft` | 从 SFT 权重出发 |
| `accumulation_steps` | 1 | |
| `save_interval` | 100 | 存得比较勤（DPO 容易训崩） |

### 4.2 双模型初始化

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)   # 同一份权重
ref_model.eval()
ref_model.requires_grad_(False)
```

**三条铁律**：

1. **必须是同一份权重**——否则「相对改进」的基线就是错的；
2. **ref 必须 `eval()`**——dropout 会让参考概率抖动；
3. **ref 必须冻结**——否则它会被优化器顺带更新（虽然本脚本的优化器只收了 `model` 的参数，
   但显式冻结更安全，也不会建梯度）。

### 4.3 优化器与恢复

```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)   # 只有策略
...
model.load_state_dict(ckp_data['model'])
optimizer.load_state_dict(ckp_data['optimizer'])
```

**注意**：resume 只恢复策略模型和优化器，**参考模型永远从 `from_weight` 重新加载**。
这是对的——参考模型在 DPO 里本来就该固定为「训练开始时」的样子。

### 4.4 DDP 只包策略

```python
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

参考模型每张卡各有一份，冻结、不通信。

---

## 5. 超参为什么这么极端

### 5.1 学习率 4e-8

对比表：

| 阶段 | lr |
| --- | --- |
| Pretrain | 5e-4 |
| SFT | 1e-5 |
| LoRA | 1e-4（只训适配器） |
| **DPO** | **4e-8** |

**为什么小到离谱**：DPO 直接优化「偏好」，梯度信号很尖锐。
学习率稍大，模型会迅速「讨好数据」而丢掉语言能力——典型表现是：
偏好准确率上去了，但生成的句子开始重复、跑题、甚至崩坏。
README 也注明了「建议 ≤5e-8 避免遗忘」。

### 5.2 只训 1 个 epoch

偏好数据集通常只有几千条。多轮会让模型**记住具体样本**（过拟合偏好对），
泛化反而下降。

### 5.3 β = 0.15

| β | 效果 |
| --- | --- |
| 太大（如 1.0） | 几乎不能偏离参考，学不动 |
| 太小（如 0.01） | 可以大幅偏离，容易训崩 |
| 0.1 ~ 0.2 | 常用区间 |

---

## 6. 显存账：4GB 能不能跑

| 组件 | 64M Dense |
| --- | --- |
| 策略权重（bf16 前向 + fp32 主权重） | ~0.4 GB |
| 参考权重（bf16） | ~0.13 GB |
| 策略的梯度 + Adam 状态 | ~0.75 GB |
| 激活（batch 8 × seq 1024） | **1.5 ~ 2.5 GB** |

**估算结论**：batch 4（拼接后 8）× 1024 长度在 4GB 上**很紧**，大概率 OOM；
本地要跑请把 `--batch_size 1 --max_seq_len 512` 起步，或用梯度累积。
比 PPO 友好得多（PPO 光奖励模型就 3.6GB）。

---

## 7. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| 参考模型权重不同源 | 训练不收敛或越训越差 | 必须用同一个 `from_weight` |
| 忘记给 ref 设 `eval()` | 参考概率有噪声 | 显式 `eval()` |
| 拼接顺序颠倒 | 损失符号反了，越训越糟 | 前半 chosen、后半 rejected |
| mask 全 0 | `dpo_loss` 恒为 0.693，无梯度 | 检查数据里的 assistant 标记与模板 |
| lr 太大 | 偏好学会了，语言能力崩 | 保持 ≤5e-8 |
| 多轮训练 | 验证集变差 | 1 epoch，或加早停 |
| 长度偏置 | 模型偏爱长回答 | 序列 logp 是**求和**，长答案天然占优；可改成长度归一化（本项目没做） |
| 两个模型显存翻倍 | OOM | 降 batch / 序列长度 |
| 参考模型每步重算 | 慢 | 可预先缓存 ref logp（本项目没做，换来代码简单） |
| `strict` 加载 | — | 策略和参考都从完整权重加载，无需 `strict=False` |

**其中「长度偏置」值得单独说**：DPO 的序列分数是 log 概率**之和**，
越长的回答 token 越多，和越大。如果数据集里 chosen 系统性比 rejected 长，
模型会学到「长就是好」。工业实现常用「长度归一化」或「长度惩罚」缓解。

---

## 8. 怎么判断 DPO 训好了 / 训崩了

不用等训练结束，看三个信号：

| 信号 | 健康 | 异常 |
| --- | --- | --- |
| `dpo_loss` | 从 ~0.69 缓慢下降 | 一直 0.69（无梯度）或剧烈震荡 |
| 隐式奖励边际 `β·(logratio差)` | 稳步上升 | 上升过快 → lr 太大 |
| 抽样生成 | 语言通顺、更符合偏好 | 重复、崩坏、答非所问 |

**特别提醒**：DPO 有一个已知现象 **likelihood displacement**——
训练中 chosen 和 rejected 的绝对概率**可能同时下降**，只有「相对差」在扩大。
所以不要只看「chosen 的 logp 有没有涨」，要看差值。

---

## 9. 与 PPO / GRPO 的对比

| 项 | DPO | PPO | GRPO |
| --- | --- | --- | --- |
| 模型数 | 2 | 4 | 3 |
| 是否采样 | ❌ 静态数据 | ✅ 每步生成 | ✅ 每组多条 |
| 奖励来源 | 数据里的偏好对 | 奖励模型 + 规则 | 规则/可验证奖励 |
| 训练稳定性 | 高 | 低 | 中 |
| 能力提升 | 偏「偏好/安全」 | 可提升推理 | 可提升推理（可验证任务） |
| 显存 | 低 | 最高 | 中 |

**一句话**：DPO 便宜、稳、适合「价值观对齐」；
PPO/GRPO 贵、难调，但能靠在线探索提升「做题能力」。

---

## 10. 自测题

1. DPO 为什么不需要奖励模型？
2. `πref` 是从哪来的？为什么必须冻结？
3. `mask` 在 DPO 里做什么？和 SFT 的 `-100` 有什么区别？
4. 为什么要先把 chosen/rejected 拼成一个 batch？
5. 拼接顺序错了会怎样？
6. `-log σ(β·x)` 在 x 很大和 x 很小时分别是什么行为？
7. 学习率为什么是 4e-8？
8. β 太大 / 太小分别会怎样？
9. 为什么只训 1 个 epoch？
10. 什么是长度偏置？怎么缓解？
11. 什么是 likelihood displacement？
12. resume 时参考模型为什么不恢复？

### 答案要点

1. 损失用「策略/参考的对数概率比」隐式表示奖励，偏好对本身提供监督信号。
2. 训练开始前的策略（同一份 SFT 权重）；冻结是为了提供固定基线。
3. 对回答的 log 概率加权求和，只比较回答部分；SFT 的 -100 是「哪些位置算损失」。
4. 批处理效率；一次前向算两倍样本。
5. 损失符号反转，模型学到相反的偏好。
6. x 大 → 损失≈0、梯度≈0；x=0 → 损失 ln2、梯度最大。
7. 偏好梯度尖锐，lr 大会破坏语言能力。
8. 太大 → 学不动；太小 → 偏离过大、训崩。
9. 数据少，多轮过拟合。
10. 序列 logp 求和，长答案天然得分高；可用长度归一化/惩罚缓解。
11. chosen 与 rejected 绝对概率同时下降，只有相对差扩大。
12. 参考模型必须固定为训练开始时的那份权重。

---

## 11. 一表速查

| 段落 | 关键点 |
| --- | --- |
| 数据 | chosen/rejected 各渲染，mask 只覆盖回答，已错位 |
| `logits_to_log_probs` | log_softmax + gather 取真实 token |
| `dpo_loss` | mask 求和 → 对半拆 → 双重 logratio → -logσ(βx) |
| 前向 | 拼接一次前向；ref 走 no_grad |
| lr | 4e-8 |
| β | 0.15 |
| epochs | 1 |
| 模型 | 策略 + 参考（同源、冻结、eval） |
| 保存 | `dpo_768.pth` + resume（只含策略） |
| 监控 | dpo_loss 缓慢下降、相对差扩大、生成不崩 |
