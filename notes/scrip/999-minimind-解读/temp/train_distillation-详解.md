# `train_distillation.py` 详解：老师带学生

> 对象：`minimind-master/trainer/train_distillation.py`（242 行）
> 配套：`SFTDataset`、`init_model`、`lm_checkpoint`
> 说明：本文按源码讲解，不跑训练。

---

## 0. 定位：把大模型的知识压进小模型

默认配置很有意思：**教师是 MoE（198M），学生是 Dense（64M）**——
这正是「MoE 蒸馏成 Dense」的真实用法（DeepSeek-R1 的蒸馏就是这么做的）。

| 项 | 学生（Dense） | 教师（MoE） |
| --- | --- | --- |
| 参数量 | 63.91M | 198.42M（激活约 64M） |
| 是否训练 | ✅ | ❌ 冻结 |
| 输出 | logits | logits |

**它和 SFT 的关系**：数据集、mask、训练循环全部复用 SFT，
只是损失从「一个 CE」变成「α·CE + (1−α)·KL」。

> 口诀：**硬标签教对错，软标签教分寸；温度放大分布，T² 补回梯度。**

---

## 1. 为什么要用「软标签」

普通 SFT 只告诉模型「正确答案是 token A」；
蒸馏额外告诉它「**B 有 10% 的可能，C 只有 0.1%**」——这些相对大小就是「暗知识」。

| 信号 | 来源 | 教什么 |
| --- | --- | --- |
| 硬标签（CE） | 数据集 | 对/错 |
| 软标签（KL） | 教师模型的完整分布 | 相似度、歧义、置信度 |

**温度的作用**：教师的分布通常很尖（top-1 0.99），软标签信息被淹没。
除以温度 T>1 把分布「摊平」，让次要类别的概率显现出来。

---

## 2. 三个公式

### 2.1 软化

```python
teacher_probs = softmax(teacher_logits / T)
student_log_probs = log_softmax(student_logits / T)
```

### 2.2 KL 散度

```python
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```

PyTorch 的 `kl_div(input, target)` 计算的是 `KL(target ‖ exp(input))`，
所以这里是 **KL(教师 ‖ 学生)**——「forward KL」，倾向于让学生覆盖教师的所有可能。

### 2.3 乘 T²

```python
return (temperature ** 2) * kl
```

**为什么乘 T²**：温度抬高后 softmax 的梯度会缩小约 `1/T²`。
乘回去，让不同温度下的损失量级可比、学习率不用重调。

---

## 3. `distillation_loss`（:25）

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    if student_logits.size(0) == 0 and reduction == 'batchmean':
        return student_logits.sum() * 0.0        # 空张量保护
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

| 细节 | 原因 |
| --- | --- |
| `no_grad` 包住教师 | 教师不参与梯度 |
| `.detach()` | 双保险 |
| 空张量返回 0 | 某批可能没有任何被监督的 token |
| `batchmean` | 除以 batch 大小，而非元素数 |

---

## 4. `train_epoch` 逐段（:41）

### 4.1 掩码对齐（关键一步）

```python
loss_mask = (labels[..., 1:] != -100).float()
```

**注意 `[..., 1:]`**：模型是「位置 t 预测 t+1」，所以训练用的标签要错开一位。
后面的 `shift_labels` 也是同一个切片：

```python
shift_labels = labels[..., 1:].contiguous()
```

**掩码的含义**：只有 assistant 回答部分是真值，问题与 padding 都是 `-100`。
蒸馏也只在这些位置做——**不让模型去模仿教师「怎么生成问题」**。

### 4.2 两个前向

```python
with autocast_ctx:
    res = model(input_ids)
    student_logits = res.logits[..., :-1, :].contiguous()      # 去掉最后一个位置

if teacher_model is not None:
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :vocab_size_student]   # 词表对齐
```

| 细节 | 原因 |
| --- | --- |
| `logits[..., :-1]` | 最后一位没有「下一个 token」可预测 |
| 教师走 `no_grad` | 省显存、省时间 |
| 教师 logits 截断到学生词表 | 防御性处理；本项目两者词表相同（6400） |

**词表截断是个隐患**：它假设「学生词表 = 教师词表的前缀」。
如果两者训练过不同的 tokenizer，截断出来的 id 对不上，蒸馏会悄悄学错。
本项目的 `SFTDataset` 用学生 tokenizer 编码，教师又用同一份词表，所以安全。

### 4.3 硬标签损失：带掩码的 CE

```python
ce_loss = F.cross_entropy(student_logits.view(-1, V), shift_labels.view(-1),
                          ignore_index=-100, reduction='none')
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
ce_loss = ce_loss_raw + res.aux_loss if student 是 MoE else ce_loss_raw
```

**为什么用 `reduction='none'` 再手动加权**：
`ignore_index=-100` 本来就会跳过这些位置，用 `mean` 也能算对。
这里手动加权的真正目的是**用同一份 mask 选出参与蒸馏的位置**——
两个损失必须落在同一批 token 上，显式写出来更不容易错。

### 4.4 软标签损失：只在被监督的位置

```python
distill_loss = distillation_loss(
    student_logits.view(-1, V)[loss_mask_flat == 1],
    teacher_logits.view(-1, V)[loss_mask_flat == 1],
    temperature=temperature)
```

**用布尔索引挑出「回答 token」的 logits**，形状 `(N_valid, V)`。
这和 `ce_loss` 用的是同一批位置——两个损失对齐。

### 4.5 合成总损失

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / accumulation_steps
```

| α | 效果 |
| --- | --- |
| 1.0 | 退化成普通 SFT |
| 0.0 | 纯蒸馏（没有标准答案） |
| **0.5**（默认） | 一半硬、一半软 |

**注意**：`aux_loss` 被包含在 `ce_loss` 里，所以它也被乘了 α。
MoE 学生的路由均衡会被「打折」——小问题，但读代码要知道。

---

## 5. 教师模型：三个必须

```python
teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
teacher_model.eval()
teacher_model.requires_grad_(False)
```

| 必须 | 原因 |
| --- | --- |
| `eval()` | dropout 会让教师的分布每次不同，学生学不到稳定信号 |
| `requires_grad_(False)` | 不做无用梯度计算 |
| 同一个 tokenizer | 数据集用学生 tokenizer 编码，教师必须认识同一套 id |

**教师不参与 DDP**：每张卡各放一份、只做前向。

---

## 6. 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `save_weight` | `full_dist` | 产出 `full_dist_768.pth` |
| `epochs` | **6** | 比 SFT（2）多：蒸馏收敛慢 |
| `batch_size` | 32 | |
| `learning_rate` | **5e-6** | 介于 SFT（1e-5）与 DPO 之间 |
| `max_seq_len` | 340 | |
| `student_hidden_size/layers` | 768 / 8 | Dense 学生 |
| `teacher_hidden_size/layers` | 768 / 8 | MoE 教师 |
| `student_use_moe` / `teacher_use_moe` | 0 / **1** | 默认 MoE→Dense |
| `from_student_weight` | `full_sft` | |
| `from_teacher_weight` | `full_sft` | 对应 `full_sft_768_moe.pth` |
| `alpha` | 0.5 | CE 权重 |
| `temperature` | **1.5** | 软化温度 |

**为什么 lr 只有 5e-6**：学生已经有 SFT 权重，蒸馏是「精修分布」，步子要小。

**为什么 epochs 是 6**：软标签的信号更「弱」但更「密」，需要多轮才能吸收。

---

## 7. 与 SFT 的对照

| 项 | full_sft | distillation |
| --- | --- | --- |
| 模型数 | 1 | 2（学生 + 教师） |
| 损失 | CE | α·CE + (1−α)·KL |
| 数据 | SFT | **同一个 SFT** |
| mask | 只算回答 | 只算回答（两个损失共用） |
| lr | 1e-5 | 5e-6 |
| epochs | 2 | 6 |
| 保存 | 学生 | 学生（教师不存） |

**循环骨架、精度、保存、DDP 逻辑完全一致。**

---

## 8. 显存与本地可行性

以默认配置（dense 学生 64M + MoE 教师 198M）、batch 32 × 340 估算：

| 项 | 量级 |
| --- | --- |
| 学生权重 + 梯度 + Adam | ~1.1 GB |
| 教师权重（fp16，冻结） | ~0.38 GB |
| 学生 logits（32×340×6400，fp16） | ~139 MB |
| 教师 logits（同上） | ~139 MB |
| 激活 | 1~2 GB |

**结论**：batch 32 在 4GB 上会 OOM；把 `--batch_size` 降到 **4~8**、
或把 `--max_seq_len` 降到 256，有机会跑通。这是本项目里**第二适合本机体验**的脚本
（第一是 LoRA）。

---

## 9. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| 教师没 `eval()` | 学生学到的分布抖动 | 必须在每轮开始设置 |
| 词表不一致 | 蒸馏「学歪」且不报错 | 教师必须用同一 tokenizer |
| 忘记 `/temperature` | 软标签退化成硬标签 | 两个模型都要除 |
| 忘记 `× T²` | 换温度后学习率失配 | 函数里已处理 |
| `alpha=1` | 变回 SFT，蒸馏无效 | 检查参数 |
| `alpha=0` | 没有标准答案，可能跑偏 | 一般留 0.3~0.7 |
| 教师权重路径 | `full_sft_768_moe.pth` 不存在 | MoE 后缀 `_moe` 不能少 |
| 蒸馏位置错 | 把问题也蒸馏了 | 依赖 SFT mask；检查数据 |
| 显存翻倍 | OOM | 降 batch / 序列长度 |
| 教师 logits 全精度 | 显存暴涨 | 已在 no_grad + autocast 下 |

---

## 10. 白盒 vs 黑盒，顺带一提

| 类型 | 需要什么 | 本项目 |
| --- | --- | --- |
| **白盒蒸馏** | 教师 logits（完整分布） | ✅ 就是这一份 |
| 黑盒蒸馏 | 只需要教师生成的文本 | 用 SFT 数据也能做 |

白盒信息更多（有「暗知识」），但要求教师和学生**在同一台机器上同时前向**；
黑盒只需要离线数据，适合教师超大（比如从 API 蒸馏）。

---

## 11. 自测题

1. 蒸馏比 SFT 多提供了什么信息？
2. 温度的作用是什么？为什么还要乘 T²？
3. `F.kl_div(student_logp, teacher_prob)` 算的是哪个方向的 KL？
4. 蒸馏为什么只在 assistant 区间做？
5. 教师模型为什么必须 `eval()` + `no_grad`？
6. 教师 logits 截断到学生词表，前提是什么？
7. 默认配置里教师和学生分别是什么结构？
8. α=1 和 α=0 分别退化成什么？
9. lr 为什么取 5e-6？
10. 续训时恢复的是谁的权重？
11. 白盒蒸馏与黑盒蒸馏的区别？
12. 本机 4GB 要改哪些参数？

### 答案要点

1. 完整分布（软标签），包含「相似答案的相对可能性」。
2. 摊平分布让次要类别显现；T² 补偿 softmax 梯度缩小。
3. KL(教师 ‖ 学生)（forward KL）。
4. 问题部分是输入，不该让模型去模仿教师「生成问题」。
5. 避免 dropout 噪声；省显存与计算。
6. 学生词表是教师词表的前缀（同一 tokenizer）。
7. 教师 MoE 198M，学生 Dense 64M。
8. α=1 → 普通 SFT；α=0 → 纯蒸馏。
9. 学生已有 SFT 权重，精修分布要小步。
10. 只恢复学生（和优化器）；教师永远重新加载并冻结。
11. 白盒要教师 logits，黑盒只要教师文本。
12. 降 batch 到 4~8、序列长度降到 256。

---

## 12. 一表速查

| 段落 | 关键点 |
| --- | --- |
| 数据 | 复用 SFTDataset，只在回答 token 上蒸馏 |
| 掩码 | `labels[..., 1:] != -100` |
| 损失 | α·CE + (1−α)·KL·T² |
| 温度 | 默认 1.5 |
| 教师 | 冻结、eval、同词表、不 DDP |
| lr / epochs | 5e-6 / 6 |
| 保存 | 只存学生 |
| 本机 | 降 batch/长度可跑 |
