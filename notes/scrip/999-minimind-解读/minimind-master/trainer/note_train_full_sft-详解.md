# `train_full_sft.py` 详解：从「学说话」到「学回答」

> 对象：`minimind-master/trainer/train_full_sft.py`（166 行）
> 配套：`dataset/lm_dataset.py` 里的 `SFTDataset`
> 说明：本文按源码讲解；前一阶段的耗时/显存实测见《train_pretrain 详解》，此处不重复跑训练。

---

## 0. 一句话定位

**它和 `train_pretrain.py` 的循环体一模一样，只换了 8 处。**
SFT 的全部难度不在训练代码，而在**数据里那张 loss mask**。

| 项 | pretrain | full_sft |
| --- | --- | --- |
| Dataset | `PretrainDataset`（整段算 loss） | `SFTDataset`（**只算回答**） |
| 学习率 | 5e-4 | **1e-5** |
| 序列长度 | 340 | **768** |
| batch | 32 | **16** |
| 累积步数 | 8 | **1** |
| 初始权重 | `none`（随机） | **`pretrain`** |
| 数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 保存前缀 | `pretrain` | `full_sft` |

**读法建议**：先看第 2 节的逐行 diff（5 分钟），再把 80% 的精力花在第 3 节的 mask 上。

---

## 1. 逐行 diff：改动只有这些

| 位置 | pretrain | full_sft |
| --- | --- | --- |
| import | `PretrainDataset` | `SFTDataset` |
| Dataset 构造 | `PretrainDataset(path, tok, max_length=340)` | `SFTDataset(path, tok, max_length=768)` |
| 默认 lr | 5e-4 | 1e-5 |
| 默认 batch | 32 | 16 |
| 默认 accum | 8 | 1 |
| 默认 seq | 340 | 768 |
| 默认数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 默认 from_weight | `none` | `pretrain` |
| 默认 save_weight | `pretrain` | `full_sft` |
| `train_epoch` 函数体 | — | **完全相同** |
| main 九段结构 | — | **完全相同** |

也就是说：**除了 8 个默认值和一行 Dataset 类名，这个文件与 pretrain 逐字节相同。**

> 口诀：**换数据、换 lr、换长度；循环不动，mask 定胜负。**

---

## 2. 参数变化的三个「为什么」

### 2.1 学习率 5e-4 → 1e-5（降 50 倍）

| 阶段 | 起点 | 可以大步走的原因 |
| --- | --- | --- |
| 预训练 | 随机权重 | 没有可破坏的知识，大 lr 收敛快 |
| SFT | 预训练权重 | 步子大了会**灾难性遗忘**：对话学会了，常识丢了 |

经验值：全参微调 lr 通常取预训练的 1/10 ~ 1/100；LoRA 因为只训适配器，可以用回 1e-4。

### 2.2 序列长度 340 → 768

预训练是「一段文本」，340 token 够；SFT 是「多轮对话 + 系统提示 + 工具定义」，
768 才装得下一轮完整问答。长度翻倍意味着**激活显存显著上升**，所以：

### 2.3 batch 32 → 16、累积 8 → 1

| 项 | pretrain | full_sft | 原因 |
| --- | --- | --- | --- |
| micro batch | 32 | 16 | 序列长了，单条更占显存 |
| 累积 | 8 | 1 | SFT 数据量小，每步都更新更划算 |
| 等效 batch | 256 | 16 | 小数据 + 小 lr，不需要大 batch 稳定 |

**注意**：等效 batch 从 256 掉到 16，是刻意为之。SFT 数据通常只有几万条，
大 batch 会让一个 epoch 只有几十步，学不充分。

---

## 3. 核心：`SFTDataset` 与 loss mask

这是两个脚本真正的分水岭。完整实现在 `dataset/lm_dataset.py:61`。

### 3.1 初始化：先算出两个「标记序列」

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

以 MiniMind 词表为例：

| 序列 | 内容 |
| --- | --- |
| `bos_id` | `<\|im_start\|>` + `assistant` + `\n`（3 个 token 左右） |
| `eos_id` | `<\|im_end\|>` + `\n`（2 个 token 左右） |

**它们不是单个 token，而是 token 序列**——所以后面要用「子序列匹配」而不是「相等判断」。

### 3.2 渲染：`create_chat_prompt`

```python
for message in conversations:
    if message.get("role") == "system" and message.get("tools"):
        tools = json.loads(...)          # tools 字段可能是字符串
    if message.get("tool_calls") and isinstance(message["tool_calls"], str):
        message["tool_calls"] = json.loads(...)
return tokenizer.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=False, tools=tools)
```

三个细节：

| 细节 | 原因 |
| --- | --- |
| `add_generation_prompt=False` | 训练数据里**已经包含**完整回答，不需要再补 assistant 开头 |
| `tools` 单独抽出来传 | 模板有专门的 tools 分支（塞工具定义） |
| 字符串形式的 json 要 `loads` | jsonl 里工具调用常被存成字符串 |

### 3.3 两个概率性增强

**① 注入 system prompt（20%）**

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations): return conversations   # 工具数据不动
    if conversations[0].get('role') != 'system' and random.random() < 0.2:
        return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
```

让模型既见过「有 system」也见过「没有 system」，推理时加不加都能工作。

**② 移除空思考块（80%）**

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content:
        if random.random() > empty_think_ratio:          # 80% 概率
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
```

**为什么需要**：模板**总是**给 assistant 加一个 `<think>` 块（见 tokenizer 详解）。
如果数据里 100% 都带空块，模型会永远输出空思考。80% 移除、20% 保留，
让模型学会「有时想、有时不想」。

### 3.4 编码、截断、padding

```python
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
```

| 动作 | 方式 | 注意 |
| --- | --- | --- |
| 截断 | **硬切片** `[:max_length]` | 不用 `truncation=True`，避免警告 |
| padding | 右侧补 `<\|endoftext\|>`（pad id 0） | 补齐到定长 |

### 3.5 生成 labels：扫描 assistant 区间

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)          # ① 先全部屏蔽
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:      # ② 找到回答开头
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:   # ③ 找到回答结尾
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]                          # ④ 只给这段真值
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

**四步翻译成人话**：

1. 默认所有位置都是 `-100`（不参与损失）；
2. 滑窗找 `<|im_start|>assistant\n`；
3. 从这里往后找 `<|im_end|>\n`；
4. 把「开头之后到结尾（含结尾）」之间的位置，标签设成真实 token。

### 3.6 mask 长什么样（示意）

一段两轮对话渲染后大致是：

```
<|im_start|>system
你是一个AI助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>
</think>
你好！有什么可以帮你？<|im_end|>
```

对应的 labels：

| 片段 | labels |
| --- | --- |
| `<\|im_start\|>system` … `<\|im_end\|>` | 全 `-100` |
| `<\|im_start\|>user` … `<\|im_end\|>` | 全 `-100` |
| `<\|im_start\|>assistant\n`（**标记本身**） | 全 `-100` |
| 紧跟其后的 `<think>`、思考内容、回答、`<\|im_end\|>` | **真实 token id** |
| padding | `-100` |

**注意第一个被监督的 token 是 `<think>`**，因为模板在 `assistant\n` 之后立刻插入思考块。
这意味着模型学的是「先出 `<think>`，再出内容」——和 tokenizer 详解里的模板设计闭环了。

### 3.7 为什么用「扫描」而不是「按 role 拼接」

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| 按 role 逐条拼接、拼接时记录 mask | 直接 | 必须复刻 chat template 的每一处格式 |
| **扫描渲染后的 token 序列** | 自动适配任何模板 | 依赖标记序列的唯一性与完整性 |

作者选了后者：**只要模板格式稳定，mask 永远对**。代价是——模板一改，扫描就失效（见第 6 节坑）。

### 3.8 与模型内部 shift 的配合

模型里是：

```python
loss = CE(logits[..., :-1, :], labels[..., 1:], ignore_index=-100)
```

所以 labels 里第 `j` 个位置的真值，监督的是**位置 `j-1` 的预测**。

对 assistant 区间来说：header 最后一个 token（`\n`）的预测目标是 `<think>`——
**模型要学会「看到 assistant 开头就该开始思考」**。这正是我们要的。

---

## 4. 训练循环：与 pretrain 完全相同

`train_epoch` 的函数体一个字符都没变：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()
if step % args.accumulation_steps == 0 or step == iters:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

所以《train_pretrain 详解》里讲的五步顺序、`unscale_` 必须在裁剪前、日志乘回累积步数、
双文件保存，全部原样适用。这里只说 SFT 特有的两点：

### 4.1 日志里的 `aux_loss` 一定是 0

默认 `use_moe=0`，`res.aux_loss` 是 0 维零张量。日志里 `logits_loss == loss`，
看起来像「冗余」，其实是给 MoE 版本复用的同一段代码。

### 4.2 loss 的参考值不同了

| 阶段 | 初始 loss 参考 |
| --- | --- |
| 预训练（随机权重） | ≈ `ln(6400)` = 8.76 |
| SFT（已载入 pretrain 权重） | **明显低于 8.76**，通常在 1~4 之间，取决于数据 |

**不要拿 SFT 的 loss 和 8.76 比**——它不是从零开始，而且只在回答片段上计算，
分母（有效 token 数）比预训练小得多。

---

## 5. 保存与续训：同样两份文件

```python
ckp = f'{args.save_dir}/{args.save_weight}_{hidden}{moe_suffix}.pth'   # full_sft_768.pth
torch.save({k: v.half().cpu() ...}, ckp)
lm_checkpoint(..., weight=args.save_weight, model=model, optimizer=..., scaler=...)
```

| 文件 | 用途 |
| --- | --- |
| `out/full_sft_768.pth` | 推理、作为 LoRA/DPO 的底座 |
| `checkpoints/full_sft_768_resume.pth` | 断点续训 |

后续脚本的 `--from_weight` 默认值就能对上：LoRA 默认 `full_sft`，DPO 默认 `full_sft`。

---

## 6. SFT 专属的坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| **标记序列对不上** | labels 全 -100，loss 为 `nan` | `bos_id` 用 `f'{bos_token}assistant\n'` 生成，模板里若多了/少了空格就匹配不到 |
| **截断把 eos 切掉** | 该样本只监督了半句话 | `max_length` 太小；截断后没有 `<\|im_end\|>`，扫描标到末尾 |
| **最后一条是 user** | 该样本没有监督信号 | 数据要保证 assistant 有回复；否则该条对训练无贡献 |
| **全 -100 的 batch** | loss 是 `nan` | `cross_entropy` 在忽略所有位置时分母为 0；检查数据与标记 |
| **工具数据的 tools 字段** | 模板渲染报错 | `tools` 可能是字符串，要 `json.loads` |
| **忘了 `from_weight=pretrain`** | 从随机权重开始 SFT | 检查 `out/pretrain_768.pth` 是否存在 |
| **Windows 多进程** | DataLoader 崩溃 | `num_workers=0`（脚本默认已处理） |
| **数据字段类型** | `datasets` 报 schema 错 | `Features` 声明了 string 类型，`tool_calls` 列表要先转字符串存 |

**其中第一个是最致命的**：模板或分词器一改，扫描静默失效，训练照跑、loss 看起来也不异常
（只是学不到东西）。所以下一节的「自查」比训练本身更重要。

---

## 7. 不跑训练也能做的数据自查

训练前建议先打印 3 条样本，检查三件事：

```python
ds = SFTDataset("../dataset/sft_t2t_mini.jsonl", tokenizer, max_length=768)
ids, labels = ds[0]

# 1) 有多少 token 被监督？
sup = (labels != -100).sum().item()
print(f"总长 {len(ids)}，监督 {sup} 个（占比 {sup / len(ids):.1%}）")

# 2) 把 mask 还原成文本，看看学到的是什么
print("监督内容:", tokenizer.decode([t for t, l in zip(ids, labels) if l != -100]))

# 3) 检查是否真的扫到了 assistant 段
assert sup > 0, "这条样本没有任何监督信号"
```

**期望看到**：第 2 步打印出来的应该**只有 assistant 的回答**，没有 system、没有用户提问。
如果打出来的是整段对话，说明 mask 没生效；如果是空，说明标记没匹配上。

**另一个更快的检查**：随机抽 100 条，统计 `sup > 0` 的比例，低于 95% 就该回去查数据与模板。

---

## 8. 自测题

1. 这个脚本和 `train_pretrain.py` 差哪几处？
2. SFT 的学习率为什么比预训练小 50 倍？
3. `bos_id` / `eos_id` 为什么是序列而不是单个 id？
4. `add_generation_prompt` 为什么是 False？
5. 空思考块为什么要按 80% 概率移除？
6. mask 的扫描算法分哪四步？
7. 第一个被监督的 token 是什么？为什么？
8. 为什么用「扫描 token 序列」而不是「按 role 拼接」？
9. 模型内部的 shift 和这个 mask 怎么配合？
10. SFT 的 loss 能不能和 `ln(6400)` 比？为什么？
11. labels 全为 -100 会发生什么？
12. 模板改了会有什么后果？

### 答案要点

1. Dataset、lr、seq、batch、accum、初始权重、数据、保存前缀（8 处）。
2. 预训练权重已含知识，大步会灾难性遗忘。
3. 它们是「标记 + 角色名 + 换行」的 token 序列，只能做子序列匹配。
4. 训练数据已包含完整回答，不需要再补 assistant 开头。
5. 让模型学会自适应思考，而不是永远输出空思考标签。
6. 全 -100 → 找 bos 标记 → 找 eos 标记 → 区间填真值。
7. `<think>`，因为模板紧接在 `assistant\n` 之后插入思考块。
8. 自动适配任意模板，不用复刻拼接逻辑；代价是依赖模板稳定性。
9. `logits[:-1]` 对 `labels[1:]`，header 末位预测第一个回答 token。
10. 不能；起始 loss 已低于 8.76，且监督范围只是回答片段。
11. `cross_entropy` 分母为 0，返回 `nan`。
12. 标记匹配失败，labels 全 -100，训练静默失效。

---

## 9. 一表速查

| 段落 | 关键点 |
| --- | --- |
| Dataset | `SFTDataset`：模板渲染 + 扫描 mask |
| 标记 | `bos_id = "<\|im_start\|>assistant\n"`、`eos_id = "<\|im_end\|>\n"` |
| 增强 | 20% 注入 system、80% 移除空 think |
| 编码 | 硬切片截断 + 右侧 pad |
| mask | 只有 assistant 区间是真值，其余 -100 |
| lr | 1e-5（预训练的 1/50） |
| seq / batch | 768 / 16，累积 1 |
| 初始权重 | `pretrain` |
| 保存 | `full_sft_768.pth` + resume |
| 循环 | 与 pretrain 完全相同 |
