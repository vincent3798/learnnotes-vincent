# PyTorch 源码实战：从 MiniMind 学训练系统

> **取材**：`minimind-master`（minimind-3）全部训练代码，逐个统计实际用到的 torch API
> **环境**：本机 conda `swift_env`（torch 2.6.0+cu124 / GTX 1650 4GB）
> **写法**：每个 API 都标出它在 MiniMind 里的出处，配「为什么这么写」和「坑」；
> 所有示例都可以用 `D:\conda_envs\swift_env\python.exe` 直接跑。

---

## 0. 这本书怎么用

### 0.1 和普通 PyTorch 教材的区别

普通教材按「知识点」组织：张量 → 自动求导 → 模型 → 训练。
这本书按「**一个真实训练系统里 API 出现的顺序**」组织，并且每个 API 都对应一句源码。

| 普通教材 | 这本书 |
| --- | --- |
| `torch.cat` 的用法 | MiniMind 在 RoPE 的 cos/sin 拼接里用它（`model_minimind.py:76`） |
| `F.kl_div` 的公式 | 蒸馏里配 `T²` 用（`train_distillation.py:33`），GRPO 里用 k3 估计（`train_grpo.py:127`） |
| `DistributedDataParallel` 是什么 | 为什么参数里要写 `broadcast_buffers=False`（`train_ppo.py:423`） |

### 0.2 API 全景（统计自项目源码）

项目里实际出现的 torch 系 API 约 **90 个**，分八组：

| 组 | 数量 | 代表 |
| --- | --- | --- |
| 张量与形状 | ~28 | `cat` `view` `permute` `repeat_interleave` |
| 索引与掩码 | ~12 | `gather` `scatter_` `index_add_` `topk` |
| 数学与损失 | ~20 | `cross_entropy` `log_softmax` `kl_div` `logsigmoid` |
| 模块与参数 | ~10 | `nn.Linear` `nn.Embedding` `register_buffer` |
| 自动求导 | 6 | `requires_grad_` `detach` `no_grad` `inference_mode` |
| 优化与精度 | ~10 | `AdamW` `clip_grad_norm_` `GradScaler` `autocast` |
| 数据与分布式 | ~12 | `DataLoader` `DistributedSampler` `DDP` `all_reduce` |
| 保存与调试 | ~8 | `save` `load` `compile` `empty_cache` |

### 0.3 三条阅读建议

1. **先读第 1、2 章**：张量形状和 dtype 是所有报错的根源。
2. **第 5、6 章最重要**：MiniMind 的「loss mask + gather + scatter」是 LLM 训练的核心手艺。
3. **第 9、10 章按需读**：单卡训练可以先跳过分布式。

### 0.4 配套脚本（都已在本机跑通）

| 脚本 | 对应章节 | 演示什么 |
| --- | --- | --- |
| `code/01_shapes_index.py` | 第 1、5、6 章 | cat/stack、GQA 复制、gather/scatter/index_add_、top-p、repeat_interleave |
| `code/02_autograd.py` | 第 3 章 | requires_grad 冻结、detach、no_grad/inference_mode、梯度累加 |
| `code/03_train_step.py` | 第 7、8 章 | autocast + GradScaler + clip + step 的完整顺序 |

运行：

```bash
D:\conda_envs\swift_env\python.exe code\01_shapes_index.py
```

---

## 目录

- 第 1 章 张量与形状：把「几维、多长」刻进脑子
- 第 2 章 dtype、device 与精度
- 第 3 章 自动求导的六个开关
- 第 4 章 用 nn.Module 搭模型
- 第 5 章 数学与损失函数
- 第 6 章 索引、掩码与 gather/scatter
- 第 7 章 优化器、梯度裁剪与学习率
- 第 8 章 混合精度与 GradScaler
- 第 9 章 数据管线与批采样
- 第 10 章 分布式：DDP 与集合通信
- 第 11 章 保存、加载与断点续训
- 第 12 章 随机性与复现
- 第 13 章 性能与显存
- 第 14 章 三个完整模式（把 API 串起来）
- 附录 A API 速查表 / B 与项目其他教材的对照 / C 练习

---

## 第 1 章 张量与形状：把「几维、多长」刻进脑子

> 口诀：**见张量先问三件事——shape、dtype、device。**

### 1.1 创建

| API | MiniMind 出处 | 用途 |
| --- | --- | --- |
| `torch.tensor(data)` | `lm_dataset.py` 各 Dataset | 列表 → 张量 |
| `torch.zeros_like(x)` | `model_minimind.py:167` | MoE 输出缓冲 |
| `torch.zeros(B)` | `train_ppo.py:46` | 奖励向量 |
| `torch.ones_like(x)` | `train_agent.py:113` | attention_mask |
| `torch.full((L,L), -inf)` | `model_minimind.py:129` | 因果掩码 |
| `torch.arange(n)` | `model_minimind.py:74`；`train_ppo.py:114` | 位置、索引 |
| `torch.randperm(n)` | 各训练脚本 | 每轮打乱数据 |
| `torch.randint(0, V, (B,L))` | 探针/测试 | 假数据 |
| `torch.outer(t, freqs)` | `model_minimind.py:75` | RoPE 的「位置 × 频率」 |

**关键点**：`torch.full(..., float("-inf"))` 是 causal mask 的标准写法；
用 `-1e9` 的地方（`:130`）是为了避免 fp16 下 `-inf` 产生 NaN。

### 1.2 形状变换

| API | 形状语义 | MiniMind 出处 |
| --- | --- | --- |
| `view(...)` | 重组维度（要求连续） | `model_minimind.py:114` 拆头 |
| `reshape(...)` | 同上，必要时拷贝 | `:89` repeat_kv |
| `permute(2,0,1)` | 换轴顺序 | `:124` 转 SDPA 布局 |
| `transpose(1,2)` | 交换两轴 | `:124` |
| `unsqueeze(dim)` | 插一个长度为 1 的维度 | `:82` 广播 cos/sin |
| `squeeze(-1)` | 去掉长度 1 的维 | `:41` value head |
| `flatten(1)` | 从第 1 维起拉平 | 数据侧 |
| `contiguous()` | 保证内存连续（`view` 的前置） | `:256` 损失前 |
| `expand(...)` | 零拷贝广播 | `:89` GQA 复制 KV |
| `repeat_interleave(G, 0)` | 每个元素连续复制 G 份 | `rollout_engine.py:76` |
| `numel()` / `shape` | 元素数 / 形状 | 参数统计 |

**`repeat_interleave` 是 GRPO/PPO 的关键**：每个 prompt 复制 6 份/1 份，
顺序必须是 `[p0,p0,p0,p1,p1,p1]`，训练脚本后面才能 `view(B, G)`。

### 1.3 拼接与切分

| API | 用途 | 出处 |
| --- | --- | --- |
| `torch.cat([...], dim)` | 沿已有维度拼接 | RoPE 的 cos 拼两遍（`:76`）、KV cache（`:121`） |
| `torch.stack([...], dim)` | 新建维度堆叠 | `rollout_engine.py:50` 逐行 logp |
| `[..., :-1]` / `[..., 1:]` | 切片 | 语言模型的错位 |
| `x.chunk(n)` | 等分 | 探针演示分片 |

**易错**：`cat` 不增加维度，`stack` 增加维度。把 `stack` 当 `cat` 用，形状会多一维。

> 第 1 章口诀：**view 要连续，cat 不加维，stack 才加维；repeat_interleave 保分组。**

---

## 第 2 章 dtype、device 与精度

> 口诀：**模型 float，标签 long；算完转回原精度。**

### 2.1 常用 dtype

| dtype | MiniMind 用途 | 出处 |
| --- | --- | --- |
| `torch.long` | token id、labels | `lm_dataset.py` |
| `torch.float32` | scaling、掩码加法 | `model_lora.py:11` |
| `torch.float16` / `bfloat16` | 训练与推理 | 各训练脚本 |
| `torch.bool` | finished 标记 | `model_minimind.py:267` |

### 2.2 转换

| API | 语义 | 出处 |
| --- | --- | --- |
| `.to(device)` | 搬设备 | 到处 |
| `.to(dtype)` | 转精度 | — |
| `.half()` / `.float()` | 半/全精度 | 保存权重 `.half().cpu()` |
| `.type_as(x)` | 转成与 x 同精度 | `model_minimind.py:60` RMSNorm 出口 |
| `x.float()` | 升到 fp32 再算 | `:60` 归一化；`:128` softmax 前 |

**RMSNorm 的经典写法**：

```python
return (self.weight * self.norm(x.float())).type_as(x)
```

先升 fp32 算方差（避免 fp16 溢出），算完用 `type_as` 落回原精度。

### 2.3 设备与能力探测

| API | 用途 | 出处 |
| --- | --- | --- |
| `torch.device(args.device).type` | 判断 cuda/cpu | 各脚本 |
| `torch.cuda.is_available()` | 能否用 GPU | 参数默认值 |
| `torch.cuda.get_device_name(0)` | 打印显卡名 | `resolve_dtype` |
| `torch.cuda.is_bf16_supported(including_emulation=False)` | 原生 bf16 判断 | `trainer_utils.py:50` |
| `torch.cuda.set_device(local_rank)` | DDP 绑定卡 | `trainer_utils.py:72` |

**`including_emulation=False` 的意义**：只认硬件原生支持，避免「能跑但慢十倍」。

> 第 2 章口诀：**索引 long、参数 float；归一化升 fp32，出口 type_as。**

---

## 第 3 章 自动求导的六个开关

> 口诀：**学习参数要 requires，推理上下文要 no_grad；取日志先 detach。**

### 3.1 六个开关对照

| API | 作用 | MiniMind 出处 |
| --- | --- | --- |
| `p.requires_grad = True/False` | 标记参数是否训练 | LoRA 冻结（`train_lora.py:138`） |
| `model.requires_grad_(False)` | 整模型冻结 | 参考模型（`train_dpo.py:181`） |
| `loss.backward()` | 反向传播 | 所有训练脚本 |
| `.detach()` | 切断计算图 | 日志累加、CISPO 的 ratio（`train_grpo.py:130`） |
| `torch.no_grad()` | 上下文内不建图 | 参考模型前向、rollout |
| `torch.inference_mode()` | 更强的不建图 | `MiniMind.generate`（`:261`） |

### 3.2 为什么参考模型要 `no_grad`

```python
with torch.no_grad():
    ref_logits = ref_model(x).logits          # train_dpo.py:72
```

不建图 = 不存激活、不存中间结果。DPO 每步都要跑参考模型，
不开 no_grad 会让显存直接翻倍。

### 3.3 `detach` 的三个真实用途

| 用途 | 代码 | 原因 |
| --- | --- | --- |
| 日志累加 | `total += loss.detach().item()` | 别把计算图挂住 |
| 教师目标 | `teacher_probs.detach()` | 教师不参与梯度 |
| CISPO 的 ratio | `clamp(ratio, max=5.0).detach()` | 只作权重，不传梯度 |

### 3.4 一个容易忽略的细节

`MiniMind.generate` 标了 `@torch.inference_mode()`，
所以它返回的是 **inference tensor**。本地 rollout 引擎因此要 `.clone()`
（`rollout_engine.py:84`），否则后续 `torch.cat` 等操作可能报错。

> 第 3 章口诀：**训练建图，推理关图；日志 detach，参考冻结。**

---

## 第 4 章 用 nn.Module 搭模型

> 口诀：**init 里建层，forward 里连线；子模块注册，buffer 不训练。**

### 4.1 项目里用到的层

| 层 | 出处 | 说明 |
| --- | --- | --- |
| `nn.Module` | 全部结构 | 基类 |
| `nn.ModuleList` | `model_minimind.py:153` | MoE 的 4 个专家 |
| `nn.Linear(in, out, bias=False)` | 所有投影 | 无偏置 |
| `nn.Embedding(V, d)` | `:206` | token → 向量 |
| `nn.Dropout(p)` | `:106` | 默认 p=0 |
| `nn.Parameter(torch.ones(dim))` | RMSNorm `:54` | 可学习缩放 |
| `register_buffer(name, t, persistent=False)` | `:211` | RoPE 的 cos/sin |

**为什么 RoPE 用 buffer 而不是 Parameter**：

| 属性 | 进 state_dict | 参与梯度 | 随 `.to()` 搬动 |
| --- | --- | --- | --- |
| Parameter | ✅ | ✅ | ✅ |
| buffer | ✅（`persistent=True`） | ❌ | ✅ |
| buffer（`persistent=False`） | **❌** | ❌ | ✅ |

RoPE 的 cos/sin 由 config 决定、不需要保存，所以用 `persistent=False`。

### 4.2 自定义模块：RMSNorm 的完整写法

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.float()).type_as(x)
```

**三个要点**：`super().__init__()` 必须先调；参数必须赋给 `self.*`；
计算在 `forward` 里。

### 4.3 权重管理

| API | 用途 | 出处 |
| --- | --- | --- |
| `model.state_dict()` | 取全部参数 | 保存 |
| `model.load_state_dict(sd, strict=False)` | 加载（允许键缺失） | `init_model` |
| `model.eval()` / `train()` | 切换 dropout/BN 行为 | 每轮训练、保存时 |
| `model.named_modules()` | 遍历子模块 | LoRA 注入、weight 绑定 |
| `model.named_parameters()` | 遍历参数 | 冻结、统计 |
| `model.parameters()` | 只取参数张量 | 优化器、裁剪 |

**`strict=False` 的真实用途**（`trainer_utils.py:152`）：
加载 SFT 权重去训练 MoE 版本时，键对不齐；或者 LoRA 续训时新增了 `lora.*` 键。
**但它不会放过形状不匹配**。

### 4.4 LoRA 的「猴补丁」（`:31`）

```python
original_forward = module.forward
def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)
module.forward = forward_with_lora
```

两个知识点：
- `setattr(module, "lora", lora)` 让 LoRA 成为注册子模块，才会出现在 `parameters()`；
- 默认参数 `layer1=...` 是**早绑定**，避开闭包陷阱。

> 第 4 章口诀：**层在 init 建，buffer 不进优化器；strict=False 只放键、不放形状。**

---

## 第 5 章 数学与损失函数

> 口诀：**CE 吃 logits，KL 要配 T²；DPO 用 logsigmoid，top-p 靠 cumsum。**

### 5.1 交叉熵：三种用法

| 用法 | 代码 | 出处 |
| --- | --- | --- |
| 标准（跳过 -100） | `F.cross_entropy(x.view(-1,V), y.view(-1), ignore_index=-100)` | `model_minimind.py:257` |
| 逐元素（自己加权） | `reduction='none'` 再乘 mask 求和 | `train_distillation.py:72` |
| 概率比形式的 CE | 先 `log_softmax` 再 `gather` | `train_dpo.py:29` |

**为什么语言模型要错位**：

```python
x, y = logits[..., :-1, :], labels[..., 1:]
```

位置 t 的预测，对应位置 t+1 的标签。**数据不动，模型错位**——所以
`PretrainDataset` 里 labels 与 input_ids 完全相同。

**`ignore_index=-100` 的作用**：跳过 padding 与「不训练的位置」（如用户提问）。
如果所有位置都被跳过，`cross_entropy` 会返回 **nan**——这是 SFT 标记写错时的典型症状。

### 5.2 log_softmax + gather：取出真实 token 的概率

```python
log_probs = F.log_softmax(logits, dim=2)                          # (B,L,V)
pick = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1) # (B,L)
```

出处：`train_dpo.py:25`、`train_grpo.py:95`。

**为什么不用 `logits.gather`**：必须先归一化成概率，取出的才是 log 概率。

### 5.3 KL 散度：两种场景

| 场景 | 代码 | 出处 |
| --- | --- | --- |
| 蒸馏（整分布） | `F.kl_div(student_logp, teacher_prob, reduction='batchmean') * T**2` | `train_distillation.py:33` |
| 策略约束（逐 token） | `exp(kl) - kl - 1`（k3 估计） | `train_grpo.py:127` |

**蒸馏的温度三件套**：

```python
teacher_probs = softmax(teacher_logits / T)
student_log_probs = log_softmax(student_logits / T)
loss = T**2 * F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```

**k3 估计**为什么更好：非负、无偏、数值稳定。直接 `-kl` 会有负数与噪声。

### 5.4 logsigmoid：DPO 的四行核心

```python
pi_logratios  = chosen_policy - reject_policy
ref_logratios = chosen_ref    - reject_ref
loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

出处：`train_dpo.py:46-50`。

### 5.5 注意力与采样用的数学

| API | 用途 | 出处 |
| --- | --- | --- |
| `F.scaled_dot_product_attention(q,k,v,dropout_p,is_causal)` | 融合注意力 | `model_minimind.py:126` |
| `torch.outer(t, freqs)` | RoPE 位置 × 频率 | `:75` |
| `torch.sin` / `torch.cos` | 旋转编码 | `:76` |
| `torch.rsqrt` | 1/√x | `:57` |
| `torch.exp` | k3、ratio | `train_grpo.py:127-128` |
| `torch.clamp(x, min, max)` | 裁剪 | `train_grpo.py:133` |
| `torch.topk(x, k)` | top-k / MoE 路由 | `:160` |
| `torch.sort(x, descending=True)` | top-p 排序 | `:280` |
| `torch.cumsum(p, dim)` | top-p 累积 | `:281` |
| `torch.multinomial(p, 1)` | 按概率采样 | `:283` |
| `torch.argmax(x, dim)` | 贪心 | `:283` |
| `F.one_hot(idx, n).float().mean(0)` | MoE 负载统计 | `:177` |
| `F.pad(t, (1,0))` | top-p 掩码右移 | `:282` |

**top-p 的三行**（`:280-282`）值得单独背：

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
mask = torch.cumsum(torch.softmax(sorted_logits, -1), -1) > top_p
logits[mask.scatter(1, sorted_indices, F.pad(mask[..., :-1], (1, 0)))] = -float('inf')
```

`F.pad` 把 mask 右移一位，保证「跨过阈值」的那个 token 被保留。

### 5.6 `torch.where` 与 `torch.unique`

| API | 用途 | 出处 |
| --- | --- | --- |
| `torch.where(cond, a, b)` | 条件选择 | 重复惩罚（`:276`）、finished 行强制 EOS（`:284`） |
| `torch.unique(ids)` | 去重 | 重复惩罚只惩罚出现过的 token（`:276`） |

> 第 5 章口诀：**CE 吃 logits 跳过 -100；KL 乘 T²，k3 非负；top-p 排序累加再移位。**

---

## 第 6 章 索引、掩码与 gather/scatter

> 口诀：**gather 取指定位置，scatter 写指定位置，index_add_ 累加。**

### 6.1 `torch.gather`：按索引取值

```python
# 从 (B,L,V) 里，为每个 (b,l) 取出 labels[b,l] 那一项
torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
```

| 参数 | 要求 |
| --- | --- |
| 输入与索引 | 除 `dim` 外形状一致 |
| 索引 dtype | **必须是 long** |
| 索引范围 | 不能越界 |

**MiniMind 的高级用法**（`train_grpo.py:95`）：

```python
logps = F.log_softmax(res.logits[:, :-1], -1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

两次 gather：先取真实 token 的 logp（`(B,L-1)`），再按 `logp_pos` 取出回答位置。

### 6.2 `scatter_`：按索引写入

```python
full_mask.scatter_(1, logp_pos + 1, completion_mask)   # train_grpo.py:88
```

把「回答位置」的 mask 覆盖成 1。`scatter_` 是**原地**操作，会修改原张量。

**top-p 里的 `scatter`** 用的是非原地版本 + 返回值（`:282`），因为要基于排序后的索引还原顺序。

### 6.3 `index_add_`：按索引累加（MoE 的核心）

```python
y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)   # model_minimind.py:173
```

| 特点 | 说明 |
| --- | --- |
| 累加而非覆盖 | 同一 token 命中多个专家时正确 |
| 原地 | 返回自身 |
| 索引 long | `token_idx` 必须是整型 |

### 6.4 布尔掩码

| 写法 | 用途 | 出处 |
| --- | --- | --- |
| `x[mask]` | 选出 True 位置 | 蒸馏 `[loss_mask_flat == 1]` |
| `(topk_idx == i).any(-1).nonzero().flatten()` | 命中的 token 行号 | MoE `:171` |
| `x.eq(eos)` / `x == eos` | 找终止符 | `train_ppo.py:118` |
| `is_eos.any(dim=1)` | 是否出现过 | `:119` |
| `mask.any()` | 非空判断 | `:170` |

**常见坑**：布尔索引后**维度会变**（变成一维），后续 `view` 前要想清楚。

### 6.5 掩码对齐：三个必须一致的形状

以 GRPO 为例（`train_grpo.py:86-96`）：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `outputs` | `(B, P+R)` | prompt + 回答 |
| `completion_mask` | `(B, R)` | 回答有效位置 |
| `logp_pos` | `(B, R)` | 回答在 logits 中的下标 |
| `per_token_logps` | `(B, R)` | 每个回答 token 的 logp |

**只要这四个有一个错一位，loss 就会「看起来正常但学不到东西」。**

> 第 6 章口诀：**gather/scatter 索引必须 long；布尔索引会降维；四个张量要同长。**

---

## 第 7 章 优化器、梯度裁剪与学习率

> 口诀：**unscale 再裁剪，step 前先清零；余弦留一成，任务换量级。**

### 7.1 优化器

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
```

出处：所有训练脚本。项目只用 `AdamW`，没有 SGD。

**PPO 有两套**（`train_ppo.py:397-398`）：

```python
actor_optimizer  = optim.AdamW(actor_model.parameters(),  lr=3e-7)
critic_optimizer = optim.AdamW(critic_model.parameters(), lr=5e-7)
```

### 7.2 手动改学习率

项目**不用** `scheduler.step()` 来驱动主训练（除 PPO/GRPO 的余弦外），
而是每步手写覆盖：

```python
for param_group in optimizer.param_groups:
    param_group['lr'] = lr        # train_pretrain.py:30-31
```

| 方式 | 优点 | 缺点 |
| --- | --- | --- |
| `param_groups['lr'] = lr` | 无状态，续训天然正确 | 曲线要自己写（`get_lr`） |
| `scheduler.step()` | 现成曲线多 | 状态要一起存取 |

### 7.3 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

出处：所有训练脚本。**必须在 `scaler.unscale_()` 之后**（见第 8 章）。
LoRA 版本只裁适配器（`train_lora.py:43`）：

```python
clip_grad_norm_(lora_params, 1.0)
```

### 7.4 学习率调度：余弦

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
```

出处：`train_ppo.py:403`、`train_grpo.py:286`、`train_agent.py:452`。

| 参数 | 含义 | 坑 |
| --- | --- | --- |
| `T_max` | 总优化步数 | 算少了会提前归零 |
| `eta_min` | 地板值（lr/10） | 不建议为 0 |

**PPO 的步数最容易算错**：

```python
total = ceil(iters × epochs × ppo_update_iters × mb_factor / accum)
```

漏乘 `ppo_update_iters` 或 `mb_factor`，学习率曲线就崩了。

### 7.5 手写余弦（`trainer_utils.py:57`）

```python
lr * (0.1 + 0.45 * (1 + math.cos(angle)))
```

从 1.0 降到 0.1，保留 10% 地板。**为什么留一成**：避免后期梯度噪声主导。

> 第 7 章口诀：**zero_grad 放最后；clip 在 unscale 后；T_max 算总数。**

---

## 第 8 章 混合精度与 GradScaler

> 口诀：**autocast 管前向，scaler 管反向；scale → unscale → clip → step → update。**

### 8.1 精度上下文

```python
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
```

出处：所有训练脚本。CPU 上用 `nullcontext()` 占位，代码路径统一。

| dtype | loss scaling | 适用 |
| --- | --- | --- |
| bf16 | 不需要 | Ampere+ |
| fp16 | **必须** | 老卡 |

### 8.2 GradScaler 的五个动作

```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

scaler.scale(loss).backward()        # ① 放大损失再反向
scaler.unscale_(optimizer)           # ② 梯度还原本来的大小
torch.nn.utils.clip_grad_norm_(...)  # ③ 裁剪真实梯度
scaler.step(optimizer)               # ④ 检查 inf/nan 再更新
scaler.update()                      # ⑤ 调整下一次的 scale
optimizer.zero_grad(set_to_none=True)# ⑥ 清空
```

出处：`train_pretrain.py:38-47`，所有脚本一致。

**为什么顺序不能变**：

| 错误顺序 | 后果 |
| --- | --- |
| 先裁剪再 unscale | 裁的是放大后的梯度，阈值失效 |
| 用 `optimizer.step()` 代替 `scaler.step()` | 溢出时污染权重 |
| 先 `zero_grad` 再 `step` | 梯度被清空，等于没更新 |

### 8.3 `enabled` 开关

```python
GradScaler(enabled=(args.dtype == 'float16'))
```

bf16 下 scaler 自动禁用（空操作），所以同一套代码能跑两种精度。

### 8.4 本机实测提醒

在 GTX 1650（Turing）上，**fp16 autocast 反而比 fp32 慢约 4~5 倍**——
`resolve_dtype` 会把你的卡降成 fp16，反而变慢。
预算紧张时可以手动改回 fp32（见第 2 章教材的实测表）。

> 第 8 章口诀：**scale 进、unscale 出；clip 在中间，step 会跳过 inf。**

---

## 第 9 章 数据管线与批采样

> 口诀：**Dataset 管单条，Sampler 管顺序，DataLoader 管成批。**

### 9.1 三个角色的分工

| 角色 | 你要实现 | MiniMind 出处 |
| --- | --- | --- |
| `Dataset` | `__len__` + `__getitem__` | `lm_dataset.py` 四个 Dataset |
| `Sampler` | `__iter__` + `__len__` | `SkipBatchSampler`（`trainer_utils.py:160`） |
| `DataLoader` | 传参数 | 各训练脚本 |

### 9.2 DataLoader 的关键参数

```python
loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                    num_workers=args.num_workers, pin_memory=True)
```

| 参数 | MiniMind 用法 | 坑 |
| --- | --- | --- |
| `batch_sampler=` | 传 `SkipBatchSampler`（自带打乱与跳过） | 与 `batch_size` 互斥 |
| `num_workers` | Windows 默认 0 | 多进程需 `if __name__ == "__main__"` |
| `pin_memory=True` | 加速 CPU→GPU | 只在 CUDA 下有意义 |
| `collate_fn` | Agent RL 自定义（dict 列表不转张量） | 默认 collate 会尝试 stack |

### 9.3 `SkipBatchSampler`：断点续训的快进键

```python
class SkipBatchSampler(Sampler):
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches: skipped += 1; batch = []; continue
                yield batch; batch = []
```

**它接受两种输入**：`DistributedSampler`（多卡）或 `randperm` 列表（单卡）。
这就是「单卡/多卡同一套代码」的实现方式。

### 9.4 多卡时的两个约定

```python
train_sampler.set_epoch(epoch)                       # 每轮必须调
setup_seed(args.seed + epoch); indices = randperm(n) # 单卡打乱
```

`DistributedSampler` 用 `set_epoch` 决定本轮的切分顺序，不调用会导致**每轮数据完全相同**。

### 9.5 先例：数据集里的错位

`lm_dataset.py` 的 DPO 数据集做了 `x = ids[:-1]`、`y = ids[1:]`，
而 SFT 数据集不做——因为模型内部已经错位。**两处只能错一次**。

> 第 9 章口诀：**set_epoch 每轮调，Windows worker 设 0；错位只做一次。**

---

## 第 10 章 分布式：DDP 与集合通信

> 口诀：**RANK 判断是否分布式；DDP 包模型，集合通信传信号。**

### 10.1 初始化

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1: return 0     # 非分布式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

出处：`trainer_utils.py:66`。

| API | 用途 |
| --- | --- |
| `dist.is_initialized()` | 判断是否已初始化（调用 46 次，最常用） |
| `dist.get_rank()` | 全局进程号 |
| `dist.get_world_size()` | 总进程数 |
| `dist.barrier()` | 同步所有进程 |
| `dist.destroy_process_group()` | 收尾 |

**`backend="nccl"` 写死**：Windows 没有 NCCL，所以这个函数在 Windows 上必然失败。

### 10.2 DDP 包装

```python
model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
```

出处：`train_ppo.py:423`、`train_grpo.py:303`、`train_agent.py:467`。

**`broadcast_buffers=False`**：默认每步把 rank0 的 buffer 广播给所有卡。
模型里唯一的 buffer 是 RoPE 的 cos/sin，由 config 确定性算出、各卡必然相同，
广播纯属浪费。**前提**：buffer 真的各卡一致。

### 10.3 集合通信：两个真实用法

**① 早停前同步 KL**（`train_ppo.py:188-193`）：

```python
approx_kl_val = approx_kl.detach().clone()
if dist.is_initialized():
    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
if approx_kl_val > args.early_stop_kl: stop_ppo = True
```

不先同步，某卡停了、其它卡继续 forward，DDP 会永久挂起。

**② 广播成功标志**（`rollout_engine.py:190-193`）：

```python
ok_t = torch.tensor(int(ok), device=...)
dist.broadcast(ok_t, src=0)
dist.barrier()
if not ok: raise RuntimeError(...)
```

rank0 上传权重失败时，所有卡一起失败，而不是一半新策略一半旧策略。

### 10.4 单卡也能跑同一套代码

所有训练脚本都写了 `if dist.is_initialized()` 判断，所以：

```bash
python train_pretrain.py                 # 单卡
torchrun --nproc_per_node 8 train_pretrain.py   # 8 卡
```

> 第 10 章口诀：**先同步再决策，否则一半卡等一半卡跑。**

---

## 第 11 章 保存、加载与断点续训

> 口诀：**推理存 half，续训存全套；先写 tmp 再 replace。**

### 11.1 两种保存

```python
# ① 推理权重：只要参数，半精度
torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()}, ckp)

# ② 续训状态：参数 + 优化器 + scaler + 进度
torch.save({'model': sd, 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'step': step, 'world_size': ws}, resume)
```

出处：`trainer_utils.py:88`（`lm_checkpoint`）。

| 文件 | 内容 | 体积 |
| --- | --- | --- |
| `out/xxx_768.pth` | 半精度权重 | ~128MB |
| `checkpoints/xxx_768_resume.pth` | fp32 + 优化器 + scaler + 进度 | ~1GB+ |

### 11.2 加载的三个参数

```python
weights = torch.load(path, map_location=device)     # ① 映射到目标设备
model.load_state_dict(weights, strict=False)        # ② 允许键缺失
```

| 参数 | 作用 | 坑 |
| --- | --- | --- |
| `map_location` | 避免「GPU 存的在 CPU 上加载不了」 | 必须给 |
| `strict=False` | 允许缺键/多键 | 形状不匹配仍报错 |
| `weights_only`（torch 2.6 默认 True） | 只允许张量/基础类型 | 存了自定义对象会失败 |

### 11.3 脱壳：DDP 与 compile

```python
raw = model.module if isinstance(model, DistributedDataParallel) else model
raw = getattr(raw, '_orig_mod', raw)          # torch.compile 的壳
```

不脱壳，state_dict 的键会带 `module.` / `_orig_mod.` 前缀，加载不回去。

### 11.4 原子写

```python
tmp = path + '.tmp'
torch.save(obj, tmp)
os.replace(tmp, path)      # 原子替换
```

写到一半断电，旧文件仍然完整。

### 11.5 换卡续训

```python
if saved_ws != current_ws:
    step = step * saved_ws // current_ws
```

4 卡跑 100 步 ≈ 8 卡跑 50 步，再配合 `SkipBatchSampler` 继续。

> 第 11 章口诀：**存前脱壳，加载给 map_location；先 tmp 后 replace。**

---

## 第 12 章 随机性与复现

> 口诀：**五个随机源全固定，cudnn 确定性关 benchmark。**

### 12.1 `setup_seed` 的完整写法

```python
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

出处：`trainer_utils.py:77`。

| 开关 | 代价 | 收益 |
| --- | --- | --- |
| `deterministic=True` | 部分卷积变慢 | 可复现 |
| `benchmark=False` | 不做算法寻优 | 可复现 |

### 12.2 调两次的原因

```python
setup_seed(args.seed + rank)     # 进程级：每卡不同
setup_seed(args.seed + epoch)    # 每轮：保证打乱顺序可复现
```

> 第 12 章口诀：**rank 区分卡，epoch 区分轮；两次都不能省。**

---

## 第 13 章 性能与显存

> 口诀：**能省则省：少算 logits、少建图、少同步。**

### 13.1 项目里用到的四个提速手段

| 手段 | 代码 | 效果 |
| --- | --- | --- |
| `torch.inference_mode()` | `generate` | 不建图，最省 |
| `logits_to_keep` | `MiniMindForCausalLM.forward` | 只算最后 k 个位置的输出层 |
| `torch.compile(model)` | 各脚本（Windows 上不可用） | 融合算子 |
| `autocast` | 各脚本 | 半精度计算 |

**`logits_to_keep` 的价值**（`rollout_engine.py:24` 用到）：
生成 1024 token、prompt 只有 10 个时，只算最后 1025 个位置的 logits，
省掉约 99% 的输出层计算。

### 13.2 显存 API

| API | 用途 |
| --- | --- |
| `torch.cuda.empty_cache()` | 释放缓存池（`lm_checkpoint` 用） |
| `torch.cuda.memory_allocated()` | 当前占用 |
| `torch.cuda.max_memory_allocated()` | 峰值 |
| `torch.cuda.reset_peak_memory_stats()` | 重置峰值 |

### 13.3 三个隐形的性能损耗

| 位置 | 问题 | 说明 |
| --- | --- | --- |
| `torch.all(attention_mask == 1)`（`:125`） | GPU→CPU 同步 | 每层每步一次 |
| `repeat_kv` 的 `expand().reshape()` | 真实拷贝 | 4→8 头 |
| 手写 causal mask 每次新建 | 分配开销 | 可用缓存优化 |

### 13.4 `torch.compile` 的壳

```python
raw = getattr(model, '_orig_mod', model)
```

`compile` 后模型的真实对象藏在 `_orig_mod` 里。保存、冻结、统计参数都要先脱壳。

> 第 13 章口诀：**推理不建图，输出只算尾；compile 要脱壳。**

---

## 第 14 章 三个完整模式（把 API 串起来）

### 14.1 模式一：训练步五件套

```python
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)

    optimizer.zero_grad(set_to_none=True)          # ①
    with autocast_ctx:
        loss = model(xb, labels=yb).loss / accum   # ②
    scaler.scale(loss).backward()                  # ③
    if step % accum == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()    # ④
        optimizer.zero_grad(set_to_none=True)      # ⑤
```

**用到的 API**：`to`、`zero_grad`、`autocast`、`scale/backward`、`unscale_`、
`clip_grad_norm_`、`step/update` — 共 8 个，覆盖了 80% 的日常训练。

### 14.2 模式二：RL 的「采样—打分—更新」

```python
rollout  = engine.rollout(prompts, ...)                     # no_grad + generate
rewards  = calculate_rewards(...)                           # 规则/RM
logps    = log_softmax(model(outputs).logits[:, :-1]).gather(...)   # 重新前向
ref_logps = compute_per_token_logps(ref_model, ...)         # no_grad
advantages = (rewards - group_mean) / (group_std + 1e-4)    # 组内标准化
loss = -(clamp(ratio,...) * advantages * logps - beta * kl) # CISPO
loss.backward(); optimizer.step()
```

**用到的 API**：`no_grad`、`generate`、`log_softmax`、`gather`、`view`、
`mean/std`、`repeat_interleave`、`clamp`、`exp`、`detach` — 这是全项目最密集的一段。

### 14.3 模式三：断点续训

```python
ckp = torch.load(resume_path, map_location="cpu")     # 读
model.load_state_dict(ckp["model"])                    # 恢复模型
optimizer.load_state_dict(ckp["optimizer"])            # 恢复优化器
start_step = ckp["step"]                               # 恢复进度
loader = DataLoader(ds, batch_sampler=SkipBatchSampler(sampler, bs, start_step))
```

**用到的 API**：`torch.load`、`load_state_dict`、`Sampler`、`DataLoader(batch_sampler=)`。

> 第 14 章口诀：**清零→前向→损失→反向→更新；采样→打分→优势→裁剪。**

---

## 附录 A API 速查表（按组）

### 张量创建与形状

| API | 一句话 | 项目出处 |
| --- | --- | --- |
| `torch.tensor` | 列表转张量 | Dataset |
| `zeros` / `zeros_like` | 零张量 | MoE 输出 |
| `ones` / `ones_like` | 全 1 | mask |
| `full` | 指定值填充 | causal mask |
| `arange` | 等差序列 | 位置/索引 |
| `randperm` | 随机排列 | 打乱数据 |
| `randint` | 随机整数 | 假数据 |
| `outer` | 外积 | RoPE |
| `cat` / `stack` | 拼接/堆叠 | cache、logp |
| `view` / `reshape` | 改形状 | 拆头 |
| `permute` / `transpose` | 换轴 | SDPA 布局 |
| `unsqueeze` / `squeeze` | 加/去维 | 广播 |
| `contiguous` | 连续化 | view 前 |
| `expand` | 零拷贝广播 | GQA |
| `repeat_interleave` | 连续复制 | 多采样 |
| `numel` / `shape` | 元信息 | 统计 |

### 索引与数学

| API | 一句话 |
| --- | --- |
| `gather` | 按索引取值（logp） |
| `scatter_` / `scatter` | 按索引写入（mask、top-p） |
| `index_add_` | 按索引累加（MoE） |
| `topk` / `sort` | 取前 k / 排序 |
| `cumsum` | 累积和（top-p） |
| `argmax` / `multinomial` | 贪心 / 采样 |
| `where` / `eq` / `any` / `all` | 条件与判断 |
| `unique` | 去重（重复惩罚） |
| `clamp` / `min` / `max` | 截断与取极值 |
| `exp` / `log` / `pow` / `rsqrt` | 数学函数 |
| `sin` / `cos` | RoPE |
| `mean` / `sum` | 聚合 |

### 损失与注意力

| API | 一句话 | 项目出处 |
| --- | --- | --- |
| `F.cross_entropy` | 交叉熵（ignore_index） | 语言模型 |
| `F.log_softmax` | 对数概率 | DPO/GRPO |
| `F.softmax` | 概率 | 蒸馏/采样 |
| `F.kl_div` | KL 散度 | 蒸馏 |
| `F.logsigmoid` | 对数 sigmoid | DPO |
| `F.one_hot` | 独热 | MoE 负载 |
| `F.pad` | 填充 | top-p 掩码 |
| `F.scaled_dot_product_attention` | 融合注意力 | Attention |

### 模块与训练

| API | 一句话 |
| --- | --- |
| `nn.Module` / `ModuleList` / `Parameter` | 模型骨架 |
| `nn.Linear` / `nn.Embedding` / `nn.Dropout` | 常用层 |
| `register_buffer` | 非训练常量 |
| `state_dict` / `load_state_dict` | 存取权重 |
| `train()` / `eval()` | 切换模式 |
| `named_modules` / `named_parameters` / `parameters` | 遍历 |
| `requires_grad_` / `detach` / `no_grad` / `inference_mode` | 梯度控制 |
| `optim.AdamW` | 优化器 |
| `lr_scheduler.CosineAnnealingLR` | 余弦调度 |
| `clip_grad_norm_` | 梯度裁剪 |
| `autocast` / `GradScaler` | 混合精度 |
| `DistributedDataParallel` | 多卡 |
| `DataLoader` / `Dataset` / `Sampler` | 数据 |
| `torch.save` / `torch.load` | 存取文件 |
| `torch.compile` | 编译加速 |

---

## 附录 B 与项目其他教材的对照

| 问题 | 去哪本看 |
| --- | --- |
| 张量/自动求导/训练循环基础 | 《PyTorch 实战》 |
| 分词器、AutoModel、generate | 《transformers 实战》 |
| ZeRO、多卡显存、DeepSpeed 配置 | 《DeepSpeed 实战》 |
| MiniMind 每个文件的设计取舍 | 《MiniMind 源码解读》系列 |

**本文的定位**：把前几本的「概念」落到「MiniMind 真实代码用了哪个 API」。

---

## 附录 C 练习

1. 找到 `train_grpo.py` 里所有用 `gather` 的地方，说出每次取的是什么。
2. `full_mask.scatter_` 为什么索引要 `+1`？
3. 为什么 `clip_grad_norm_` 必须在 `scaler.unscale_` 之后？
4. `repeat_interleave` 与 `repeat` 的区别？换成 `repeat` 会怎样？
5. `logits_to_keep` 为什么能省 99% 的计算？多算 1 个位置的目的是什么？
6. `dist.all_reduce` 在 PPO 早停里防的是什么？
7. `torch.load` 在 torch 2.6 的默认 `weights_only` 是什么？项目里为什么没出问题？
8. `broadcast_buffers=False` 的前提是什么？什么情况下会出错？
9. `strict=False` 允许什么、不允许什么？
10. 为什么 `setup_seed` 要调两次？

### 答案要点

1. 第一次取真实 token 的 logp，第二次按 `logp_pos` 取回答位置。
2. logits 与 token 错位一位，回答 token 在 `logp_pos+1`。
3. 否则裁的是被放大的梯度。
4. 复制顺序不同；`repeat` 会打乱分组，`view(B,G)` 就对不上。
5. 只算尾部位置的输出层；多算 1 个用于错位对齐。
6. 防止部分卡早停、其它卡继续导致 DDP 挂起。
7. 默认 True；项目存的是张量/基础类型，所以能加载。
8. buffer 各卡一致（RoPE 由 config 决定）。
9. 允许键缺失/多余；不允许形状不匹配。
10. 进程级区分卡，epoch 级保证打乱可复现。


