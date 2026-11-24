# MiniMind 源码解读：3 块钱，从零训练一个自己的 LLM

> **解读对象**：`minimind-master`（jingyaogong/minimind，minimind-3 版本）
> **解读环境**：本机 conda `swift_env`（Python 3.10 / torch 2.6.0+cu124 / transformers 4.57.6）
> **验证方式**：模型代码已在本机实际加载运行，本文「实测」数字均来自本机。

---

## 先看结论：这个项目到底是什么

| 维度 | 事实 |
| --- | --- |
| 定位 | 教学向的「从零手写 LLM」项目，**不依赖** trl / peft / deepspeed 的训练封装 |
| 模型 | Decoder-only Transformer，结构对齐 Qwen3 生态 |
| 规模 | Dense **63.91M**（实测）；MoE **198.42M**，4 experts / top-1 |
| 词表 | 6400（自训 BPE + ByteLevel，36 个特殊 token） |
| 结构 | 8 层 / d_model 768 / 8 Q 头 / 4 KV 头（GQA）/ FFN 2432 / RoPE θ=1e6 |
| 训练链 | Tokenizer → Pretrain → SFT → LoRA → 蒸馏 → DPO → PPO → GRPO/CISPO → Agentic RL |
| 成本 | 单卡 3090：pretrain 1.21h（¥1.57）+ SFT 1.10h（¥1.43）≈ **2.3 小时、3 元** |
| 部署 | 原生权重 → HF 格式 → llama.cpp / vLLM / ollama；自带 OpenAI 兼容 API 与 WebUI |

**一句话**：它把「一个大模型公司内部做的事」压缩到一个你一夜能读完、几小时能跑完的规模，
而且每一行都是可读的 PyTorch。

> 口诀：**结构对齐 Qwen3，代码不靠框架；数据一条 jsonl，训练一张 3090。**

---

## 这本书怎么读

按「先看骨架、再看血肉、最后看取舍」三遍：

1. **第一遍（30 分钟）**：读本文第 2 章的代码地图 + 第 5 章的训练骨架，把「一个训练脚本由哪九块组成」记住。
2. **第二遍（2 小时）**：读第 3、4 章，对照源码逐行看模型和数据。
3. **第三遍（按需）**：读第 6 章各训练阶段，挑你关心的算法（SFT / DPO / GRPO）精读。

---

## 目录

- 第 1 章 项目定位与版本演进
- 第 2 章 代码地图与阅读路线
- 第 3 章 模型结构逐层拆解
- 第 4 章 数据管线
- 第 5 章 训练脚本的统一骨架
- 第 6 章 八个训练阶段逐个拆
- 第 7 章 推理、转换与部署
- 第 8 章 值得学的设计取舍与三个坑
- 第 9 章 局限与常见误读
- 第 10 章 在 4GB 显卡上能跑到什么程度
- 第 11 章 自测问答
- 附录 A 文件速查 / B 术语 / C 与 Qwen3 的对应关系

---

## 第 1 章 项目定位与版本演进

### 1.1 它反对什么

README 里有一段很直白的话，值得原文理解：

> 第三方框架（transformers / trl / peft）只用十几行就能跑完「加载模型 + 数据 + 推理 + 强化学习」，
> 效率很高，但**把开发者和底层实现隔离开了**。

MiniMind 的选择是：**把框架里的东西自己写一遍**。所以你在项目里会看到：

| 别处用框架 | 这里自己写 | 文件 |
| --- | --- | --- |
| `peft` 的 LoRA | 70 行手写 LoRA + monkey-patch | `model/model_lora.py` |
| `trl` 的 DPOTrainer | 原生 DPO loss + 双模型前向 | `trainer/train_dpo.py` |
| `trl` 的 GRPOTrainer | 原生 GRPO/CISPO + rollout 抽象 | `trainer/train_grpo.py` |
| `transformers` 的 `generate` | 手写采样循环 | `model/model_minimind.py` |
| 第三方 tokenizer | `tokenizers` 库训练 BPE | `trainer/train_tokenizer.py` |

这既是它最大的价值（可读、可改），也是它最大的局限（性能、工程完备度不如框架）。

### 1.2 版本演进：一条「对齐主流生态」的路线

| 版本 | 时间 | 规模 | 关键变化 |
| --- | --- | --- | --- |
| minimind-v1 | 2024-08 | 108M | 首版，自定义 tokenizer |
| minimind2 | 2025-04 | 26M / 104M / 145M MoE | 代码重构，对齐 transformers 命名，支持 llama.cpp/vllm |
| minimind-3 | 2026-04 | 64M / 198M-A64M | **结构、Tokenizer、训练链全面对齐 Qwen3**，移除 shared expert，加入 Agentic RL |

从版本变化能看出一条主线：**模型结构不断向 Qwen3 靠拢**，目的是让权重能直接落到
主流推理生态（llama.cpp、vLLM、ollama）里，而不是困在自己的代码里。

### 1.3 它适合谁 / 不适合谁

| 适合 | 不适合 |
| --- | --- |
| 想搞懂 LLM 每一层怎么实现的工程师 | 想要一个能用的中文助手（64M 的能力有硬上限） |
| 想复现完整训练链的学生 | 想训练 7B 模型的人（这是教学项目） |
| 想改架构做实验的研究者 | 追求训练吞吐的团队（请用 Megatron/DeepSpeed） |

> 口诀：**读它为了「拆黑盒」，用它别指望「干重活」。**

---

## 第 2 章 代码地图与阅读路线

### 2.1 文件清单（按重要度排序）

| 文件 | 行数 | 作用 | 必读 |
| --- | --- | --- | --- |
| `model/model_minimind.py` | 292 | 模型全部结构 + 手写 generate | ⭐⭐⭐ |
| `dataset/lm_dataset.py` | 260 | 四种 Dataset + chat 模板处理 | ⭐⭐⭐ |
| `trainer/train_pretrain.py` | 165 | 预训练主循环 | ⭐⭐⭐ |
| `trainer/trainer_utils.py` | 243 | 学习率/断点/分布式/奖励模型/安全求值 | ⭐⭐⭐ |
| `trainer/train_full_sft.py` | 166 | 全参数 SFT | ⭐⭐⭐ |
| `model/model_lora.py` | 70 | 手写 LoRA（注入/保存/合并） | ⭐⭐ |
| `trainer/train_dpo.py` | 220 | DPO 偏好对齐 | ⭐⭐ |
| `trainer/train_grpo.py` | 321 | GRPO / CISPO | ⭐⭐ |
| `trainer/rollout_engine.py` | 224 | 采样引擎抽象（Torch / SGLang） | ⭐⭐ |
| `trainer/train_ppo.py` | 442 | PPO（actor + critic + GAE） | ⭐ |
| `trainer/train_agent.py` | 484 | 多轮工具调用 RL | ⭐ |
| `trainer/train_distillation.py` | 242 | 白盒蒸馏（KL + CE） | ⭐ |
| `trainer/train_lora.py` | 180 | LoRA 训练入口 | ⭐ |
| `eval_llm.py` | 99 | CLI 推理 | ⭐ |
| `scripts/convert_model.py` | 144 | 原生权重 → HF / Qwen3 兼容格式 | ⭐ |
| `scripts/serve_openai_api.py` | 277 | OpenAI 兼容 API 服务 | 按需 |
| `scripts/web_demo.py` | 425 | Streamlit 聊天界面 | 按需 |
| `trainer/train_tokenizer.py` | 189 | 训练 BPE 分词器（不建议重训） | 参考 |

### 2.2 推荐阅读顺序

```
model_minimind.py      ← 先看模型长什么样
      ↓
lm_dataset.py          ← 数据怎么变成 (input_ids, labels)
      ↓
train_pretrain.py      ← 最简单的训练循环（无 mask、无 ref 模型）
      ↓
train_full_sft.py      ← 多了一个 loss mask（只对 assistant 算）
      ↓
model_lora.py + train_lora.py   ← 只训一小部分参数
      ↓
train_dpo.py           ← 两个模型、偏好对
      ↓
train_grpo.py          ← 采样 + 组内优势（RLHF 主线）
      ↓
train_ppo.py / train_agent.py   ← 最复杂，最后看
```

**阅读技巧**：这些训练脚本有 80% 的代码是一样的（第 5 章的九步骨架），
所以第 2 个脚本以后，只需要盯「损失函数怎么算」那十几行。

### 2.3 项目的目录约定

| 目录 | 内容 | 是否入库 |
| --- | --- | --- |
| `model/` | 模型代码 + 官方 tokenizer 文件（tokenizer.json 等） | ✅ |
| `dataset/` | 数据加载代码（`lm_dataset.py`），数据文件需自行下载 | 代码✅ 数据❌ |
| `trainer/` | 全部训练脚本 | ✅ |
| `scripts/` | 转换、API、WebUI、工具评测 | ✅ |
| `out/` | 训练产出的 `.pth` 权重 | ❌ |
| `checkpoints/` | 断点续训文件 | ❌ |

**注意**：训练脚本里的相对路径都假设你在 `trainer/` 目录下执行（README 明确写了
「所有训练脚本均 `cd ./trainer` 目录执行」），否则 `../out` 会指错地方。

> 口诀：**模型 → 数据 → 预训练 → SFT → 后训练；骨架九成相同，只看损失那几行。**

---

## 第 3 章 模型结构逐层拆解

### 3.1 Config：默认值就是全部设计

`MiniMindConfig`（`model/model_minimind.py:10`）的默认值：

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `hidden_size` | 768 | 主干宽度 |
| `num_hidden_layers` | 8 | 层数（**浅而宽**，见第 8 章） |
| `vocab_size` | 6400 | 自训 BPE 词表 |
| `num_attention_heads` | 8 | Q 头 |
| `num_key_value_heads` | 4 | KV 头 → **GQA，2 个 Q 头共享 1 组 KV** |
| `head_dim` | 96 | = 768 / 8 |
| `intermediate_size` | **2432** | `ceil(768×π/64)×64`，不是常见的 4 倍 |
| `max_position_embeddings` | 32768 | 支持 YaRN 外推 |
| `rope_theta` | 1e6 | 大底数，长文本友好 |
| `tie_word_embeddings` | True | 输入 Embedding 与输出 lm_head 共享 |
| `rms_norm_eps` | 1e-6 | |
| MoE：`num_experts` / `num_experts_per_tok` | 4 / 1 | 4 个专家，每个 token 只走 1 个 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡损失系数 |

**一个容易忽略的细节**：`intermediate_size` 用 `π/64` 取整，是为了让 FFN 宽度
接近 `8/3 × d_model`（SwiGLU 的标准比例，因为有三个矩阵而不是两个），
同时保证能被 64 整除，对齐硬件。

### 3.2 RMSNorm：为什么先转 float

```python
def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

归一化在 **float32** 上算，再转回原精度。这是低精度训练（bf16/fp16）下的常见稳手：
方差计算在 fp16 下容易溢出或精度不足。

### 3.3 RoPE + YaRN：位置编码怎么外推

`precompute_freqs_cis`（`:62`）做了三件事：

1. 按 `rope_theta=1e6` 生成基础频率；
2. 如果开了 `inference_rope_scaling`，用 **YaRN** 的线性斜坡对频率做插值：
   `f' = f × (1 - γ + γ/factor)`，γ 由 `beta_fast/beta_slow` 决定哪些维度被缩放；
3. 预计算 `cos/sin` 并注册为 **非持久 buffer**（不进 state_dict，省权重体积）。

**为什么需要 YaRN**：模型只在 32768 长度内训练过，直接推理到 10 万 token 会崩。
YaRN 用插值 + 注意力温度补偿（`attention_factor`）把频率压缩到训练分布内，
属于「不改权重、只改位置编码」的低成本外推。

### 3.4 Attention：Qwen3 味的三个细节

```python
self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)   # 只有 4 个头
self.q_norm = RMSNorm(head_dim)   # ← 对每个头单独归一化
self.k_norm = RMSNorm(head_dim)
```

三个与现代 LLM 一致的细节：

| 细节 | 作用 |
| --- | --- |
| **GQA**（8 Q / 4 KV） | KV cache 直接减半，推理更省显存 |
| **q_norm / k_norm** | 对 head_dim 做 RMSNorm（Qwen3 同款），大学习率下更稳 |
| **bias=False** | 所有线性层无偏置，与主流一致 |

前向里的关键分支：

```python
if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and 掩码全为1:
    F.scaled_dot_product_attention(...)        # 快路径（PyTorch 自带融合）
else:
    scores = q @ k.T / sqrt(d)                 # 手写路径，支持 KV cache + padding mask
```

**为什么有两条路径**：SDPA 快，但不能同时处理「带 KV cache 的自回归 + 变长 padding」。
所以生成过程中（`past_key_value` 存在）会走手写路径，训练/首轮前向走 SDPA。

> 这里也是本项目唯一一处「为性能妥协」的地方——其余全部是可读的原生实现。

### 3.5 FFN：SwiGLU

```python
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

三个矩阵（gate / up / down），门控机制。参数量是普通 FFN 的 1.5 倍，
所以 `intermediate_size` 用 8/3 而不是 4 倍来平衡。

### 3.6 Block：Pre-Norm 残差

```python
hidden = hidden + self.self_attn(self.input_layernorm(hidden), ...)   # 注意力
hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))     # FFN
```

先归一化再进子层，残差直连。比 Post-Norm 更容易训深，是现代标准做法。

### 3.7 MoE 层：项目里最「有想法」的一段

```python
scores = softmax(gate(x))                       # 每个 token 对 4 个专家的分数
topk_weight, topk_idx = topk(scores, k=1)       # 每个 token 只选 1 个专家
```

三个值得注意的实现细节：

**① k=1 时的直通梯度（straight-through）**

```python
top1 = topk(softmax(gate(x.detach())), 1)[0]
topk_weight = top1 - top1.detach() + 1.0
```

前向值是 1.0（等价于直接用专家输出），反向时梯度却流向 gate——
让路由器在 k=1 时也能被训练。否则「选谁」是不可导的离散操作。

**② 负载均衡辅助损失**

```python
load = one_hot(topk_idx, num_experts).mean(0)   # 各专家实际接了多少 token
aux_loss = (load * scores.mean(0)).sum() * num_experts * coef
```

惩罚「所有 token 都挤向同一个专家」，否则专家会退化成一个，MoE 白搭。

**③ DDP 下的空专家技巧**

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

某个专家这一批没接到 token 时，加一个「值为 0 但依赖参数」的项，
避免 DDP 因为「有参数没参与计算」而卡死在同步上。这是分布式训练里的实战经验。

**MoE 的代价**：README 自己也说了，原生实现下 token 要按专家分桶再分别 forward，
kernel 启停开销大，4 专家 top-1 大约比同尺寸 Dense 慢 50%。要快得靠
DeepSpeed-MoE / Megatron / Triton 融合算子——但那会牺牲「可读性」。

### 3.8 ForCausalLM：三个工程点

```python
if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight   # 共享同一块内存
```

| 工程点 | 代码 | 为什么 |
| --- | --- | --- |
| 权重共享 | `embed_tokens.weight = lm_head.weight` | 6400×768 ≈ 4.9M 参数，省掉就是省 7.7% 总参数 |
| 损失错位 | `logits[..., :-1, :]` vs `labels[..., 1:]` | 标准 next-token 预测 |
| `logits_to_keep` | 只算最后 N 个位置的 logits | 生成时省大量显存与计算（RL 里每步都生成，收益明显） |

### 3.9 手写的 generate：一个完整的采样器

`generate`（`:261`）没有用 `GenerationMixin` 的默认实现，而是自己写了循环，包含：

| 功能 | 实现要点 |
| --- | --- |
| KV cache | 每步只喂 `input_ids[:, past_len:]`，复用 `past_key_values` |
| temperature | `logits / temperature` |
| top-k | 把低于第 k 名的分数置 `-inf` |
| top-p（核采样） | 排序 → 累积概率超过 p 的置 `-inf`（用 `scatter` 还原到原顺序） |
| repetition penalty | 对已出现 token：正分数除以系数，负分数乘系数 |
| 提前结束 | 记录 `finished`，全部到 EOS 就 break |
| 流式 | `streamer.put(...)` 逐 token 回调 |

**为什么值得读**：这是「一次自回归生成到底发生了什么」最干净的参考实现，
比读 transformers 的 `generate`（几千行、几十个可选参数）容易得多。

### 3.10 实测：模型在你机器上的样子

我在 `swift_env` 里实例化并前向了一次（随机权重）：

```
[Dense] params=63.91M  vocab=6400 layers=8 dim=768 head_dim=96 kv_heads=4 ffn=2432
  forward: input (2, 32) -> logits (2, 32, 6400), aux_loss=0.0000
  loss(labels=ids, 随机权重): 8.873  (ln(vocab)=8.764)

[MoE(4 experts/top-1)] params=198.42M
  forward: input (2, 32) -> logits (2, 32, 6400)
  loss(labels=ids, 随机权重): 8.977

GPU(fp16) 生成 32 tokens: 2297 ms (13.9 tok/s), 峰值 144 MB
```

**怎么读这组数**：

1. **63.91M** 与 README 的 64M 对得上；MoE 总参 198M，但每 token 只激活约 64M。
2. **随机权重下 loss ≈ ln(6400) = 8.76**，这是「均匀猜」的基准线。实测 8.87 略高，
   符合未训练模型的预期——**这个数字也是判断训练是否真的开始了的第一标尺**。
3. GPU 生成 13.9 tok/s、144MB 峰值：64M 模型在你的 GTX 1650 上完全跑得动。

### 3.11 与 Qwen3 的结构对应

| 组件 | MiniMind-3 | Qwen3（8B 级） | 是否一致 |
| --- | --- | --- | --- |
| 归一化 | RMSNorm（pre-norm） | 同 | ✅ |
| 位置编码 | RoPE + YaRN | 同 | ✅ |
| QK-Norm | 有（head_dim 上） | 有 | ✅ |
| 注意力 | GQA | GQA | ✅ |
| FFN | SwiGLU | SwiGLU | ✅ |
| 偏置 | 全无 | 全无 | ✅ |
| MoE | 4 experts / top-1，无 shared | 128 experts / top-8，无 shared | 结构同型、规模不同 |

正因如此，`scripts/convert_model.py` 里才能直接把权重映射成
`Qwen3Config / Qwen3ForCausalLM` 的格式。

> 口诀：**RMSNorm 打底，RoPE 定位置；QK 归一化，GQA 省缓存；SwiGLU 三矩阵，MoE 四选一。**

---

## 第 4 章 数据管线：一条 jsonl 走完全流程

### 4.1 四个 Dataset 一览

| Dataset | 输入格式 | 返回 | 用在哪 |
| --- | --- | --- | --- |
| `PretrainDataset` | `{"text": "..."}` | `(input_ids, labels)` | 预训练 |
| `SFTDataset` | `{"conversations": [...]}` | `(input_ids, labels)` | 全参 SFT、LoRA、蒸馏 |
| `DPODataset` | `{"chosen": [...], "rejected": [...]}` | 6 个张量（chosen/rejected 各 3） | DPO |
| `RLAIFDataset` | `{"conversations": [...]}` | `{"prompt": str}`（只要问题） | PPO / GRPO |
| `AgentRLDataset` | `{"conversations": [...], "gt": [...]}` | `{"messages", "tools", "gt"}` | Agentic RL |

### 4.2 预训练：最简单的标签

```python
tokens = [BOS] + tokenize(text) + [EOS]
input_ids = tokens + [PAD] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == PAD] = -100        # ← 关键：padding 不参与损失
```

**为什么 `-100`**：这是 PyTorch `cross_entropy` 的默认 `ignore_index`。
预训练时模型要学「整段文本的下一个词」，但补齐的 pad 位置没有意义，
必须排除，否则模型会学着「预测 padding」。

### 4.3 SFT：只对 assistant 的回答算损失

这是全项目最值得逐行读的数据处理。`SFTDataset.generate_labels`（`dataset/lm_dataset.py:91`）：

```python
bos_id = tokenize("<|im_start|>assistant\n")     # 回答开始
eos_id = tokenize("<|im_end|>\n")                # 回答结束

i = 0
while i < len(input_ids):
    if input_ids[i:i+len(bos_id)] == bos_id:     # 找到 assistant 回合
        ... 把从 start 到 eos 之间的位置都标成 input_ids[j]
    else:
        i += 1
```

结果：labels 里**只有 assistant 说的内容是真值，其它（system/user/模板标记）全是 -100**。

**为什么必须这样做**：如果对整段都算损失，模型会同时学习「如何生成用户的问题」，
既浪费容量，又会让模型在推理时倾向于自问自答。

| 做法 | 效果 |
| --- | --- |
| 全序列算 loss | 模型学「复述用户提问」，对话能力差 |
| 只算 assistant 片段 | 模型专注学「如何回答」✅ |

**实现上的取舍**：这里用「token 序列扫描」而不是「按 role 逐条拼接」，
好处是自动适配 chat template 的任何格式；代价是依赖模板里 assistant 标记的唯一性。

### 4.4 两个对话增强技巧

**① 概率性注入 system prompt**（`pre_processing_chat`）

```python
if conversations[0]['role'] != 'system' and random.random() < 0.2:
    插入一条随机 system prompt（10 条候选之一）
```

让模型既见过「有 system」也见过「没有 system」的输入，推理时加不加系统提示都能工作。

**② 概率性移除空思考块**（`post_processing_chat`）

```python
prompt = prompt.replace('<think>\n\n</think>\n\n', '')   # 80% 概率移除
```

这是为 minimind-3 的「自适应思考」服务：模型要学会**有时思考、有时不思考**。
如果训练数据里每条回答都带空 `<think></think>`，模型会永远输出空思考标签。

### 4.5 DPO 数据：一条样本拆成两路

```python
x_chosen, y_chosen, mask_chosen       = ids[:-1], ids[1:], mask[1:]
x_rejected, y_rejected, mask_rejected = ids[:-1], ids[1:], mask[1:]
```

**注意 `[:-1]` / `[1:]`**：模型内部已经做了错位（`logits[:-1]` 对 `labels[1:]`），
这里数据集再错一次，是为了让 `logits_to_log_probs(logits, y)` 能直接按位置 gather。
两者配合，最终对齐是「输入 t 位置 → 预测 t+1」。

mask 同样只覆盖 assistant 片段，DPO 只比较「两种回答在回答部分」的概率。

### 4.6 RL 数据：只要 prompt

```python
class RLAIFDataset:
    def create_chat_prompt(self, conversations):
        use_thinking = random.random() < self.thinking_ratio   # 50% 开启思考
        return tokenizer.apply_chat_template(conversations[:-1], ...,
                                             open_thinking=use_thinking,
                                             add_generation_prompt=True)
```

- 丢掉最后一条 assistant 回答（`[:-1]`），只留问题；
- 随机决定这次要不要「开启思考」——让模型在 RL 中同时被评估两种模式。

> 口诀：**预训练整段算，SFT 只算回答；DPO 拆两路，RL 只喂问题。**

---

## 第 5 章 训练脚本的统一骨架

打开任意一个 `train_*.py`，结构几乎一模一样，都是九步：

```python
# 1. 初始化分布式与随机种子
local_rank = init_distributed_mode()
setup_seed(args.seed + rank)

# 2. 模型配置 + 断点探测
lm_config = MiniMindConfig(hidden_size=..., num_hidden_layers=..., use_moe=...)
ckp_data = lm_checkpoint(...) if args.from_resume == 1 else None

# 3. 混合精度
autocast_ctx = torch.cuda.amp.autocast(dtype=bfloat16/fp16)

# 4. 日志（swanlab，兼容 wandb API）
wandb = None
if args.use_wandb: import swanlab as wandb

# 5. 模型 / 数据 / 优化器
model, tokenizer = init_model(lm_config, args.from_weight)
train_ds = XxxDataset(...)
optimizer = optim.AdamW(model.parameters(), lr=...)

# 6. 恢复训练状态
model.load_state_dict(ckp_data['model']); optimizer.load_state_dict(...)

# 7. torch.compile（可选）+ DDP 包装
model = DistributedDataParallel(model, device_ids=[local_rank])

# 8. 训练循环（每个 epoch 一个 DataLoader）
for epoch in range(start_epoch, args.epochs):
    train_epoch(epoch, loader, ...)

# 9. 清理分布式
dist.destroy_process_group()
```

**记忆方法**：**种子 → 配置 → 精度 → 日志 → 模型数据 → 断点 → 包装 → 循环 → 收尾**。

### 5.1 学习率：余弦 + 10% 地板

```python
def get_lr(current_step, total_steps, lr, warmup_ratio=0.0):
    if current_step < warmup: return lr * (current_step + 1) / warmup
    angle = math.pi * (current_step - warmup) / max(1, total_steps - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(angle)))
```

`0.1 + 0.45×(1+cos)` 在 0.1 和 1.0 之间余弦下降——**不是降到 0，而是保留 10%**。
大模型训练常用这种「有地板」的余弦，避免最后阶段学习率太小、梯度噪声主导。

### 5.2 断点续训：两个文件、一个换算

```python
ckp      = out/xxx_768.pth                  # 只有半精度权重，用于推理
resume   = checkpoints/xxx_768_resume.pth   # 权重 + 优化器 + scaler + epoch/step + world_size
```

**最巧妙的一点**：

```python
if saved_ws != current_ws:
    ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
```

换卡数续训时自动换算 step（4 卡跑 100 步 ≈ 8 卡跑 50 步），
再配合 `SkipBatchSampler` 跳过已训批次。这是很多教学项目没有的工程细节。

### 5.3 三个平台适配点

| 代码 | 解决的问题 |
| --- | --- |
| `num_workers = 0 if os.name == 'nt' else 8` | Windows 多进程 DataLoader 容易出问题 |
| `import datasets  # noqa` 放在 torch 之前 | 规避 Windows 下 pyarrow 与 torch 的 DLL 冲突 |
| `resolve_dtype()` 在 bf16 不支持时回退 fp16 | 老卡（如你的 1650）自动降级 |

> 口诀：**种子配置精度日志，模型数据断点包装；余弦留底一成，换卡自动折步。**

---

## 第 6 章 八个训练阶段逐个拆

### 6.1 阶段总览

| 阶段 | 脚本 | 需要几个模型 | 损失 | 数据 | 关键超参 |
| --- | --- | --- | --- | --- | --- |
| 预训练 | `train_pretrain.py` | 1 | 交叉熵（全序列） | pretrain.jsonl | lr 5e-4, seq 340 |
| SFT | `train_full_sft.py` | 1 | 交叉熵（仅回答） | sft.jsonl | lr 1e-5, seq 768 |
| LoRA | `train_lora.py` | 1（只训适配器） | 同 SFT | sft/lora.jsonl | r=16, lr 1e-4 |
| 蒸馏 | `train_distillation.py` | 2（教师+学生） | α·CE + (1-α)·KL | sft.jsonl | α, T |
| DPO | `train_dpo.py` | 2（策略+参考） | -logσ(β·Δ) | dpo.jsonl | β=0.15, lr 4e-8 |
| PPO | `train_ppo.py` | 4（actor/critic/ref/RM） | clip + value + KL | rlaif.jsonl | γ=1, λ=0.95, KL 0.02 |
| GRPO | `train_grpo.py` | 3（policy/ref/RM） | clip 或 CISPO + KL | rlaif.jsonl | 每组 6 条, β=0.1 |
| Agentic RL | `train_agent.py` | 3 | 规则奖励 + GT 校验 | agent_rl.jsonl | 多轮, max_turns=3 |

**一句话记住区别**：预训练学「说话」，SFT 学「听指令」，LoRA 学「低成本加技能」，
蒸馏学「模仿」，DPO 学「偏好」，PPO/GRPO 学「试错」，Agentic RL 学「用工具」。

### 6.2 预训练：学词语接龙

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss        # MoE 才有 aux_loss
```

数据是 340 token 的短片段，batch 32，梯度累积 8，学习率 5e-4，训 2 轮。
目标就是让 loss 从 ln(6400) ≈ 8.76 降下来。

### 6.3 SFT：学「怎么回答」

差别只有两点：

1. Dataset 换成 `SFTDataset`（只对回答算 loss）；
2. 学习率从 5e-4 降到 **1e-5**，序列长度从 340 升到 768。

**为什么 SFT 学习率要小两个数量级**：预训练是「从随机到有知识」，可以大步走；
SFT 是「在已有能力上微调行为」，步子大了会破坏预训练学到的知识（灾难性遗忘）。

### 6.4 LoRA：70 行手写实现

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(0, 0.02)   # A 高斯初始化
        self.B.weight.data.zero_()            # B 全零
```

**为什么 B 初始化为 0**：训练开始时 `B·A = 0`，LoRA 对原模型**零扰动**，
模型从「和底座完全一样」的状态开始学。如果 A、B 都随机初始化，
一开始就注入了噪声，微调会先「修复噪声」再学新知识。

**注入方式**（`apply_lora`）：

```python
if isinstance(module, nn.Linear) and module.in_features == module.out_features:
    setattr(module, "lora", lora)
    module.forward = lambda x: original_forward(x) + lora(x)
```

只给**方阵**线性层加 LoRA——在这套结构里就是每层的 `q_proj` 和 `o_proj`（768→768）。
`k_proj`、`v_proj` 是 768→384（GQA），`lm_head` 是 768→6400，都不是方阵。

**训练的四个动作**：

```python
apply_lora(model)                                  # 1. 注入
for p in model.parameters(): p.requires_grad = 'lora' in name
optimizer = AdamW(lora_params)                     # 2. 只优化 LoRA 参数
save_lora(model, path)                             # 3. 只存适配器
merge_lora(model, lora_path, save_path)            # 4. 合并回完整权重
```

`merge_lora` 的核心三行：

```python
delta = (lora.B.weight @ lora.A.weight) * lora.scaling
state_dict[name].weight += delta
```

推理时零额外开销——适配器被「烤」进原权重。

### 6.5 蒸馏：学生模仿老师的完整分布

```python
distill_loss = KL(softmax(student/T), softmax(teacher/T)) * T²
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

| 概念 | 作用 |
| --- | --- |
| 硬标签（CE） | 学「标准答案」 |
| 软标签（KL, T>1） | 学老师的「可能性分布」（哪些错答案更像对的） |
| `T²` | 温度抬高后梯度会变小，乘 T² 补偿 |
| `alpha` | 两者权重 |

**代码细节**：教师 logits 截断到学生词表大小（`teacher_logits[..., :vocab]`），
处理「老师词表比学生大」的情况；loss 只在 label ≠ -100 的位置计算。

### 6.6 DPO：不用奖励模型的偏好对齐

核心 5 行（`train_dpo.py:34`）：

```python
pi_logratios  = logp_pi_chosen   - logp_pi_rejected
ref_logratios = logp_ref_chosen  - logp_ref_rejected
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits).mean()
```

**逐项解释**：

| 项 | 含义 |
| --- | --- |
| `pi_logratios` | 策略模型觉得「好回答」比「坏回答」好多少 |
| `ref_logratios` | 参考模型原本觉得好多少（基线） |
| 差值 | 策略**相对参考模型**的改进量 |
| `β` | 允许偏离参考模型的程度（越小越保守） |
| `-logσ` | 越大越好的量取反当损失 |

**参考模型从哪来**：同一个权重再加载一份，`eval()` + `requires_grad_(False)`。它提供
「原始偏好」的基线，防止策略为了讨好数据把语言能力训崩。

**两个超参值得注意**：

- `beta` 默认 0.15（CLI），函数默认 0.1；
- **学习率 4e-8**——比 SFT 又小了两个数量级。DPO 极易训崩，README 也注明「建议 ≤5e-8 避免遗忘」。

### 6.7 PPO：四个模型一起转

PPO 是项目里最复杂的脚本（442 行），需要四份模型：

| 模型 | 作用 |
| --- | --- |
| actor（策略） | 生成回答，被训练 |
| critic（价值） | 估计「当前状态值多少」，用 `value_head` 输出标量 |
| ref（参考） | 算 KL 惩罚 |
| reward model | 给回答打分（外部模型，`LMForRewardModel`） |

流程：

```
采样 → 计算奖励 → GAE 算优势 → 归一化 → 多个 mini-batch 更新 → 早停
```

**GAE（广义优势估计）** 的核心三行：

```python
delta = token_rewards[:, t] + gamma * v[t+1] - v[t]
lastgaelam = delta + gamma * lam * lastgaelam
```

从序列末尾倒推，把「即时奖励 + 未来价值」平滑成每个 token 的优势。

**四个防崩设计**（值得抄的工程细节）：

| 设计 | 代码 | 作用 |
| --- | --- | --- |
| 优势归一化 | `(adv - mean) / std` | 稳定梯度尺度 |
| 策略裁剪 | `clamp(ratio, 1±ε)` | 防止单步更新过大 |
| 价值裁剪 | `clamp(value, old±range)` | 同上，作用在 critic |
| KL 早停 | `approx_kl > 0.25` 停止本轮更新 | 偏离太远就收手 |

**多卡上的一个坑**（代码里有注释）：

```python
# 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
dist.all_reduce(approx_kl_val, op=ReduceOp.AVG)
```

早停必须全卡同步——否则一张卡停了、其它卡还在 forward，DDP 会永久挂起。

### 6.8 GRPO：去掉 critic 的 PPO

GRPO 的核心洞察：**不需要价值模型，用「同一问题的多条回答」的相对好坏当基线**。

```python
grouped = rewards.view(B, num_generations)        # 每个 prompt 生成 6 条
advantages = (rewards - mean) / (std + 1e-4)      # 组内标准化
```

| 对比 | PPO | GRPO |
| --- | --- | --- |
| 基线来源 | critic 模型估计 | 同组其它回答的平均分 |
| 模型数量 | 4 | 3（省掉 critic） |
| 显存 | 高 | 低 |
| 适合 | 通用 | 可验证奖励（数学/代码） |

**两种损失实现**（`args.loss_type`）：

```python
# 标准 GRPO：带裁剪的 PPO 目标
per_token_loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + β * KL

# CISPO：只裁剪分母，保留所有 token 的梯度
clamped_ratio = clamp(ratio, max=ε_high).detach()
per_token_loss = -(clamped_ratio * A * logp - β * KL)
```

**KL 估计**用的是 k3 估计量：`exp(kl) - kl - 1`，比直接减 log 比值更稳定，
且始终非负。

### 6.9 rollout 抽象：同一套算法，两种后端

`rollout_engine.py` 定义了 `RolloutEngine` 抽象类，两个实现：

| 引擎 | 实现 | 用途 |
| --- | --- | --- |
| `TorchRolloutEngine` | 直接调 `model.generate` | 默认，简单 |
| `SGLangRolloutEngine` | HTTP 请求 SGLang 服务 | 高吞吐，推理与训练分离 |

**为什么值得学**：RL 训练里「采样」往往占 80% 的时间。把采样抽成接口，
就能在不改算法的情况下把生成换成高性能引擎（vLLM/SGLang）。这也是
`verl`、`OpenRLHF` 这类工业框架的标准做法。

### 6.10 Agentic RL：多轮工具调用

`train_agent.py` 在 GRPO 基础上加了「环境交互」：

```
模型输出 <tool_call> → 解析 → 执行工具 → 把结果作为 <tool_response> 塞回对话 → 继续生成
```

奖励是**规则化的**（`calculate_rewards`）：

| 规则 | 分数 |
| --- | --- |
| 标签不成对（`<tool_call>` 与 `</tool_call>` 数量不等） | -0.5 × 差值 |
| 工具名不在允许列表 / 参数非法 | 每个 -0.5 |
| 调用次数与 GT 不一致 | -0.5 × 差值 |
| 最终文本包含 GT 答案 | +2.5 × 命中比例 |
| 未完成（超过 max_turns） | -0.5 |
| 重复 n-gram | 按比例扣分 |
| 总分 | 裁剪到 [-3, 3] |

**这就是「RLVR」的典型形态**：奖励不是人类打分，而是**可自动验证的信号**
（工具能不能执行、答案对不对）。

> 口诀：**预训练算全部，SFT 只算回答；LoRA 零扰动起步，DPO 只动偏好；PPO 四模型，GRPO 组内比；Agent 靠规则，奖励自动给。**

---

## 第 7 章 推理、转换与部署

### 7.1 两种加载方式

`eval_llm.py` 支持两条路径：

```python
# 方式 1：原生 PyTorch 权重（./out/xxx_768.pth）
model = MiniMindForCausalLM(MiniMindConfig(...))
model.load_state_dict(torch.load(ckp))

# 方式 2：transformers 格式目录（./minimind-3）
model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
```

判断逻辑很实用：`if os.path.basename(path) == 'model'` 就用原生权重，
否则当成 HF 目录。**这也意味着项目自带的 `model/` 目录名不能改**。

### 7.2 对话推理的三个关键调用

```python
prompt = tokenizer.apply_chat_template(conversation, tokenize=False,
                                       add_generation_prompt=True,
                                       open_thinking=bool(args.open_thinking))
generated = model.generate(..., streamer=streamer, temperature=0.85, top_p=0.95)
```

| 参数 | 作用 |
| --- | --- |
| `open_thinking` | 是否让模型先输出 `<think>...</think>` 再回答 |
| `streamer` | 逐 token 打印（TextStreamer） |
| `repetition_penalty=1` | 这个小模型不需要惩罚，开了反而降低表达 |

### 7.3 权重转换：两个方向

`scripts/convert_model.py` 提供两种转换：

| 函数 | 目标 | 用途 |
| --- | --- | --- |
| `convert_torch2transformers_minimind` | MiniMind 自己的 AutoModel 格式 | 可直接 `from_pretrained` |
| `convert_torch2transformers` | **Qwen3 / Qwen3-MoE 兼容配置** | 直接吃 vLLM / llama.cpp 生态 |

后者是「结构对齐 Qwen3」的兑现：把 `MiniMindConfig` 的字段映射成
`Qwen3Config`，再把 state_dict 的键名对齐，就能让 Qwen3 的实现加载 MiniMind 的权重。

同文件里还有 `merge_lora` 的调用，训练完 LoRA 后一步导出完整模型。

### 7.4 服务化

| 脚本 | 能力 |
| --- | --- |
| `serve_openai_api.py` | FastAPI，`/v1/chat/completions`，SSE 流式，返回 `reasoning_content` 与 `tool_calls` |
| `web_demo.py` | Streamlit 界面，支持思考展示、工具选择、多轮 Tool Call |
| `scripts/chat_api.py` | 调用任意 OpenAI 兼容服务的极简客户端 |

服务端做了一件很多教学项目不做的事：**把 `<think>` 和 `<tool_call>` 从文本里解析出来，
按 OpenAI 协议字段返回**，这样 Open-WebUI、FastGPT 这类客户端可以直接接。

### 7.5 第三方生态

| 引擎 | 前提 |
| --- | --- |
| llama.cpp | 转成 GGUF（项目提供转换说明） |
| vLLM | 用 transformers 格式目录直接 `--model` |
| ollama | 基于 GGUF |

> 口诀：**原生权重调试，HF 格式部署；think 与 tool_call 解析成字段，才接得进生态。**

---

## 第 8 章 值得学的设计取舍与四个坑

### 8.1 为什么是「8 层 × 768 宽」，而不是「4 层 × 1024」

README 里引用了 MobileLLM 的结论：**在参数量固定时，深度比宽度更重要**。
作者自己的实验补充了两个边界：

| 观察 | 结论 |
| --- | --- |
| 深而窄 > 宽而浅（同参数量） | 优先加深 |
| `d_model < 512` 时 | 词嵌入太窄，加深也补不回来 |
| `d_model > 1536` 时 | 再加宽不如继续加层 |

所以 64M 的配置选 `8 × 768`：能加深（8 层不算浅），又不至于窄到 512 以下。

### 8.2 词表 6400 的取舍

| 词表大小 | 优点 | 缺点 |
| --- | --- | --- |
| 6400（本项目） | Embedding 小（4.9M，占 7.7%），训练快 | 中文压缩率低，序列更长 |
| 15 万（Qwen） | 压缩率高，token 少 | Embedding 巨大，小模型负担重 |

对小模型来说，**省下来的 Embedding 参数可以拿去加深网络**，这个取舍是合理的。

### 8.3 四个真实的坑

| 坑 | 现象 | 源码里的处理 |
| --- | --- | --- |
| Windows 多进程 DataLoader | 崩溃/卡死 | `num_workers=0 if os.name=='nt'` |
| pyarrow 与 torch 的 DLL 冲突 | import 时报错 | `import datasets` 放在 `torch` 之前（文件顶部注释写明 issue #771） |
| LoRA 的 monkey-patch 与 torch.compile 不兼容 | 编译报错 | `train_lora.py` 里自动关闭 compile 并打印提示 |
| 换卡数续训 step 对不上 | 重复训练/跳过数据 | 按 `saved_ws/current_ws` 换算 step |

### 8.4 文档与代码的一处不一致

README 写「支持 DDP、DeepSpeed」，但仓库里**只有 DDP 的实现**：
训练脚本用 `DistributedDataParallel` + `torchrun`，没有任何 `deepspeed` 配置或启动脚本
（全仓库搜索 `deepspeed` 只在 README 出现）。

**读法建议**：把 README 的功能列表当成「作者验证过可行」而不是「开箱即用」。
用 DeepSpeed 需要自己补 `ds_config` 和启动方式——正好可以接上你之前那本 DeepSpeed 教材。

> 口诀：**深度优先于宽度，词表要小而够用；文档当方向，代码当事实。**

---

## 第 9 章 局限与常见误读

### 9.1 能力边界：64M 能做什么

| 能做 | 不能做 |
| --- | --- |
| 流畅的中文对话（短问题） | 可靠的事实知识 |
| 简单指令、格式遵循 | 复杂推理、多步数学 |
| 基本的工具调用格式 | 长文档理解 |
| 情感/风格模仿 | 专业领域准确性 |

README 自己也贴了一段 early Zero 模型的对话，里面英文回答直接崩坏——
这不是 bug，是 64M 参数 + 小数据的必然结果。

### 9.2 四个常见误读

| 误读 | 事实 |
| --- | --- |
| 「3 块钱训出 GPT」 | 3 块钱训出的是「会聊天的小模型」，不是 GPT |
| 「DPO 能提升智力」 | DPO 改变的是偏好（更符合人类喜好），不提升事实能力 |
| 「MoE 推理更快」 | 原生的 MoE 实现训练更慢（分桶 + kernel 启停），推理也需要融合算子才快 |
| 「代码里支持多卡就够了」 | 只做了 DDP；真正的显存优化（ZeRO/FSDP）要自己接 |

### 9.3 工程上的简化

| 简化 | 后果 |
| --- | --- |
| 没有评测流水线内置（有 C-Eval 等脚本但需自备数据） | 不好做严格的模型对比 |
| 没有 tensor/pipeline parallel | 上限就是单卡能放下的规模 |
| 数据清洗脚本不在仓库（只给成品 jsonl） | 复现「数据质量」这一环有缺口 |
| 奖励模型是外部模型，不在训练链内 | RM 质量决定 RL 上限 |

> 口诀：**能力看参数量，对齐看后训练；三块钱买的是流程，不是智能。**

---

## 第 10 章 在 4GB 显卡上能跑到什么程度

这一节专门回答「你的 GTX 1650 / 4GB 能不能玩」。以下数字来自本机实测与按比例的估算。

### 10.1 模型本身：完全放得下

| 项目 | 64M Dense | 198M MoE |
| --- | --- | --- |
| fp16 权重 | 122 MB | 约 380 MB |
| 全量训练（AdamW，16B/参） | 0.95 GB | 约 2.95 GB |
| 加上激活值（batch 4~8） | 约 2~3 GB | **超过 4 GB** |
| 结论 | ✅ 能训 | ❌ 不建议 |

**实测参考**：fp16 推理峰值只有 144 MB，生成速度 13.9 tok/s（32 个 token 用了 2.3 秒）。
推理完全没问题，训练要小心 batch。

### 10.2 训练时间估算

| 阶段 | 3090（README 实测） | 你的 1650（估算） |
| --- | --- | --- |
| pretrain_t2t_mini | 1.21 h | 约 10 ~ 14 h |
| sft_t2t_mini | 1.10 h | 约 9 ~ 13 h |

估算依据：3090 的 fp32 算力约为 1650 的 12 倍，显存带宽约 7 倍，
实际训练差距通常取中间值。**「挂着跑一夜」是可行的**，但别指望一小时出结果。

### 10.3 两个需要注意的点

**① 你的卡会被自动降成 fp16**

```python
torch.cuda.is_bf16_supported(including_emulation=False)   # 1650 → False
```

`resolve_dtype` 会打印「不原生支持 bfloat16，已回退为 float16」。我们在 PyTorch 教材里
实测过：这块卡上 fp16 autocast 比 fp32 慢约 5 倍——所以上面的时间估算**可能还要更悲观**。

**② 脚本没有 fp32 训练路径**

```python
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
```

传 `--dtype float32` 也会被当成 fp16。想强制 fp32 需要自己改这一行。

### 10.4 建议的上手路径

```bash
# 1) 先把仓库拷到自己的工作目录，避免污染
# 2) 安装依赖（建议在你的 conda 环境里单独装，requirements 里钉死了版本）
cd minimind-master
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 先只做推理：下载 64M 的 minimind-3 权重（约 130 MB）
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3

# 4) 想自己训：下载 mini 数据集，进 trainer 目录
cd trainer
python train_pretrain.py --batch_size 4 --accumulation_steps 16 --max_seq_len 256
```

**顺序建议**：先推理（10 分钟）→ 再跑 10 个 step 看显存 → 最后才放长跑。
不要一上来就 `--epochs 2` 跑通宵，先确认显存峰值与 loss 曲线。

> 口诀：**推理十分钟，训练一整夜；先看显存峰值，再决定跑多久。**

---

## 第 11 章 自测问答

合上源码，能答上来再往下走。

### 一、模型结构

1. MiniMind-3 的 GQA 是几比几？影响什么？
2. 为什么 `intermediate_size` 是 2432 而不是 3072？
3. `q_norm` / `k_norm` 加在哪里？为什么加？
4. MoE 在 k=1 时怎么让路由器拿到梯度？
5. `tie_word_embeddings=True` 省了多少参数？

### 二、数据

6. 预训练和 SFT 的 labels 最大区别是什么？
7. SFT 怎么定位「哪些 token 该算损失」？
8. DPO 数据为什么要把 ids 左移一位？
9. 为什么要概率性移除空的 `<think></think>`？
10. 为什么 RL 数据只保留 prompt？

### 三、训练

11. `get_lr` 为什么保留 10% 地板而不是降到 0？
12. 换卡数续训时 step 怎么处理？
13. LoRA 的 B 矩阵为什么初始化为 0？
14. LoRA 在这套结构里加在哪些层？为什么只有这些层？
15. DPO 的参考模型从哪来？为什么需要它？
16. PPO 的四个模型分别是什么？
17. GRPO 相比 PPO 省掉了什么？用什么替代？
18. GRPO 的 KL 估计量为什么用 `exp(kl)-kl-1`？
19. rollout 引擎抽象的意义是什么？
20. Agentic RL 的奖励为什么可以不要奖励模型？

### 参考答案要点

1. 8 Q / 4 KV = 2:1；KV cache 减半，推理省显存。
2. SwiGLU 有三个矩阵，`8/3×d` 等效普通 FFN 的 4 倍，2432 ≈ 8/3×768 且能被 64 整除。
3. 在 head_dim 上做 RMSNorm；大学习率下稳定注意力分数。
4. 直通估计：`top1 - top1.detach() + 1.0`，前向为 1，反向有梯度。
5. Embedding 与 lm_head 共享，6400×768 ≈ 4.9M，占总参数约 7.7%。
6. 预训练对整段文本算损失；SFT 只对 assistant 回答算，其余位置 -100。
7. 用 `<|im_start|>assistant\n` 与 `<|im_end|>\n` 的 token 序列扫描。
8. 模型内部已经是「位置 t 预测 t+1」，数据再移位一次配合 gather 取 logp。
9. 让模型学会「有时思考、有时不思考」，避免永远输出空思考标签。
10. RL 要模型自己生成回答再评分，给答案就变成 SFT 了。
11. 避免末期学习率过小、被梯度噪声主导；保留探索能力。
12. 按 `step × saved_world / current_world` 换算，再用 SkipBatchSampler 跳过。
13. 保证初始时 `B·A=0`，对原模型零扰动。
14. 只加方阵（`q_proj`、`o_proj`），因为 `in_features == out_features`。
15. 同权重再加载一份并冻结；提供偏好基线，约束策略不偏离语言能力。
16. actor、critic（value head）、ref、reward model。
17. 省掉 critic；用同一 prompt 的 N 条回答的组内均值/方差做优势基线。
18. k3 估计量无偏且非负，数值更稳。
19. 采样占 RL 大部分时间，抽象后可以换 SGLang/vLLM 而不改算法。
20. 奖励来自可验证信号（工具执行结果、GT 匹配、格式规则），不需要人类/RM 打分。

---

## 附录 A 训练超参速查

| 阶段 | lr | epochs | batch | 累积 | seq len | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain | 5e-4 | 2 | 32 | 8 | 340 | grad_clip 1.0 |
| full_sft | 1e-5 | 2 | 16 | 1 | 768 | |
| lora | 1e-4 | 10 | 32 | 1 | 340 | r=16, 只训适配器 |
| distillation | 5e-5 量级 | — | — | — | 768 | α、T 需调 |
| dpo | 4e-8 | 1 | 4 | 1 | 1024 | β=0.15 |
| ppo | — | — | — | — | 1024 | γ=1.0, λ=0.95, clip 0.2, kl_coef 0.02 |
| grpo | — | — | — | — | — | 每组 6 条, β=0.1, ε=0.2 |

> 表内 lr 取自各脚本 argparse 默认值；实际跑之前用 `--help` 确认当前版本。

## 附录 B 术语中英对照

| 中文 | 英文 | 在本项目里的位置 |
| --- | --- | --- |
| 预训练 | pretraining | `train_pretrain.py` |
| 监督微调 | SFT | `train_full_sft.py` |
| 低秩适配 | LoRA | `model_lora.py` |
| 知识蒸馏 | knowledge distillation | `train_distillation.py` |
| 直接偏好优化 | DPO | `train_dpo.py` |
| 近端策略优化 | PPO | `train_ppo.py` |
| 组相对策略优化 | GRPO | `train_grpo.py` |
| 可验证奖励 | RLVR | `train_agent.py` 的规则奖励 |
| 分组查询注意力 | GQA | `Attention` 的 kv_heads |
| 旋转位置编码 | RoPE / YaRN | `precompute_freqs_cis` |
| 混合专家 | MoE | `MOEFeedForward` |
| 负载均衡损失 | aux loss | `router_aux_loss_coef` |
| 广义优势估计 | GAE | `train_ppo.py` |
| 采样引擎 | rollout engine | `rollout_engine.py` |
| 断点续训 | resume | `lm_checkpoint` |

## 附录 C 一句话记住每个文件

| 文件 | 一句话 |
| --- | --- |
| `model_minimind.py` | 模型 + 手写 generate，全部结构都在这 |
| `model_lora.py` | 70 行注入/保存/合并 LoRA |
| `lm_dataset.py` | 四种数据格式 + 只算回答的 mask |
| `train_pretrain.py` | 最小训练循环 |
| `train_full_sft.py` | 多一个 loss mask |
| `train_lora.py` | 冻结主干只训适配器 |
| `train_distillation.py` | CE + KL 双损失 |
| `train_dpo.py` | 两模型、四行损失 |
| `train_ppo.py` | 四模型 + GAE |
| `train_grpo.py` | 组内比较 + 无 critic |
| `train_agent.py` | 多轮工具调用 + 规则奖励 |
| `rollout_engine.py` | 采样后端抽象 |
| `trainer_utils.py` | 学习率/断点/奖励模型/安全求值 |
| `convert_model.py` | 原生 → HF → Qwen3 兼容 → 生态 |

---

## 结语

MiniMind 的价值不在于「小」，而在于**把每个环节都写成了能读的代码**：

- 结构：8 层、GQA、RoPE、SwiGLU——和 Qwen3 同型，读到的东西能迁移；
- 数据：四种 jsonl，覆盖从接龙到工具调用的全部形态；
- 训练：八个阶段，每个阶段只差「损失函数那十几行」；
- 部署：转 HF、转 Qwen3 兼容、OpenAI 兼容服务，一条龙。

**推荐的下一步**：把 `train_pretrain.py` 和 `train_full_sft.py` 的 diff 看一遍——
你会发现从「学说话」到「学回答」，代码只改了 6 行。这 6 行就是大模型训练里
最值钱的部分之一。


