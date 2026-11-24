# `model_minimind.py` 逐类详解

> 对象：`minimind-master/model/model_minimind.py`（292 行）
> 环境：本机 conda `swift_env`（torch 2.6.0+cu124 / transformers 4.57.6）
> 本文按「文件全景 → 三个模块级函数 → 七个类」的顺序，逐个讲：**职责、初始化、前向逐步、设计意图、坑、改法**。

---

## 0. 文件全景

### 0.1 类与函数的依赖关系

```
                 MiniMindConfig（所有类的参数来源）
                        │
     ┌──────────────────┼───────────────────┐
     ▼                  ▼                   ▼
  RMSNorm          precompute_freqs_cis   ACT2FN
     │             apply_rotary_pos_emb
     │             repeat_kv
     ▼                  │
  Attention ◄───────────┘
     │
     ├──────────► MiniMindBlock ──► MiniMindModel ──► MiniMindForCausalLM
     │                  ▲                 ▲                    ▲
  FeedForward ──────────┤                 │                    │
  MOEFeedForward ───────┘                 │                    │
                                          └────────────────────┘
```

### 0.2 七个类一览

| 类 | 行 | 职责 | 类型 |
| --- | --- | --- | --- |
| `MiniMindConfig` | 10 | 所有结构参数的唯一来源 | 配置对象（继承 `PretrainedConfig`） |
| `RMSNorm` | 50 | 归一化 | `nn.Module` |
| `Attention` | 91 | 自注意力（GQA + QK-Norm + RoPE + KV cache） | `nn.Module` |
| `FeedForward` | 136 | SwiGLU 前馈层 | `nn.Module` |
| `MOEFeedForward` | 148 | 4 专家、top-1 路由的前馈层 | `nn.Module` |
| `MiniMindBlock` | 183 | 一层 Transformer（注意力 + 前馈 + 两个残差） | `nn.Module` |
| `MiniMindModel` | 201 | 主干：Embedding + N 层 + 最终归一化 | `nn.Module` |
| `MiniMindForCausalLM` | 239 | 语言模型头 + 损失 + 手写 generate | `PreTrainedModel` |

**设计上最关键的一条**：只有最外层是 `PreTrainedModel`，中间全部是普通 `nn.Module`、返回普通元组。这让代码读起来和「手写 PyTorch 教学模型」一样，同时又能享受 `from_pretrained` / `save_pretrained`。

---

## 1. 三个模块级函数

它们不是类，但被 Attention 和 Model 依赖，必须先讲清楚。

### 1.1 `precompute_freqs_cis`（:62）——预计算旋转位置编码

**职责**：一次性算出所有位置的 `cos/sin`，之后每层直接查表。

**逐步拆解**：

```python
freqs = 1.0 / (rope_base ** (arange(0, dim, 2)[:dim//2].float() / dim))
```

- `arange(0, dim, 2)` 取出偶数下标 `[0,2,4,...,94]`，共 `dim/2 = 48` 个（dim=head_dim=96）；
- 除以 `dim` 得到指数，`rope_base=1e6` 做底 → 48 个频率，低频到高频；
- **这就是 RoPE 的「频率带」**：不同维度以不同速度旋转。

```python
if rope_scaling is not None and end / orig_max > 1.0:
    inv_dim = lambda b: (dim * log(orig_max / (b * 2π))) / (2 * log(rope_base))
    low, high = floor(inv_dim(beta_fast)), ceil(inv_dim(beta_slow))
    ramp = clamp((arange(dim//2) - low) / max(high - low, 0.001), 0, 1)
    freqs = freqs * (1 - ramp + ramp / factor)
```

这是 **YaRN** 的核心：只在高频段（低索引）做插值、低频段保持不变，中间用线性斜坡过渡。

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `factor` | 16 | 长度扩展倍数 |
| `beta_fast` | 32 | 高频边界（这部分**不**压缩） |
| `beta_slow` | 1 | 低频边界（这部分**全**压缩） |
| `attention_factor` | 1.0 | 注意力温度补偿 |

**为什么需要外推**：模型训练时只见过 2048/32768 的位置，直接推理到 10 万 token 时，
高频维度会「转得太快」导致注意力错乱。YaRN 把频率压缩回训练分布。

```python
t = arange(end)
freqs = outer(t, freqs)                      # (end, dim/2)
freqs_cos = cat([cos(freqs), cos(freqs)], -1) * attn_factor   # (end, dim)
```

**为什么要拼接两次**：`apply_rotary_pos_emb` 里的 `rotate_half` 采用「前后两半互换」的写法，
所以 cos/sin 需要铺成与 `dim` 等宽（前后两半相同）。

**返回**：`(freqs_cos, freqs_sin)`，形状都是 `(max_position_embeddings, head_dim)`，
float32，在 CPU 上算好再注册成 buffer。

**内存代价**：32768 × 96 × 4B ≈ 12.6 MB / 个，两个共约 25 MB。
注册为 `persistent=False`，所以**不进权重文件**，但**要占显存/内存**。

### 1.2 `apply_rotary_pos_emb`（:80）——把旋转作用到 q/k 上

```python
def rotate_half(x):
    return cat((-x[..., d//2:], x[..., :d//2]), dim=-1)

q_embed = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
```

| 步骤 | 说明 |
| --- | --- |
| `rotate_half` | 把最后一维劈成两半，交换位置并给前半取负 |
| `cos.unsqueeze(1)` | cos 形状 `(L, D)` → `(L, 1, D)`，广播到 `(B, L, H, D)` |
| `.to(q.dtype)` | cos/sin 是 fp32 buffer，转回 q 的精度 |

**调用时机**：在 `Attention.forward` 里，**reshape 之后、拼接 KV cache 之前**。
新 token 才需要编码；缓存里的旧 token 早就编好了——这个顺序决定了 cache 的语义。

### 1.3 `repeat_kv`（:86）——GQA 的展开

```python
x[:, :, :, None, :].expand(B, L, KVH, n_rep, D).reshape(B, L, KVH*n_rep, D)
```

把 4 个 KV 头复制成 8 个，供 8 个 Q 头直接做矩阵乘。

| 细节 | 说明 |
| --- | --- |
| `expand` 再 `reshape` | `expand` 是零拷贝视图，但随后的 `reshape` 会真实复制 |
| `n_rep == 1` 直接返回 | 避免无谓复制 |
| 调用位置 | 在存完 cache 之后 → **缓存里只有 4 个头**，显存减半 |

---

## 2. `MiniMindConfig`（:10）

### 2.1 职责

一个「纯数据对象」，被其余所有类读取。继承 `PretrainedConfig`，因此天然支持
`save_pretrained` / `from_pretrained` / `register_for_auto_class`。

### 2.2 初始化逐行

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)          # ① 先让父类吃掉通用字段
        self.hidden_size = hidden_size      # ② 再覆盖/补齐自定义字段
        ...
```

| 字段 | 默认值 | 类型 | 作用 |
| --- | --- | --- | --- |
| `hidden_size` | 768 | 位置参数 | 主干宽度 |
| `num_hidden_layers` | 8 | 位置参数 | 层数 |
| `use_moe` | False | 位置参数 | 是否用 MoE 前馈 |
| `dropout` | 0.0 | kwargs | 丢弃率 |
| `vocab_size` | 6400 | kwargs | 词表 |
| `bos_token_id` / `eos_token_id` | 1 / 2 | kwargs | 特殊 token |
| `flash_attn` | True | kwargs | 是否允许走 SDPA 快路径 |
| `num_attention_heads` | 8 | kwargs | Q 头数 |
| `num_key_value_heads` | 4 | kwargs | KV 头数（GQA） |
| `head_dim` | `hidden//heads` = 96 | kwargs | 每头维度 |
| `hidden_act` | 'silu' | kwargs | FFN 激活 |
| `intermediate_size` | `ceil(768π/64)*64 = 2432` | kwargs | FFN 宽度 |
| `max_position_embeddings` | 32768 | kwargs | 预计算 RoPE 的长度 |
| `rms_norm_eps` | 1e-6 | kwargs | 归一化 epsilon |
| `rope_theta` | 1e6 | kwargs | RoPE 底数 |
| `tie_word_embeddings` | True | kwargs | 输入输出词嵌入共享 |
| `inference_rope_scaling` | False | kwargs | 是否启用 YaRN 外推 |

**MoE 专属（use_moe=False 时被忽略）**：

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| `num_experts` | 4 | 专家数量 |
| `num_experts_per_tok` | 1 | 每个 token 激活几个专家 |
| `moe_intermediate_size` | = intermediate_size | 每个专家的 FFN 宽度 |
| `norm_topk_prob` | True | 是否对 top-k 权重归一化 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡损失系数 |

### 2.3 两个推导公式

```python
head_dim          = hidden_size // num_attention_heads        # 768/8 = 96
intermediate_size = ceil(hidden_size * π / 64) * 64           # 2432
```

- `head_dim` 显式存下来（Qwen 风格），而不是每次现算——便于像 Qwen3 一样**解耦**
  「hidden_size」与「heads × head_dim」。
- `π/64` 的来历：SwiGLU 有三个矩阵，等效 FFN 宽度约为 `8/3 × d`，
  且取整到 64 的倍数对齐硬件。`8/3 ≈ 2.67`，`π/64×64 ≈ 3.14`，接近但不相等——
  这是作者的一个偏保守取值。

### 2.4 YaRN 配置是「开关 + 内置字典」

```python
self.rope_scaling = {
    "beta_fast": 32, "beta_slow": 1, "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0, "type": "yarn"
} if self.inference_rope_scaling else None
```

注意它**只影响推理**：训练时 `inference_rope_scaling=False`，
位置编码就是原始 RoPE；推理时打开才插值。

### 2.5 设计意图

| 做法 | 好处 |
| --- | --- |
| 继承 `PretrainedConfig` | 与 transformers 生态互通，`convert_model.py` 才能注册 |
| 默认值即「参考配置」 | 不传任何参数就得到 64M 模型 |
| MoE 字段与 Dense 字段共存 | 一个配置类管两种架构，`use_moe` 切换 |
| `head_dim` 显式 | 换 `hidden_size` 时不必担心整除问题（但仍会算出一个值） |

### 2.6 坑

1. **`super().__init__(**kwargs)` 在赋值之前**：如果 kwargs 里带了 `hidden_size`，
   会被后面的赋值覆盖——位置参数优先。
2. `tie_word_embeddings` 同时也是父类字段，这里用 `kwargs.get` 取默认 True，
   与 HF 默认 False 不同，**转 HF 格式时要小心**。
3. `intermediate_size` 只在 `use_moe=False` 时决定 FFN；
   MoE 时每个专家用 `moe_intermediate_size`（默认相同）。

### 2.7 可以怎么改

```python
# 换成更深的窄模型（对照 MobileLLM 的结论）
MiniMindConfig(hidden_size=512, num_hidden_layers=16)

# 换成 MHA（去掉 GQA）
MiniMindConfig(num_key_value_heads=8)

# 开 MoE
MiniMindConfig(use_moe=True, num_experts=8, num_experts_per_tok=2)
```

---

## 3. `RMSNorm`（:50）

### 3.1 职责

均方根归一化：**不减均值、不加偏置**，只用 RMS 缩放。

### 3.2 初始化

```python
self.eps = eps                                  # 注意默认 1e-5，但 config 传 1e-6
self.weight = nn.Parameter(torch.ones(dim))     # 可学习的缩放向量，初始为 1
```

### 3.3 前向

```python
def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

| 步骤 | 形状 | 说明 |
| --- | --- | --- |
| `x.pow(2).mean(-1, keepdim=True)` | `(..., D)` → `(..., 1)` | 最后一维求均方 |
| `rsqrt(· + eps)` | `(..., 1)` | 1/√(ms + ε)，防止除零 |
| `x *` | `(..., D)` | 广播回去 |
| `weight *` | `(..., D)` | 逐维缩放 |
| `.type_as(x)` | 转回原精度 | fp32 算完降回 fp16/bf16 |

### 3.4 设计意图

| 对比 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 去均值 | ✅ | ❌ |
| 偏置 | ✅ | ❌ |
| 计算量 | 高 | 低约 10–15% |
| 效果 | 基准 | 大模型上基本持平 |

**为什么先转 float**：`mean(x²)` 在 fp16 下可能上溢（值域 65504），
转到 fp32 算完再转回来，几乎零成本，但能避免训练后期的数值事故。

### 3.5 坑

- `eps` 的默认值（1e-5）与 config 传的值（1e-6）不同。模型里**到处都显式传 config 的 eps**，
  所以默认值只在单独实例化时生效。
- `type_as(x)` 对 bf16 输入会转成 bf16——如果上游是 fp32（比如手算），行为一致。

### 3.6 可以怎么改

改用 `nn.LayerNorm(dim, eps=..., elementwise_affine=True)` 即可做 A/B 对比，
参数量不变，但会多一组 bias。

---

## 4. `Attention`（:91）——全文件最核心的一次

### 4.1 职责

多头自注意力，同时承担四件事：**GQA 投影、QK 归一化、RoPE、KV cache 管理**。

### 4.2 初始化逐行

```python
self.num_key_value_heads = heads if config.num_key_value_heads is None else config.num_key_value_heads
self.n_local_heads    = config.num_attention_heads      # 8
self.n_local_kv_heads = self.num_key_value_heads        # 4
self.n_rep            = self.n_local_heads // self.n_local_kv_heads   # 2
self.head_dim         = config.head_dim                 # 96
self.is_causal        = True                            # 写死，永远是因果注意力
```

| 子层 | 形状 | 偏置 | 说明 |
| --- | --- | --- | --- |
| `q_proj` | 768 → 8×96 = 768 | 无 | Q |
| `k_proj` | 768 → 4×96 = 384 | 无 | K，**只有 4 个头** |
| `v_proj` | 768 → 4×96 = 384 | 无 | V |
| `o_proj` | 768 → 768 | 无 | 输出投影 |
| `q_norm` | RMSNorm(96) | — | 每个头单独归一化 |
| `k_norm` | RMSNorm(96) | — | 同上 |
| `attn_dropout` | Dropout(config.dropout) | — | 只作用于注意力权重 |
| `resid_dropout` | Dropout(config.dropout) | — | 作用于输出 |

```python
self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn
```

这是一个**能力探测 + 配置开关**的组合：torch 版本够新、且用户没关 `flash_attn`，
才允许走融合路径。

### 4.3 前向：12 步与形状

| # | 操作 | 形状变化 |
| --- | --- | --- |
| 1 | 取出 `bsz, seq_len` | — |
| 2 | `q/k/v = proj(x)` | → (B,L,768) / (B,L,384) / (B,L,384) |
| 3 | reshape | q (B,L,8,96)，k/v (B,L,4,96) |
| 4 | `q_norm` / `k_norm` | 不变（在最后一维 96 上归一化） |
| 5 | `apply_rotary_pos_emb` | 不变，旋转 q/k |
| 6 | 拼接历史 | k/v → (B, L+past, 4, 96) |
| 7 | 存 cache | `past_kv = (xk, xv)`，**存的是 4 头版本** |
| 8 | `repeat_kv` + transpose | q (B,8,L,96)，k/v (B,8,L+past,96) |
| 9 | 注意力计算 | scores (B,8,L,L+past) → out (B,8,L,96) |
| 10 | transpose + reshape | (B,L,768) |
| 11 | `o_proj` | (B,L,768) |
| 12 | `resid_dropout` | 返回 `(out, past_kv)` |

### 4.4 第 9 步：两条计算路径

```python
if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) \
        and (attention_mask is None or torch.all(attention_mask == 1)):
    output = F.scaled_dot_product_attention(xq, xk, xv,
                 dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
else:
    scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
    if self.is_causal:
        scores[:, :, :, -seq_len:] += full((seq_len, seq_len), -inf).triu(1)
    if attention_mask is not None:
        scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
    output = self.attn_dropout(softmax(scores, -1).type_as(xq)) @ xv
```

**为什么条件里有个 `past_key_value is None`**：因为 `is_causal=True` 写死了，
一旦有 cache，SDPA 的 `is_causal` 语义就和「历史 + 新 token」对不上，必须走手写路径。

**手写路径的因果掩码为什么是 `[:, :, :, -seq_len:]`**：

```
scores 形状: (B, H, seq_len, past+seq_len)
                                       └──────┬──────┘
                                        只在这块加三角掩码
```

前 `past` 列是历史 token，对每个新 token 都可见；只有「新 token 之间」需要因果屏蔽。

**掩码为什么用 `-1e9` 而不是 `-inf`**：fp16 下 `-inf` 参与 softmax 容易产生 NaN，
用一个大负数在数值上等价且更安全。

### 4.5 设计意图

| 设计 | 目的 |
| --- | --- |
| q_norm / k_norm | 压低注意力 logits 的方差，大 lr 下训练更稳（Qwen3 同款） |
| cache 存 4 头 | KV cache 显存直接减半 |
| RoPE 在拼 cache 之前 | 旧 token 的旋转只算一次，永久复用 |
| SDPA 快路径 + 手写慢路径 | 训练快、生成正确 |
| 所有投影无 bias | 与 Qwen/Llama 一致，参数量更省 |

### 4.6 坑

| 坑 | 说明 |
| --- | --- |
| `torch.all(attention_mask == 1)` | 每层每次前向都触发一次 GPU→CPU 同步，**是隐藏的性能损耗**；长训练时能占到可观比例 |
| `full(..., -inf).triu(1)` | 手写路径每次前向都新建一个 (L,L) 张量 |
| `repeat_kv` 会真实复制 | 4→8 头的展开不是零拷贝，显存与带宽都有代价 |
| `self.is_causal` 写死 True | 想改成双向注意力（如 BERT）必须同时改 SDPA 与手写两条路径 |

### 4.7 可以怎么改

```python
# 1) 换回标准 MHA（去掉 GQA）
MiniMindConfig(num_key_value_heads=8)

# 2) 关闭 QK-Norm 做消融
self.q_norm = nn.Identity(); self.k_norm = nn.Identity()

# 3) 加滑动窗口（只保留最近 W 个位置的 cache）
#    在拼接后截断 xk/xv，并在手写掩码里多遮一段
```

---

## 5. `FeedForward`（:136）

### 5.1 职责

SwiGLU 前馈层：三个线性层 + 门控。

### 5.2 初始化

```python
intermediate_size = intermediate_size or config.intermediate_size
self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
self.down_proj = nn.Linear(intermediate, hidden, bias=False)
self.act_fn = ACT2FN[config.hidden_act]        # 默认 silu
```

**为什么要收一个可选的 `intermediate_size`**：MoE 的每个专家要用
`moe_intermediate_size`，复用同一个类，避免再写一份。

### 5.3 前向

```python
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

| 分支 | 形状 |
| --- | --- |
| `gate_proj(x)` | (B,L,768) → (B,L,2432) |
| `act(...)` | 同上，逐元素 SiLU |
| `up_proj(x)` | (B,L,2432) |
| 相乘 | (B,L,2432) 门控 |
| `down_proj` | (B,L,768) |

### 5.4 为什么是 2432

| FFN 类型 | 矩阵数 | 等效宽度 | 本配置 |
| --- | --- | --- | --- |
| 普通 FFN（ReLU/GELU） | 2 | 4d | 3072 |
| SwiGLU | 3 | ≈ 8/3 d | 2432 |

三个矩阵 × 2432 的参数量 ≈ 两个矩阵 × 3072，**参数不变、表达更强**——这是
LLaMA 之后的标准做法。

### 5.5 坑

- `ACT2FN['silu']` 来自 transformers，若以后要脱离该依赖，需自己写 `F.silu`。
- 没有 dropout（config.dropout 只作用在注意力和嵌入）。

---

## 6. `MOEFeedForward`（:148）——项目里最「有想法」的一段

### 6.1 职责

4 个独立 FFN 专家 + 一个路由器（gate），每个 token 只走 top-1 个专家。

### 6.2 初始化

```python
self.gate    = nn.Linear(hidden_size, num_experts, bias=False)   # 768 → 4
self.experts = nn.ModuleList([FeedForward(config, moe_intermediate_size)
                              for _ in range(num_experts)])       # 4 个专家
```

**参数量**：4 × 专家参数 ≈ 4 × 4.7M ≈ 19M/层，8 层 ≈ 150M——
这就是 MoE 总参 198M 而激活只有约 64M 的来源。

### 6.3 前向：七步

```python
x_flat = x.view(-1, hidden)                              # (B*L, 768)
scores = softmax(self.gate(x_flat), -1)                  # (B*L, 4)
topk_weight, topk_idx = topk(scores, k=1, sorted=False)  # 各 (B*L, 1)
```

**① k=1 的直通梯度**

```python
if norm_topk_prob:
    if num_experts_per_tok > 1:
        topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
    else:
        top1 = topk(softmax(self.gate(x_flat.detach()), -1), k=1)[0]
        topk_weight = top1 - top1.detach() + 1.0
```

拆开看这句：

| 部分 | 数值 | 梯度 |
| --- | --- | --- |
| `top1` | 路由器对所选专家的概率 | 有（对 gate 参数） |
| `- top1.detach()` | 抵消数值 | 无 |
| `+ 1.0` | 前向结果恒为 1.0 | 无 |

结果：**前向权重 = 1.0，反向梯度 = d(top1)/d(gate)**。
这是直通估计（straight-through）：`topk` 的「选哪个」不可导，就用「概率本身」的梯度代替。

`x_flat.detach()` 是关键：梯度**只更新路由器的参数，不传给 token 表示**。
因为路由权重最终乘在专家输出上，如果让这部分梯度回传，会干扰隐藏状态的学习。

**② 按专家分桶、累加输出**

```python
y = zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)                   # (N, k)
    token_idx = mask.any(-1).nonzero().flatten()
    weight = topk_weight[mask].view(-1, 1)
    y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

| 步骤 | 说明 |
| --- | --- |
| `mask` | 哪些 token 选了专家 i |
| `token_idx` | 这些 token 的行号 |
| `expert(...)` | 只对这批 token 做前向（**这就是省算力的地方**） |
| `index_add_` | 累加回去，k>1 时同一 token 会落在多个专家里 |

**③ 空专家的 DDP 补丁**

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

某专家本批一个 token 都没接到时，`expert` 的参数没参与计算图。
DDP 在反向时会发现「有参数没被用到」而报错或卡死，所以加一个值为 0、
但依赖参数的项，把参数「挂」进计算图。

**④ 负载均衡损失**

```python
load = one_hot(topk_idx, num_experts).float().mean(0)   # 各专家接单比例 (4,)
self.aux_loss = (load * scores.mean(0)).sum() * num_experts * coef
```

| 项 | 含义 |
| --- | --- |
| `load` | 实际分配比例 |
| `scores.mean(0)` | 路由器给出的平均概率 |
| 点积 | 两者越「一致且均匀」越小 |
| `× num_experts × coef` | 归一化到与主损失同量级 |

**为什么要它**：没有这项，模型会迅速把所有 token 塞给同一个专家（路由坍缩），
MoE 退化成「一个大 FFN + 3 个闲置参数」。

### 6.4 坑

| 坑 | 说明 |
| --- | --- |
| Python for 循环遍历专家 | 4 个专家就是 4 次 kernel 启动，训练比 Dense 慢约 50%（README 承认） |
| `y[0,0] += ...` | 假设 batch×seq > 0，空张量会报错 |
| `aux_loss` 只在 training 时非零 | 推理时是 0 维零张量，日志里要区分 |
| k=1 时 `topk_weight` 前向恒 1 | 调试时看到「权重都是 1」不是 bug |

### 6.5 可以怎么改

```python
# 1) 4 专家 top-2
MiniMindConfig(use_moe=True, num_experts_per_tok=2)

# 2) 加共享专家（Qwen2-MoE 风格，本项目刻意去掉了）
#    在 gate 之外加一个总是激活的 FeedForward，输出相加

# 3) 换成批量矩阵乘（grouped GEMM）可提速，但可读性下降
```

---

## 7. `MiniMindBlock`（:183）

### 7.1 职责

一层完整的 Transformer：注意力 + 前馈 + 两个维护残差。

### 7.2 初始化

```python
self.self_attn = Attention(config)
self.input_layernorm = RMSNorm(hidden, eps)
self.post_attention_layernorm = RMSNorm(hidden, eps)
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

**这是唯一按 `use_moe` 分流的地方**——其余代码对两种架构完全透明。

### 7.3 前向

```python
residual = hidden_states
hidden_states, present_kv = self.self_attn(
    self.input_layernorm(hidden_states), position_embeddings,
    past_key_value, use_cache, attention_mask)
hidden_states += residual                                   # 残差 1
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 残差 2
```

| 设计 | 说明 |
| --- | --- |
| Pre-Norm | 先归一化再进子层，残差通路干净，深网络更稳 |
| 两条独立 RMSNorm | 注意力和 FFN 各一套，互不干扰 |
| 残差直连 | 子层只学「增量」 |
| 返回 `(hidden, present_kv)` | 让上层收集 cache |

### 7.4 坑

- `hidden_states += residual` 是**原地操作**。这里合法（张量不是叶子、也没被别处引用），
  但如果将来在这行之后还要用「残差相加前的 hidden_states」，就会踩到版本计数错误。
- 没有 dropout 夹在残差之间（注意力内部已有）。

---

## 8. `MiniMindModel`（:201）

### 8.1 职责

主干网络：Embedding → 8 层 Block → 最终归一化；管理 RoPE buffer 与 KV cache 列表。

### 8.2 初始化

```python
self.embed_tokens = nn.Embedding(vocab_size, hidden)      # 6400 × 768
self.dropout = nn.Dropout(config.dropout)
self.layers = ModuleList([MiniMindBlock(l, config) for l in range(num_layers)])
self.norm = RMSNorm(hidden, eps)
freqs_cos, freqs_sin = precompute_freqs_cis(...)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

**`persistent=False` 的三重含义**：

1. 不进 `state_dict` → 权重文件小 25 MB；
2. 但仍随 `.to(device)` 搬动 → 推理时在 GPU 上；
3. 换长度/换 rope 参数时不会「加载到旧的错误 buffer」。

### 8.3 前向逐步

| # | 代码 | 说明 |
| --- | --- | --- |
| 1 | `if hasattr(past_key_values, 'layers'): past_key_values = None` | 兼容 HF 新版 `DynamicCache` 对象：不认识就丢弃，退回自己算 |
| 2 | `past_key_values = past_key_values or [None] * len(layers)` | 统一成列表 |
| 3 | `start_pos = past[0][0].shape[1] if past[0] else 0` | 已有多少 token |
| 4 | `hidden = dropout(embed(input_ids))` | (B,L,768) |
| 5 | meta-device 修复 | buffer 为 0 时重建（transformers 5.x 的 meta 初始化会丢 buffer） |
| 6 | 切位置编码 | `freqs_cos[start:start+L]` → (L,96) |
| 7 | 逐层前向 | 收集 `presents` |
| 8 | `hidden = self.norm(hidden)` | 最终归一化 |
| 9 | 汇总 aux_loss | `sum(层.mlp.aux_loss for MoE 层)`，默认 0 维零张量 |
| 10 | 返回三元组 | `(hidden, presents, aux_loss)` |

**第 1 步那句 `hasattr(..., 'layers')`** 是很有代入感的兼容补丁：
新版 transformers 的 `generate` 会塞一个 `DynamicCache` 进来，而本模型用的是
「列表套元组」的自定义格式，两者不通，所以干脆丢掉重算。

### 8.4 设计意图

| 做法 | 目的 |
| --- | --- |
| 返回普通元组而非 `ModelOutput` | 简单、无依赖；代价是与 HF 生态约定不一致 |
| RoPE 在主干算一次 | 所有层共享，避免每层重复 |
| aux_loss 用 `isinstance` 过滤 | 同时兼容 Dense 与 MoE |
| cache 以列表传入传出 | 结构直观，便于手写 generate |

### 8.5 坑

| 坑 | 说明 |
| --- | --- |
| 返回元组 | 上层 `ForCausalLM` 必须按位置解包，改签名会连锁 |
| `start_pos` 只看第 0 层 | 各层 cache 长度必须一致（生成时天然满足） |
| `freqs_cos[0,0] == 0` 判断 | 依赖「第 0 位 cos 恒为 1」这一性质；若哪天换了编码方式，这个修复会失效 |
| 无 `output_hidden_states` / `output_attentions` | 想要中间层输出需自己改签名 |

---

## 9. `MiniMindForCausalLM`（:239）——唯一继承 `PreTrainedModel` 的类

### 9.1 类声明与两个类属性

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

| 声明 | 作用 |
| --- | --- |
| `PreTrainedModel` | 获得 `from_pretrained` / `save_pretrained` / `to()` 等 |
| `GenerationMixin` | 获得 `chat_template` 等生成生态方法（**但 generate 被覆盖了**） |
| `config_class` | Auto 类据此识别配置类型 |
| `_tied_weights_keys` | 告诉 transformers「这两个键是同一块权重」，保存/加载时去重 |

### 9.2 初始化

```python
def __init__(self, config=None):
    self.config = config or MiniMindConfig()
    super().__init__(self.config)
    self.model = MiniMindModel(self.config)
    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)   # 768 → 6400
    if self.config.tie_word_embeddings:
        self.model.embed_tokens.weight = self.lm_head.weight        # 同一块内存
    self.post_init()
```

三个细节：

| 细节 | 说明 |
| --- | --- |
| **先赋 config 再 super()** | 父类 `PreTrainedModel.__init__` 需要 config |
| **直接赋值绑定权重** | 两个 Parameter 指向同一张量，任意一个更新另一个同步变 |
| **`post_init()`** | 父类的收尾：应用 `_init_weights`、调用 `tie_weights()`、设置 generation 配置 |

**共享收益**：6400×768 = 4.9M 参数，占 64M 总参数的 **7.7%**。

### 9.3 `forward`：五行核心

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None,
            use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids, attention_mask, past_key_values, use_cache, **kwargs)

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

    return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                     past_key_values=past_key_values, hidden_states=hidden_states)
```

| 参数 | 作用 | 典型场景 |
| --- | --- | --- |
| `logits_to_keep=0` | 0 表示全算；k 表示只算最后 k 个位置 | RL 里每步只取最后一个 token 的 logits，省显存 |
| `labels` | 传了才计算损失 | 训练 |
| `**kwargs` | 透传给主干（实际没用） | 兼容 HF 的额外参数 |

**损失的四步拆解**：

1. `logits[..., :-1, :]`：位置 0..L-2 的预测；
2. `labels[..., 1:]`：位置 1..L-1 的真值；
3. `view(-1, V)` / `view(-1)`：摊平成 (N, V) 与 (N,)；
4. `ignore_index=-100`：跳过 padding 与被 mask 的位置。

**返回值里的 `hidden_states` 是「最终隐藏状态」一个张量**，不是 HF 约定的
元组（每层一个）。这是刻意的简化，也意味着传 `output_hidden_states=True` 不会有分层输出。

### 9.4 `generate`：手写采样的完整流程

```python
@torch.inference_mode()
def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
             temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
             streamer=None, use_cache=True, num_return_sequences=1,
             do_sample=True, repetition_penalty=1.0, **kwargs):
```

**① 准备阶段**

| 步骤 | 代码 | 说明 |
| --- | --- | --- |
| 参数校验 | `if do_sample and not 0 < temperature < inf: raise` | 防止温度非法 |
| 兼容入参 | `kwargs.pop("input_ids", inputs)` | 两种调用风格都支持 |
| 并行采样 | `input_ids.repeat(num_return_sequences, 1)` | 同一 prompt 生成多条 |
| 状态 | `finished = zeros(B, dtype=bool)` | 记录哪些样本已结束 |
| 流式 | `streamer.put(input_ids.cpu())` | 先把 prompt 推出去 |

**② 主循环（每一步）**

```
past_len = 缓存长度
outputs = forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache)
attention_mask 追加一列 1
logits = outputs.logits[:, -1] / temperature
[重复惩罚] → [top-k] → [top-p] → 采样/贪心 → [已完成的行强制 EOS]
拼接到 input_ids → 更新缓存 → streamer → 更新 finished → 全部结束则 break
```

**③ 三个采样技巧的实现细节**

**重复惩罚**（CTRL 风格的对称处理）：

```python
seen = unique(input_ids[i])
score = logits[i, seen]
logits[i, seen] = where(score > 0, score / penalty, score * penalty)
```

正分数除以系数（压低）、负分数乘以系数（更负），保证「越重复越不划算」。

**top-k**：

```python
logits[logits < topk(logits, k)[0][..., -1, None]] = -inf
```

取出第 k 大的值，比它小的全部置 `-inf`。

**top-p（核采样）** 是全文件最绕的三行：

```python
sorted_logits, sorted_indices = sort(logits, descending=True)
mask = cumsum(softmax(sorted_logits)) > top_p
logits[mask.scatter(1, sorted_indices, F.pad(mask[..., :-1], (1, 0)))] = -inf
```

为什么 `scatter` 里要 `pad` 一位：`cumsum > p` 第一次为 True 的那个 token
**本身是要保留的**（它是「跨过阈值」的那个），所以把 mask 右移一位，
让「第一个超过 p 的 token」不被屏蔽。

**④ 结束条件**

```python
next_token = where(finished.unsqueeze(-1), full(eos_token_id), next_token)
finished |= next_token.squeeze(-1).eq(eos_token_id)
if finished.all(): break
```

已结束的样本继续输出 EOS 占位（保持 batch 形状），全部结束才退出。

**⑤ 可选返回 KV**

```python
if kwargs.get("return_kv"):
    return {'generated_ids': input_ids, 'past_kv': past_key_values}
```

RL 训练里用得着（`rollout_engine` 会取 `per_token_logps`）。

### 9.5 与 HF `generate` 的差异

| HF 支持 | 本实现 | 影响 |
| --- | --- | --- |
| `do_sample` / `temperature` / `top_k` / `top_p` | ✅ | 一致 |
| `repetition_penalty` | ✅ | 一致（略简化） |
| `num_beams` / `length_penalty` | ❌ | 不能做 beam search |
| `stop_strings` / `StoppingCriteria` | ❌ | 只能靠 EOS 或长度停 |
| `min_new_tokens` | ❌ | 可能一上来就 EOS |
| `max_length` | ❌ | 只有 `max_new_tokens` |
| `use_cache` / `past_key_values` | ✅ | 一致 |
| 流式 | ✅（`streamer`） | 仅支持 `put/end` 协议 |

### 9.6 坑

| 坑 | 说明 |
| --- | --- |
| `@torch.inference_mode()` 装饰器 | 整个 generate 都在无梯度模式，训练时不能拿它算 loss |
| `attention_mask` 每步增长 | 但 `input_ids` 只喂新 token，两者长度不同（手写路径需要全长 mask） |
| `temperature` 只在 `do_sample=True` 时生效 | 贪心模式下看 temperature 没反应 |
| `eos_token_id` 默认写死 2 | 换 tokenizer 必须显式传 |
| `finished` 的行仍在算 | 省的是「采样」，没省前向计算 |

---

## 10. 类之间的四个隐式约定

读懂这个文件，关键是记住这四条「不成文契约」。

### 约定一：Config 是唯一真源

所有类都从 `config` 读参数，**没有任何硬编码的维度**（除了 `-1e9` 这类数值常量）。
所以改模型 = 改 config 一个地方。

### 约定二：张量布局

| 位置 | 布局 | 原因 |
| --- | --- | --- |
| Attention 内部（进 SDPA 前） | `(B, L, H, D)` | q/k/v 投影出来的自然形状 |
| SDPA 后 | `(B, H, L, D)` | 矩阵乘要求头维度在前 |
| KV cache | `(B, L, KVH, D)` | **未展开**，省显存 |
| 输出 | `(B, L, 768)` | reshape 回去 |

**记住这条**：`Attention.forward` 里 `transpose(1,2)` 出现了两次（进、出各一次），
就是为了在这两种布局之间往返。

### 约定三：返回值

| 层 | 返回 |
| --- | --- |
| `Attention` | `(output, past_kv)` |
| `Block` | `(hidden, present_kv)` |
| `Model` | `(hidden, presents_list, aux_loss)` |
| `ForCausalLM` | `MoeCausalLMOutputWithPast` |

越往上越「像 HF」，但只在最顶层遵守 HF 的标准。

### 约定四：权重绑定

```python
self.model.embed_tokens.weight = self.lm_head.weight
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

两处必须同时正确：初始化时赋值绑定，类属性告诉 HF 工具链去重。
**只做前者**会导致 `save_pretrained` 存两份、`from_pretrained` 时后加载的覆盖前者。

---

## 11. 逐类阅读检查清单

读完一个类，问自己这几个问题，答不上就回去再看。

### Config
1. `intermediate_size` 为什么是 2432 而不是 3072？
2. `head_dim` 为什么不直接每次用 `hidden // heads`？
3. MoE 的哪个字段决定「每 token 激活多少参数」？

### RMSNorm
4. 和 LayerNorm 差在哪两步？
5. 为什么先转 float32？

### precompute_freqs_cis / RoPE
6. 为什么 cos/sin 要拼接两次？
7. YaRN 的「斜坡」在哪些维度上起作用？
8. buffer 的 `persistent=False` 带来什么好处？

### Attention
9. KV cache 存的是 4 个头还是 8 个头？
10. 为什么有 cache 时不能走 SDPA？
11. 因果掩码为什么只加在 `[:, :, :, -seq_len:]`？
12. `q_norm` 归一化的是哪个维度？

### FeedForward / MoE
13. SwiGLU 为什么是三个矩阵？
14. k=1 时 `top1 - top1.detach() + 1.0` 前向值是多少？梯度从哪来？
15. `x_flat.detach()` 的作用是什么？
16. 没有 aux loss 会发生什么？
17. 空专家那句 `y[0,0] += 0 * ...` 是为了防什么？

### Block / Model
18. Pre-Norm 和 Post-Norm 的区别？
19. `start_pos` 从哪来？
20. 为什么 `Model` 返回普通元组而不是 `ModelOutput`？

### ForCausalLM
21. `logits_to_keep` 省的是什么？
22. 损失的错位是怎么做的？
23. `hidden_states` 返回的是所有层还是一个张量？
24. top-p 里 `scatter` 为什么要 pad 一位？
25. generate 为什么标了 `@torch.inference_mode()`？

---

## 12. 五个可以动手做的修改实验

| 实验 | 改哪里 | 预期现象 |
| --- | --- | --- |
| ① 加深变窄 | `MiniMindConfig(hidden_size=512, num_hidden_layers=16)` | 参数量接近，但 `d_model<512` 的边缘，观察 loss 是否更差 |
| ② 去掉 GQA | `num_key_value_heads=8` | 参数量 +约 10%，KV cache 翻倍 |
| ③ 关掉 QK-Norm | `q_norm/k_norm = nn.Identity()` | 大学习率下 loss 更容易抖 |
| ④ MoE top-2 | `use_moe=True, num_experts_per_tok=2` | 激活参数上升，训练变慢，aux loss 行为变化 |
| ⑤ 换 GeGLU | `hidden_act='gelu_pytorch_tanh'` | 几乎无感，作为对照 |

每个实验都只需要改一行，而且 `Model` 到 `ForCausalLM` 的代码完全不用动——
这正是「Config 单一真源 + 层间约定清晰」带来的好处。

---

## 13. 一表总结

| 类 | 一句话 | 最容易被问到的点 |
| --- | --- | --- |
| `MiniMindConfig` | 参数总表，默认即 64M | 2432、96、tie 默认 True |
| `RMSNorm` | 不减均值的归一化 | 为什么转 fp32 |
| `precompute_freqs_cis` | 一次算好所有位置的旋转 | YaRN 斜坡、拼接两次 |
| `apply_rotary_pos_emb` | 把旋转应用到 q/k | 在拼 cache 之前 |
| `repeat_kv` | 4 头复制成 8 头 | 为什么 cache 存 4 头 |
| `Attention` | GQA + QK-Norm + 双路径 | SDPA 条件、因果掩码偏移 |
| `FeedForward` | SwiGLU 三矩阵 | 8/3 宽度 |
| `MOEFeedForward` | 4 专家 top-1 | 直通梯度、aux loss、空专家补丁 |
| `MiniMindBlock` | 一层 Transformer | Pre-Norm 与两条残差 |
| `MiniMindModel` | 主干 + RoPE + cache 管理 | 返回元组、DynamicCache 兼容 |
| `MiniMindForCausalLM` | 输出头 + 损失 + generate | logits_to_keep、手写采样 |

> 一句话：**这个文件的精妙之处不在任何单个类，而在于「每一层只依赖下一层、
> 所有参数只来自 config、所有形状只有一种约定」——所以 292 行能装下一个完整的 LLM。**


