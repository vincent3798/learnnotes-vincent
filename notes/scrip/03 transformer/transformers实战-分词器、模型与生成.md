# transformers 实战：分词器、模型与生成

> **适用环境（本机实测）**
> conda 环境 `swift_env` ｜ Python 3.10.22 ｜ torch 2.6.0+cu124 ｜ transformers 4.57.6
> GPU：GTX 1650（4 GB 显存）
> 本书代码全部在本机跑通；凡标注「实测」的输出都是原样复制。

---

## 这本书怎么用

这是「模型相关 Python 包」系列的第二本，上一本是《PyTorch 实战》。
如果你还没打通 `loss.backward()` / `optimizer.step()` 这条链，先回去补。

本书只讲一件事：**transformers 这个包到底把 PyTorch 的哪些东西封装了、怎么用**。
读完你应该能：

1. 看懂任何开源模型的目录结构和加载代码；
2. 自己写分词、前向、生成三段代码，不依赖 `pipeline`；
3. 在 4 GB 显存上判断「这个模型能不能跑」「多大 batch 会爆」。

配套脚本在 `code/`，统一用这个解释器：

```bash
D:\conda_envs\swift_env\python.exe code\00_env.py
```

---

## 目录

- 第 0 章 心智模型：transformers 到底封装了什么
- 第 1 章 Config：模型的身份证
- 第 2 章 Tokenizer：文本与数字世界的桥
- 第 3 章 AutoModel：一行代码加载模型
- 第 4 章 前向与损失：labels 的魔力
- 第 5 章 generate：生成全解析
- 第 6 章 批处理与 padding：一个必踩的坑
- 第 7 章 4 GB 显存的现实
- 第 8 章 报错速查
- 第 9 章 小项目：完整的文本分类流水线
- 第 10 章 衔接下一本：peft
- 第 11 章 练习与答案
- 附录 A API 速查表
- 附录 B 术语中英对照

---

## 第 0 章 心智模型：transformers 到底封装了什么

### 0.1 一句话回答

**transformers 是建在 PyTorch 之上的一层「模型应用层」**：它把「某种网络结构 +
配套的分词规则 + 预训练权重 + 配置文件」打包成一个可下载的目录，并提供统一的
加载接口，让你三行代码就能用上别人训练好的模型。

| 你以前手写 | transformers 里对应的东西 |
| --- | --- |
| `class MLP(nn.Module)` | `AutoModel.from_pretrained(...)` |
| 自己写的词表 + 字符表 | `AutoTokenizer.from_pretrained(...)` |
| `nn.Embedding(V, d)` 和 d 的来源 | `config.vocab_size` / `config.hidden_size` |
| 训练循环 | `Trainer` 或 `model.generate()` |
| 自己保存的 `.pt` | `save_pretrained()` 目录（config + 权重 + 分词器） |

### 0.2 四个核心对象

```
      配置 Config（结构参数：几层、多宽、词表多大）
            │ 决定
            ▼
   模型 Model（nn.Module 子类，一堆权重） ◄──── 分词器 Tokenizer（文本 ↔ token id）
            │
            ▼
   输出（logits / hidden_states / loss）──► generate 生成文本
```

| 对象 | 类 | 你平时怎么拿 |
| --- | --- | --- |
| 配置 | `*Config` | `AutoConfig.from_pretrained(path)` |
| 分词器 | `*Tokenizer` | `AutoTokenizer.from_pretrained(path)` |
| 模型 | `*Model` / `*ForCausalLM` 等 | `AutoModel*` `.from_pretrained(path)` |
| 生成结果 | `GenerateDecoderOnlyOutput` 等 | `model.generate(...)` 的返回值 |

**四个对象必须来自同一个模型目录**。混用不同模型的分词器和权重，
轻则输出乱码，重则直接报 `index out of range`。

### 0.3 模型从哪来：三种来源

| 来源 | 写法 | 适用 |
| --- | --- | --- |
| 在线（HF 镜像） | `"Qwen/Qwen2.5-0.5B-Instruct"` | 有网时，首次下载后进缓存 |
| 本地目录 | `r"E:\models\Qwen3-ASR-0.6B"` | 离线、内网、已下载 |
| ModelScope | `modelscope` 包下载后本地加载 | 国内网络更稳（你已装） |

### 0.4 本机现状：离线能跑什么

**实测环境**：

```
python        : 3.10.22
torch         : 2.6.0+cu124
transformers  : 4.57.6
cuda          : True
gpu           : NVIDIA GeForce GTX 1650
HF 缓存目录   : D:\hf_cache\hub
HF 镜像       : https://hf-mirror.com
```

**本地可用模型**：

| 模型 | 位置 | 用途 |
| --- | --- | --- |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | HF 缓存 | 生成/前向 API 练习（权重随机，输出无意义） |
| Qwen2 分词器 | `E:\models\Qwen3-ASR-0.6B` | 真实中文 BPE，15 万词表 |
| Qwen3-ASR-0.6B 权重 | 同上 | 音频模型，**不能**当纯文本模型用 |

### 0.5 两个环境坑（你已经踩到其中一个）

**坑一：本机代理是 socks5，但没装 PySocks。** 只要 transformers 尝试联网，
就会抛：

```
requests.exceptions.InvalidSchema: Missing dependencies for SOCKS support.
```

两种解决方式，二选一：

```bash
# 方式 A：临时去掉代理（hf-mirror 通常不需要代理）
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 方式 B：让 requests 支持 socks
D:\conda_envs\swift_env\python.exe -m pip install "requests[socks]"
```

**坑二：缓存里有模型，但加载时仍去联网。** 强制离线：

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 必须在 import transformers 之前设置
```

> **口诀：配置定结构，分词管文本，模型算张量；三者同源，加载才稳。**

---

## 第 1 章 Config：模型的身份证

### 1.1 Config 里有什么

Config 是一个纯 Python 对象（本质是 dataclass），描述模型结构。
**它不包含权重**，只告诉 `AutoModel` 怎么搭骨架。

| 字段 | 含义 | 影响 |
| --- | --- | --- |
| `vocab_size` | 词表大小 | Embedding 行数、输出层维度 |
| `hidden_size` | 隐藏层宽度 | 每层参数量 ∝ 它的平方 |
| `num_hidden_layers` | 层数 | 参数量、显存、速度 |
| `num_attention_heads` | 注意力头数 | 必须能整除 `hidden_size` |
| `intermediate_size` | FFN 中间层宽度 | 通常是 `4 × hidden_size` |
| `max_position_embeddings` | 最大序列长度 | 超过要截断或扩位置编码 |
| `num_labels` | 分类类别数 | 只对分类模型有意义 |
| `architectures` | 模型类名列表 | **Auto 类靠它推断用哪个类** |
| `tie_word_embeddings` | 输入输出词嵌入是否共享 | 省参数量（GPT 系常见） |
| `num_key_value_heads` | KV 头数 | GQA：小于 Q 头数时省显存 |

### 1.2 手写配置造模型（完全离线）

不需要下载任何东西，就能练习整个流程：

```python
from transformers import BertConfig, BertForSequenceClassification

cfg = BertConfig(
    vocab_size=100, hidden_size=64,
    num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=128, num_labels=2,
)
model = BertForSequenceClassification(cfg)
print(sum(p.numel() for p in model.parameters()))
```

**实测**：参数量 `110,658`。

### 1.3 保存与加载：一个模型目录长什么样

```python
model.save_pretrained("tiny-bert")
```

**实测目录内容**：

```
config.json              615 字节
model.safetensors    447,032 字节
```

真实模型的目录还会多这些东西：

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `config.json` | 结构参数 | ✅ |
| `model.safetensors` / `pytorch_model.bin` | 权重 | ✅ |
| `tokenizer.json` / `vocab.txt` / `tokenizer.model` | 分词规则 | ✅（要分词时） |
| `tokenizer_config.json` | 特殊 token、模板配置 | ✅ |
| `generation_config.json` | 生成默认参数 | 生成时常用 |
| `special_tokens_map.json` | pad/eos/bos 映射 | 常用 |

### 1.4 Auto 类是怎么知道该用哪个模型的

```python
from transformers import AutoConfig, AutoModelForSequenceClassification

cfg = AutoConfig.from_pretrained("tiny-bert")
print(cfg.architectures)
```

**实测**：你手写 Config 时 `architectures = None`；经 `save_pretrained` 后
变成 `['BertForSequenceClassification']`。`AutoModel*` 就是读这个字段，
动态导入对应的类。

**推论**：想给自己的自定义模型接入 Auto 体系，只要在 `config.json` 里写对
`architectures`，并让自定义类能被 transformers 找到（或注册）。

### 1.5 解码器架构的两个关键参数

```python
from transformers import LlamaConfig
lcfg = LlamaConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2,
                   num_attention_heads=2, num_key_value_heads=2,
                   intermediate_size=128, max_position_embeddings=512)
```

| 参数 | 含义 | 为什么重要 |
| --- | --- | --- |
| `num_key_value_heads` | KV 头数 | 小于 Q 头数 = GQA，KV cache 显存成比例下降 |
| `tie_word_embeddings` | 词嵌入共享 | 词表大时能省下整个输入 Embedding 的参数量 |

真实例子：Llama-3-8B 的 `num_attention_heads=32`、`num_key_value_heads=8`，
KV cache 省 4 倍。

> **第 1 章口诀：层数定深度，宽度定体积；architectures 定类，vocab 定两端。**

---

## 第 2 章 Tokenizer：文本与数字世界的桥

### 2.1 三种主流分词算法

| 算法 | 代表模型 | 切分方式 | 词表大小 | 特点 |
| --- | --- | --- | --- | --- |
| BPE（字节级） | GPT / Qwen / Llama | 从字节开始合并高频组合 | 5 万 ~ 15 万 | 任意文本都能编码，不会 OOV |
| WordPiece | BERT 系 | 贪心最长匹配子词 | 3 万 | 英文强，中文常按字切 |
| SentencePiece | T5 / Llama | 直接在原始文本上建模（含空格） | 3 万 ~ 25 万 | 与语言无关 |

**你环境里能直接用的**：Qwen 字节级 BPE（15.1 万词表）。

### 2.2 实测：中文到底被切成什么样

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"E:\models\Qwen3-ASR-0.6B")
enc = tok("模型训练很慢，怎么办？")
```

**实测输出**：

```
类名        : Qwen2TokenizerFast
词表大小    : 151,643
padding_side: right
pad_token   : '<|endoftext|>' (id=151643)
eos_token   : '<|im_end|>' (id=151645)

原文   : 模型训练很慢，怎么办？  (11 个字)
token 数: 7
ids    : [104949, 104034, 99165, 99843, 3837, 102572, 11319]
tokens : ['æ¨¡åŀĭ', 'è®Ńç»ĥ', 'å¾Ī', 'æħ¢', 'ï¼Į', 'æĢİä¹ĪåĬŀ', 'ï¼Ł']
decode(ids) == 原文 ? True
```

三个要点：

1. **11 个汉字 → 7 个 token**。中文在 Qwen 词表里大量以「词」为单位合并
   （"模型"、"训练"、"怎么办"各占一个 token）。
2. **token 显示成乱码是正常的**。字节级 BPE 的 token 是「utf-8 字节序列」，
   直接显示就是这种样子；`decode` 后会还原成中文。
3. **编码可逆**：`decode(encode(x)) == x`，这是能放心用 token 数的前提。

### 2.3 token 效率实测

| 文本 | token 数 |
| --- | --- |
| `'hello'` | 1 |
| `'Hello world!'` | 3 |
| `'人工智能'` | 1 |
| `'大语言模型（Large Language Model）'` | 8 |
| `'1234567890'` | 10 |
| `'    '`（四个空格） | 1 |

**结论**：中文约 1 字 ≈ 0.6 ~ 1 token，英文 1 词 ≈ 1 ~ 1.5 token，
**数字每位一个 token**（所以模型做算术吃力）。这直接决定上下文长度和计费。

> **口诀：中文按词合并，英文按子词切，数字一位一个；token 数 ≠ 字数。**

### 2.4 批量处理：padding、truncation、attention_mask

模型只接受矩形张量，长度不一的句子必须补齐：

```python
batch = tok(["短句", "这是一个明显更长的句子，用来制造长度差异"],
            padding=True, return_tensors="pt")
```

**实测**：

```
批量 input_ids: (2, 11)
attention_mask: [[1,1,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1,1]]
```

`attention_mask` 告诉模型哪些位置是真 token、哪些是填充。**没有它，模型会把
填充位当成内容，输出全错。**

右侧填充（默认）：

```
短句        -> [99534, 99700, 151643, 151643, ..., 151643]
长句子      -> [105464, 100687, ..., 102478]
```

左侧填充：

```
短句        -> [151643, ..., 151643, 99534, 99700]
长句子      -> [105464, 100687, ..., 102478]
```

**同一个短句，填充位置不同，ID 序列完全不同。** 第 6 章会看到这直接改变生成结果。

### 2.5 truncation：超长截断

**实测**：50 个"长"字共 25 个 token，`truncation=True, max_length=16` 后
变成 16 个。

```python
tok(long_text, truncation=True, max_length=16)
```

**注意**：不设 `max_length` 时，`truncation=True` 用模型的
`model_max_length`（可能是 32k，等于没截）。显存不够时先检查这里。

### 2.6 chat template：把对话变成提示词

聊天模型的输入是消息列表，但模型只认一串文本。`chat_template` 负责转换：

```python
messages = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "你好"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**实测（Llama-2 风格模板）**：

```
'<s>[INST] <<SYS>>\n你是一个简洁的助手。\n<</SYS>>\n\n你好 [/INST]'
```

`add_generation_prompt=True` 的作用：在末尾补上「助手回合」的起始标记，
模型从这里接着写。**评估/预览时用 False，实际生成必须用 True**——
少了它，模型会接着复述用户的话，而不是回答。

### 2.7 模板本质是 Jinja2，可以自己写

```python
qwen_style = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
tok.chat_template = qwen_style
print(tok.apply_chat_template([{"role": "user", "content": "你好"}],
                              tokenize=False, add_generation_prompt=True))
```

**实测**：

```
'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
```

模板来源优先级：`tokenizer.chat_template` 属性 > `tokenizer_config.json`
里的 `chat_template` > 单独的 `chat_template.json` 文件。

**本机的坑**：`E:\models\Qwen3-ASR-0.6B\chat_template.json` 是多模态模板，
要求 `content` 是列表（文本块 + 音频块）。直接套纯文本消息会渲染成空字符串。
这类模板要配合模型的 processor 使用，不是通用聊天模板。

> **第 2 章口诀：先分词再喂模型，长度不一要 padding；mask 标真伪，模板定回合。**

---

## 第 3 章 AutoModel：一行代码加载模型

### 3.1 Auto 家族对照表

| 类 | 输出 | 用途 |
| --- | --- | --- |
| `AutoModel` | `last_hidden_state (B,L,H)` | 只要特征，自己接下游 |
| `AutoModelForCausalLM` | `logits (B,L,V)` + `past_key_values` | 文本生成（GPT/Qwen/Llama） |
| `AutoModelForSequenceClassification` | `logits (B, num_labels)` | 整句分类（情感、意图） |
| `AutoModelForTokenClassification` | `logits (B,L,num_labels)` | 序列标注（NER） |
| `AutoModelForQuestionAnswering` | `start_logits` / `end_logits` | 抽取式问答 |
| `AutoModelForMaskedLM` | `logits (B,L,V)` | 填空（BERT 系） |
| `AutoModelForSeq2SeqLM` | `logits (B,L,V)` | 翻译/摘要（T5、BART） |

规律：**`AutoModelForXxx` = 底座模型 + 一个任务头**。

### 3.2 from_pretrained 的常用参数

```python
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.float16,          # 4.57 推荐写法
    device_map="auto",            # 需要 accelerate（你已装）
    attn_implementation="sdpa",   # 默认就是 sdpa
    low_cpu_mem_usage=True,       # 大模型加载时省内存
)
```

| 参数 | 作用 | 注意 |
| --- | --- | --- |
| `dtype` | 权重精度 | **4.56 起用 `dtype=`，旧的 `torch_dtype=` 仍兼容但会提示弃用** |
| `device_map` | 自动放置到 GPU/CPU | 需要 `accelerate`；4GB 卡上大模型会自动 offload |
| `attn_implementation` | 注意力实现 | `sdpa`（默认，快）/ `eager`（可看注意力）/ `flash_attention_2`（需装 flash-attn） |
| `low_cpu_mem_usage` | 逐层加载 | 加载 7B+ 时基本必开 |
| `trust_remote_code` | 允许执行仓库自定义代码 | **有安全风险**，只对可信仓库开 |
| `revision` | 指定分支/commit | 复现实验时用 |

**实测**：`dtype=torch.float16, device_map="auto"` 加载 tiny llama 后，

```
first param: cuda:0 torch.float16
```

模型被自动放到了 GPU 上。

### 3.3 输出对象：不是裸张量

```python
out = model(**batch)
```

| 字段 | 形状 | 何时存在 |
| --- | --- | --- |
| `logits` | `(B, L, V)` 或 `(B, C)` | 大部分模型 |
| `last_hidden_state` | `(B, L, H)` | `AutoModel` |
| `hidden_states` | 长度 = 层数+1 的元组 | `output_hidden_states=True` |
| `attentions` | 每层 `(B, H, L, L)` | `output_attentions=True`（sdpa 下不可用） |
| `past_key_values` | 元组 | 生成/`use_cache=True` |
| `loss` | 标量 | 传入 `labels` |

### 3.4 实测：分类前向

```python
cfg = BertConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=128, num_labels=2)
model = AutoModelForSequenceClassification.from_config(cfg).eval()
batch = {"input_ids": torch.randint(0, 100, (4, 16)),
         "attention_mask": torch.ones(4, 16, dtype=torch.long)}
out = model(**batch)
```

**实测**：

```
logits: (4, 2)  (= batch × num_labels)
probs : [0.5109, 0.4891]
传 labels 后自动算 loss: 0.6865
等价于（手算 cross_entropy）: 0.6865
```

注意最后两行：**传 `labels` 时模型内部就是 `cross_entropy(logits, labels)`**，
不用自己写损失函数。

### 3.5 冻结参数：只训练任务头

```python
for p in model.bert.parameters():
    p.requires_grad = False
```

**实测**：

```
可训练 130 / 总计 110,658  (0.12%)
可训练参数名字: ['classifier.weight', 'classifier.bias']
```

这是「冻结底座 + 训练头」的最小版本，也是下一本 LoRA 的思想起点。

### 3.6 实测：因果语言模型前向

```python
tok = AutoTokenizer.from_pretrained(TINY)
model = AutoModelForCausalLM.from_pretrained(TINY).eval()
out = model(**tok("Hello world", return_tensors="pt"), output_hidden_states=True)
```

**实测**：

```
logits        : (1, 3, 32000)   (B, L, vocab)
hidden_states : 3 个（含 embedding 输出）
past_key_values: 有
传 labels 自动计算错位交叉熵: 10.3800
模型参数量: 1,032,272
```

`hidden_states` 有 3 个 = 2 层 + 1 个 Embedding 输出。这个元组是「探测模型
内部」最常用的抓手（可视化、探针实验、层间差异分析）。

> **第 3 章口诀：Auto 家族看后缀，后缀决定输出头；dtype 定精度，device_map 定位置。**

---

## 第 4 章 前向与损失：labels 的魔力

### 4.1 两类任务的损失约定

| 任务 | 模型类 | labels 形状 | 内部损失 |
| --- | --- | --- | --- |
| 单句分类 | `ForSequenceClassification` | `(B,)` int64 | CrossEntropy(logits, labels) |
| 多标签分类 | `ForSequenceClassification`（`problem_type` 配置） | `(B, C)` float | BCEWithLogits |
| 因果语言模型 | `ForCausalLM` | `(B, L)` int64 | 错位交叉熵 |
| 序列标注 | `ForTokenClassification` | `(B, L)` int64 | 逐位置 CE |
| 填空 | `ForMaskedLM` | `(B, L)` int64（非 mask 位置 -100） | 带 ignore_index 的 CE |

**关键机制**：`-100` 是默认 `ignore_index`，损失会跳过这些位置。
做指令微调时，只有回答部分算损失——就是把问题部分的 label 设成 `-100`。

### 4.2 因果语言的「错位」是什么

模型在每个位置输出「下一个 token 的分布」。所以：

```python
# 内部等价于
shift_logits = logits[:, :-1, :]     # 位置 0..L-2 的预测
shift_labels = labels[:, 1:]          # 位置 1..L-1 的真实答案
loss = F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1))
```

自己实现一遍，就能理解为什么输入和标签是同一个序列——只是错开一位。
第 9 章会用到这个机制做「只对回答算损失」。

### 4.3 什么时候不用 labels

| 场景 | 做法 |
| --- | --- |
| 推理/生成 | 不传 labels，用 `generate` 或取 logits |
| 自定义损失（对比学习、蒸馏） | 不传 labels，`out.logits` 自己算 |
| 多任务加权 | 各任务分开前向，loss 加权求和后再 backward |

### 4.4 与 Trainer 的关系

`Trainer` 内的默认行为就是：把 batch 喂给模型 → 取 `outputs.loss` →
backward → step。你在 PyTorch 教材里学的五步循环，一行没变。

区别只在于：`Trainer` 会自动处理 `labels` 的搬运、混合精度、梯度累积、
分布式和日志。**看懂 `outputs.loss` 从哪来，就看懂了 Trainer 的一半。**

> **第 4 章口诀：分类标量标签，语言模型错一位；-100 跳过不计分，labels 一传损失自算。**

---

## 第 5 章 generate：生成全解析

### 5.1 最小调用

```python
enc = tok("Hello", return_tensors="pt").to("cuda")
ids = model.generate(**enc, max_new_tokens=10)
text = tok.decode(ids[0], skip_special_tokens=True)
```

`generate` 返回的是 **prompt + 新生成 token 的完整序列**，不是只有新部分。
解码时要么 `skip_prompt=True`（流式），要么自己切掉前 `enc["input_ids"].shape[1]` 个。

### 5.2 参数全景表

| 参数 | 作用 | 常用值 | 说明 |
| --- | --- | --- | --- |
| `max_new_tokens` | 最多生成多少新 token | 128 ~ 1024 | **优先用它，别用 max_length** |
| `do_sample` | 是否采样 | `False` 贪心 / `True` 采样 | 采样才有随机性 |
| `temperature` | 分布锐化/平滑 | 0.7 ~ 1.0 | 越小越保守 |
| `top_k` | 只从概率最高的 k 个里选 | 40 ~ 50 | 砍掉长尾 |
| `top_p` | 累积概率到 p 为止 | 0.9 ~ 0.95 | 与 top_k 二选一常用 |
| `repetition_penalty` | 惩罚重复 token | 1.05 ~ 1.2 | 1.0 = 不惩罚 |
| `num_beams` | beam 搜索宽度 | 1（默认）/ 3 ~ 5 | 越大越慢越稳 |
| `length_penalty` | beam 的长度偏好 | 1.0 | >1 偏好长句 |
| `eos_token_id` | 停止标记 | 自动 | 生成到它就停 |
| `pad_token_id` | 填充标记 | 自动 | 批量生成时必需 |
| `use_cache` | KV 缓存 | True | 关掉会明显变慢 |
| `return_dict_in_generate` | 返回对象而非张量 | True | 要 scores 时必须 |
| `output_scores` | 返回每步 logits | True | 用于置信度分析 |

> **高频错误**：用 `max_length` 控制生成长度。它限制的是**总长度**，
> prompt 长一点就会生成 0 个 token。统一用 `max_new_tokens`。

### 5.3 三种解码策略（实测）

模型是 tiny-random-Llama（**权重随机，文本没有意义**，这里只看行为差异）：

```python
model.generate(**enc, max_new_tokens=10, do_sample=False)                    # 贪心
model.generate(**enc, max_new_tokens=10, do_sample=True,
               temperature=1.5, top_k=50, top_p=0.9)                          # 采样
model.generate(**enc, max_new_tokens=10, do_sample=False, num_beams=3)       # beam
```

**实测输出**：

```
device=cuda  dtype=torch.float16  params=1,032,272
greedy         'Helloino película sums single mehrere Mathemat jaarєдна Cultural improved'
sampling T=1.5 'Hello Narerenfen identity overlayhindengo Category desp règ'
beam=3         'Helloinoesoтил sod族 Warren](/ årsLLкі'
```

| 策略 | 确定性 | 质量 | 速度 | 适用 |
| --- | --- | --- | --- | --- |
| 贪心 | 完全确定 | 易重复、易陷入循环 | 最快 | 分类式任务、评测基线 |
| 采样 | 随机 | 多样，可能跑偏 | 最快 | 对话、创作 |
| beam | 完全确定 | 全局更优 | 慢 `num_beams` 倍 | 翻译、摘要 |

### 5.4 从 scores 看模型有多「确信」

```python
out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                     return_dict_in_generate=True, output_scores=True)
probs = out.scores[0].softmax(-1)[0]
```

**实测**：

```
第 1 个生成 token 的 top-5：
  id=  1789  p=4.32e-05  token='ino'
  id=  3038  p=4.28e-05  token='▁cell'
  id= 30564  p=4.21e-05  token='社'
  id= 14836  p=4.20e-05  token='bits'
  id= 28241  p=4.14e-05  token='▁mv'
scores 元组长度=5，每个形状=(1, 32000)
```

**怎么读**：32000 个 token 上最大概率只有 `4.3e-5`，约等于 `1/23000`——
这模型对「下一个词」几乎没有偏好（随机权重，理应如此）。真实模型在
常见上下文下 top-1 概率可以到 0.3 ~ 0.9。**用这个指标就能判断模型是不是
真的「知道」**，比看输出文本客观。

### 5.5 KV cache：生成加速的关键

**原理**：自回归生成每步只多一个 token，但注意力需要看全部历史。
不缓存的话每步都要重算所有历史位置的 K、V；缓存后每步只算新 token 的 K、V，
历史直接复用。

**实测（2 层小模型，生成 60 个 token）**：

```
use_cache=True    388.7 ms
use_cache=False   431.0 ms
```

这里差距只有 11%，因为模型只有 2 层、prompt 只有 1 个 token——缓存没多少可省的。
**层数越多、prompt 越长，差异越夸张**：7B 模型长上下文下，关掉 cache 会慢十几倍。

代价是显存。KV cache 大小估算：

```
bytes = 2(K和V) × 层数 × KV头数 × head_dim × 序列长 × batch × 每元素字节
```

例：Llama-3-8B（32 层、8 个 KV 头、head_dim 128），fp16、4096 长度、batch 1：

```
2 × 32 × 8 × 128 × 4096 × 1 × 2 ≈ 0.5 GB
```

这还没算权重（16 GB）——**长上下文推理时，KV cache 往往是大头**。

### 5.6 流式输出

```python
from transformers import TextStreamer
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
model.generate(**enc, max_new_tokens=12, do_sample=False, streamer=streamer)
```

**实测**：token 逐个打印，而不是等全部生成完。

Web 服务里更常用 `TextIteratorStreamer` + 后台线程，把每个 chunk 推给前端。

### 5.7 停止条件

| 方式 | 写法 | 何时停 |
| --- | --- | --- |
| 长度 | `max_new_tokens=N` | 生成够 N 个 |
| EOS | 自动 | 模型输出结束符 |
| 指定字符串 | `stop_strings=["\n\n"]`（需传 `tokenizer=tok`） | 输出包含该串 |
| 自定义 | `StoppingCriteria` 子类 | 任意条件 |

```python
model.generate(**enc, max_new_tokens=64, stop_strings=["。\n"], tokenizer=tok)
```

### 5.8 generate 的几条常见提示

1. `Setting pad_token_id to eos_token_id:...` —— 正常提示，说明没显式给 pad。
   批量生成时建议显式设置：`tok.pad_token = tok.eos_token`。
2. `The following generation flags are not valid and may be ignored` ——
   你传了当前配置不认的参数（如没开采样却传 `temperature`），不致命但应清理。
3. 生成结果重复 —— 加 `repetition_penalty=1.1` 或改用采样。

> **第 5 章口诀：max_new_tokens 管长度，do_sample 定随机；top_p 收尾巴，cache 省时间。**

---

## 第 6 章 批处理与 padding：一个必踩的坑

### 6.1 两条规则

| 任务 | padding_side | 原因 |
| --- | --- | --- |
| 分类 / 标注 / 训练 | **right**（默认） | 真实 token 在前，池化取 `[CLS]` 或末位不受影响 |
| 批量生成 | **left** | 生成从最后一个位置往后接，填充必须在前面 |

### 6.2 实测：同一句话，不同 padding，结果不同

两条 prompt：`"Hello"`（短）和 `"Hello there my friend how are you"`（长）。

**单条推理（标准答案）**：

```
Hello                          -> [15043, 1789, 19053, 25470, 2323, 21742]
Hello there my friend how ar   -> [366, 21910, 10370, 16406, 1404, 5661]
```

**右侧 padding 批量生成**：

```
Hello                          -> [2, 20141, 31458, 17245, 22444, 6745]   与单条一致: False
Hello there my friend how ar   -> [366, 21910, 10370, 16406, 1404, 5661]   与单条一致: True
```

**左侧 padding 批量生成**：

```
Hello                          -> [15043, 1789, 19053, 25470, 2323, 21742]   与单条一致: True
Hello there my friend how ar   -> [366, 21910, 10370, 16406, 1404, 5661]   与单条一致: True
```

**结论**：

- 长的那条两种 padding 都正确（它本来就不需要填充）。
- 短的那条在右侧 padding 下**生成结果完全变了**（`1789...` vs `20141...`）。
- 左侧 padding 下两条都与单条推理一致。

### 6.3 为什么会错

生成时 `generate` 从序列的**最后一个位置**继续。右侧填充时，最后一个位置
是 pad token，模型从 pad 后面接着写；同时后续 token 的位置编号也整体错位。
`attention_mask` 能屏蔽 pad 的内容，但**屏蔽不了位置错位**。

### 6.4 正确的批量生成写法

```python
tok.pad_token = tok.eos_token          # 解码器模型通常没有 pad，借用 eos
tok.padding_side = "left"              # 关键一行

batch = tok(prompts, padding=True, return_tensors="pt").to(device)
out = model.generate(**batch, max_new_tokens=64, do_sample=False)

# 解码时要去掉 pad
texts = tok.batch_decode(out, skip_special_tokens=True)
```

### 6.5 每批里有不同长度的输出

`generate` 会返回等长张量，短的可能在后面补 pad。解码后按需清理：

```python
for t in texts:
    print(t.replace(tok.pad_token, "").strip())
```

更严格的写法是用 `eos_token_id` 定位结束位置后切片。

> **第 6 章口诀：训练右填充，生成左填充；填错位置，结果全歪。**

---

## 第 7 章 4 GB 显存的现实

### 7.1 推理显存的三块账

```
总显存 ≈ 权重 + KV cache + 激活/临时张量
```

| 组成 | 估算方式 | 4GB 上的地位 |
| --- | --- | --- |
| 权重 | 参数量 × 每参数字节（fp32=4, fp16/bf16=2, int8=1, int4=0.5） | 通常最大 |
| KV cache | 见 5.5 公式 | 长上下文时很大 |
| 激活 | 与 batch × 序列长 成正比 | 短输入时很小 |

### 7.2 实测算账

加载 fp16 的 tiny llama（1,032,272 参数）跑一次生成：

```
峰值显存 11.7 MB
```

理论权重 = 1.03M × 2 字节 ≈ 2 MB，其余是 CUDA 上下文和临时张量。
**小模型上，固定开销远大于权重本身。**

### 7.3 真实模型在这块卡上的可行性

| 模型规模 | fp16 权重 | KV cache（2k ctx） | 4GB 能否推理 | 建议 |
| --- | --- | --- | --- | --- |
| 0.5B | 1.0 GB | ~0.1 GB | ✅ 可以 | 本地学习首选 |
| 1.5B | 3.0 GB | ~0.3 GB | ⚠️ 勉强，ctx ≤ 1k | 短对话可以 |
| 7B | 14 GB | ~1 GB | ❌ | 必须量化或上云 |
| 13B+ | 26 GB+ | — | ❌ | 上云 |

**注意**：这是**推理**。训练需要额外存梯度、优化器状态和激活，
能训的规模要再小一个数量级（见 PyTorch 教材第 7 章）。

### 7.4 下载模型：本机的真实情况

**实测**：`hf-mirror.com` 直连约 **420 KB/s**。0.5B 模型约 1 GB，
需要 **40 分钟左右**。7B 模型 14 GB，不现实。

| 方案 | 做法 | 适合 |
| --- | --- | --- |
| HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com`（你已设置） | 小模型、有耐心 |
| ModelScope | `modelscope download --model Qwen/Qwen2.5-0.5B-Instruct` | 国内更稳 |
| 离线拷贝 | 从别的机器拷整个模型目录 | 大模型首选 |

**代理坑**（本机实测）：`HTTPS_PROXY=socks5://...` 但没装 PySocks，
所有 HF 请求直接失败。下载前先 `unset` 代理，或安装 `requests[socks]`。

**给这套环境的推荐命令**：

```bash
D:\conda_envs\swift_env\python.exe -m modelscope.cli.cli download --model Qwen/Qwen2.5-0.5B-Instruct
```

### 7.5 加载时的省显存开关

| 开关 | 效果 | 代价 |
| --- | --- | --- |
| `dtype=torch.float16` | 权重减半 | 数值稳定性略降 |
| `device_map="auto"` | 超显存自动切到 CPU | 变慢（PCIe 传输） |
| `low_cpu_mem_usage=True` | 逐层加载，峰值内存低 | 无 |
| `attn_implementation="sdpa"` | 注意力更省显存更快 | 看不到 attentions |

> **第 7 章口诀：权重看参数量乘字节，KV 看层数和长度；4G 只跑五亿以内，七 B 请上云。**

---

## 第 8 章 报错速查

| 报错 / 现象 | 原因 | 修法 |
| --- | --- | --- |
| `InvalidSchema: Missing dependencies for SOCKS support` | 代理是 socks5 但没装 PySocks | 取消代理环境变量，或 `pip install "requests[socks]"` |
| `OSError: We couldn't connect to ...` 但模型明明下过 | 没开离线模式，联网失败 | `HF_HUB_OFFLINE=1` |
| `Cannot use chat template functions because tokenizer.chat_template is not set` | 模型没有聊天模板 | 自己设置 `tok.chat_template` 或用普通拼接 |
| `Token indices sequence length is longer than ... model_max_length` | 输入超长 | `truncation=True, max_length=...` |
| `index out of range in self` / `device-side assert triggered` | 分词器与模型词表不匹配 | 用同一目录的 tokenizer + model |
| `The model does not support ...` / 输出乱码 | 混用了不同模型的分词器 | 同上 |
| 右侧 padding 下批量生成结果异常 | padding_side 不对 | `tok.padding_side="left"` |
| `CUDA out of memory` | 权重+KV+激活超限 | 降 `max_new_tokens`、短上下文、fp16、换小模型 |
| 生成内容重复循环 | 贪心 + 模型偏保守 | `repetition_penalty=1.1`、采样 |
| `You are using the default legacy behaviour ...` | Llama 分词器的普通提示 | 可忽略；`legacy=False` 启用新行为 |
| `incorrect regex pattern ... fix_mistral_regex=True` | 本机 Qwen3-ASR 的 tokenizer_config 问题 | 加载时加 `fix_mistral_regex=True`，或忽略该提示 |
| `attn_implementation="flash_attention_2"` 报错 | 没装 flash-attn | 用默认 `sdpa` |
| `output_attentions=True` 在 sdpa 下无效 | sdpa 不返回注意力矩阵 | 加载时 `attn_implementation="eager"` |
| `KeyError: 'xxx'` 读 config 时 | 字段名因模型而异 | `getattr(cfg, name, default)` |

**排错顺序**：先看最后一行报错 → 再看是网络、token 还是显存 →
最后才怀疑模型本身。

> **第 8 章口诀：报错看末行，网络、词表、显存三连查。**

---

## 第 9 章 小项目：完整的文本分类流水线

目标：用**真实的 Qwen 分词器** + 一个同词表的小模型，跑通
「文本 → token → 模型 → 概率 → 决策」全流程。模型权重是随机的，
所以预测没有意义，但流程和形状完全真实；换成真实模型只需改一行。

### 9.1 五步流程

```
原始文本
   │ tokenizer(padding/truncation)          ← 第 2 章
   ▼
input_ids + attention_mask
   │ model(**batch)                          ← 第 3 章
   ▼
logits (B, num_labels)
   │ softmax                                 ← 第 4 章
   ▼
概率 → argmax → 标签
```

### 9.2 代码（code/08_pipeline.py 核心）

```python
tok = AutoTokenizer.from_pretrained(QWEN_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = BertConfig(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                 num_attention_heads=4, intermediate_size=256, num_labels=2)
model = AutoModelForSequenceClassification.from_config(cfg).to(device).eval()

batch = tok(TEXTS, padding=True, truncation=True, max_length=64,
            return_tensors="pt").to(device)

with torch.inference_mode():
    logits = model(**batch).logits
probs = logits.softmax(-1).cpu()
```

### 9.3 实测输出

```
tok.vocab_size=151643  len(tok)=151705
device=cuda  params=19,766,018
input_ids (3, 8)  attention_mask (3, 8)
logits (3, 2)

分类结果：
  这个产品质量很好，物流也快   -> 正面  (p=0.5308)  分布=[0.4692, 0.5308]
  用了一周就坏了，非常失望     -> 正面  (p=0.5361)  分布=[0.4639, 0.5361]
  一般般，没什么特别的感觉     -> 正面  (p=0.5301)  分布=[0.4699, 0.5301]
```

**怎么读这份输出**：

1. **三条都判成"正面"、概率都≈0.5**——这正是随机权重的表现，模型没有任何
   判断力。真实模型会拉开差距（比如 0.95 / 0.05）。
2. **形状全对**：3 条文本、最长 8 个 token、2 个类别。
3. **`input_ids (3, 8)`**：短句被 padding 到最长那条的长度。

### 9.4 换成真实模型：只改一行

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english").eval()
```

其余代码完全不动。**模型换了，分词器必须一起换**——这是最容易忽略的一条。

### 9.5 一个必须记住的坑：`vocab_size` ≠ `len(tokenizer)`

本次实测正好踩到了：

```
tok.vocab_size=151643
len(tok)=151705
```

差 62 个，是 `tokenizer_config.json` 里新增的特殊 token（`<|im_start|>`、
`<|im_end|>`、`<|endoftext|>` 等）。分词器能产出 id 151704，
但模型 Embedding 只有 151643 行 → **CUDA 直接断言失败**：
`Assertion 'srcIndex < srcSelectDimSize' failed`。

| 用哪个 | 场景 |
| --- | --- |
| `tok.vocab_size` | 通常等于基础词表大小 |
| `len(tok)` | **含特殊 token，建模型时用这个** |
| `model.config.vocab_size` | 以模型为准；两者不一致就是隐患 |

> **口诀：建模型用 len(tok)，加载模型看 config；分词器与权重同源，越界断言马上报。**

---

## 第 10 章 衔接下一本：peft

### 10.1 LoRA 挂载：四行代码

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "v_proj"],
                 task_type="CAUSAL_LM")
peft_model = get_peft_model(base_model, cfg)
peft_model.print_trainable_parameters()
```

### 10.2 实测：秩 r 决定可训练参数

**实测（tiny llama，1,032,272 参数）**：

| r | 可训练参数 | 占比 |
| --- | --- | --- |
| 4 | 512 | 0.05% |
| 8 | 1,024 | 0.10% |
| 32 | 4,096 | 0.40% |

**LoRA 层的真实形状（r=8，hidden=16）**：

```
layers.0.self_attn.q_proj.lora_A.weight  (8, 16)
layers.0.self_attn.q_proj.lora_B.weight  (16, 8)
layers.0.self_attn.v_proj.lora_A.weight  (8, 16)
layers.0.self_attn.v_proj.lora_B.weight  (16, 8)
... 每层 q_proj / v_proj 各一对
```

原权重是 `(16, 16)`，LoRA 只加 `(8,16)` 和 `(16,8)` 两个小矩阵——
这就是「低秩」：`ΔW = B·A`，秩不超过 r。

### 10.3 适配器文件有多小

**实测保存到 `lora_demo/`**：

```
README.md                 5,250 字节
adapter_config.json       1,146 字节
adapter_model.safetensors 5,120 字节
```

**5 KB 的适配器，配上原来的 2 MB 底座模型**。真实场景里，7B 模型的 LoRA 适配器
通常只有 10 ~ 200 MB，而底座是 14 GB——这就是为什么一个底座可以挂几十个
不同任务的适配器。

### 10.4 加载适配器

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(底座路径)
model = PeftModel.from_pretrained(base, "lora_demo")
# 需要部署时合并回一个模型：
model = model.merge_and_unload()
```

### 10.5 下一本该学什么

| 主题 | 包 | 关键问题 |
| --- | --- | --- |
| LoRA 原理与调参 | peft | r、alpha、target_modules 怎么选 |
| 监督微调全流程 | trl / ms-swift | 数据格式、chat template、只对回答算损失 |
| 数据集处理 | datasets | map / filter / 拼接 |
| 多卡与显存 | accelerate | device_map、梯度累积、deepspeed |

> **第 10 章口诀：冻结底座挂低秩，q/v 两处最常用；适配器只存几 MB，合并之后即原样。**

---

## 第 11 章 练习与答案

### 一、概念题

1. `AutoModel` 和 `AutoModelForCausalLM` 的区别是什么？
2. 为什么 `decode(encode(text))` 必须等于 `text`？不满足说明什么？
3. `attention_mask` 里 0 的位置，模型实际做了什么？
4. `max_length` 和 `max_new_tokens` 有什么区别？为什么推荐后者？
5. 为什么批量生成要用左侧 padding？
6. KV cache 缓存的是什么？为什么能加速？
7. `tok.vocab_size` 和 `len(tok)` 为什么可能不相等？
8. `labels` 传进去后模型内部做了什么？

### 二、动手题

1. 用 `code/02_tokenizer.py` 统计你自己的一段中文的 token 数，和字数对比。
2. 把 `code/07_padding_generate.py` 改成 3 条不同长度的 prompt，验证只有左填充
   与单条推理一致。
3. 加载 tiny llama，设置 `output_attentions=True, attn_implementation="eager"`，
   打印第一层注意力矩阵形状。
4. 用 `code/06_generate.py` 对比 `repetition_penalty=1.0` 与 `1.3` 的差异。
5. 把 `code/09_lora_preview.py` 的 `target_modules` 改成 `["q_proj","k_proj","v_proj","o_proj"]`，
   看可训练参数怎么变。
6. 给 `code/08_pipeline.py` 加一个「预测概率低于 0.6 就输出『不确定』」的规则。

### 参考答案要点

1. 前者输出 `last_hidden_state`（特征），后者在底座上加语言模型头，输出
   `logits (B,L,V)` 并支持 generate。
2. 编码必须无损，否则文本被静默破坏；不满足通常是分词器与文本编码不匹配
   （如字节级 BPE 处理非法 UTF-8）。
3. 注意力里屏蔽掉对应位置（权重置为极小），使其不影响结果。
4. `max_length` 限制总长度（prompt + 生成），prompt 长时可能生成 0 个；
   `max_new_tokens` 只限制新生成部分，行为可预测。
5. 生成从最后一个位置继续，左侧填充保证最后一个位置是真实 token，
   且位置编号正确。
6. 缓存每一层历史 token 的 Key 和 Value，生成新 token 时直接复用，
   避免重算整段注意力。
7. 分词器可能动态添加特殊 token，`len(tok)` 包含它们，而 `vocab_size`
   只统计基础词表。
8. 计算损失（分类用交叉熵，因果语言模型用错位交叉熵），结果放在
   `outputs.loss`。

---

## 附录 A API 速查表

### 分词器

| 需求 | API |
| --- | --- |
| 加载 | `AutoTokenizer.from_pretrained(path)` |
| 编码 | `tok(text, padding=True, truncation=True, max_length=N, return_tensors="pt")` |
| 解码 | `tok.decode(ids, skip_special_tokens=True)` / `tok.batch_decode(...)` |
| 查看 token | `tok.convert_ids_to_tokens(ids)` / `convert_tokens_to_ids` |
| 对话模板 | `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)` |
| 设置填充 | `tok.pad_token = tok.eos_token`；`tok.padding_side = "left"` |
| 词表大小 | `len(tok)`（含特殊 token） |

### 模型

| 需求 | API |
| --- | --- |
| 配置 | `AutoConfig.from_pretrained(path)`；`BertConfig(...)` / `LlamaConfig(...)` |
| 加载 | `AutoModel*.from_pretrained(path, dtype=..., device_map="auto")` |
| 构造 | `AutoModel*.from_config(cfg)` |
| 前向 | `model(**batch)` → `outputs.logits` / `.loss` / `.hidden_states` |
| 冻结 | `p.requires_grad = False` |
| 保存 | `model.save_pretrained(dir)`；分词器 `tok.save_pretrained(dir)` |
| 设备 | `model.to(device)`；`next(model.parameters()).device` |

### 生成

| 需求 | API / 参数 |
| --- | --- |
| 基本 | `model.generate(**batch, max_new_tokens=64)` |
| 采样 | `do_sample=True, temperature=0.8, top_p=0.9` |
| beam | `num_beams=4, length_penalty=1.0` |
| 防重复 | `repetition_penalty=1.1` |
| 返回分数 | `return_dict_in_generate=True, output_scores=True` |
| 流式 | `TextStreamer(tok, skip_prompt=True)` / `TextIteratorStreamer` |
| 停止串 | `stop_strings=["\n\n"], tokenizer=tok` |
| 批量解码 | `tok.batch_decode(out.sequences, skip_special_tokens=True)` |

---

## 附录 B 术语中英对照

| 中文 | 英文 | 一句话 |
| --- | --- | --- |
| 分词器 | tokenizer | 文本 ↔ token id 的双向转换器 |
| 词表 | vocabulary | 模型认识的全部 token |
| 字节级 BPE | byte-level BPE | 从字节开始合并的 BPE，永不 OOV |
| 子词 | subword | 介于字符与词之间的切分单位 |
| 特殊 token | special token | pad/eos/bos 等控制符号 |
| 注意力掩码 | attention mask | 标记哪些位置是真实内容 |
| 填充 / 截断 | padding / truncation | 补齐到等长 / 截短到上限 |
| 对话模板 | chat template | 消息列表 → 提示词的 Jinja2 规则 |
| 因果语言模型 | causal LM | 只能看左侧上下文的生成模型 |
| 自回归生成 | autoregressive generation | 一次一个 token 往后接 |
| 贪心解码 | greedy decoding | 每步取概率最大 |
| 采样 | sampling | 按分布随机取 |
| 束搜索 | beam search | 同时保留多条候选路径 |
| KV 缓存 | KV cache | 缓存历史注意力键值以加速 |
| 温度 | temperature | 采样时分布锐化/平滑的系数 |
| 核采样 | top-p / nucleus sampling | 从累积概率 p 以内采样 |
| 词嵌入 | word embedding | token id → 向量 |
| 隐藏状态 | hidden state | 每层输出的向量表示 |
| 适配器 | adapter | 附加在底座上的小型可训练模块 |
| 低秩适配 | LoRA | 用低秩矩阵近似参数增量 |

---

## 结语

transformers 把深度学习的复杂度压缩成了三个对象：**Config 定结构、
Tokenizer 管文本、Model 算张量**，外加一个 `generate` 负责自回归循环。

这本书里所有数字都可以用 `code/` 下的脚本复现。等你把这 9 个脚本的每一行
都看懂了，再去读 transformers 官方文档，会发现大部分内容你已经在形状层面
见过一遍。

下一站：**peft 与 LoRA 微调**——你已经有底座模型、有分词器、知道 loss 怎么来，
剩下的就是「冻结 + 加适配器 + 只训一小部分」。


