# `train_tokenizer.py` 详解：从零训一个 6400 词表的 BPE 分词器

> 对象：`minimind-master/trainer/train_tokenizer.py`（189 行）
> 环境：本机 conda `swift_env`（transformers 4.57.6 / tokenizers 0.22.2）
> 本文所有「实测」都来自本机完整跑通一次（语料：项目自带的两份 README，约 290 KB，955 段），
> 复现脚本见 `train_tokenizer_probe.py`。

---

## 0. 先读文件头那两行警告

```python
# 注：不建议再重复训练tokenizer（"词典"），MiniMind已自带，此脚本仅供学习和参考。
# 基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
```

**为什么「重训词表 = 权重全废」**：

| 事实 | 后果 |
| --- | --- |
| Embedding 是按 token id 索引的 | id 变了，权重对应的词就换了 |
| lm_head 输出维度 = 词表大小 | 词表一变，输出层尺寸就对不上 |
| chat template 依赖特定特殊 token | 模板里的 `<|im_start|>` 找不到 |

所以这个脚本的正确定位是：

- **学**：看清 BPE 训练、特殊 token、chat template 是怎么拼起来的；
- **用**：给一个**全新领域**（比如某种专业符号体系）造词表，并**从头训练模型**；
- **不要**：拿它重训一个词表去加载官方权重。

---

## 1. 全景：一个脚本干三件事

```
get_texts()            读语料，产出纯文本流（兼容两种 jsonl 格式）
      ↓
train_tokenizer()      训 BPE → 保存 tokenizer.json / vocab.json / merges.txt
                       → 手写 tokenizer_config.json（含 chat_template）
      ↓
eval_tokenizer()       用 transformers 加载 → 渲染模板 → 查一致性 → 测压缩率 → 试流式解码
```

**`__main__` 里两行就串起来**：

```python
train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
eval_tokenizer(TOKENIZER_DIR)
```

---

## 2. 五个全局常量

```python
DATA_PATH        = '../dataset/sft_t2t_mini.jsonl'   # 训练语料
TOKENIZER_DIR    = '../model_learn_tokenizer/'       # 输出目录（注意不是 model/）
VOCAB_SIZE       = 6400
SPECIAL_TOKENS_NUM = 36
MAX_LINES        = 0                                 # 0 = 全部；正数 = 只取前 N 条试跑
```

| 常量 | 说明 |
| --- | --- |
| `DATA_PATH` | 默认指向 SFT 数据；也可指向预训练数据（两种格式都支持） |
| `TOKENIZER_DIR` | 刻意输出到 `model_learn_tokenizer/`，**避免覆盖官方 `model/`** |
| `VOCAB_SIZE` | 6400，和 `MiniMindConfig.vocab_size` 必须一致 |
| `SPECIAL_TOKENS_NUM` | 36，特殊 token 的**总数**（不是额外数） |
| `MAX_LINES` | 调试开关：先拿 100 条跑一遍看流程 |

---

## 3. `get_texts`：读语料，兼容两种格式

```python
def get_texts(data_path, max_lines=MAX_LINES):
    used = 0
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if max_lines and used >= max_lines: break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'text' in data:
                text = str(data['text'])
            else:
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                text = "\n".join(contents) if contents else ''
            if text.strip():
                used += 1
                yield text
    if used == 0:
        raise ValueError(f'{data_path} 中没有可用文本：...')
```

**四个设计点**：

| 设计 | 原因 |
| --- | --- |
| 用 `yield` 生成器 | 语料可能几个 GB，不能一次性读进内存 |
| 同时兼容 `{"text"}` 与 `{"conversations"}` | 预训练语料是前者，SFT 语料是后者 |
| 跳过坏行（`JSONDecodeError`） | 单行损坏不该让整个训练失败 |
| `used == 0` 时抛异常 | 见下面的历史教训 |

**注释里那段历史 bug 值得单独讲**：

> 原先只解析 conversations，所以把 DATA_PATH 指向预训练语料时一条文本都取不到，
> BPE 仅基于 initial_alphabet 训练，产出一个没有任何 merge 的词表，**且全程不报错**。

也就是说：语料读空了 → BPE 只拿到 256 个字节初始字符 → 训出一个「只有字节、没有合并」的废词表 →
后面训练模型时也不报错，只是效果极差。**这是最阴的一类 bug：静默失败**。
现在用「取不到文本就抛 ValueError」把问题挡在源头。

**实测**：

```
语料段落数: 955
get_texts 产出条数: 955
```

---

## 4. `train_tokenizer`：六步

### 4.1 建 BPE + ByteLevel 预分词

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

**为什么选 byte-level BPE**（而不是 WordPiece / SentencePiece）：

| 方案 | 未登录词（OOV） | 任意 Unicode | 备注 |
| --- | --- | --- | --- |
| 词表匹配 | ❌ 大量 OOV | 差 | 最老 |
| WordPiece | ❌ 有 `[UNK]` | 一般 | BERT |
| SentencePiece BPE | ❌ 有 `<unk>` | 好 | Llama 常用 |
| **ByteLevel BPE** | **✅ 永不 OOV** | **✅ 任意字节** | GPT-2/Qwen/MiniMind |

原理：先把文本按 UTF-8 拆成字节（0–255），任何字符都能用字节组合表示，
所以**永远不会有「不认识的字符」**。代价是：中文一个常见字可能要 2–3 个 token
（一个汉字 UTF-8 占 3 字节，若没被合并就拆成 3 个字节 token）。

`add_prefix_space=False`：不在文本前自动加空格——中文场景不需要。

### 4.2 特殊 token 分三组，凑够 36 个

```python
special_tokens_list = [                       # 21 个「核心」特殊 token
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
    "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
]
additional_tokens_list = [                    # 6 个「标签」token
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<think>", "</think>"
]
num_buffer = 36 - (21 + 6) = 9
buffer_tokens = ["<|buffer1|>", ..., "<|buffer9|>"]   # 预留
```

**为什么要预留 buffer**：以后要加新标记（比如新的模态、新的工具标记）时，
可以直接用空闲槽位，**不用重训词表、不用重训模型**。

**实测 id 分配**（顺序即 id）：

```
<|endoftext|> 0   <|im_start|> 1   <|im_end|> 2   ...（21 个核心）
<tool_call> 21   </tool_call> 22   <tool_response> 23   ...
<think> 25   </think> 26
<|buffer1|> 27   ...   <|buffer9|> 35
```

**注意**：特殊的 id 0/1/2 被固定成了 pad/bos/eos，这对后续训练和推理很关键。

### 4.3 训练器：从 256 个字节开始合并

```python
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # 256 个字节 token
    special_tokens=all_special_tokens                       # 36 个
)
```

`initial_alphabet` 是 byte-level BPE 的**地基**：先放 256 个字节 token，
再通过「合并高频字节对」往上长到 6400。

**6400 = 256 字节 + 36 特殊 + 约 6108 个合并出来的子词**。

### 4.4 流式训练

```python
texts = get_texts(data_path)
tokenizer.train_from_iterator(texts, trainer=trainer)
```

`train_from_iterator` 接受**迭代器**，边读边统计，不把语料读进内存。

**实测**：955 段、约 290 KB 语料，就把 6400 词表填满了（`len(tokenizer) = 6400`）。
如果语料太小（比如只有几十条），BPE 会「有多少合并算多少」，最终词表少于 6400——
**不会报错，但后面模型配置里的 `vocab_size` 就对不上了**。

### 4.5 保存，然后把「标签 token」的 special 标志翻掉

```python
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)          # vocab.json + merges.txt
```

然后做一件很关键的事：

```python
for token_info in tokenizer_data.get('added_tokens', []):
    if token_info['content'] not in special_tokens_list:
        token_info['special'] = False        # 21 个核心之外的都改成 False
```

**为什么要翻转**：训练器把 36 个 token 全登记成 `special=True`，
但 `<think>` / `<tool_call>` 这些是**内容的一部分**，不应该在解码时被 `skip_special_tokens=True` 删掉。

| token | 期望行为 | 原因 |
| --- | --- | --- |
| `<|im_end|>` | 解码时**丢掉** | 控制符，不该出现在回答里 |
| `<think>` / `</think>` | 解码时**保留** | 是模型输出的一部分 |
| `<tool_call>` / `<tool_response>` | **保留** | 前端要解析 |
| `<|bufferN|>` | 保留 | 将来用 |

**实测验证**：

```
<|im_end|>       id=2      decode后保留=False
<think>          id=25     decode后保留=True
<tool_call>      id=21     decode后保留=True
<|buffer1|>      id=27     decode后保留=True
```

同一个 `special` 标志还要在 `tokenizer_config.json` 的 `added_tokens_decoder` 里同步一遍
（第 96 行），两处保持一致，transformers 加载时才不会打架。

### 4.6 手写 `tokenizer_config.json`

这是全脚本信息量最大的一段。逐字段看：

| 字段 | 值 | 作用 |
| --- | --- | --- |
| `add_bos_token` / `add_eos_token` | False | **不自动加**，由 chat template 负责 |
| `bos_token` | `<|im_start|>` | 对话开始的标记 |
| `eos_token` | `<|im_end|>` | 对话结束 |
| `pad_token` / `unk_token` | `<|endoftext|>` | 复用同一个，省一个 id |
| `additional_special_tokens` | 21 个核心（去掉 endoftext） | 告诉 transformers 哪些是特殊 token |
| `model_max_length` | 131072 | 理论最大长度（实际训练用 32768） |
| `added_tokens_decoder` | 36 个，带 special 标志 | 与 tokenizer.json 同步 |
| `image_token` / `audio_token` / `video_token` | 对应 `<|*_pad|>` | 多模态扩展预留 |
| `chat_template` | 一大段 Jinja2 | 见 4.7 |
| `tokenizer_class` | `PreTrainedTokenizerFast` | 让 transformers 用 fast 实现加载 |

**为什么 pad 和 unk 都指向 `<|endoftext|>`**：ByteLevel BPE 永远不会产生未登录词，
`unk` 只是占位；pad 复用 endoftext 则让「补齐」和「文档结束」共享一个 id，
在追求小词表的场景下是常见做法。

### 4.7 `chat_template`：整块模板逐段拆

这段 Jinja2 模板本质上是 **Qwen2 模板的 MiniMind 定制版**，分五段：

**① tools 分支**（有工具时）

```jinja
{%- if tools %}
    <|im_start|>system
    {系统提示}
    # Tools ... <tools>[{...},{...}]</tools>
    For each function call, return a json object ... <tool_call>{"name":...,"arguments":...}</tool_call>
{%- else %}
```

把工具定义以 JSON 形式塞进 system 消息，并**明确教模型输出格式**——
这就是 Tool Call 能力的来源。

**② 找「最后一条真实用户问题」**

```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- if ns.multi_step_tool and message.role == "user"
           and not(message.content.startswith('<tool_response>') ...) %}
        {%- set ns.last_query_index = index %}
```

从后往前找第一条**不是工具返回**的 user 消息，用来识别多步工具调用场景。

**③ 普通消息**

| role | 渲染 |
| --- | --- |
| user | `<|im_start|>user\n内容<|im_end|>\n` |
| system（非首条） | 同上，按 user 处理 |
| assistant | 见 ④ |
| tool | 见 ⑤ |

**④ assistant 的「思考 + 回答」拆分**

```jinja
{%- if '</think>' in content %}
    reasoning_content = content.split('</think>')[0].split('<think>')[-1]
    content = content.split('</think>')[-1]
{%- endif %}
<|im_start|>assistant
<think>{{ reasoning_content }}</think>

{{ content }}
```

**这是 minimind-3「自适应思考」的核心**：无论数据里有没有思考内容，
模板**总是**输出一个 `<think>` 块（空的就是空块）。这样模型在训练时
同时见过「有思考」和「空思考」两种形态，推理时就能靠 `open_thinking` 开关切换。

**⑤ tool 消息合并**

```jinja
{%- if loop.first or messages[i-1].role != "tool" %}<|im_start|>user{% endif %}
\n<tool_response>\n{{ content }}\n</tool_response>
{%- if loop.last or messages[i+1].role != "tool" %}<|im_end|>\n{% endif %}
```

连续多条 tool 结果会被**合并进同一个 user 回合**，避免每个工具结果都占一个完整回合。

**⑥ 生成提示**

```jinja
{%- if add_generation_prompt %}
    <|im_start|>assistant\n
    {%- if open_thinking %}<think>\n{%- else %}<think>\n\n</think>\n\n{%- endif %}
{%- endif %}
```

- 训练/预览：`add_generation_prompt=False`，不追加；
- 推理：`True`，追加 assistant 开头；
- `open_thinking=True` 时让模型从 `<think>` 内开始写，`False` 时先给一个**空的完整思考块**，
  逼模型直接回答。

---

## 5. `eval_tokenizer`：四项检查

### 5.1 模板渲染

```python
new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

**实测渲染结果**：

```
<|im_start|>system
你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
<|im_start|>user
你来自哪里？<|im_end|>
<|im_start|>assistant
<think>
</think>
我来自月球<|im_end|>
...
```

注意 assistant 的**空思考块**被原样渲染出来——和 4.7 的设计对应。

### 5.2 编解码一致性

```python
response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
print('decoder一致性：', response == new_prompt)
```

**实测**：`decoder一致性： True`，编码长度 76。

**为什么这是最重要的检查**：如果 byte-level 的 decoder 配置错了（比如少了 `decoders.ByteLevel()`），
空格、换行、中文都会在解码时变形，而训练 loss 看起来完全正常——
**这是分词器最隐蔽的 bug**。

### 5.3 压缩率：字符数 / token 数

**实测（本机，用 README 语料训练的词表）**：

| 样本 | 类型 | 字符数 | Tokens | 压缩率 |
| --- | --- | --- | --- | --- |
| 1 | 中文（通用） | 206 | 149 | 1.38 |
| 2 | 中文（科幻） | 185 | 187 | **0.99** |
| 3 | 英文 | 573 | 161 | 3.56 |
| 4 | 英文 | 533 | 194 | 2.75 |
| 5 | 中英混合 | 269 | 116 | 2.32 |
| — | **平均** | | | **2.20** |

**怎么读这张表**：

- 英文约 3–4 字符/token，符合预期；
- 中文只有 1.0–1.4，**明显偏低**——因为我的语料是 README（技术+代码），
  中文合并规律没学到位，生僻字被拆成 2–3 个字节 token；
- 样本 2 的 0.99 意味着「一个字要一个多 token」——**比一个字一个 token 还差**；
- 官方 tokenizer 在真实 SFT 语料上训练，README 提到中文约 1.5–1.7 字符/token。

**结论**：压缩率不只由分词器决定，**更由「训练语料与目标文本的相似度」决定**。
这也是为什么这个脚本不建议用来重训。

### 5.4 流式解码与 `�`

```python
for tid in input_ids:
    token_cache.append(tid)
    current_decode = tokenizer.decode(token_cache)
    if current_decode and '�' not in current_decode:
        print(...); token_cache = []
```

**为什么需要缓冲**：byte-level 的一个 token 可能只是**半个汉字**的 UTF-8 字节。
单独解码会得到 `�`（U+FFFD 替换字符）；必须等后续 token 补齐，才能输出完整字符。

**实测片段**：

```
Token ID: [548, 258]        -> Raw: ['ç§', 'Ģ']           -> Decode Str: 秀
Token ID: [1266, 259, 5738] -> Raw: ['çļĦè', 'ģ', 'Ĭå¤©'] -> Decode Str: 的聊天
```

三个 token 才拼出「的聊天」——这就是流式服务端必须做字节缓冲的原因。

---

## 6. 关键设计点汇总

| 设计 | 一句话理由 |
| --- | --- |
| byte-level BPE | 永不 OOV，任何 Unicode 都能编码 |
| 词表 6400 | 小模型省 Embedding 参数（4.9M，占 7.7%） |
| 预留 9 个 buffer | 将来加标记不用重训 |
| 翻转 tag 的 special 标志 | `<think>`/`<tool_call>` 是内容，不是控制符 |
| pad = unk = endoftext | 省 id，小词表场景常见 |
| add_bos/eos = False | 由 chat template 统一负责 |
| 模板总是输出 `<think>` 块 | 支持「自适应思考」 |
| 独立输出目录 | 防止覆盖官方 `model/` |

---

## 7. 坑与注意事项

| 坑 | 说明 |
| --- | --- |
| **重训词表后权重作废** | 文件头已警告；官方权重只能配官方词表 |
| **语料太少 → 词表不满 6400** | BPE 不报错，只是没长到目标大小；后续 `config.vocab_size` 会不一致 |
| **`errors='ignore'` 静默丢字符** | 非 UTF-8 文件会被悄悄截断，不报错 |
| **词表大小必须与模型一致** | `MiniMindConfig(vocab_size=6400)` 与此处的 `VOCAB_SIZE` 必须相等 |
| **模板假设 `messages[0]` 存在** | 空对话会渲染异常 |
| **`<|im_end|>` 被 unk 复用为 pad** | 做 padding 时要注意 mask 掉，否则模型会学「预测结束符」 |
| **压缩率可能 < 1** | 见样本 2；说明词表与该领域文本不匹配 |

---

## 8. 自测题

1. 为什么用 byte-level BPE 而不用 WordPiece？
2. `initial_alphabet` 放的是什么？为什么是 256 个？
3. 6400 词表由哪几部分组成？
4. 为什么要预留 buffer token？
5. `<think>` 为什么设置成 `special=False`？
6. `pad` 和 `unk` 为什么可以共用一个 id？
7. `add_bos_token=False` 之后，BOS 由谁负责？
8. 模板为什么总是输出 `<think>` 块（哪怕是空的）？
9. `decoder一致性：True` 说明什么？不 True 会怎样？
10. 压缩率 0.99 意味着什么？
11. 流式解码为什么要等 `�` 消失？
12. 语料不够会怎样？为什么危险？

### 答案要点

1. 永不 OOV，任意 Unicode 可编码；代价是中文可能多 token。
2. 256 个字节 token，作为 BPE 合并的地基。
3. 256 字节 + 36 特殊 + 约 6108 个合并子词。
4. 以后加新标记可直接用，不必重训词表与模型。
5. 它是模型输出内容的一部分，`skip_special_tokens=True` 时不应被删除。
6. Byte-level 不会产生未登录词，unk 仅占位；小词表省 id。
7. chat template（模板里显式写 `<|im_start|>`）。
8. 训练时同时覆盖「有思考 / 空思考」，推理用开关切换。
9. 编码解码可逆；不 True 说明 decoder 配置错，文本会被破坏而 loss 正常。
10. 平均一个字要 1.01 个 token，压缩效率低于「一字一 token」。
11. token 可能只包含半个字符的字节，提前解码会出现替换字符。
12. 词表长不满 6400，但不报错；与模型配置不一致，后续会形状错误或效果异常。

---

## 9. 一表速查

| 部分 | 输入 | 输出 | 关键点 |
| --- | --- | --- | --- |
| `get_texts` | jsonl | 文本生成器 | 兼容 text/conversations，空语料抛错 |
| `train_tokenizer` | 语料 + 目录 + 词表大小 | tokenizer 目录 | BPE + ByteLevel + 36 特殊 + 模板 |
| `eval_tokenizer` | tokenizer 目录 | 终端报告 | 一致性、压缩率、流式解码 |
