# `train_lora.py` 详解：只训练 0.6% 的参数

> 对象：`minimind-master/trainer/train_lora.py`（180 行）
> 配套：`model/model_lora.py`（70 行，LoRA 全部实现）、`SFTDataset`
> 说明：本文按源码讲解，不跑训练。

---

## 0. 定位：SFT 的「低成本替代版」

| 项 | 全参 SFT | LoRA |
| --- | --- | --- |
| 训练参数 | 63.91M（100%） | **约 0.39M（0.62%）** |
| 优化器状态 | 全模型 | 只有适配器 |
| 产出文件 | `full_sft_768.pth`（~128MB） | `lora_xxx_768.pth`（**~0.8MB**） |
| 学习率 | 1e-5 | **1e-4**（大 10 倍） |
| 训练轮数 | 2 | **10** |
| 数据 | 通用 SFT | 领域数据（医疗/自我认知） |

**它和 SFT 共用同一个 Dataset、同一套 mask、同一个训练循环**，
差别只在「哪些参数参与优化」和「保存什么」。

> 口诀：**冻结底座，旁挂低秩；A 随机 B 归零，只存适配器。**

---

## 1. LoRA 的三行数学

```
原始：  y = Wx
LoRA： y = Wx + (B·A·scaling)x     其中 A ∈ R^{r×d}, B ∈ R^{d×r}, r ≪ d
```

| 矩阵 | 形状（d=768, r=16） | 初始化 |
| --- | --- | --- |
| A | 16 × 768 | 高斯 `N(0, 0.02²)` |
| B | 768 × 16 | **全零** |
| scaling | 标量 | `alpha / r` |

**B 初始化为零的意义**：训练第 0 步时 `B·A = 0`，LoRA 对原模型**零扰动**。
模型从「和底座完全一样」的状态开始学，不会先被随机噪声破坏。
如果 A、B 都随机初始化，训练前期会花时间「修复噪声」。

**参数量计算**（本模型）：

```
每个方阵模块：768×16 + 16×768 = 24,576
注入模块数：q_proj + o_proj = 2 × 8 层 = 16
总计：24,576 × 16 = 393,216 ≈ 0.39M
占 63.91M 的 0.62%
```

---

## 2. `model_lora.py` 逐个拆

### 2.1 `class LoRA`（:6）

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.register_buffer("scaling", torch.tensor(scaling, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(0.0, 0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling
```

| 细节 | 说明 |
| --- | --- |
| `scaling` 用 buffer | 会进 state_dict（保存/加载时带上），但不参与梯度 |
| `lora_alpha=None` | 默认 scaling = 1（alpha 等于 rank） |
| 两个无 bias 的 Linear | A 降维、B 升维 |

### 2.2 `apply_lora`（:23）：注入 + 猴补丁

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and module.in_features == module.out_features:
        lora = LoRA(...).to(model.device)
        setattr(module, "lora", lora)                      # ① 注册成子模块
        original_forward = module.forward
        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)                   # ② 原输出 + 增量
        module.forward = forward_with_lora                 # ③ 猴补丁
```

**三个关键点**：

| 点 | 为什么 |
| --- | --- |
| `setattr(module, "lora", lora)` | 让 LoRA 成为已注册子模块，`.parameters()` / `state_dict()` 才能看到它 |
| 闭包默认参数 `layer1=..., layer2=...` | **早绑定**。如果直接引用外层变量，所有模块会共享最后一次循环的值（经典 Python 闭包陷阱） |
| `module.forward = ...` | 不用改模型源码，也不新增模块层级，模型其他部分完全无感 |

**代价**：`forward` 不再是 `nn.Module` 的标准方法，`torch.compile` 无法追踪——
所以 `train_lora.py` 里会强制关掉 compile（见 3.4）。

### 2.3 `load_lora`（:37）

```python
state_dict = torch.load(path, map_location=model.device)
state_dict = {(k[7:] if k.startswith('module.') else k): v ...}
for name, module in model.named_modules():
    if hasattr(module, 'lora'):
        lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
        module.lora.load_state_dict(lora_state, strict=False)
```

- 去掉 DDP 的 `module.` 前缀；
- 按模块名前缀把 key 分发到各适配器；
- `strict=False`：允许部分缺失（**但形状不匹配仍会报错**）。

### 2.4 `save_lora`（:47）：只存适配器

```python
lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() if k.endswith('.weight')
              else v.cpu().float() for k, v in module.lora.state_dict().items()}
```

| 细节 | 说明 |
| --- | --- |
| 只遍历 `hasattr(module,'lora')` 的模块 | 底座权重一个字节都不存 |
| 权重转 half，buffer（scaling）转 float | 权重是主体，scaling 只需一个标量 |
| 体积约 0.8 MB | 相当于底座 128MB 的 0.6% |

### 2.5 `merge_lora`（:59）：把增量「烤」回原权重

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and '.lora.' not in name:
        delta = (module.lora.B.weight.float() @ module.lora.A.weight.float()) * module.lora.scaling.float()
        state_dict[f'{name}.weight'] += delta
```

`ΔW = B·A·scaling`，加到原权重上，导出完整模型。**推理时零额外开销**。

---

## 3. `train_lora.py` 逐段

### 3.1 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `lora_rank` | 16 | 低秩维数 |
| `lora_alpha` | None | 默认等于 rank（scaling=1） |
| `lora_name` | `lora_medical` | 适配器文件名前缀 |
| `epochs` | **10** | 领域数据少，多轮 |
| `batch_size` | 32 | 底座冻结，显存压力小 |
| `learning_rate` | **1e-4** | 比 SFT 大 10 倍 |
| `max_seq_len` | 340 | 领域问答较短 |
| `data_path` | `lora_medical.jsonl` | |
| `from_weight` | `full_sft` | 在 SFT 基础上加技能 |

### 3.2 注入 + 冻结 + 统计

```python
model, tokenizer = init_model(lm_config, args.from_weight)
apply_lora(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)

total = sum(p.numel() for p in model.parameters())
lora_count = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
Logger(f"LoRA 参数占比: {lora_count / total * 100:.2f}%")

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False

optimizer = optim.AdamW(lora_params, lr=args.learning_rate)     # 只收适配器
```

**三件事缺一不可**：注入 → 冻结非 LoRA → 优化器只收 LoRA。
如果忘了冻结，底座参数也会被更新（虽然不在优化器里就不会更新，
但显式冻结能避免梯度计算浪费显存）。

### 3.3 训练循环：与 SFT 相同，只有两处变化

```python
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0 or step == iters:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)   # ← 只裁 LoRA
    scaler.step(optimizer)
    ...
```

| 变化 | 原因 |
| --- | --- |
| `clip_grad_norm_(lora_params, ...)` | 其他参数没有梯度，传全体会浪费时间 |
| 保存改为 `save_lora(...)` | 只存适配器 |

数据侧完全复用 `SFTDataset`：同样只对 assistant 区间算损失。

### 3.4 强制关闭 compile

```python
if args.use_compile == 1:
    args.use_compile = 0
    Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
```

**这是很贴心的处理**：猴补丁的 `forward` 是普通 Python 函数，inductor 无法追踪；
与其报错，不如直接关掉并提示。

### 3.5 保存与续训

```python
save_lora(model, lora_save_path)          # out/lora_medical_768.pth
lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, ...)
```

**注意顺序**：先存适配器，再存 resume。resume 里包含完整 state_dict（含 LoRA 参数），
所以「续训」和「分发适配器」是两件事。

### 3.6 恢复

```python
model.load_state_dict(ckp_data['model'], strict=False)   # ← strict=False
optimizer.load_state_dict(ckp_data['optimizer'])
```

**为什么 `strict=False`**：续训时模型是刚注入 LoRA 的结构，与保存时结构一致，
但如果配置里 `lora_rank` 改了，键就少/多了——用 `strict=False` 至少能加载上主干部分。
（真正换 rank 续训是错误用法，形状不匹配仍会报错。）

---

## 4. 与 SFT 的 diff

| 项 | full_sft | lora |
| --- | --- | --- |
| 训练参数 | 全部 | 仅适配器 |
| lr | 1e-5 | 1e-4 |
| epochs | 2 | 10 |
| 保存 | 完整权重 | 适配器 |
| compile | 可用 | 强制关闭 |
| clip 范围 | 全模型 | 仅 LoRA |
| 数据 | 通用 SFT | 领域数据 |

**其余（Dataset、mask、五步循环、断点、DDP）完全一致。**

---

## 5. 训练完之后怎么用

### 5.1 直接推理（不合并）

```bash
python eval_llm.py --load_from model --weight full_sft \
                   --lora_weight lora_medical --lora_rank 16
```

`eval_llm.py` 会：加载底座 → `apply_lora(rank)` → `load_lora(适配器)` → 推理。

**注意**：`--lora_rank` 必须与训练时一致（默认 16）。

### 5.2 合并成完整权重

```python
from model.model_lora import apply_lora, merge_lora
model = ...                      # 先建好底座并 apply_lora(rank)
merge_lora(model, "out/lora_medical_768.pth", "out/merged_768.pth")
```

或走 `scripts/convert_model.py`，一步导出成 transformers 格式。

### 5.3 部署时的取舍

| 方式 | 优点 | 缺点 |
| --- | --- | --- |
| 带着适配器 | 一个底座 + 多个小适配器 | 推理要多加载一次 |
| 合并 | 推理零开销、可转 GGUF/vLLM | 每个任务一份完整权重 |

---

## 6. 坑

| 坑 | 现象 | 原因 / 处理 |
| --- | --- | --- |
| **rank 不一致** | 加载报形状错误，或效果全无 | 训练/推理/合并三处的 rank 必须一致 |
| **只注入方阵** | 换配置后 LoRA 没加到 q_proj | 条件 `in_features == out_features`；若 `head_dim ≠ hidden/heads`，q_proj 不再是方阵 |
| compile 开启 | 报错或静默无效 | 脚本已自动关闭；别手改回来 |
| 优化器收了全模型 | 显存暴涨 | 必须只传 `lora_params` |
| 忘记冻结 | 梯度计算浪费显存 | 显式 `requires_grad=False` |
| 合并时忘了 apply_lora | `module.lora` 不存在 | `merge_lora` 前先注入 |
| `save_lora` 后改名 | 加载时 key 对不上 | 前缀名与模块名绑定 |
| 多领域适配器 | 不能同时加载两个 | 需要分别合并，或用多适配器方案（peft 支持） |

**「只注入方阵」这条最隐蔽**：MiniMind 默认 `head_dim = hidden/heads = 96`，
所以 q_proj/o_proj 是 768→768 的方阵，正好被注入。
如果哪天把 `head_dim` 解耦（像 Qwen3 那样），`q_proj` 变成 768→1024，
`in_features != out_features`，**LoRA 会静默地不注入 q_proj**——训练照跑，效果变差。

---

## 7. 显存与本地可行性

| 项 | 说明 |
| --- | --- |
| 底座权重 | 冻结，但仍需存放（fp16 ~128MB） |
| 底座梯度 | **不产生**（冻结） |
| 适配器 + 优化器状态 | 0.39M × 16B ≈ 6 MB |
| 激活 | 主要开销（batch 32 × seq 340） |

**结论**：比全参 SFT 省得多。4GB 卡上把 `batch_size` 降到 8~16 即可跑，
这是本项目**最适合本机体验**的训练脚本。

---

## 8. 自测题

1. LoRA 的参数量怎么算？本模型是多少？
2. 为什么 B 初始化为 0？
3. `scaling` 是什么？`lora_alpha=None` 时等于多少？
4. 为什么只给方阵 Linear 加 LoRA？
5. 猴补丁里为什么要用默认参数 `layer1=..., layer2=...`？
6. 为什么优化器只传 `lora_params`？
7. 为什么 clip 也只作用在 LoRA 参数上？
8. 训练输出为什么只有 0.8MB？
9. 加载适配器时 rank 为什么必须一致？
10. 什么时候需要 merge？
11. compile 为什么必须关？
12. `strict=False` 在这里允许什么、不允许什么？

### 答案要点

1. `(d×r + r×d) × 模块数 = 24,576 × 16 ≈ 0.39M`，占 0.62%。
2. 保证初始时对原模型零扰动。
3. `alpha / rank`；alpha 为 None 时等于 rank，即 scaling=1。
4. 保证输入输出维度一致，增量才能直接相加。
5. 早绑定，避免闭包捕获循环变量。
6. 冻结参数没梯度，收进来只会浪费显存。
7. 其他参数没有梯度。
8. 只保存 A/B 权重（约 0.39M 参数）。
9. A/B 的形状由 rank 决定，不一致就加载不进。
10. 部署时想零额外开销，或要转 GGUF/vLLM。
11. 猴补丁的 forward 无法被 inductor 追踪。
12. 允许键缺失/多余；不允许形状不匹配。

---

## 9. 一表速查

| 段落 | 关键点 |
| --- | --- |
| LoRA 结构 | A(降维) + B(升维) + scaling |
| 初始化 | A 高斯、B 零 |
| 注入 | 仅 `in == out` 的 Linear（q_proj / o_proj） |
| 猴补丁 | 闭包默认参数早绑定 |
| 训练 | 冻结底座、只优化适配器、只裁适配器 |
| 保存 | 只存适配器（~0.8MB） |
| 合并 | `ΔW = B@A*scaling` 加回原权重 |
| 学习率 | 1e-4（比 SFT 大 10 倍） |
| 轮数 | 10（数据少） |
| 本机 | 4GB 可跑，降 batch 即可 |
