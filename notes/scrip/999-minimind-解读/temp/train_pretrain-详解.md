# `train_pretrain.py` 详解：最小可用的预训练闭环

> 对象：`minimind-master/trainer/train_pretrain.py`（165 行）
> 环境：本机 conda `swift_env`（torch 2.6.0+cu124 / GTX 1650 4GB）
> 说明：仓库未附带数据集，本文的训练耗时与显存数字来自「同配置模拟步」的一次性实测，
> 模型结构、超参、循环逻辑均按源码原文。

---

## 0. 定位：整条训练链的第一站

MiniMind 有 8 个训练脚本，**`train_pretrain.py` 是其中最短、也最干净的一个**：
没有 loss mask、没有参考模型、没有奖励模型。后面所有脚本都是它的「加法」。

| 脚本 | 相对 pretrain 多了什么 |
| --- | --- |
| `train_full_sft` | 换 Dataset（只对回答算 loss）+ 降学习率 |
| `train_lora` | 注入 LoRA、只优化适配器 |
| `train_distillation` | 多一个教师模型 + KL 损失 |
| `train_dpo` | 多一个参考模型 + 偏好对 |
| `train_ppo` / `train_grpo` | 采样 + 奖励 + 优势估计 |
| `train_agent` | 多轮工具调用 + 规则奖励 |

**所以读它的价值是**：把「一个训练脚本骨架」刻进脑子，后面每加一个算法，
你只需要盯「损失函数那十几行」。

---

## 1. 文件结构：一个函数 + 一个 main

```python
train_epoch(epoch, loader, iters, start_step=0, wandb=None)   # 165 行里的第 24~72 行
if __name__ == "__main__":
    ... 9 段初始化 ... 然后 for epoch: train_epoch(...)
```

顶层还有三行「脚手架」：

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
```

| 行 | 作用 |
| --- | --- |
| `__package__ = "trainer"` | 让脚本以「包内模块」身份运行，相对导入才不报错 |
| `sys.path.append(...)` | 把项目根目录加进搜索路径，才能 `from model.xxx import` |
| `import datasets` 放在 torch 前 | Windows 下 pyarrow 与 torch 的 DLL 顺序冲突，先导入 pyarrow 系可绕开 |

最后一行是**平台特定 workaround**，注释里甚至带上了 issue 编号——说明是踩过的真实坑。

---

## 2. 参数表：每一个默认值都有理由

| 参数 | 默认 | 含义 | 为什么是这个值 |
| --- | --- | --- | --- |
| `save_dir` | `../out` | 权重目录 | 相对 trainer/ 运行 |
| `save_weight` | `pretrain` | 文件名前缀 | 产出 `pretrain_768.pth` |
| `epochs` | 2 | 轮数 | mini 数据集 2 轮足够冒烟 |
| `batch_size` | **32** | 每卡微批 | 3090 24GB 下的甜点 |
| `learning_rate` | **5e-4** | 初始 lr | 从零训练要大步 |
| `warmup_ratio` | 0.0 | 预热占比 | 默认关，小模型可省 |
| `dtype` | bfloat16 | 混合精度 | 会被 `resolve_dtype` 降级 |
| `accumulation_steps` | **8** | 梯度累积 | 等效 batch 32×8=256 |
| `grad_clip` | 1.0 | 梯度裁剪 | 防早期爆炸 |
| `max_seq_len` | **340** | 序列长度 | 中文 1 token ≈ 1.5~1.7 字 |
| `hidden_size` / `num_hidden_layers` | 768 / 8 | 模型规模 | 对应 64M |
| `use_moe` | 0 | 是否 MoE | 默认 Dense |
| `data_path` | `pretrain_t2t_mini.jsonl` | 语料 | 需自行下载 |
| `from_weight` | `none` | 初始权重 | 预训练当然从随机开始 |
| `from_resume` | 0 | 断点续训 | 见第 5 节 |
| `use_compile` | 0 | torch.compile | Windows 上开了会失败 |
| `log_interval` / `save_interval` | 100 / 1000 | 打印/保存间隔 | |

**两个最容易被忽略的默认值**：

- `batch_size=32` 是按 3090 定的，**在 4GB 卡上必 OOM**（见第 7 节实测）；
- `max_seq_len=340` 不是随便取的，注释里写了换算：340 token ≈ 500~600 个汉字。

---

## 3. `main` 的九段：顺序本身就是知识

```python
# 1. 分布式 + 种子
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

# 2. 目录 + 配置 + 探测断点
os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(hidden_size=..., num_hidden_layers=..., use_moe=...)
ckp_data = lm_checkpoint(...) if args.from_resume == 1 else None

# 3. 精度
args.dtype = resolve_dtype(args.dtype, args.device)
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

# 4. 日志
wandb = None
if args.use_wandb and is_main_process():
    import swanlab as wandb
    wandb.init(..., id=wandb_id, resume=...)

# 5. 模型 / 数据 / 优化器
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

# 6. 恢复状态
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    if ckp_data.get('scaler'): scaler.load_state_dict(ckp_data['scaler'])
    start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)

# 7. 编译 + DDP
if args.use_compile == 1: model = torch.compile(model)
if dist.is_initialized(): model = DistributedDataParallel(model, device_ids=[local_rank])

# 8. 训练
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
    train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)

# 9. 收尾
if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()
```

**为什么是这个顺序**（每一处顺序错了都会出问题）：

| 顺序 | 如果错了 |
| --- | --- |
| 先起分布式，再设种子 | 每张卡的随机种子会相同，数据增强/初始化失去差异性 |
| 先探测断点，再建模型 | 需要知道 `lm_config` 才能拼出检查点文件名 |
| 先解析 dtype，再建 autocast | 会用错精度 |
| 先建模型，再恢复状态 | 没模型就没处加载 |
| `torch.compile` 在 DDP 之前 | 先 DDP 后 compile 会包装错层级 |
| `setup_seed(seed + epoch)` 在 `randperm` 前 | 每轮的打乱顺序就不可复现 |

### 3.1 第 8 段里那个 `train_sampler or indices`

```python
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

- 多卡：`train_sampler` 是 `DistributedSampler`，负责把数据切给各卡；
- 单卡：`indices` 是 `randperm` 的结果，自己打乱。

**两者二选一**，都交给 `SkipBatchSampler` 再切成 batch。这是全脚本最巧的一处「单卡/多卡统一」。

---

## 4. `train_epoch` 逐行

### 4.1 学习率：每步手写进 param_group

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

**为什么不用 `torch.optim.lr_scheduler`**：

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| `scheduler.step()` | 现成、支持多种曲线 | 需要与优化器状态一起存取、顺序易错 |
| **手写覆盖 `param_group['lr']`** | 无状态、断点续训时天然正确 | 需要自己实现曲线 |

断点续训时，手写方案只要知道「全局第几步」就能算出 lr，**不需要保存 scheduler 状态**。
代价是曲线只有一种实现（见 `trainer_utils.get_lr`）。

### 4.2 前向与损失：除以累积步数

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss      # MoE 的负载均衡损失；Dense 时为 0
    loss = loss / args.accumulation_steps
```

**为什么要除**：梯度会累积，不除等于把学习率放大 8 倍。

**为什么要把 `aux_loss` 加上**：MoE 的路由均衡损失必须参与反向，否则路由会坍缩。
源码里 `MiniMindModel` 对 Dense 返回 0 维零张量，所以这行对两种架构都成立。

### 4.3 反向与更新：五个动作的顺序不能乱

```python
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0 or step == iters:
    scaler.unscale_(optimizer)          # ① 先把梯度除以 scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # ② 再裁剪
    scaler.step(optimizer)              # ③ 内部检查 inf/nan 后才真正更新
    scaler.update()                     # ④ 调整 scale 系数
    optimizer.zero_grad(set_to_none=True)  # ⑤ 清梯度
```

| 步骤 | 为什么必须在这个位置 |
| --- | --- |
| `unscale_` 必须在 clip 前 | 否则裁的是「被放大的梯度」，阈值失效 |
| `scaler.step` 而不是 `optimizer.step` | scale 太大产生 inf 时跳过该步，避免污染权重 |
| `update` 在 `step` 后 | 根据是否溢出调整下一次的 scale |
| `zero_grad` 在最后 | 清得太早会把累积的梯度也清掉 |

**`step % accumulation_steps == 0 or step == iters`**：正常情况下每累积 8 步更新一次；
`or step == iters` 保证**最后不足 8 步的部分也会被应用**，否则尾部梯度白算。
（`iters` 传入的是 `len(loader) + skip`，所以带续训时也成立。）

**一个续训细节**：`enumerate(loader, start=start_step + 1)` 让 step 从断点继续，
所以 `step % 8` 的相位也延续——第一次更新可能发生在恢复后的第 3 个微批，
这不影响正确性（梯度从零开始累积）。

### 4.4 日志：三个数字的拆解

```python
current_loss = loss.item() * args.accumulation_steps
current_aux_loss = res.aux_loss.item()
current_logits_loss = current_loss - current_aux_loss
```

| 字段 | 含义 |
| --- | --- |
| `loss` | 总损失 = 语言模型损失 + 均衡损失 |
| `logits_loss` | 纯语言模型损失（能直接和 ln(vocab) 比较） |
| `aux_loss` | MoE 路由均衡损失（Dense 恒为 0） |

**为什么要乘回来**：`loss` 在前面被除以累积步数了，`loss.item()` 拿到的是「微型损失」，
乘回去才是这一批的真实值。

**ETA 公式**：

```python
eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
```

用「已用时间 / 已走步数」估每步耗时，再乘剩余步数。冷启动阶段前几步会偏慢，所以界面上
第一个 ETA 往往偏大。

### 4.5 保存：半精度 + 双文件

```python
model.eval()
ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
raw_model = model.module if isinstance(model, DistributedDataParallel) else model
raw_model = getattr(raw_model, '_orig_mod', raw_model)
state_dict = raw_model.state_dict()
torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
              scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
model.train()
```

| 动作 | 原因 |
| --- | --- |
| 脱 DDP / compile 壳 | 否则键名带 `module.` / `_orig_mod.` |
| `.half()` | 推理权重只需 fp16 |
| 再存一份 resume | 包含优化器、scaler、epoch/step，能续训 |
| `model.eval()` 后 `model.train()` | 保存期间关 dropout，保存完恢复 |

**文件名不带 step**：每 `save_interval` 步覆盖同一个文件，**只保留最新一份**。
这是刻意的取舍——省磁盘，代价是「想要中间某一步」拿不到。

### 4.6 最后两行 `del`

```python
del input_ids, labels, res, loss
```

**作用**：及时释放这批张量的引用，让显存尽早回收。
注意它**不会立刻释放计算图**——`res` 里还挂着图，但循环下一轮重新赋值后引用归零，
`del` 只是把「这一轮结束」提前了一点。在 4GB 卡上这个习惯有意义。

---

## 5. 数据与损失是怎么配合的

### 5.1 `PretrainDataset` 做了什么

```python
tokens = tokenizer(text, add_special_tokens=False, max_length=max_length - 2, truncation=True).input_ids
tokens = [BOS] + tokens + [EOS]
input_ids = tokens + [PAD] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == PAD] = -100
```

| 动作 | 说明 |
| --- | --- |
| `max_length - 2` | 给 BOS/EOS 留位 |
| 加 BOS/EOS | 模型要学会「何时开始、何时结束」 |
| `labels = input_ids` | 预训练是**自监督**：整段都是监督信号 |
| pad → `-100` | 不学「预测 padding」 |

### 5.2 模型内部的错位

`MiniMindForCausalLM.forward` 里：

```python
loss = F.cross_entropy(logits[..., :-1, :].view(-1, V), labels[..., 1:].view(-1), ignore_index=-100)
```

**数据不动，模型错位**：位置 0 的预测对位置 1 的标签，依此类推。
所以数据集里 `labels` 与 `input_ids` 完全相同，不需要手动移位。

### 5.3 为什么要看 `logits_loss` 而不是总 loss

随机初始化时，语言模型损失的期望值是 `ln(vocab_size) = ln(6400) ≈ 8.76`。
训练开始后这个数应该**从 8.76 往下走**；如果它不降，说明学习率、数据或代码有问题。
`aux_loss` 只对 MoE 有意义，混在一起看会干扰判断。

---

## 6. 断点续训：三个组件配合

| 组件 | 作用 |
| --- | --- |
| `lm_checkpoint` 存的 `step` | 知道「训到哪了」 |
| `SkipBatchSampler` | 跳过前面已训的 batch |
| `setup_seed(seed + epoch)` + `randperm` | 保证同一 epoch 的数据顺序可复现 |

**换卡数时的近似**：`step × saved_world / current_world`，再跳过对应批次。
数据顺序并非严格等价，但比从头再来好得多。

**注意**：`skip` 只在**恢复时的那一个 epoch** 生效：

```python
skip = start_step if (epoch == start_epoch and start_step > 0) else 0
```

后续 epoch 从头开始，这是正确的——每个 epoch 本来就该看全量数据。

---

## 7. 本机实测：4GB 卡到底能不能跑

### 7.1 初始 loss 校验

用随机权重 + 随机 token 单次前向：

```
loss = 8.873   (ln(6400) = 8.764)
```

符合「均匀猜」的预期。这是判断预训练是否真的在学的第一把尺子。

### 7.2 每步耗时与显存（同配置模拟，seq=340，lr=5e-4）

| batch | 精度 | ms/step | 峰值显存 |
| --- | --- | --- | --- |
| 4 | fp16 | 1814 | 1559 MB |
| 4 | fp32 | **395** | 1766 MB |
| 8 | fp16 | 3564 | 2202 MB |
| 8 | fp32 | **719** | 2735 MB |

**三个结论**：

1. **`resolve_dtype` 会把你的卡降成 fp16，而 fp16 在这里比 fp32 慢 4.6 倍**——
   和 PyTorch 教材里测到的「Turing 卡 fp16 无加速」完全一致。想快就得手动改一行：

   ```python
   # train_pretrain.py:115 附近
   dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
   # 改成显式 fp32 分支，或者：
   # 直接用 fp32 时把 autocast_ctx 设为 nullcontext()
   ```

2. **batch 8 fp32 已占 2.7 GB**，batch 32（脚本默认）会突破 4GB 直接 OOM。
   本地必须改成：

   ```bash
   python train_pretrain.py --batch_size 4 --accumulation_steps 64 --max_seq_len 256
   ```

   把「32×8=256」换成「4×64=256」，等效 batch 不变，显存能塞下。

3. **速度换算**：README 说单卡 3090 跑完 pretrain 约 1.21 小时。
   按每步耗时与等效步数折算，本地是**「一天量级」**——挂一夜能跑完，
   但不要指望上午开始下午出结果。

### 7.3 一个容易误读的现象

如果拿**固定的几条数据**连跑几十步，loss 会掉到 0.0x 甚至 0.007。
**这不是训练成功**，是模型把这几个样本背下来了（64M 参数背 4 条 340 token 的序列绰绰有余）。
真实训练中数据不断变化，loss 会稳定在「与数据难度匹配」的水平。

---

## 8. 坑与注意事项

| 坑 | 说明 | 处理 |
| --- | --- | --- |
| 默认 batch 32 在 4GB 上 OOM | 按 3090 定的 | 降到 4~8，用累积补 |
| fp16 在 Turing 上更慢 | `resolve_dtype` 自动降级 | 手动改 fp32 或忍受 |
| `use_compile=1` 在 Windows 失败 | 无 Triton、缺 MSVC | 保持 0 |
| 保存文件名固定 | 只留最新一份 | 需要中间权重就自己改文件名 |
| 日志的 `loss` 是除过的 | 乘回 `accumulation_steps` 才是真值 | 已处理 |
| 最后不足 accum 的批 | `or step == iters` 强制更新 | 已处理 |
| 续训只跳过首个 epoch | 设计如此 | 不需要改 |
| `import datasets` 必须在 torch 前 | Windows DLL 冲突 | 别调换顺序 |
| 权重和模型结构必须匹配 | `strict=False` 也会因形状报错 | 用同一 `hidden_size`/`use_moe` |

---

## 9. 它和 `train_full_sft.py` 差多少

把两个文件 diff 一下，**只差这些**：

| 项 | pretrain | full_sft |
| --- | --- | --- |
| Dataset | `PretrainDataset`（整段算 loss） | `SFTDataset`（只算回答） |
| 学习率 | 5e-4 | **1e-5** |
| 序列长度 | 340 | 768 |
| batch | 32 | 16 |
| 累积步数 | 8 | 1 |
| 初始权重 | `none`（随机） | `pretrain` |
| 数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 保存前缀 | `pretrain` | `full_sft` |

**循环体、精度处理、保存逻辑、分布式逻辑一行没变。**

这就是为什么说「看懂 pretrain，后面都是加法」——
SFT 的本质是「换一个会屏蔽提问的 Dataset + 把学习率降两个数量级」。

---

## 10. 自测题

1. `__package__ = "trainer"` 和 `sys.path.append` 各自解决什么问题？
2. `import datasets` 为什么必须放在 `import torch` 之前？
3. 为什么先初始化分布式再设随机种子？
4. `setup_seed(args.seed + epoch)` 放在 `randperm` 前的作用？
5. `train_sampler or indices` 这行在单卡和多卡下分别是哪个分支？
6. `loss / accumulation_steps` 为什么不能省？
7. `unscale_` 为什么必须在 `clip_grad_norm_` 之前？
8. `or step == iters` 解决了什么边界问题？
9. 日志里为什么要乘回 `accumulation_steps`？
10. 保存时为什么要脱 `module.` / `_orig_mod.` 前缀？
11. 随机权重时 loss 的期望值是多少？为什么？
12. 固定数据训 20 步 loss 掉到 0.007 说明了什么？
13. 在你的 4GB 卡上，脚本默认参数会遇到什么问题？怎么改？
14. pretrain 和 SFT 的循环体差在哪？

### 答案要点

1. 前者让脚本以包内模块运行；后者让项目根目录可被导入。
2. Windows 下 pyarrow 与 torch 的 DLL 加载顺序冲突。
3. 分布式初始化会设置各卡的环境，种子要按 rank 区分。
4. 保证每个 epoch 的打乱顺序可复现。
5. 多卡走 `DistributedSampler`，单卡走 `randperm` 列表。
6. 梯度会累积，不除等于学习率放大 N 倍。
7. 裁剪必须作用在真实梯度上，否则阈值失效。
8. 最后不足一个累积周期的梯度也能被应用。
9. 日志值在前面被除过。
10. 否则加载时键名对不上。
11. `ln(6400) ≈ 8.76`，因为输出近似均匀分布。
12. 过拟合了几条固定样本，不代表训练有效。
13. batch 32 OOM；改小 batch + 增大累积；fp16 反而慢，可改 fp32。
14. 只差 Dataset、学习率、序列长度、batch、初始权重 —— 循环体相同。

---

## 11. 一表速查

| 段落 | 关键代码 | 一句话 |
| --- | --- | --- |
| 脚手架 | `__package__` / `sys.path` / `import datasets` | 让脚本能当模块跑 |
| 参数 | 20 个 argparse | 默认值按 3090 定 |
| 初始化 | 分布式 → 配置 → 精度 → 模型 → 恢复 | 顺序不能乱 |
| 训练步 | `zero_grad → forward → loss/accum → backward → step` | 五步循环 |
| 更新 | `unscale → clip → step → update → zero` | 顺序不能乱 |
| 日志 | `loss × accum`、拆 aux | 看 `logits_loss` |
| 保存 | half 权重 + resume | 双文件 |
| 续训 | `SkipBatchSampler` + step 换算 | 近似恢复 |
| 分布式 | DDP 包在 compile 之后 | 单卡跳过 |
