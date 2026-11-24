# `trainer_utils.py` 详解：12 个训练脚本共用的工具箱

> 对象：`minimind-master/trainer/trainer_utils.py`（243 行）
> 环境：本机 conda `swift_env`（torch 2.6.0+cu124）
> 本文所有「实测」输出均来自本机实际调用，脚本见 `trainer_utils_probe.py`。

---

## 0. 定位：它解决的是「复制粘贴」问题

项目里有 **8 个训练脚本**（pretrain / sft / lora / distillation / dpo / ppo / grpo / agent），
它们的循环骨架几乎一样。如果不抽公共函数，每加一个算法就要复制一遍：

- 日志打印
- 学习率调度
- 分布式初始化
- 断点保存与恢复
- 模型加载

`trainer_utils.py` 就是把这些「与算法无关」的部分收拢到一个文件里。
**判断标准很清晰：任何和「具体训练目标」有关的东西（loss、数据、奖励）都不在这里。**

### 0.1 谁在用哪些函数

| 使用方 | 用到的函数 |
| --- | --- |
| `train_pretrain` / `train_full_sft` / `train_lora` / `train_distillation` / `train_dpo` | `get_lr`、`Logger`、`is_main_process`、`lm_checkpoint`、`init_distributed_mode`、`setup_seed`、`init_model`、`SkipBatchSampler`、`resolve_dtype` |
| `train_ppo` / `train_grpo` / `train_agent` | 上面全部 + `LMForRewardModel`、`rep_penalty`（agent 另有 `safe_math_eval`） |
| `eval_llm.py` | `setup_seed`、`get_model_params` |
| `scripts/web_demo.py` / `scripts/eval_toolcall.py` | `safe_math_eval`、`get_model_params` |

### 0.2 14 个符号，分四组

| 分组 | 符号 | 一句话 |
| --- | --- | --- |
| **环境** | `is_main_process` | 是不是 0 号卡 |
| | `Logger` | 只在主进程打印 |
| | `setup_seed` | 固定所有随机源 |
| | `init_distributed_mode` | 读环境变量起 NCCL |
| | `resolve_dtype` | bf16 不支持就降级 fp16 |
| **调度** | `get_lr` | warmup + 余弦（10% 地板） |
| | `SkipBatchSampler` | 续训时跳过已训 batch |
| | `init_model` | 建模型 + 载权重 + 打印参数量 |
| | `get_model_params` | 打印总参数 / 激活参数 |
| **断点** | `lm_checkpoint` | 存两份：推理权重 + 续训状态 |
| **业务** | `LMForRewardModel` | 外部奖励模型封装 |
| | `safe_math_eval` | AST 白名单安全求值 |
| | `rep_penalty` | n-gram 重复度打分 |

---

## 1. 环境组

### 1.1 `is_main_process()`（:36）

```python
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
```

**语义**：单卡训练恒为 True；多卡时只有 rank 0 为 True。

**为什么重要**：多卡下每个进程都会执行同一份代码，如果都打印/都保存，
日志会重复 N 遍、文件会互相覆盖。所以凡是「只该做一次」的事（打印、存盘、上传）
都必须过这一层判断。

**坑**：它没有缓存 `dist.get_rank()`，高频调用（比如每个 step 打日志）会有微小开销；
在主循环里没问题，在每 token 循环里就有影响。

### 1.2 `Logger(content)`（:41）

```python
def Logger(content):
    if is_main_process():
        print(content)
```

**它做了什么**：只是「加了一层主进程判断的 print」。
**它没做什么**：没有时间戳、没有文件落盘、没有等级、没有结构化字段。

**为什么可以这么简单**：训练日志真正的归宿是 swanlab/wandb（脚本里单独处理），
终端打印只是给人看的副产物。

### 1.3 `setup_seed(seed)`（:77）

```python
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**五个随机源全部固定**：Python 的 random、NumPy、CPU 的 torch、当前 GPU、所有 GPU。

**两个 cudnn 开关的含义**：

| 开关 | 值 | 代价 | 收益 |
| --- | --- | --- | --- |
| `deterministic=True` | 强制确定性算法 | 部分卷积会变慢 | 可复现 |
| `benchmark=False` | 不做算法自动寻优 | 慢一点 | 同上 |

**训练脚本里的实际用法**（两处）：

```python
setup_seed(args.seed + rank)     # 进程级：每张卡不同
setup_seed(args.seed + epoch)    # 每个 epoch：保证各轮 shuffle 顺序可复现
```

### 1.4 `init_distributed_mode()`（:66）

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0                    # 不是 torchrun 启动的 → 单卡
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

**它靠什么判断该不该起分布式**：`RANK` 环境变量。`torchrun` 会自动注入
`RANK / LOCAL_RANK / WORLD_SIZE`，直接 `python train.py` 则没有——**一个变量区分两种启动方式**。

| 环境变量 | 含义 |
| --- | --- |
| `RANK` | 全局进程序号 |
| `LOCAL_RANK` | 本机内的卡号（用于 `set_device`） |
| `WORLD_SIZE` | 总进程数（本函数没读，DDP 自己读） |

**三个局限**：

1. `backend="nccl"` 写死——**Windows 没有 NCCL，这个方法在 Windows 上必然失败**（这也是为什么你的机器只能用单卡）。
2. 没有 `timeout`，默认 30 分钟，长任务卡住时很难排查。
3. 没有打印 rank 信息，出问题只能靠 `torchrun` 的日志。

### 1.5 `resolve_dtype(dtype, device)`（:47）

```python
def resolve_dtype(dtype, device):
    if dtype != "bfloat16" or not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return dtype
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return dtype
    Logger(f"{torch.cuda.get_device_name(device)} 不原生支持 bfloat16，已回退为 float16")
    return "float16"
```

**逻辑拆解**：

| 输入 | 结果 |
| --- | --- |
| 不是 bf16 | 原样返回（fp16/fp32 不做检查） |
| 没有 CUDA / 不是 GPU | 原样返回（CPU 上 bf16 其实能跑） |
| 卡原生支持 bf16 | 原样返回 |
| 卡不支持 | **降级 fp16** 并打印警告 |

**`including_emulation=False` 是关键**：torch 在较老的卡上可以用软件模拟 bf16，
但速度极差。传 False 表示「只认硬件原生支持」，避免「能跑但慢十倍」。

**本机实测**：

```
NVIDIA GeForce GTX 1650 不原生支持 bfloat16，已回退为 float16
  bfloat16 -> float16
  float16  -> float16
```

**这对你的意义**：MiniMind 在 1650 上拿到了 fp16 训练路径，而我们在 PyTorch 教材里
实测过这块卡 fp16 autocast 比 fp32 慢约 5 倍——**能跑，但别指望速度**。

---

## 2. 调度组

### 2.1 `get_lr`（:57）——学习率曲线

```python
def get_lr(current_step, total_steps, lr, warmup_ratio=0.0):
    warmup = max(1, int(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0
    if current_step < warmup:
        return lr * (current_step + 1) / warmup
    angle = math.pi * (current_step - warmup) / max(1, total_steps - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(angle)))
```

**两段式**：

| 阶段 | 公式 | 端点值 |
| --- | --- | --- |
| warmup（前 `w` 步） | `lr × (step+1)/w` | 起点 `lr/w`，终点 `lr` |
| 余弦下降 | `lr × (0.1 + 0.45×(1+cos θ))` | θ=0 → `lr`；θ=π → `0.1·lr` |

**注意两个细节**：

1. **warmup 起点不是 0，而是 `lr/w`**（因为分子是 `step+1`）。warmup 100 步、lr=1e-3 时，
   第一步就是 1e-5，而不是 0。
2. **余弦段用的是「剩余步数」**（`total - warmup`），所以开启 warmup 后曲线整体左移。

**本机实测（lr=1e-3, total=1000）**：

*不 warmup*：

```
step    0: 0.001000
step  250: 0.000868
step  500: 0.000550
step  750: 0.000232
step 1000: 0.000100      ← 10% 地板
```

*warmup_ratio=0.1（前 100 步线性升）*：

```
step    0: 0.000010      ← lr/100，不是 0
step   50: 0.000510
step   99: 0.001000
step  500: 0.000628      ← 注意与不 warmup 时的 0.000550 不同
step 1000: 0.000100
```

**为什么保留 10% 地板**：大模型训练后期如果 lr 趋近 0，梯度噪声会主导更新，
模型在局部乱跳。留 10% 相当于保留一点「探索」，也让最后阶段的 batch 仍有意义。

**坑**：

- 调用方传的是 `epoch * iters + step`，而 `step` 从 1 开始（`enumerate(loader, start=1)`），
  所以实际第一个 step 传进来的是 1，导致 warmup 进度**偏快一步**。无伤大雅，但做精确复现时要知道。
- `warmup_ratio=0` 时 `warmup=0`，`current_step < 0` 永远为假，直接走余弦。

### 2.2 `SkipBatchSampler`（:160）——断点续训的「快进键」

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0): ...
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches: skipped += 1; batch = []; continue
                yield batch; batch = []
        if batch and skipped >= self.skip_batches: yield batch
```

**它解决的场景**：从第 500 步续训时，如果从头再跑一遍数据，既浪费时间又打乱
「同一步看到同一批数据」的复现性。所以**直接跳过前面的批次**。

**它接受的输入有两种**：

| 输入 | 场景 |
| --- | --- |
| `DistributedSampler` | 多卡（每个 rank 只拿自己那份） |
| `list`（`randperm` 的结果） | 单卡（自己打乱） |

**本机实测**：

```python
s = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=2)
len(s) == 2
list(s) == [[6, 7, 8], [9]]
```

前两个 batch（`[0,1,2]`、`[3,4,5]`）被吃掉，从第 6 个样本开始。

**坑**：

- 只按 batch 数跳过，最后一个不完整 batch 也计入。
- 如果换卡数，`lm_checkpoint` 会把 step 换算成新卡数下的步数，再喂给这里——
  两者配合才成立（见 3.1）。

### 2.3 `init_model`（:144）——一行搞定「建模型 + 载权重 + 打印」

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model',
               save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: ...')
    return model.to(device), tokenizer
```

**四件事**：

| 步骤 | 说明 |
| --- | --- |
| 载分词器 | 固定从 `../model` 读（**依赖「在 trainer/ 目录下运行」**） |
| 建模型 | 由 config 决定结构 |
| 载权重 | 文件名由 `from_weight + hidden_size + _moe` 拼成 |
| 打印 | 总参数 + 可训练参数 |

**`strict=False` 的取舍**：允许键名不完全匹配（比如 Dense 权重载入时忽略 MoE 键）。
但**形状不匹配仍会报错**，所以拿 512 的权重加载到 768 模型上不会被静默接受。

**坑**：所有路径都是相对路径，换目录运行就会 `FileNotFoundError`。
README 特意强调「所有训练脚本均 `cd ./trainer` 执行」，就是为这个。

### 2.4 `get_model_params`（:22）——MoE 的「总参数 / 激活参数」

```python
total = sum(p.numel()) / 1e6
n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
n_active = getattr(config, 'num_experts_per_tok', 0)
n_shared = getattr(config, 'n_shared_experts', 0)
expert = 名称含 'mlp.experts.0.' 的参数 / 1e6      # 单个专家
shared_expert = 名称含 'mlp.shared_experts.0.' 的参数 / 1e6
base = total - expert*n_routed - shared_expert*n_shared
active = base + expert*n_active + shared_expert*n_shared
```

**思路**：从总参数里**扣掉所有专家**，得到「底座」（embedding + 注意力 + 路由）；
再加回「实际会激活的专家」数，就是每 token 的计算量。

**本机实测**：

```
Model Params: 63.91M              ← Dense
Model Params: 198.42M-A63.94M     ← MoE：总 198M，激活 64M
```

**这个数字和 README 的「198M-A64M」对得上**，也解释了 MoE 的卖点：
**参数量 3 倍，算力几乎不变**。

**两个坑**：

1. **硬编码了模块名** `'mlp.experts.0.'`。重命名专家层后，这里会静默算成 0，
   打印出错误的数字而不报错。
2. `getattr` 的默认值链（`n_routed_experts` → `num_experts` → 0）是为了兼容不同版本配置；
   MiniMind 的 config 里两者都叫 `num_experts`，Dense 模式下它是 4 但**没有实际专家参数**，
   所以 `expert=0`，最终只打印总数。

---

## 3. 断点组

### 3.1 `lm_checkpoint`（:88）——全文件最值得读的一段

**两个文件，两种用途**：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `out/{weight}_{hidden}{_moe}.pth` | **半精度**权重（`.half()`） | 推理、发布 |
| `checkpoints/{weight}_{hidden}{_moe}_resume.pth` | fp32 权重 + 优化器 + scaler + epoch/step + world_size + wandb_id | 断点续训 |

**保存分支（`model is not None`）逐步**：

```python
raw_model = model.module if isinstance(model, DistributedDataParallel) else model
raw_model = getattr(raw_model, '_orig_mod', raw_model)      # ① 剥掉 DDP / torch.compile
state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
os.replace(ckp_tmp, ckp_path)                               # ② 原子写
```

| 步骤 | 为什么 |
| --- | --- |
| 剥 DDP / `_orig_mod` | 否则键名会带 `module.` / `_orig_mod.` 前缀，加载不回去 |
| 先 `.cpu()` 再 `.half()` | 省显存；顺序反了会先在 GPU 上生成半精度副本 |
| `.tmp` + `os.replace` | **原子写**：写到一半断电，旧文件仍完整 |
| `resume` 里存 fp32 | 续训要恢复精度，不能是 half |

**断点里存了什么**：

```python
resume_data = {
    'model': fp32 state_dict,
    'optimizer': optimizer.state_dict(),
    'epoch': epoch, 'step': step,
    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    'wandb_id': ...,
}
# 额外 kwargs（如 scaler）如果有 state_dict 也一起存
```

**加载分支（`model is None`）最聪明的一处**：

```python
saved_ws = ckp_data.get('world_size', 1)
current_ws = dist.get_world_size() if dist.is_initialized() else 1
if saved_ws != current_ws:
    ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
```

**换卡数续训**：4 卡跑了 100 步 ≈ 8 卡跑 50 步。换算后的 step 再喂给
`SkipBatchSampler`，就能从「大致相同的数据位置」继续。这不是精确等价，
但比从头开始或重复训练要好得多。

**坑**：

| 坑 | 说明 |
| --- | --- |
| 双份磁盘占用 | 半精度 + fp32 各一份 |
| `torch.load` 未传 `weights_only` | torch 2.6 默认 True；这里存的都是张量/基础类型，能正常加载；但如果哪天往 resume 里塞了自定义对象就会失败 |
| 每保存一次都 `empty_cache` | 单卡 CPU 训练时是无操作 |
| world_size 换算是有损近似 | 数据顺序并非严格等价 |
| 路径同样依赖 cwd | `save_dir='../checkpoints'` 默认值 |

---

## 4. 业务组：RL 与工具调用

### 4.1 `LMForRewardModel`（:187）

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype,
                                               trust_remote_code=True).to(device).eval()
    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [{"role": "user", "content": message_context},
                         {"role": "assistant", "content": response}]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
```

**它假设什么**：外部奖励模型实现了 `get_score(tokenizer, messages) -> float`
这个方法（MiniMind 作者发布的 RM 就是这样），所以必须 `trust_remote_code=True`。

**为什么把历史拼成一段**：做成了「单轮」的评分格式——
把多轮对话历史压进一条 user 消息里，再让 RM 给 assistant 回答打分。
这样可以用单轮奖励模型评多轮对话。

**分数裁剪 `[-3, 3]`**：防止异常样本（RM 打出 -10 或 20）把 RL 的梯度带飞。

**坑**：

| 坑 | 说明 |
| --- | --- |
| `trust_remote_code=True` | **会执行仓库里的任意 Python 代码**，只对可信模型开 |
| `torch_dtype=` | transformers 4.56 起推荐 `dtype=`，旧参数仍可用但已弃用 |
| 评分格式是「约定的」 | 换一个只支持 `sequence_classification` 的 RM，`get_score` 不存在会直接 AttributeError |
| fp16 加载 | 在 1650 上 RM 推理同样不快 |

### 4.2 `safe_math_eval`（:208）——不用 `eval()` 的算式求值

**背景**：Agentic RL 里模型会调用 `calculate_math` 工具，参数是它自己生成的字符串。
如果直接 `eval()`，模型可以输出 `__import__('os').system('rm -rf /')`。

**实现思路**：把表达式解析成 **AST**，只允许白名单里的节点类型。

```python
def walk(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
    name = resolve(node)
    if name in consts: return consts[name]
    if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
    if isinstance(node, ast.BinOp):  return bin_ops[type(node.op)](walk(node.left), walk(node.right))
    if isinstance(node, ast.Call) and not node.keywords:
        return funcs[resolve(node.func)](*map(walk, node.args))
    raise ValueError(...)
```

**白名单**：

| 类别 | 内容 |
| --- | --- |
| 常量 | `pi`、`e`、`tau` |
| 运算 | `+ - * / // % **` |
| 函数 | `pow`、`sqrt/exp/log/log2/log10`、三角/反三角、`floor/ceil/trunc`、`fabs/fmod/hypot/gcd`、`degrees/radians` |
| 写法兼容 | `sqrt(4)` 与 `math.sqrt(4)` 都行 |

**两层防护**：

```python
def pow_guard(base, exp):
    if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * log10(abs(base)) > 100):
        raise ValueError('幂运算结果过大')
```

防止 `9**9**9` 这种把进程算死的输入。

**本机实测**：

```
'2^10'          -> 1024
'（1+2）×3'      -> 9            ← 全角符号被 translate 成半角
'sqrt(16)'      -> 4.0
'math.log(e)'   -> 1.0
'2**-3'         -> 0.125
'10**50'        -> 1e50          ← 允许
'9**9**9'       -> ValueError: 幂运算结果过大
'1/0'           -> ZeroDivisionError: division by zero    ← 没兜住！
"__import__('os')" -> ValueError: 不支持的表达式语法
'2+'            -> ValueError: 不支持的表达式语法
```

**一个真实的漏洞**：`except (KeyError, SyntaxError, TypeError)` **没有捕获
`ZeroDivisionError` 和 `OverflowError`**。`1/0` 会把异常抛给调用方。
（在上层 `execute_tool` 里有 `try/except Exception` 兜底，所以不会崩训练，
但如果你单独复用这个函数，要自己包一层。）

**其他边界**：

- `type(node.value) in (int, float)` 排除了 `bool`（`True` 的 type 是 bool），避免 `True*5`；
- 表达式长度上限 512，深度嵌套最多几十层，不会爆栈；
- 支持 `**` 的右结合，`2**3**2 = 512`。

### 4.3 `rep_penalty`（:238）——重复度奖励项

```python
def rep_penalty(text, n=3, cap=0.5):
    cjk = "CJK 与假名等大段 Unicode 区间"
    toks = re.findall(rf"[{cjk}]|[^\W{cjk}]+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return min(cap, (len(grams)-len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

**分三步**：

| 步骤 | 说明 |
| --- | --- |
| 分词 | 中文按**单字**切、英文按**单词**切、标点单独成 token（CJK 区间写得很全，含扩展区） |
| 取 3-gram | 连续 3 个 token 为一组 |
| 算重复率 | `(总数 - 去重后数) / 总数`，再乘 `cap×2` 并截断到 `cap` |

**本机实测**：

```
0.5000  今天天气很好，今天天气很好，今天天气很好     ← 重复率高，触顶 0.5
0.0000  今天天气很好，我们去公园散步吧              ← 无重复
0.0000  好的                                      ← 短于 3 个 token
```

**在 RL 里怎么用**：作为**负奖励**叠加（`rewards -= rep_penalty(answer)`），
让模型学会别复读。乘以 2 是为了让「重复率 25%」就触顶，属于比较激进的惩罚。

**坑**：

- 中文按字切，所以「今天天气」这种重复词会被抓到；但同义改写抓不到；
- 只要 3-gram 重复就扣分，**合理的结构重复**（如列表项）也会被误伤；
- 文本不足 3 个 token 直接返回 0。

---

## 5. 设计评价

### 5.1 五个值得学的做法

| 做法 | 价值 |
| --- | --- |
| **只收「与算法无关」的工具** | 12 个脚本复用，改一处全生效 |
| **原子写检查点** | `.tmp` + `os.replace`，断电不毁旧档 |
| **双文件策略** | 推理要半精度、续训要完整状态，各取所需 |
| **跨卡数续训换算** | 换机器/换卡不丢进度 |
| **AST 白名单求值** | 比 `eval` 安全，比正则解析可靠 |

### 5.2 五个可以改进的地方

| 问题 | 影响 | 建议 |
| --- | --- | --- |
| 路径全是相对路径 | 换目录就崩 | 由调用方传入绝对路径或用 `pathlib` |
| 模块名硬编码（`mlp.experts.0.`） | 改名后统计静默出错 | 按 `isinstance(MOEFeedForward)` 统计 |
| `init_distributed_mode` 只支持 NCCL | Windows/CPU 不可用 | 加 gloo 回退 |
| `safe_math_eval` 漏捕 `ZeroDivisionError` | 单测/复用时会冒泡 | `except (KeyError, SyntaxError, TypeError, ZeroDivisionError, OverflowError)` |
| 无类型标注、无单测 | 重构风险高 | 至少给纯函数补 `-> float` 与 pytest |

---

## 6. 自测题

1. `Logger` 在多卡下为什么不会重复打印？
2. `setup_seed` 为什么要设两次（进程级 + epoch 级）？
3. `init_distributed_mode` 靠哪个环境变量判断「是不是分布式」？
4. `resolve_dtype` 里 `including_emulation=False` 的作用是什么？
5. `get_lr` 在 warmup=100、lr=1e-3 时，第 0 步的学习率是多少？
6. 余弦为什么保留 10% 而不是降到 0？
7. `SkipBatchSampler` 能接受哪两种输入？
8. `init_model` 的 `strict=False` 允许什么、不允许什么？
9. MoE 的「198M-A64M」是怎么算出来的？
10. 检查点为什么要存两份文件？
11. 换卡数续训时 step 怎么换算？为什么只是近似？
12. `safe_math_eval` 为什么不能用 `eval()`？
13. 它漏掉了哪类异常？
14. `rep_penalty` 的中文是怎么切词的？
15. 三段 3-gram 文本，重复率多少会触顶？

### 答案要点

1. 全部走 `is_main_process()`，只有 rank 0 打印。
2. 进程级保证各卡初始化一致，epoch 级保证每轮 shuffle 可复现。
3. `RANK` 环境变量（`torchrun` 注入）。
4. 只认硬件原生 bf16，避免软件模拟导致「能跑但极慢」。
5. `1e-3 × 1/100 = 1e-5`（不是 0）。
6. 避免后期梯度噪声主导；保留探索能力。
7. `DistributedSampler` 或普通 `list`。
8. 允许缺失/多余键；**不允许形状不匹配**。
9. 总参数扣掉全部专家、再加回 1 个专家。
10. 推理要半精度，续训要 fp32 权重 + 优化器状态。
11. `step × saved_world / current_world`；数据顺序不严格等价。
12. 防提示注入/代码执行。
13. `ZeroDivisionError`、`OverflowError`。
14. 按单字切（CJK 区间），英文按单词，标点单独。
15. 25%（因为乘了 2 倍系数）。

---

## 7. 一表速查

| 符号 | 输入 | 输出 | 关键行为 |
| --- | --- | --- | --- |
| `is_main_process` | — | bool | 单卡 True / 多卡 rank0 True |
| `Logger` | 任意 | None | 仅主进程 print |
| `setup_seed` | int | None | 固定 5 个随机源 + cudnn 确定性 |
| `init_distributed_mode` | — | local_rank | 看 `RANK`，NCCL 后端 |
| `resolve_dtype` | str, device | str | bf16 不支持 → fp16 |
| `get_lr` | step, total, lr, ratio | float | warmup + 余弦（10% 地板） |
| `SkipBatchSampler` | sampler/list, bs, skip | batch 迭代器 | 跳过前 N 批 |
| `init_model` | config, weight, ... | (model, tok) | 建模型 + 载权重 + 打印 |
| `get_model_params` | model, config | None | 打印总参数/激活参数 |
| `lm_checkpoint` | ... | ckp 或 None | 双文件 + 原子写 + 换算 |
| `LMForRewardModel` | path | 对象 | 外部 RM，`get_score` |
| `safe_math_eval` | str | number | AST 白名单 |
| `rep_penalty` | str | float | 3-gram 重复率，上限 0.5 |
