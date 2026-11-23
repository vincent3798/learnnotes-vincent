# DeepSpeed 实战：ZeRO、显存算账与分布式训练

> **适用环境**
> 本机：conda `swift_env`（Python 3.10 / torch 2.6.0+cu124 / transformers 4.57.6）
> GPU：GTX 1650 单卡 4 GB，Windows 11，**无 CUDA 工具包**
> 云端：Linux + 多卡（第 9 章给出从零到跑通的步骤）
>
> 本书的诚实声明：**DeepSpeed 在这台机器上无法运行**（原因见 0.6，实测报错原文附上）。
> 因此本书分两部分——概念与算账全部本地实测；真正跑 DeepSpeed 的代码标注「云端」。

---

## 这本书怎么用

这是系列第三本。前两本讲单卡上的「怎么写模型」，这一本讲**多卡上怎么把模型
塞进去**。核心只有一个问题：

> 一个 7B 模型全量训练要 104 GB 显存，而你只有 8 张 24GB 的卡。
> 怎么让它跑起来？

DeepSpeed 的回答是：**别让每张卡都存一份完整的东西，切开分着存**。

读完你应该能：

1. 拿到任何模型，算出训练需要多少显存、几张卡、用哪个 ZeRO 阶段；
2. 写出正确的 `ds_config.json`，并用体检器自查；
3. 分清 ZeRO-2 和 ZeRO-3 的取舍，不被「ZeRO-3 更省」一句话带偏；
4. 知道自己这台机器该用什么替代方案。

配套脚本在 `code/`：

```bash
D:\conda_envs\swift_env\python.exe code\01_memory_math.py
```

---

## 目录

- 第 0 章 DeepSpeed 是什么，以及它在这台机器上的结局
- 第 1 章 先算账：训练显存到底花在哪
- 第 2 章 ZeRO：把三样东西切开分给多张卡
- 第 3 章 Offload：拿 CPU 和硬盘换显存
- 第 4 章 精度、激活检查点与梯度累积
- 第 5 章 ds_config.json 逐字段详解
- 第 6 章 三条集成路径
- 第 7 章 推理侧：ZeRO-Inference 与 vLLM
- 第 8 章 你这台机器的现实选择
- 第 9 章 云端实操：从零跑通 7B 全量微调
- 第 10 章 排错速查
- 第 11 章 练习与答案
- 附录 A 配置速查 / B 术语 / C 方案对比

---

## 第 0 章 DeepSpeed 是什么，以及它在这台机器上的结局

### 0.1 一句话

**DeepSpeed 是微软的分布式训练引擎**，解决的核心问题是：模型和优化器状态
太大，一张卡放不下。它的招牌技术叫 **ZeRO**（Zero Redundancy Optimizer，
零冗余优化器）——把原本每张卡重复存的东西切开，让每张卡只存一份。

### 0.2 和普通 DDP 的差别

```
普通 DDP（数据并行）:
  每张卡: [完整参数] + [完整梯度] + [完整优化器状态]
  每步:   梯度 all-reduce，参数各自更新 → 每卡内容完全一样

ZeRO-3:
  每张卡: [1/N 参数] + [1/N 梯度] + [1/N 优化器状态]
  每步:   需要时把别人的分片临时取过来算，算完还回去
```

| 维度 | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 每卡优化器状态 | 完整 | 1/N | 1/N | 1/N |
| 每卡梯度 | 完整 | 完整 | 1/N | 1/N |
| 每卡参数 | 完整 | 完整 | 完整 | 1/N |
| 每步通信 | 2P | 2P | 2P | 3P |
| 代码改动 | 无 | 配置文件 | 配置文件 | 配置文件 |

（P = 参数量，通信量单位见 2.4）

### 0.3 DeepSpeed 的组件

| 组件 | 作用 | 你会不会用 |
| --- | --- | --- |
| ZeRO-1/2/3 | 分片显存 | ✅ 本书重点 |
| ZeRO-Offload | 把状态搬到 CPU/NVMe | ✅ 4GB / 小显存必备 |
| Activation Checkpointing | 用重算换激活显存 | ✅ 大模型标配 |
| 混合精度 | bf16/fp16 训练 | ✅ 默认开 |
| DeepSpeed-Inference | 推理加速与分片 | ⚠️ 了解一下，现在多用 vLLM |
| MoE / 1-bit Adam / 稀疏注意力 | 特定场景优化 | ❌ 暂时不用 |

### 0.4 一个重要的认知：ZeRO ≠ 多卡并行

ZeRO 仍然是**数据并行**：每张卡处理不同的数据，模型是同一份（只是分片存放）。
它不改变「所有卡算的是同一个模型」这件事。真正的模型并行（把不同层放不同卡）
是另一套东西（Pipeline Parallel / Tensor Parallel），DeepSpeed 也支持，但本书不讲。

### 0.5 它适合谁

| 场景 | 是否值得用 |
| --- | --- |
| 单卡训练，模型放得下 | ❌ 不需要，纯增加复杂度 |
| 单卡训练，模型放不下 | ⚠️ 只有 ZeRO-Offload 有意义，且很慢 |
| 多卡全量微调 7B+ | ✅ 这正是它的主场 |
| 多卡 LoRA 微调 | ⚠️ 能用，但 FSDP / accelerate 往往更简单 |
| 多机训练 | ✅ 支持，配置更复杂 |

### 0.6 本机实测：装不上，而且这不是配置问题

我在独立 venv 里试过：

```
pip download deepspeed --only-binary=:all:
  → 找到 deepspeed-0.16.5-cp310-cp310-win_amd64.whl（56.3 MB）
pip install deepspeed-0.16.5-...whl
  → 安装成功
import deepspeed
  → MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)
```

**原因链**：

1. DeepSpeed 的一部分算子（如 `cpu_adam`，CPU 上的 Adam 实现）需要在首次使用时
   **现场用 nvcc 编译**。
2. 本机 `torch.utils.cpp_extension.CUDA_HOME` 为 `None`，
   `where nvcc` 找不到，`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
   不存在——**只装了显卡驱动和 torch 自带的 CUDA 运行时，没装 CUDA 工具包**。
3. 即使装上工具包，Windows 也不是 DeepSpeed 的官方支持平台，
   多卡通信（NCCL）在 Windows 上不可用。

**再加上硬件条件**：

| 条件 | 本机 | DeepSpeed 需要 |
| --- | --- | --- |
| GPU 数量 | 1 | ≥ 2（ZeRO 分片才有意义） |
| 显存 | 4 GB | 通常 ≥ 24 GB/卡（7B 全量） |
| 操作系统 | Windows | 官方支持 Linux |
| CUDA 工具包 | 无 | 有（编译算子） |

**结论**：本书的代码在你本机跑不了 DeepSpeed 本身，这是环境事实，不是教学省略。
所以：

- **概念、算账、配置**：本地全部可验证（`code/` 下三个脚本，不依赖 deepspeed）；
- **真正训练**：按第 9 章上云。

> **口诀：ZeRO 切的是冗余，不是模型；单卡无冗余可切，Windows 无算子可编。**

---

## 第 1 章 先算账：训练显存到底花在哪

### 1.1 四个字节数：16 B / 参数

用 Adam（或 AdamW）+ 混合精度训练时，**每个参数要占 16 字节**：

| 组成 | 精度 | 每参数字节 |
| --- | --- | --- |
| 模型参数（推理用） | bf16/fp16 | 2 |
| 梯度 | bf16/fp16 | 2 |
| 优化器里的 fp32 主权重 | fp32 | 4 |
| Adam 一阶动量 m | fp32 | 4 |
| Adam 二阶动量 v | fp32 | 4 |
| **合计** | | **16** |

```python
PARAM_B, GRAD_B, MASTER_B, M_B, V_B = 2, 2, 4, 4, 4
OPT_B = MASTER_B + M_B + V_B      # 12
```

用 SGD 不用动量时是 8 B/参数（参数 2 + 梯度 2 + 主权重 4），
所以「换优化器」本身就是一种省显存手段。

### 1.2 还有两块常被忘掉

| 组成 | 规模 | 何时成为大头 |
| --- | --- | --- |
| 激活值 | 与 batch × 序列长 × 层宽 成正比 | batch 大、序列长时 |
| KV cache | 仅推理/生成 | 长上下文推理 |

**16 B/参数只是下限**，后面两项加上去，实际需求往往是它的 1.3 ~ 2 倍。
第 4 章的激活检查点就是专门打这一块的。

### 1.3 实测：各模型单卡要多大的卡

运行 `code/01_memory_math.py`，**实测输出**：

```
每张卡的非激活显存（GB）｜bf16 混合精度 + AdamW｜不含激活值
 模型     ZeRO-0    ZeRO-1    ZeRO-2    ZeRO-3      8卡 Z2     8卡 Z3
 0.5B       7.5G      7.5G      7.5G      7.5G       1.7G      0.9G
 1.5B      22.4G     22.4G     22.4G     22.4G       5.2G      2.8G
   7B     104.3G    104.3G    104.3G    104.3G      24.4G     13.0G
  13B     193.7G    193.7G    193.7G    193.7G      45.4G     24.2G
  70B    1043.1G   1043.1G   1043.1G   1043.1G     244.5G    130.4G
```

三列一样不是 bug——**单卡没有东西可以分片**，ZeRO 在 1 张卡上等于没开。

这张表能直接回答几个常见问题：

| 问题 | 答案 |
| --- | --- |
| 7B 全量微调要几张卡？ | 8 张 24GB + ZeRO-3（13 GB/卡），或 8 张 40GB + ZeRO-2 |
| 13B 呢？ | 8 卡 ZeRO-3 是 24.2 GB/卡，24GB 卡刚好不够，要 40GB 或 16 卡 |
| 70B？ | 8 卡 ZeRO-3 要 130 GB/卡，需要 8×H100 80GB 都不够 → 必须更多卡或 3D 并行 |

### 1.4 反过来：4 GB 卡能全量训练多大模型

**实测**：

```
ZeRO-0/2/3 结果相同: 约 0.23B 参数（不含激活）
```

也就是 2.3 亿参数——还没算激活值。这就是为什么你的卡只能做学习实验或
LoRA（LoRA 只训练几十万到几百万参数，优化器状态按可训练参数算）。

> **第 1 章口诀：十六字节每参数，激活 KV 另外算；单卡不分片，多卡才谈 ZeRO。**

---

## 第 2 章 ZeRO：把三样东西切开分给多张卡

### 2.1 三个东西，三个阶段

| 阶段 | 分片什么 | 剩余未分片 |
| --- | --- | --- |
| ZeRO-1 | 优化器状态（12 B/参数） | 参数、梯度 |
| ZeRO-2 | 优化器状态 + 梯度 | 参数 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 无 |

**实测（`code/02_zero_sim.py`，7B 模型，8 卡）**：

```
    阶段       参数       梯度        优化器状态   每卡总量
ZeRO-0       完整       完整           完整    104.31 GB
ZeRO-1       完整       完整           分片     35.86 GB
ZeRO-2       完整       分片           分片     24.45 GB
ZeRO-3       分片       分片           分片     13.04 GB
```

**省了多少（相对 ZeRO-0）**：

```
ZeRO-1:   35.9 GB/卡  (省 66%)
ZeRO-2:   24.4 GB/卡  (省 77%)
ZeRO-3:   13.0 GB/卡  (省 88%)
```

### 2.2 「分片」到底是什么意思

不是抽象概念，就是把张量切片：

```
总量: 100,000,000 个元素 = 381.5 MB
卡0 拿到 12,500,000 个元素，首尾 0.0 ~ 12499999.0
卡1 拿到 12,500,000 个元素，首尾 12500000.0 ~ 25000000.0
```

每张卡拿 1/N，合起来还原。**ZeRO-3 的参数分片就是这个**——
平时每张卡只存自己那份，计算到某一层时临时从别的卡取。

### 2.3 代价：显存换通信

| 阶段 | 每步通信量 | 按 7B bf16 估算 | 说明 |
| --- | --- | --- | --- |
| ZeRO-0（DDP） | 2P | 26.08 GB | 梯度 all-reduce |
| ZeRO-1 | 2P | 26.08 GB | reduce-scatter + all-gather |
| ZeRO-2 | 2P | 26.08 GB | 同 ZeRO-1，省更多显存 |
| ZeRO-3 | **3P** | **39.12 GB** | 多一次参数 all-gather |

**关键结论**：

- ZeRO-1/2 的通信量和普通 DDP **完全一样**——所以「开 ZeRO-2」几乎是无脑赚。
- ZeRO-3 每步多传 50%，换来的是「参数也能分片」，把能训的模型规模再翻一倍。
- **所以选择顺序是：显存够 → ZeRO-2；ZeRO-2 不够 → 才上 ZeRO-3。**

这正是工业界默认用 ZeRO-2 的原因。

### 2.4 为什么是 2P 和 3P

| 操作 | 传输量 | 何时发生 |
| --- | --- | --- |
| reduce-scatter 梯度 | P | 反向后，把梯度归约并只保留自己分片 |
| all-gather 参数 | P | 前向前，各卡取齐自己需要的参数分片 |
| all-gather 梯度 | P | ZeRO-2 反向前取回完整梯度的一小段 |

DDP：all-reduce ≈ 2P（reduce-scatter + all-gather 的组合）。
ZeRO-1/2：同样是 2P，只是把顺序和分片利用起来。
ZeRO-3：前向多一次参数 all-gather（P），加反向的一次（已在 2P 内），共 3P。

### 2.5 什么时候选哪个

| 条件 | 选择 |
| --- | --- |
| 模型能单卡放下 | ZeRO-0 都不用，直接 DDP |
| 单卡放不下参数，但装得下梯度+优化器 | ZeRO-2 |
| 参数本身单卡就放不下 | ZeRO-3 |
| ZeRO-3 还爆 | 开 offload（第 3 章）或激活检查点（第 4 章） |
| 换了更大显存的卡 | 降回 ZeRO-2，省 50% 通信 |

> **第 2 章口诀：一阶分状态，二阶分梯度，三阶连参数也分；二阶等通信，三阶多五成。**

---

## 第 3 章 Offload：拿 CPU 和硬盘换显存

### 3.1 三个层级

| 存放位置 | 带宽 | 相对 GPU 显存 | 适合放 |
| --- | --- | --- | --- |
| GPU 显存 | ~300-900 GB/s（1650 约 128 GB/s） | 1× | 参数、当前要算的数据 |
| CPU 内存 | ~10-25 GB/s（PCIe） | 约 1/20 | 优化器状态（每步只用一次） |
| NVMe 硬盘 | ~3-7 GB/s | 约 1/100 | 参数（只在需要时取） |

**核心思路**：优化器状态每步只用一次，放 CPU 最划算；参数每次前向都要用，
放 NVMe 会很慢，只作为最后手段。

### 3.2 两种 offload

```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": { "device": "cpu", "pin_memory": true }
}
```

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": { "device": "cpu", "pin_memory": true },
  "offload_param": { "device": "cpu", "pin_memory": true }
}
```

| 配置 | 效果 | 速度影响 |
| --- | --- | --- |
| `offload_optimizer` + ZeRO-2 | 优化器状态放 CPU，省 12 B/参数的大头 | 慢 10% ~ 30% |
| `offload_optimizer` + ZeRO-3 | 同上，配合参数分片 | 慢 20% ~ 40% |
| `offload_param` + ZeRO-3 | 参数也放 CPU | 慢 2 ~ 5 倍 |
| `device: "nvme"` | 放硬盘 | 再慢一个数量级 |

### 3.3 三个参数细节

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `pin_memory` | 锁页内存，加速 CPU↔GPU 拷贝 | `true`（内存够时） |
| `nvme_path` | 硬盘路径 | 用本地 NVMe，不要用网络盘 |
| `sub_group_size` | 一次 offload 多少参数 | 默认 1e9，太大易 OOM |

### 3.4 什么时候值得开

| 情况 | 结论 |
| --- | --- |
| 8 卡 24GB 训 7B，ZeRO-2 刚好爆 | ✅ 开 `offload_optimizer`，通常就够了 |
| 8 卡 24GB 训 13B | ✅ 要 ZeRO-3 + offload_optimizer |
| 单卡 4GB | ❌ 别指望——offload 也需要编译 `cpu_adam`，Windows 上更不可行 |
| 追求最快训练速度 | ❌ offload 一定变慢 |

> **第 3 章口诀：状态放 CPU 最划算，参数放硬盘最肉疼；开 offload 先开优化器那个。**

---

## 第 4 章 精度、激活检查点与梯度累积

### 4.1 bf16 还是 fp16

| 维度 | bf16 | fp16 |
| --- | --- | --- |
| 数值范围 | 同 fp32（8 位指数） | 窄（5 位指数） |
| 精度 | 低（7 位尾数） | 高（10 位尾数） |
| 需要 loss scaling | ❌ 不需要 | ✅ 必须 |
| 硬件要求 | Ampere（SM 8.0）及以上原生支持 | 更老的卡也支持 |
| 稳定性 | 好 | 需要调 scaler |

**对你环境的结论**（承接前两本教材的实测）：

- GTX 1650 是 Turing（SM 7.5），**没有 bf16 原生加速**；实测 fp16 autocast
  比 fp32 慢约 5 倍、bf16 略慢于 fp32。
- 但这只影响本地实验。**云端 A100/H100/4090 上 bf16 是首选**，
  配置里直接 `"bf16": {"enabled": true}`，不需要 loss scaler。

配置里的对应写法：

```json
"bf16": { "enabled": true }
```

```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "initial_scale_power": 16,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "min_loss_scale": 1
}
```

`loss_scale: 0` 表示动态缩放（推荐），DeepSpeed 会自动调。

### 4.2 激活检查点：用时间换显存

原理：前向时不保存中间激活，反向时重新算一遍。

```json
"activation_checkpointing": {
  "partition_activations": true,
  "contiguous_memory_optimization": false,
  "cpu_checkpointing": false,
  "synchronize_checkpoint_boundary": false
}
```

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `partition_activations` | 激活值也分片到各卡 | `true`（配合 ZeRO-3） |
| `cpu_checkpointing` | 激活值存 CPU | 只在极度缺显存时 |
| `contiguous_memory_optimization` | 连续内存，缓解碎片 | 有碎片问题时开 |

代价：训练慢 20% ~ 30%，显存省 30% ~ 70%（取决于模型和序列长度）。

### 4.3 梯度累积与 batch size 的硬性关系

DeepSpeed 会严格校验这个等式：

```
train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × world_size
```

**实测**：用体检器检查一个故意写错的配置：

```
❌ train_batch_size=1，但 micro(1) × gas(1) × world_size(8) = 8，DeepSpeed 会报错
```

这是新手最常见的报错来源。改法只有两种：改 `train_batch_size`，
或改微批/累积步数。

### 4.4 梯度裁剪

```json
"gradient_clipping": 1.0
```

ZeRO 会和普通的 `clip_grad_norm_` 行为一致，但必须写在配置里，
不要在代码里再裁一次。

> **第 4 章口诀：Ampere 用 bf16，老卡 fp16 加缩放；激活检查点省三成显存，batch 等式必须对。**

---

## 第 5 章 ds_config.json 逐字段详解

### 5.1 一份完整配置的骨架

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,

  "bf16": { "enabled": true },

  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 2e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0, "warmup_max_lr": 2e-5,
      "warmup_num_steps": 500, "total_num_steps": 10000
    }
  },

  "zero_optimization": { "stage": 2 },

  "steps_per_print": 50
}
```

按块读，一共五块：**批量 → 精度 → 优化 → ZeRO → 杂项**。

### 5.2 批量三兄弟

| 字段 | 含义 | 谁来定 |
| --- | --- | --- |
| `train_batch_size` | 全局等效 batch | 你（必须等于下面三者相乘） |
| `train_micro_batch_size_per_gpu` | 每卡每次前向多少条 | 显存决定 |
| `gradient_accumulation_steps` | 累积几步更新一次 | 用 micro 补够总 batch |

设 `micro=4, gas=8, world=8` → 总 batch 256。
显存不够时**只降 micro**，用 gas 补回来，总 batch 保持不变——这是标准做法。

### 5.3 优化器与调度器

| 配置 | 说明 |
| --- | --- |
| `optimizer.type: "AdamW"` | 让 DeepSpeed 自己造（推荐，能配合 offload） |
| `optimizer.type: "Adam"` | 同上 |
| 直接传 PyTorch 优化器对象 | 也能用，但 offload 等特性受限 |

调度器常见类型：

| 类型 | 曲线 | 适用 |
| --- | --- | --- |
| `WarmupLR` | 线性升到最大值后保持 | 短训练、微调 |
| `WarmupDecayLR` | 升到最大值后线性降 | 长训练 |
| `OneCycle` | 先升后降的三角形 | 追求快速收敛 |

### 5.4 ZeRO 块：最常改的部分

```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "allgather_bucket_size": 5e8,
  "overlap_comm": true,
  "reduce_scatter": true,
  "reduce_bucket_size": 5e8,
  "contiguous_gradients": true,
  "offload_optimizer": { "device": "cpu", "pin_memory": true },
  "offload_param": { "device": "cpu", "pin_memory": true },
  "stage3_gather_16bit_weights_on_model_save": true
}
```

| 字段 | 作用 | 建议 |
| --- | --- | --- |
| `stage` | 0/1/2/3 | 见第 2 章 |
| `overlap_comm` | 通信与计算重叠 | `true`，基本无脑开 |
| `allgather_bucket_size` / `reduce_bucket_size` | 通信分桶大小 | 默认 5e8；显存吃紧时调小 |
| `contiguous_gradients` | 梯度放连续内存 | `true` |
| `offload_optimizer` | 优化器状态放 CPU | 显存不够时开 |
| `offload_param` | 参数放 CPU | 只在 ZeRO-3 下有效 |
| `stage3_gather_16bit_weights_on_model_save` | 保存时合并成完整权重 | ZeRO-3 下**必须 true**，否则存出来的分片没法直接用 |

**最后一个坑要特别记住**：ZeRO-3 训练完保存时，如果不设
`stage3_gather_16bit_weights_on_model_save: true`，你会得到一堆分片文件，
用普通 `from_pretrained` 加载不了。

### 5.5 四份可直接用的配置

本书 `configs/` 目录下有四份，用体检器逐份检查过：

| 文件 | 场景 | 关键设置 |
| --- | --- | --- |
| `ds_zero2.json` | 8 卡 24GB，7B 全量微调 | stage 2 + bf16 |
| `ds_zero3_offload.json` | 8 卡 24GB，13B 或显存更紧 | stage 3 + 双 offload + 激活检查点 |
| `ds_zero2_fp16.json` | 老卡（Turing / V100） | fp16 + 动态 loss scale |
| `ds_inference_zero3.json` | 推理（了解为主） | stage 3 + CPU offload |

**实测：体检器检查 `ds_zero2.json`**：

```
== configs/ds_zero2.json ==
  ZeRO 阶段      : 2
  精度           : bf16
  微批 × 累积    : 4 × 8
  卡数           : 8
  等效总 batch   : 256
  优化器         : AdamW
  调度器         : WarmupDecayLR

体检结果:
  ✅ 未发现问题
```

**实测：检查 `ds_zero3_offload.json`**：

```
  ZeRO 阶段      : 3
  精度           : bf16
  微批 × 累积    : 1 × 8
  等效总 batch   : 64
  优化器 offload : cpu
  参数 offload   : cpu
  激活检查点      : 已开启

体检结果:
  ✅ 未发现问题
```

### 5.6 体检器能发现什么

`code/03_config_lint.py` 检查六类问题：

| 检查项 | 例子 |
| --- | --- |
| batch 等式 | `train_batch_size` ≠ micro × gas × world |
| 精度冲突 | fp16 与 bf16 同时开 |
| 阶段非法 | `stage: 4` |
| offload 放错阶段 | ZeRO-2 配 `offload_param` |
| nvme 缺路径 | `device: "nvme"` 但没有 `nvme_path` |
| 拼错的字段 | 顶层出现不认识的键 |

训练前先跑一遍，能省掉一轮多机排队的时间：

```bash
D:\conda_envs\swift_env\python.exe code\03_config_lint.py configs/ds_zero3_offload.json --world_size 8
```

> **第 5 章口诀：批量三兄弟必须自洽，精度二选一；三阶保存要合并，拼错字段先体检。**

---

## 第 6 章 三条集成路径

### 6.1 裸 PyTorch：`deepspeed.initialize`

```python
import deepspeed
import torch
from torch import nn

model = MyModel()
ds_config = "configs/ds_zero2.json"

model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,          # 路径或 dict
)

for batch in loader:
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = nn.functional.cross_entropy(outputs, labels)

    model_engine.backward(loss)     # 不是 loss.backward()
    model_engine.step()             # 内部做了梯度累积/裁剪/更新
```

**三处必须改**：

| 原来 | 改成 | 原因 |
| --- | --- | --- |
| `loss.backward()` | `model_engine.backward(loss)` | 处理梯度累积与分片归约 |
| `optimizer.step()` | `model_engine.step()` | 只在累积够时才真正更新 |
| `optimizer.zero_grad()` | 删掉 | 内部自动处理 |

**这是最容易出错的地方**：保留了 `optimizer.zero_grad()`，梯度就一直不累积。

### 6.2 transformers Trainer：一行参数

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    deepspeed="configs/ds_zero2.json",   # ← 就这一行
    num_train_epochs=1,
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

**注意**：`TrainingArguments` 里写的 batch/精度会覆盖配置文件里的对应项。
社区惯例是配置文件里写 `"auto"`，让 Trainer 填：

```json
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"gradient_accumulation_steps": "auto"
```

启动（单机多卡）：

```bash
deepspeed --num_gpus=8 train.py --deepspeed configs/ds_zero2.json
```

或者直接 `python train.py`，让 Trainer 自己起进程——单机多卡时推荐前者。

### 6.3 accelerate：不改代码的多卡开关

```bash
accelerate config        # 交互式：多卡？→ DeepSpeed？→ ZeRO 阶段？
accelerate launch --num_processes 8 train.py
```

生成的配置在 `~/.cache/huggingface/accelerate/default_config.yaml`。
好处是**训练脚本里完全不出现 deepspeed**，换回单卡只要 `accelerate launch`
不加 `--num_processes`。

| 路径 | 代码侵入 | 灵活性 | 适合 |
| --- | --- | --- | --- |
| `deepspeed.initialize` | 大（改训练循环） | 最高 | 自定义训练逻辑 |
| Trainer + deepspeed | 一行 | 中 | 标准微调 |
| accelerate | 零（配置驱动） | 中 | 已有脚本快速上多卡 |

### 6.4 启动命令速查

| 场景 | 命令 |
| --- | --- |
| 单机 8 卡 | `deepspeed --num_gpus=8 train.py --deepspeed cfg.json` |
| 指定卡号 | `deepspeed --include localhost:0,1 train.py ...` |
| 多机 | `deepspeed --hostfile hosts.txt train.py ...` |
| torchrun 方式 | `torchrun --nproc_per_node=8 train.py` |
| 检查环境 | `ds_report` |

> **第 6 章口诀：backward 和 step 都换成 engine 的，zero_grad 删掉；Trainer 一行，accelerate 不改代码。**

---

## 第 7 章 推理侧：ZeRO-Inference 与 vLLM

### 7.1 ZeRO-Inference 是什么

把训练时的「分片 + offload」思路搬到推理：参数放在 CPU/NVMe，
用到哪层取哪层。**能让你在很小的显存上跑很大的模型，但非常慢。**

| 方案 | 显存需求 | 速度 | 适合 |
| --- | --- | --- | --- |
| 普通加载（fp16） | 参数量 × 2 B | 基准 | 模型放得下 |
| ZeRO-Inference + CPU offload | 很小 | 慢 10 倍以上 | 单卡跑超大模型做实验 |
| ZeRO-Inference + NVMe | 极小 | 更慢 | 极端场景 |
| **vLLM** | 参数量 × 2 B + KV | **快（PagedAttention）** | **生产服务** |

### 7.2 现实建议

| 你的需求 | 用什么 |
| --- | --- |
| 本地小模型试跑 | transformers 直接 `device_map="auto"` |
| 想跑比显存大的模型 | accelerate 的 `device_map="auto"`（自动 offload） |
| 要对外提供 API 服务 | **vLLM**（吞吐高一个量级） |
| 研究 DeepSpeed 的推理特性 | ZeRO-Inference（现在已经不是主流） |

**结论**：DeepSpeed 的强项在训练。推理侧除非有特殊需求，直接用 vLLM 或
transformers + accelerate，不要为了统一技术栈硬上 ZeRO-Inference。

> **第 7 章口诀：推理分片换显存，速度掉一个量级；要做服务上 vLLM，别拿 ZeRO 硬撑。**

---

## 第 8 章 你这台机器的现实选择

### 8.1 先明确结论

| 目标 | 本机（4GB / Windows / 单卡） | 建议 |
| --- | --- | --- |
| 学 DeepSpeed 概念 | ✅ 本书 + 三个脚本 | 已完成 |
| 跑 DeepSpeed 训练 | ❌ 无 CUDA 工具包、单卡、Windows | 上云 |
| 微调 0.5B ~ 1.5B | ⚠️ LoRA 勉强 | accelerate + QLoRA（需 bitsandbytes） |
| 微调 7B 及以上 | ❌ | 云 GPU（第 9 章） |

### 8.2 本地能做的三件事

**一、算账与配置**（不需要 GPU 多卡）

```bash
D:\conda_envs\swift_env\python.exe code\01_memory_math.py
D:\conda_envs\swift_env\python.exe code\03_config_lint.py configs/ds_zero3_offload.json
```

**二、单卡 LoRA 小实验**（承接前两本书）

```python
from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=1,
    gradient_accumulation_steps=8, bf16=False, fp16=False,
    num_train_epochs=1,
)
```

单卡时 **不要加 `deepspeed=`**——ZeRO 没有可切的东西，只会增加出错面。

**三、用现成框架做 LoRA**

你环境里已有 `ms_swift`（ms-swift 4.6.0），它的命令式接口比自己拼 DeepSpeed 简单：

```bash
swift sft --model Qwen/Qwen2.5-0.5B-Instruct --train_type lora --dataset your_data.jsonl
```

具体参数以 `swift sft --help` 为准（版本间有差异）。

### 8.3 什么时候该升级环境

| 信号 | 该做什么 |
| --- | --- |
| 想全量微调 7B | 租 8×4090 24GB 或 4×A100 40GB |
| 想跑 70B | 8×H100 80GB + ZeRO-3 + offload，或纯推理用 vLLM 多卡 |
| 只是学习概念 | 本机足够，写配置 + 算账 + 体检 |

> **第 8 章口诀：单卡不谈 ZeRO，四 G 只做实验；真要微调七 B，云端八卡起步。**

---

## 第 9 章 云端实操：从零跑通 7B 全量微调

> ⚠️ 本章所有命令**未在本机执行**（本机不具备条件）。命令按 Linux + NVIDIA 云 GPU
> 的通用流程编写，平台面板路径各家的说法不同，但 shell 里的步骤一致。

### 9.1 选机器：按你的目标倒推

| 目标 | 配置 | 依据（第 1 章表） |
| --- | --- | --- |
| 7B LoRA 微调 | 1×24GB（4090） | LoRA 只训练少量参数，不进 ZeRO-3 |
| 7B **全量**微调 | 8×24GB | ZeRO-3 需 13 GB/卡 + 激活 |
| 13B 全量 | 8×40GB 或 8×80GB | 24.2 GB/卡 + 激活，24GB 很紧 |
| 70B 全量 | 8×H100 80GB 起 | 130 GB/卡，需要更多卡或 3D 并行 |

**先算再租**：用 `code/01_memory_math.py` 把你的模型代进去，别凭感觉选配置。

### 9.2 环境准备

```bash
# 1) 建环境（云平台一般自带 conda）
conda create -n ds python=3.10 -y
conda activate ds

# 2) 装 PyTorch（按平台给的 CUDA 版本选，别抄别人的）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) 装 DeepSpeed 与依赖
pip install deepspeed transformers datasets accelerate peft

# 4) 关键一步：确认算子状态
ds_report
```

`ds_report` 的输出重点看三行：

| 输出项 | 期望 |
| --- | --- |
| `torch` / `cuda` | 版本匹配 |
| `cpu_adam ... [OK]` | ✅ 有它才能 offload |
| `nccl ... [OK]` | ✅ 多卡通信 |

如果算子显示 `[NO]`，装完后**重新编译**：

```bash
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed
```

### 9.3 一个最小可跑的分布式脚本

`train_7b_zero3.py`（模板，删减版）：

```python
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen2.5-7B-Instruct"          # 首次会下载约 15 GB

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

ds = load_dataset("json", data_files="data.jsonl", split="train")

def tokenize(ex):
    text = tok.apply_chat_template(ex["messages"], tokenize=False)
    ids = tok(text, truncation=True, max_length=1024)["input_ids"]
    return {"input_ids": ids, "labels": ids}

ds = ds.map(tokenize, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    deepspeed="configs/ds_zero3_offload.json",
    logging_steps=10,
    save_strategy="epoch",
)

Trainer(model=model, args=args, train_dataset=ds).train()
```

启动：

```bash
deepspeed --num_gpus=8 train_7b_zero3.py
```

**跑之前先体检配置**（把 world_size 改成 8）：

```bash
python code/03_config_lint.py configs/ds_zero3_offload.json --world_size 8
```

### 9.4 下载与网络

| 问题 | 做法 |
| --- | --- |
| HuggingFace 慢/连不上 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 想要国内源 | `modelscope download --model Qwen/Qwen2.5-7B-Instruct` |
| 模型太大占网费 | 平台通常有「公共模型目录」，直接挂载 |

### 9.5 跑起来之后先看什么

| 顺序 | 看什么 | 正常表现 |
| --- | --- | --- |
| 1 | `nvidia-smi` | 8 张卡都占用了显存 |
| 2 | 日志里 `total_loss` / `grad_norm` | 平稳下降，不突然飙高 |
| 3 | 每卡显存 | 接近均匀，没有一张卡爆 |
| 4 | `steps/s` | 与平台标称吞吐对比，差太多说明通信或 IO 瓶颈 |

**第一轮训练一定先跑 20 步就停**，确认配置正确再放开跑。

### 9.6 训练完怎么保存和加载

**ZeRO-3 必坑**：确保配置里有

```json
"stage3_gather_16bit_weights_on_model_save": true
```

然后：

```python
trainer.save_model("out/final")      # 会合并成完整的 safetensors
```

加载：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("out/final", torch_dtype=torch.bfloat16)
```

如果没有合并，`out/final` 里是一堆 `mp_rank_00_model_states.pt`，
普通方式加载不了，需要用 DeepSpeed 的 `zero_to_fp32.py` 手动合并。

### 9.7 成本控制

| 做法 | 省多少 |
| --- | --- |
| 先跑 20 步验证再长跑 | 避免几天白烧 |
| LoRA 代替全量（能接受效果时） | 卡数 8→1 |
| 用小模型验证流水线（0.5B） | 调试成本接近 0 |
| 关掉不必要的 `save` / `logging` | IO 时间 |

> **第 9 章口诀：先算账再租卡，ds_report 看算子；二十步验证，三阶保存要合并。**

---

## 第 10 章 排错速查

| 报错 / 现象 | 原因 | 修法 |
| --- | --- | --- |
| `MissingCUDAException: CUDA_HOME does not exist` | 没有 CUDA 工具包（本机实测） | 装 CUDA Toolkit；Windows 建议改用 WSL/Linux |
| `AssertionError: DeepSpeed backend not set` | 没用 deepspeed 启动器 | 用 `deepspeed train.py` 而不是 `python train.py` |
| `train_batch_size` 与 micro×gas×world 不匹配 | 配置自洽性错误 | 跑体检器，改成相等 |
| `cpu_adam` 未编译 | offload 需要它 | `DS_BUILD_CPU_ADAM=1 pip install deepspeed` |
| OOM 但显存看着没满 | 碎片 / 激活峰值 | 开 `contiguous_gradients`、激活检查点、降 micro |
| ZeRO-3 比 ZeRO-2 慢很多 | 3P 通信 + 参数 all-gather | 显存够就退回 ZeRO-2 |
| 训练卡住不动（rank 0 hang） | 端口冲突 / NCCL 超时 | `--master_port 6000`、检查防火墙与 IB |
| 加载 checkpoint 报缺少权重 | ZeRO-3 未合并保存 | 配 `stage3_gather_16bit_weights_on_model_save` |
| `loss` 一直是 0 或 NaN | fp16 溢出 / 学习率过大 | 换 bf16、加 `gradient_clipping`、降 lr |
| offload 后慢到无法接受 | CPU/硬盘带宽瓶颈 | 减 offload 内容，或换更大显存 |
| `pip install deepspeed` 编译几十分钟 | 在本地编译算子 | 用预编译 wheel，或设置 `DS_BUILD_OPS=0`（不用需编译的特性时） |

**排错顺序**：环境（ds_report）→ 配置（体检器）→ 通信（NCCL 日志）→ 显存/精度。

> **第 10 章口诀：先 ds_report 再体检，最后才怀疑算法。**

---

## 第 11 章 练习与答案

### 一、算账题

1. 一个 3B 模型用 AdamW + bf16 全量训练，单卡需要多少 GB（不含激活）？
2. 8 张 24GB 卡训 7B，ZeRO-2 是 24.45 GB/卡，为什么实跑还是容易 OOM？
3. 把 ZeRO-2 换成 ZeRO-3，每步通信量增加多少百分比？
4. 4 张卡、micro=2、gas=16，train_batch_size 应该是多少？
5. 70B 模型要几张 80GB 卡才能 ZeRO-3 放下（不含激活）？

### 二、判断题

6. ZeRO-1 和 DDP 的通信量一样，所以 ZeRO-1 是纯赚。
7. 单卡开 ZeRO-3 能省显存。
8. `offload_param` 在 ZeRO-2 下也能生效。
9. bf16 需要 loss scaling。
10. ZeRO-3 保存的检查点可以直接 `from_pretrained` 加载（默认配置下）。

### 三、动手题

11. 用 `01_memory_math.py` 算出「4 张 40GB 卡 + ZeRO-3」能全量训练多大模型。
12. 修改 `configs/ds_zero2.json`，让总 batch 保持 256 但 micro 降到 2。
13. 给 `ds_zero2.json` 故意写错 `train_batch_size`，用体检器看它怎么报。
14. 用 `02_zero_sim.py` 把卡数改成 4，观察各阶段每卡显存。

### 参考答案要点

1. 3e9 × 16 B ≈ 48 GB。
2. 24.45 GB 只是参数/梯度/优化器；激活值、临时张量、CUDA 上下文还要占几 GB。
3. 从 2P 到 3P，增加 50%。
4. 2 × 16 × 4 = 128。
5. 130.4 GB/卡 ÷ 80 GB ≈ 2 张，但激活和碎片要留余量，实际 ≥ 4 张更稳。
6. ✅ 对。通信量相同、显存更省，所以只要有多卡就该开。
7. ❌ 错。单卡没有可分的对象，显存不变。
8. ❌ 错。只有 ZeRO-3 会分片参数，所以 offload_param 只在 stage 3 生效。
9. ❌ 错。bf16 动态范围同 fp32，不需要；fp16 才需要。
10. ❌ 错。默认存的是分片，要开 `stage3_gather_16bit_weights_on_model_save`。
11. 4 卡 ZeRO-3 是 16 B/参数 ÷ 4 = 4 B/参数；每卡 40 GB → 约 10B 参数理论值，
    实际按 7 折算约 7B。
12. `micro=2, gas=16`（2 × 16 × 8 = 256）。
13. 会输出「train_batch_size=X，但 micro × gas × world_size = Y」。
14. 每卡显存翻倍（分片数从 8 变 4）。

---

## 附录 A 配置速查

### 顶层字段

| 字段 | 含义 | 常用值 |
| --- | --- | --- |
| `train_batch_size` | 全局 batch | micro × gas × world |
| `train_micro_batch_size_per_gpu` | 每卡微批 | 1 ~ 8（显存决定） |
| `gradient_accumulation_steps` | 累积步数 | 4 ~ 32 |
| `gradient_clipping` | 梯度裁剪 | 1.0 |
| `bf16.enabled` | bf16 | Ampere+ |
| `fp16.enabled` | fp16 | 老卡 |
| `optimizer.type` | 优化器 | `AdamW` |
| `scheduler.type` | 调度器 | `WarmupLR` / `WarmupDecayLR` |
| `zero_optimization.stage` | ZeRO 阶段 | 2 或 3 |
| `steps_per_print` | 打印间隔 | 50 |
| `activation_checkpointing` | 激活检查点 | 大模型开 |

### ZeRO 字段

| 字段 | 阶段 | 建议 |
| --- | --- | --- |
| `overlap_comm` | 1/2/3 | true |
| `contiguous_gradients` | 1/2/3 | true |
| `allgather_bucket_size` | 1/2/3 | 5e8 |
| `reduce_bucket_size` | 1/2/3 | 5e8 |
| `offload_optimizer` | 1/2/3 | 显存不足时 |
| `offload_param` | 3 | 最后手段 |
| `stage3_gather_16bit_weights_on_model_save` | 3 | **必须 true** |
| `stage3_max_live_parameters` | 3 | 1e9 |
| `stage3_prefetch_bucket_size` | 3 | 5e7 |

---

## 附录 B 术语中英对照

| 中文 | 英文 | 一句话 |
| --- | --- | --- |
| 数据并行 | data parallel (DP) | 每卡一份模型，喂不同数据 |
| 零冗余优化器 | ZeRO | 把优化器状态/梯度/参数切开分存 |
| 分片 | sharding / partition | 张量切片后分给不同卡 |
| 卸载 | offload | 把数据搬到 CPU 或硬盘 |
| 激活检查点 | activation checkpointing | 不存激活，反向时重算 |
| 梯度累积 | gradient accumulation | 多批梯度累加后再更新 |
| 微批 | micro batch | 每卡每次前向的条数 |
| 全局批 | global batch | 所有卡、所有累积的总量 |
| 归约 | reduce | 多卡的梯度求和/平均 |
| 全收集 | all-gather | 各卡取回完整分片 |
| 归约分散 | reduce-scatter | 归约后各卡只留自己那份 |
| 混合精度 | mixed precision | 部分环节用低精度 |
| 损失缩放 | loss scaling | fp16 下放大损失防下溢 |
| 通信重叠 | communication overlap | 边算边传 |
| 检查点合并 | weight gathering | ZeRO-3 保存时还原完整权重 |

---

## 附录 C 方案对比：DeepSpeed / FSDP / accelerate / vLLM

| 方案 | 定位 | 多卡显存优化 | 上手难度 | 适合 |
| --- | --- | --- | --- | --- |
| **DeepSpeed** | 分布式训练引擎 | ZeRO 0-3 + offload | 中（配置文件） | 7B+ 全量微调、多机 |
| **FSDP**（PyTorch 原生） | 分片数据并行 | 参数/梯度/优化器分片 | 中 | 官方推荐，生态整合好 |
| **accelerate** | 启动器 + 配置层 | 转发给 DeepSpeed/FSDP | 低 | 已有脚本快速上多卡 |
| **vLLM** | 推理服务 | KV cache 管理 | 低 | 生产推理 |

**怎么选**：

- 已经有 transformers Trainer → **Trainer + deepspeed 配置**，最快；
- 想少改代码 → **accelerate**；
- 只用 PyTorch、不想装 DeepSpeed → **FSDP**；
- 做推理服务 → **vLLM**，与训练栈无关。

---

## 结语

DeepSpeed 的全部魔法可以压缩成三句话：

1. **显存里最贵的是优化器状态**（12 B/参数），所以先分它（ZeRO-1）；
2. **梯度次之**（2 B/参数），再分它（ZeRO-2）——通信量不变，纯赚；
3. **参数也能分**（ZeRO-3），代价是多传 50%，只在单卡放不下参数时才用。

你本机跑不了它，但**算账、写配置、体检**这三件事已经能完整做完。
等你有 8 张卡的时候，这份配置和这套算法直接就能用。


