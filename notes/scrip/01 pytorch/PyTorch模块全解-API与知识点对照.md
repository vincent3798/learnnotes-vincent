# PyTorch 模块全解：API 与神经网络知识点的对照手册

> **环境**：本机 conda `swift_env`（torch 2.6.0+cu124）
> **写法**：先按模块列出「有哪些函数/类」，每个都给出
> **① 用法 ② 对应的神经网络知识点 ③ 关键参数与坑**；
> 每章末尾有一张「知识点 → API」的反向索引。
> **目标**：让你看到一个神经网络概念，立刻知道该调哪个 API；看到一个 API，立刻知道它对应哪块理论。

---

## 第 0 章 怎么用这本手册

### 0.1 三栏读法

每个 API 都用三栏对照：

| 栏目 | 回答的问题 |
| --- | --- |
| **API** | 叫什么、怎么调 |
| **这是什么操作** | 数学上在做什么（矩阵乘、归一化、采样……） |
| **对应知识点** | 它在神经网络理论里叫什么、解决什么问题 |

例子：

| API | 操作 | 知识点 |
| --- | --- | --- |
| `nn.Linear(in, out)` | `y = x Wᵀ + b` | 仿射变换 / 全连接层 |
| `nn.Conv2d(...)` | 局部加权求和 + 滑窗 | 局部连接 + 权重共享 |
| `nn.LayerNorm(d)` | 对最后一维做 `(x-μ)/σ·γ+β` | 归一化、训练稳定性 |
| `F.cross_entropy` | `-log softmax(logits)[target]` | 最大似然估计 |
| `optim.AdamW` | 一阶/二阶矩 + 解耦衰减 | 自适应学习率 |

### 0.2 模块地图（本手册覆盖的范围）

```
torch                 张量创建、运算、随机、设备
torch.Tensor          形状、索引、归约、就地操作
torch.autograd        计算图、梯度
torch.nn              Module / Parameter / 容器
  ├─ linear / conv / rnn / norm / dropout / embedding
  ├─ attention / transformer
  └─ loss
torch.nn.functional   无状态版本（F.xxx）
torch.optim           优化器
torch.optim.lr_scheduler  学习率策略
torch.nn.utils        梯度裁剪、参数化、剪枝
torch.utils.data      Dataset / Sampler / DataLoader
torch.amp             混合精度
torch.cuda            设备与显存
torch.distributed     多卡通信
torch.save / load     序列化
torch.profiler / compile  性能
```

### 0.3 一条主线：一个神经网络模型由什么组成

```
输入张量 → [层] → [激活] → [归一化] → … → 输出
              ↑ 参数由 nn.Parameter 管理
损失函数 → backward() → 优化器 → 学习率调度
数据由 Dataset/DataLoader 提供；多卡由 DDP 扩展
```

**整本手册就是这条主线上的每个环节。**

### 0.4 本手册的四条使用建议

1. **先查表，再读代码**：第 21 章有完整的「知识点 → API」总表。
2. **形状永远先看**：每章都标了输入/输出形状，这是最常错的地方。
3. **注意默认值**：PyTorch 很多默认值反直觉（如 `LSTM(batch_first=False)`）。
4. **每章末尾的坑**：都是真实项目里踩过的。

### 0.5 配套验证脚本

`code/01_layer_shapes.py` 会逐个打印第 5~10 章里所有层的真实输入/输出形状
（已在本机跑通，与表格一致）：

```bash
D:\conda_envs\swift_env\python.exe code\01_layer_shapes.py
```

```
Linear(4,3) on (2,5,4)      (2,5,4) -> (2,5,3)
Conv2d(3,8,3,pad=1)         (1,3,32,32) -> (1,8,32,32)
Conv2d(3,8,3,stride=2)      (1,3,32,32) -> (1,8,15,15)
MaxPool2d(2)                (1,8,32,32) -> (1,8,16,16)
AdaptiveAvgPool2d(1)        (1,8,32,32) -> (1,8,1,1)
LSTM 默认 (L,N,H)            (10,2,16) -> (10,2,32)
MHA 默认 (L,N,E)             (10,2,32) -> (10,2,32)
```

---

## 第 1 章 `torch` 顶层：张量与运算

### 1.1 模块地图

| 子族 | 代表函数 |
| --- | --- |
| 创建 | `tensor` `zeros` `ones` `full` `empty` `arange` `linspace` `eye` `rand` `randn` `randint` `randperm` |
| 形状 | `cat` `stack` `split` `chunk` `reshape` `flatten` `unsqueeze` `squeeze` `permute` `transpose` `tril` `triu` `diag` `broadcast_to` |
| 线性代数 | `matmul` `mm` `bmm` `dot` `outer` `einsum` `inverse` `svd` `qr` `solve` `cholesky` `norm` |
| 元素运算 | `exp` `log` `pow` `sqrt` `rsqrt` `abs` `clamp` `sin` `cos` `sigmoid` `tanh` `softmax` |
| 归约 | `sum` `mean` `max` `min` `amax` `amin` `argmax` `argmin` `prod` `cumsum` `norm` |
| 比较/索引 | `eq` `gt` `lt` `where` `masked_fill` `index_select` `gather` `scatter` `topk` `sort` `unique` `nonzero` |
| 随机 | `manual_seed` `normal` `uniform` `bernoulli` |
| 设备 | `device` `cuda.is_available` `cuda.device_count` |

### 1.2 创建：dtype 决定一切

| API | 用法 | 知识点 |
| --- | --- | --- |
| `torch.tensor([[1,2]])` | 从 Python 数据创建 | 整数默认 `int64` |
| `torch.zeros/ones/full` | 初始化 | 权重初始化、掩码 |
| `torch.randn` | 标准正态 | 权重初始化（Xavier/He 的前置） |
| `torch.rand` | [0,1) 均匀 | 采样、Dropout 掩码 |
| `torch.randint(low, high, size)` | 随机整数 | 假数据、token 采样 |
| `torch.randperm(n)` | 随机排列 | 打乱数据集 |
| `torch.arange(start, end, step)` | 等差 | 位置索引、RoPE |
| `torch.linspace` | 等距 n 个点 | 可视化、位置编码 |

**关键**：`torch.tensor([1,2])` 是 **int64**；送进 `nn.Linear` 会报 dtype 错。
模型输入必须 `float32`，索引/标签必须 `int64`。

### 1.3 形状操作：五个必须分清

| API | 语义 | 是否拷贝 | 知识点 |
| --- | --- | --- | --- |
| `view` | 改形状（要求连续） | 否（视图） | 张量布局 |
| `reshape` | 改形状（必要时拷贝） | 可能 | 同上 |
| `permute` | 换轴顺序 | 否（视图） | 维度对齐（如 NCHW ↔ NHWC） |
| `transpose` | 交换两轴 | 否（视图） | 同上 |
| `contiguous` | 变连续内存 | 可能 | `view` 的前置 |

```python
x = torch.randn(2, 3, 4)
x.view(2, -1).shape          # (2, 12)
x.permute(2, 0, 1).shape     # (4, 2, 3)
x.transpose(1, 2).shape      # (2, 4, 3)
```

**为什么 `view` 有时报错**：`permute`/`transpose` 后内存不连续，必须先 `.contiguous()`。

### 1.4 拼接与切分

| API | 语义 | 知识点 |
| --- | --- | --- |
| `torch.cat([a,b], dim)` | 沿已有维拼 | 特征拼接、残差 |
| `torch.stack([a,b], dim)` | 新增一维 | 批、序列堆叠 |
| `torch.split(x, size, dim)` | 按大小切 | 多头拆分 |
| `torch.chunk(x, n, dim)` | 等分 | 分片、并行 |

**易错**：`cat` 不增加维度，`stack` 增加维度。把 `stack` 当 `cat` 用，形状会多一维。

### 1.5 矩阵运算

| API | 语义 | 知识点 |
| --- | --- | --- |
| `torch.matmul(a,b)` / `a @ b` | 批量矩阵乘 | 全连接、注意力 |
| `torch.mm` | 二维矩阵乘 | 同上（只 2D） |
| `torch.bmm` | 批量矩阵乘 | 多头注意力 |
| `torch.dot` | 向量内积 | 相似度 |
| `torch.outer` | 外积 | RoPE、秩一更新 |
| `torch.einsum("bhd,bjd->bhj", q, k)` | 爱因斯坦求和 | 任意缩并、注意力分数 |
| `torch.norm` | 范数 | 梯度裁剪、正则 |

**广播规则**：从最后一维往左对齐，每维要么相等，要么有一方为 1。

```python
a = torch.randn(3, 4); b = torch.randn(4)      # (3,4)+(4,) → (3,4)
c = torch.randn(3, 1)                          # (3,4)+(3,1) → (3,4)
```

### 1.6 归约与索引

| API | 语义 | 知识点 |
| --- | --- | --- |
| `sum(dim, keepdim)` | 求和 | 损失聚合、池化 |
| `mean(dim)` | 均值 | 归一化统计量 |
| `max/min(dim)` | 极值 | 最大池化 |
| `argmax(dim)` | 极值下标 | 分类预测、贪心解码 |
| `cumsum(dim)` | 累积和 | top-p 采样、位置编码 |
| `topk(k)` | 前 k 大 | top-k 采样、MoE 路由 |
| `gather`/`scatter` | 按索引取值/写入 | 标签对齐、掩码 |
| `masked_fill(cond, v)` | 条件填充 | `-inf` 掩码 |
| `where(cond, a, b)` | 条件选择 | 掩码、门控 |
| `unique` | 去重 | 重复惩罚 |

```python
scores = torch.randn(2, 5)
scores.masked_fill(torch.tensor([[0,1,0,0,0],[0,0,0,0,1]], dtype=torch.bool), float("-inf"))
```

> 第 1 章口诀：**建张量先想 dtype，改形状先想连续；cat 不加维，stack 才加维。**

---

## 第 2 章 `torch.Tensor` 方法：对象级操作

### 2.1 方法与顶层函数的对应

| 方法 | 等价顶层函数 | 说明 |
| --- | --- | --- |
| `x.add(y)` | `torch.add(x, y)` | 逐元素 |
| `x.matmul(y)` | `torch.matmul(x, y)` | 矩阵乘 |
| `x.sum()` | `torch.sum(x)` | 归约 |
| `x.mean(0)` | `torch.mean(x, 0)` | 归约 |
| `x.max(1)` | `torch.max(x, 1)` | 返回 (values, indices) |

### 2.2 属性与方法速查

| 成员 | 作用 |
| --- | --- |
| `.shape` / `.size()` | 形状 |
| `.dtype` / `.device` | 类型 / 设备 |
| `.ndim` / `.numel()` | 维度数 / 元素数 |
| `.to(device/dtype)` | 搬运或转型 |
| `.float()` `.half()` `.long()` `.bool()` | 转 dtype |
| `.item()` | 单个元素 → Python 数（**会同步 GPU**） |
| `.tolist()` | → Python 列表 |
| `.numpy()` | → NumPy（需先 `.cpu()`） |
| `.clone()` | 复制（保留在计算图里） |
| `.detach()` | 断开计算图（共享内存） |
| `.requires_grad_()` | 原地设置是否求导 |
| `.backward()` | 反向传播 |
| `.grad` | 梯度 |

### 2.3 视图 vs 拷贝：最容易混淆的一对

```python
a = torch.arange(6).view(2, 3)
b = a.view(3, 2)      # 共享内存：改 b 会改 a
c = a.clone()         # 独立内存
d = a.detach()        # 共享内存，但脱离计算图
```

| 操作 | 共享内存 | 在计算图内 |
| --- | --- | --- |
| `view` / `reshape` | 通常是 | 是 |
| `clone` | 否 | 是 |
| `detach` | 是 | **否** |

**知识点**：PyTorch 的「视图」对应计算机里的**内存别名（alias）**；
梯度计算依赖版本计数，原地修改共享张量可能导致反向报错。

> 第 2 章口诀：**视图共享内存，clone 独立，detach 断图；item 会同步。**

---

## 第 3 章 `torch.autograd`：计算图与梯度

### 3.1 API

| API | 作用 | 知识点 |
| --- | --- | --- |
| `requires_grad=True` | 标记需要求导 | 可训练参数 |
| `loss.backward()` | 反向传播 | 链式法则 |
| `.grad` | 梯度存放处 | 叶子张量 |
| `torch.no_grad()` | 不建图 | 推理、参考模型 |
| `torch.inference_mode()` | 更彻底的不建图 | 纯推理 |
| `torch.autograd.grad(outputs, inputs)` | 显式求梯度（不累积） | 高阶梯度、雅可比 |
| `retain_graph=True` | 保留计算图 | 多次反向 |
| `torch.autograd.set_detect_anomaly(True)` | 定位 NaN 来源 | 调试 |

### 3.2 计算图长什么样

```
参数 w ──► 线性层 ──► 激活 ──► 输出 ──► loss
              ↑ 前向时记录 op 与输入（用于反向）
              ↓ backward 时按链式法则倒推
```

| 概念 | 含义 | 对应 API |
| --- | --- | --- |
| 叶子张量 | 用户创建的、存梯度的张量 | `nn.Parameter`、`requires_grad=True` 的张量 |
| 非叶子张量 | 中间结果，默认不存梯度 | 需要时 `retain_grad()` |
| 梯度累积 | gradient 是累加不是覆盖 | 所以每步要 `zero_grad()` |

### 3.3 三种「不建图」的区别

| 方式 | 记录操作 | 可再参与求导 | 用途 |
| --- | --- | --- | --- |
| 正常 | ✅ | ✅ | 训练 |
| `no_grad` | ❌ | ❌ | 评估、参考模型 |
| `inference_mode` | ❌（更严格） | ❌ | 纯推理，最快 |

**知识点**：反向传播的本质是**链式法则的动态规划实现**；
省掉建图 = 省掉保存中间激活的显存。

> 第 3 章口诀：**叶子存梯度，梯度会累加；评估关图，日志 detach。**

---

## 第 4 章 `torch.nn` 骨架：Module、Parameter 与容器

### 4.1 核心类

| 类 | 作用 | 知识点 |
| --- | --- | --- |
| `nn.Module` | 所有层的基类 | 层 = 可组合的函数 |
| `nn.Parameter` | 可训练张量 | 模型参数 θ |
| `nn.Sequential` | 顺序容器 | 函数复合 |
| `nn.ModuleList` | 列表容器（会被注册） | 多层堆叠 |
| `nn.ModuleDict` | 字典容器 | 多分支/多任务 |
| `nn.ParameterList` | 参数列表 | 手工管理的参数 |
| `nn.Identity` | 恒等层 | 占位、消融实验 |

### 4.2 Module 的四个必备动作

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()                  # ① 必须先调
        self.fc = nn.Linear(10, 3)          # ② 子模块自动注册
        self.register_buffer("scale", torch.tensor(1.0))   # ③ 非训练常量
    def forward(self, x):
        return self.fc(x) * self.scale      # ④ 计算逻辑
```

| 动作 | 为什么 |
| --- | --- |
| `super().__init__()` | 初始化内部的 `_modules` / `_parameters` |
| 赋给 `self.*` | 才会被注册、被 `.to()` 搬动、被优化器看到 |
| `register_buffer` | 进 state_dict、随设备搬动、不参与梯度 |
| 实现 `forward` | `model(x)` 会触发 hooks |

### 4.3 状态管理

| API | 作用 |
| --- | --- |
| `state_dict()` | 参数 + buffer 的字典 |
| `load_state_dict(sd, strict=True)` | 加载；`strict=False` 允许键不匹配 |
| `train()` / `eval()` | 切换 Dropout/BN 行为 |
| `named_parameters()` | 遍历（冻结、分组） |
| `named_modules()` | 遍历子模块（注入、替换） |
| `apply(fn)` | 递归初始化 |
| `requires_grad_(False)` | 整模型冻结 |

**知识点**：`state_dict` 是「**参数与结构分离**」的体现——
结构由代码定义，权重只是一组键值对。

> 第 4 章口诀：**层在 init 里建，buffer 不进优化器；train/eval 切换行为。**

---

## 第 5 章 全连接族：`nn.Linear` / `Bilinear` / `LazyLinear`

### 5.1 `nn.Linear`：最基础的层

```python
nn.Linear(in_features, out_features, bias=True)
```

| 项 | 内容 |
| --- | --- |
| 数学 | `y = x Wᵀ + b`，`W ∈ R^{out×in}` |
| 形状 | `(*, in_features) → (*, out_features)`（只改最后一维） |
| 参数量 | `in×out + out` |
| 知识点 | **仿射变换**、全连接层、特征线性组合 |

```python
layer = nn.Linear(4, 3)
x = torch.randn(2, 5, 4)       # 任意前导维度
y = layer(x)                    # (2, 5, 3)
```

**为什么最后一维不变以外都能用**：全连接层对每个位置独立作用，
等价于对最后一维做一次线性映射（这也是 Transformer 里 FFN 的用法）。

**`bias` 的作用**：没有 bias 时，所有超平面必须过原点，表达能力受限。

### 5.2 `nn.Bilinear`：双线性交互

```python
nn.Bilinear(in1_features, in2_features, out_features)
# y = x1ᵀ W x2 + b
```

| 用途 | 知识点 |
| --- | --- |
| 特征交叉、关系建模 | 双线性形式（推荐系统、视觉问答） |

### 5.3 `nn.LazyLinear`：延迟初始化

```python
nn.LazyLinear(out_features)     # in_features 由第一次 forward 推断
```

**用途**：不确定输入维度时的快速原型；第一次前向才会创建参数。
**坑**：`state_dict` 在第一次 forward 之前是空的。

### 5.4 参数量估算

| 层 | 参数量 |
| --- | --- |
| `Linear(d, h)` | `d×h + h` |
| `Bilinear(d1, d2, h)` | `d1×d2×h + h` |

Transformer 里 FFN 通常是 `d → 4d → d`，参数量 `8d²`；这也是为什么
「加宽一层 FFN」比「加一层注意力」更贵。

> 第 5 章口诀：**Linear 只改最后一维；bias 决定超平面能否平移。**

---

## 第 6 章 卷积与池化族

### 6.1 卷积家族

| 类 | 输入形状 | 知识点 |
| --- | --- | --- |
| `nn.Conv1d` | `(N, C_in, L)` | 一维卷积（文本、音频） |
| `nn.Conv2d` | `(N, C_in, H, W)` | 二维卷积（图像） |
| `nn.Conv3d` | `(N, C_in, D, H, W)` | 三维卷积（视频、体素） |
| `nn.ConvTranspose1d/2d/3d` | 同上 | 转置卷积（上采样） |
| `nn.LazyConv2d` | 自动推断输入通道 | 原型 |

```python
nn.Conv2d(in_channels, out_channels, kernel_size,
          stride=1, padding=0, dilation=1, groups=1,
          bias=True, padding_mode='zeros')
```

### 6.2 输出尺寸与参数量

**输出尺寸**：

```
H_out = floor((H_in + 2·padding - dilation·(kernel-1) - 1) / stride + 1)
```

**参数量**：

```
out_channels × (in_channels / groups) × kh × kw + out_channels
```

| 概念 | 公式 | 知识点 |
| --- | --- | --- |
| 感受野 | 逐层累积 | 深层看更大范围 |
| 权重共享 | 同一卷积核全图滑动 | 平移等变、参数大减 |
| `groups=C_in=C_out` | 深度可分离卷积 | MobileNet |
| `dilation` | 空洞卷积 | 不降分辨率扩大感受野 |
| `padding` | 边界补零 | 保持尺寸 |

### 6.3 池化家族

| 类 | 作用 | 知识点 |
| --- | --- | --- |
| `nn.MaxPool2d` | 取窗口最大 | 下采样、保留显著特征 |
| `nn.AvgPool2d` | 窗口平均 | 平滑下采样 |
| `nn.AdaptiveAvgPool2d(output_size)` | 输出固定尺寸 | 全局平均池化（GAP） |
| `nn.AdaptiveMaxPool2d` | 同上，取最大 | 同上 |
| `nn.LPPool2d` | Lp 范数池化 | 广义池化 |
| `nn.MaxUnpool2d` | 反池化 | 分割网络 |
| `nn.FractionalMaxPool2d` | 分数池化 | 随机下采样 |

**全局平均池化**（`AdaptiveAvgPool2d(1)`）是分类网络的标准收尾：
把 `(N,C,H,W)` 压成 `(N,C,1,1)`，让分类头与输入尺寸无关。

### 6.4 上下采样工具（F 里）

| API | 作用 |
| --- | --- |
| `F.interpolate(x, size/scale_factor, mode='nearest'/'bilinear')` | 插值缩放 |
| `F.unfold` / `F.fold` | 展开成列 / 还原（im2col） |
| `F.pixel_shuffle` / `pixel_unshuffle` | 亚像素上/下采样 |
| `F.grid_sample` | 可变形采样 |

**知识点**：`unfold` 揭示了卷积的「矩阵乘法本质」——im2col 后就是一次 GEMM。

## 第 7 章 循环族：`RNN` / `LSTM` / `GRU`

### 7.1 类与形状

```python
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        batch_first=False, dropout=0.0, bidirectional=False)
```

| 默认 `batch_first=False` | 形状 |
| --- | --- |
| 输入 | `(L, N, input_size)` |
| 输出 `output` | `(L, N, hidden_size × 方向数)` |
| `h_n` | `(num_layers × 方向数, N, hidden_size)` |
| `c_n`（仅 LSTM） | 同 `h_n` |

`batch_first=True` 时把前两维换成 `(N, L, ...)`。

**这是全 PyTorch 最反直觉的默认值**：绝大多数框架和代码用 batch 在前，
但 `nn.LSTM` 默认 seq 在前。MiniMind 里就必须显式写 `batch_first=True`。

### 7.2 三个循环单元的差别

| 类 | 核心机制 | 知识点 |
| --- | --- | --- |
| `nn.RNN` | `h_t = tanh(W x_t + U h_{t-1} + b)` | 最朴素循环，易梯度消失 |
| `nn.GRU` | 重置门 + 更新门 | 门控保留长依赖，参数少 |
| `nn.LSTM` | 输入门 / 遗忘门 / 输出门 + 细胞状态 | 长短期记忆，最稳但最慢 |

**LSTM 的门**（同一份权重，四个门）：

```
i_t = σ(W_i x_t + U_i h_{t-1})      输入门：写多少
f_t = σ(W_f x_t + U_f h_{t-1})      遗忘门：旧细胞状态保留多少
g_t = tanh(W_g x_t + U_g h_{t-1})   候选值
o_t = σ(W_o x_t + U_o h_{t-1})      输出门
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     细胞状态（信息高速公路）
h_t = o_t ⊙ tanh(c_t)
```

**知识点**：循环网络的训练用 **BPTT**（沿时间反向传播），
梯度连乘导致**梯度消失/爆炸**；门控的本质是给梯度一条「加法通路」。

### 7.3 配套工具

| API | 作用 |
| --- | --- |
| `nn.RNNCell` / `LSTMCell` / `GRUCell` | 单步单元（自己写循环） |
| `nn.utils.rnn.pad_sequence(seqs, batch_first, padding_value)` | 变长序列补齐 |
| `nn.utils.rnn.pack_padded_sequence` | 打包，跳过 padding 计算 |
| `nn.utils.rnn.pad_packed_sequence` | 解包 |
| `nn.utils.rnn.pack_sequence` | 自动按长度打包 |

### 7.4 坑

| 坑 | 说明 |
| --- | --- |
| `batch_first` 默认 False | 输入写反会得到「看似正确」的错误结果 |
| `dropout` 只作用于层间 | 单层 LSTM 设 dropout 无效 |
| 不传 `h_0` 时默认全零 | 长序列建议手动初始化 |
| 变长序列不 pack | padding 会污染最后一步的隐状态 |
| `bidirectional=True` 时输出维度翻倍 | 下游层输入维度要改 |

> 第 7 章口诀：**LSTM 三门一记忆，batch_first 记得开。**

---

## 第 8 章 归一化族

### 8.1 全家福

| 类 | 归一化维度 | 典型场景 |
| --- | --- | --- |
| `nn.BatchNorm1d/2d/3d(num_features)` | 对 batch（+空间）统计 | CNN |
| `nn.SyncBatchNorm` | 跨卡统计 | 多卡小 batch |
| `nn.LayerNorm(normalized_shape)` | 对每个样本的最后若干维 | Transformer |
| `nn.GroupNorm(num_groups, num_channels)` | 通道分组内统计 | 小 batch CNN |
| `nn.InstanceNorm2d(num_features)` | 单样本单通道 | 风格迁移 |
| `nn.RMSNorm(normalized_shape)` | 只除以均方根 | LLaMA/Qwen |
| `nn.LocalResponseNorm` | 邻近通道 | AlexNet 时代 |

### 8.2 公式对照

| 方法 | 公式 | 统计范围 |
| --- | --- | --- |
| BatchNorm | `(x - μ_B)/√(σ_B²+ε) · γ + β` | 整个 batch + 空间 |
| LayerNorm | 同上 | 样本内最后 d 维 |
| GroupNorm | 同上 | 样本内每组通道 |
| InstanceNorm | 同上 | 样本内单通道 |
| RMSNorm | `x/√(mean(x²)+ε) · γ` | 最后 d 维，**不减均值、无 β** |

**BatchNorm 的输入**：`(N, C, ...)`，按通道统计；`(N, C)` 也可（1d）。
**LayerNorm 的输入**：最后一维必须等于 `normalized_shape`。

### 8.3 训练/推理的差别（只对 BN/IN 有）

| 模式 | 用什么统计量 |
| --- | --- |
| `model.train()` | 当前 batch 的均值方差，并更新 `running_mean/var` |
| `model.eval()` | 用累积的 `running_mean/var` |

`momentum=0.1` 的含义：

```
running = (1 - momentum) × running + momentum × batch_stat
```

**知识点**：
- 归一化的最初解释是「缓解内部协变量偏移」；
- 现代理解更接近「**平滑损失曲面、允许更大学习率**」；
- BN 依赖 batch 统计 → 小 batch 不稳；LN 与 batch 无关 → 适合变长序列；
- `γ, β`（仿射参数）让网络能「撤销」归一化，保留表达能力。

### 8.4 坑

| 坑 | 说明 |
| --- | --- |
| BN 用 batch=1 训练 | 方差为 0，报错或退化 |
| 忘了 `model.eval()` | 推理时用了 batch 统计，结果抖动 |
| LayerNorm 的 `normalized_shape` 写错 | 必须匹配最后几维 |
| RMSNorm 的 `eps=None` | 会用当前 dtype 的极小值，fp16 下要注意 |
| 分布式下 BN 不同步 | 需要 `SyncBatchNorm` |

> 第 8 章口诀：**BN 看批，LN 看样本；训练用实时统计，推理用滑动平均。**

---

## 第 9 章 激活函数与 Dropout

### 9.1 激活函数全家福

| 类 / 函数 | 公式要点 | 知识点 |
| --- | --- | --- |
| `nn.ReLU` / `F.relu` | `max(0,x)` | 稀疏、缓解梯度消失；可能「死亡」 |
| `nn.LeakyReLU(0.01)` | 负半轴小斜率 | 缓解死亡 ReLU |
| `nn.PReLU()` | 负半轴斜率可学习 | 同上，参数化 |
| `nn.ELU(alpha)` | 负半轴指数衰减 | 均值接近 0 |
| `nn.SELU` | 自归一化 | 配合 AlphaDropout |
| `nn.GELU(approximate='none')` | 高斯误差线性单元 | Transformer/BERT 常用 |
| `nn.SiLU` / `F.silu` | `x·σ(x)` | SwiGLU 的门控单元 |
| `nn.Mish` | `x·tanh(softplus(x))` | 平滑非单调 |
| `nn.Sigmoid` | `1/(1+e^{-x})` | 二分类输出、门控；饱和区梯度小 |
| `nn.Tanh` | `(e^x-e^{-x})/(e^x+e^{-x})` | 值域 [-1,1]，零中心 |
| `nn.Softmax(dim)` | 归一化成概率 | 多分类输出（**不是激活**，是归一化） |
| `nn.LogSoftmax(dim)` | `log softmax` | 配 NLLLoss，数值更稳 |
| `nn.GLU(dim)` | `a ⊙ σ(b)` | 门控线性单元（SwiGLU 的祖先） |
| `nn.Softplus` / `Softsign` | 平滑版 ReLU / tanh | 回归输出 |
| `nn.Hardswish` / `Hardsigmoid` / `ReLU6` | 近似版本 | 移动端 |

**知识点**：
- 没有激活函数，多层线性堆叠仍等价于一层线性；
- 非线性是**万能逼近**的前提；
- 饱和激活（sigmoid/tanh）在大 |x| 时梯度趋 0 → 梯度消失；
- 现代大模型主流：GELU / SiLU（SwiGLU）。

### 9.2 Dropout 家族

| 类 | 作用 | 知识点 |
| --- | --- | --- |
| `nn.Dropout(p=0.5)` | 训练时随机置零并缩放 | 正则化、隐式集成 |
| `nn.Dropout1d/2d/3d` | 按通道/特征图丢弃 | CNN 的空间 dropout |
| `nn.AlphaDropout` | 保持均值方差的 dropout | 配 SELU |
| `nn.FeatureAlphaDropout` | 特征级版本 | CNN |

**训练与推理**：

```
训练：y = x ⊙ mask / (1-p)     （inverted dropout，推理时不用缩放）
推理：y = x
```

**为什么除以 (1-p)**：保持期望不变。

**知识点**：Dropout ≈ 训练指数级多个子网络并做集成；
它和 weight decay、数据增强同属正则化手段。

### 9.3 坑

| 坑 | 说明 |
| --- | --- |
| 推理忘 `model.eval()` | dropout 仍在随机丢神经元 |
| 用 `F.dropout` 没传 `training` | 默认 `training=True`，评估时仍生效 |
| Softmax 后再算 CE | 重复 softmax，梯度变小 |
| ReLU 学习率过大 | 大量神经元永久死亡 |
| Dropout 放在注意力输出前 | 与注意力 dropout 重复 |

> 第 9 章口诀：**非线性靠激活，正则化靠 dropout；训练缩放，推理直通。**

---

## 第 10 章 Embedding、注意力与 Transformer

### 10.1 `nn.Embedding`：查表

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
             max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
```

| 项 | 内容 |
| --- | --- |
| 输入 | 整数索引 `(...,)`，dtype **long** |
| 输出 | `(..., embedding_dim)` |
| 参数量 | `num_embeddings × embedding_dim` |
| 知识点 | 离散符号 → 连续向量（词嵌入）；等价于 one-hot × 权重矩阵 |

```python
emb = nn.Embedding(100, 8)
x = torch.tensor([[1, 2, 3]])      # (1,3)
emb(x).shape                       # (1,3,8)
```

| 参数 | 作用 |
| --- | --- |
| `padding_idx` | 该行权重固定为 0、梯度为 0 |
| `max_norm` | 超过范数时原地归一化 |
| `sparse=True` | 稀疏梯度（配 SparseAdam） |
| `nn.EmbeddingBag` | 对一组索引做 sum/mean/max，省内存 |

**权重共享**：语言模型常把 `embed_tokens.weight` 与 `lm_head.weight` 指向同一张量
（MiniMind 的 `tie_word_embeddings=True`），省参数量。

### 10.2 注意力：从公式到 API

**缩放点积注意力**：

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

| API | 说明 |
| --- | --- |
| `F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0.0, is_causal=False)` | 融合实现，自动选后端 |
| `nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False)` | 完整多头层 |

**`MultiheadAttention` 的形状**（默认 `batch_first=False`）：

| 输入 | 形状 |
| --- | --- |
| `query` | `(L, N, E)` |
| `key` / `value` | `(S, N, E)`（可用 `kdim`/`vdim` 不同维度） |
| 输出 | `(L, N, E)` + 注意力权重 `(N, L, S)` |

**三个 mask 参数**：

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `attn_mask` | bool / float | bool：True 表示**屏蔽**；float：直接加到分数上 |
| `key_padding_mask` | bool `(N,S)` | True 表示该位置是 padding |
| `need_weights` | bool | 是否返回注意力矩阵 |

**为什么要除以 √d_k**：点积的方差随维度线性增长，不缩放会让 softmax 进入饱和区，
梯度趋 0。缩放让方差回到 1 附近。

**多头为什么有用**：把 `E` 拆成 `h` 个 `d_k=E/h` 的子空间，
不同头学习不同的关系模式；拼接后再线性变换。

### 10.3 Transformer 模块

| 类 | 作用 |
| --- | --- |
| `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=False)` | 一层编码器：自注意力 + FFN |
| `nn.TransformerDecoderLayer(...)` | 一层解码器：自注意力 + 交叉注意力 + FFN |
| `nn.TransformerEncoder(layer, num_layers, norm=None)` | 堆叠编码器 |
| `nn.TransformerDecoder(layer, num_layers, norm=None)` | 堆叠解码器 |
| `nn.Transformer(...)` | 完整编码器-解码器 |

**`norm_first` 是 pre-LN 与 post-LN 的开关**：

| 配置 | 结构 | 特点 |
| --- | --- | --- |
| `norm_first=False`（默认） | Post-LN | 原始论文，需 warmup |
| `norm_first=True` | Pre-LN | 训练更稳，现代大模型标配 |

### 10.4 位置编码

注意力本身对顺序不敏感，必须显式注入位置信息：

| 方式 | 做法 | 知识点 |
| --- | --- | --- |
| 正弦位置编码 | 固定 sin/cos 表 | 原始 Transformer |
| 可学习位置嵌入 | `nn.Embedding(max_len, d)` | BERT/GPT-2 |
| RoPE | 旋转 q/k 向量 | LLaMA/Qwen |
| ALiBi | 给注意力分数加线性偏置 | 长文本 |

**MiniMind 用的是 RoPE**：`precompute_freqs_cis` 预计算 cos/sin，
`apply_rotary_pos_emb` 对 q/k 做旋转。

### 10.5 坑

| 坑 | 说明 |
| --- | --- |
| `MultiheadAttention` 默认 seq 在前 | 与 LSTM 同样的坑 |
| `attn_mask` 的 bool 语义是「屏蔽」 | 与很多库相反 |
| `is_causal=True` 与自定义 mask 混用 | 可能重复屏蔽 |
| 多头数不整除维度 | `embed_dim % num_heads != 0` 直接报错 |
| 忘记位置编码 | 模型变成「词袋」 |

> 第 10 章口诀：**Embedding 查表，注意力算相关；除以根号 d，多头看子空间。**

---

## 第 11 章 损失函数全表

### 11.1 分类损失

| 类 | 输入 | 目标 | 知识点 |
| --- | --- | --- | --- |
| `nn.CrossEntropyLoss` | `(N,C)` **logits** | `(N,)` long | 多分类；内部含 softmax |
| `nn.NLLLoss` | `(N,C)` **log 概率** | `(N,)` long | 配 `log_softmax` |
| `nn.BCEWithLogitsLoss` | 任意形状 **logits** | 同形状 0/1 float | 二分类/多标签，数值稳定 |
| `nn.BCELoss` | `(0,1)` 概率 | 同形状 | 需先 sigmoid |
| `nn.MultiLabelSoftMarginLoss` | logits | 多标签 | 多标签分类 |
| `nn.MultiMarginLoss` | logits | `(N,)` | 多类 SVM 式 |
| `nn.SoftMarginLoss` | logits | ±1 | 二分类 hinge 平滑版 |

**CrossEntropyLoss 的关键参数**：

| 参数 | 作用 |
| --- | --- |
| `weight` | 每类权重（类别不平衡） |
| `ignore_index=-100` | 跳过指定目标（padding、不训练位置） |
| `label_smoothing` | 标签平滑，防过拟合 |
| `reduction` | `'mean'` / `'sum'` / `'none'` |

**知识点**：交叉熵 = 负对数似然 = 最大似然估计的损失形式；
`CE(logits, y) ≡ NLL(log_softmax(logits), y)`。

### 11.2 回归损失

| 类 | 公式 | 特点 |
| --- | --- | --- |
| `nn.MSELoss` | `(y-ŷ)²` | 对大误差敏感 |
| `nn.L1Loss` | `|y-ŷ|` | 对异常值稳健，零点不平滑 |
| `nn.SmoothL1Loss(beta=1.0)` | Huber 形式 | 小误差 L2、大误差 L1 |
| `nn.HuberLoss(delta=1.0)` | 同上（另一实现） | 目标检测常用 |
| `nn.PoissonNLLLoss` | 泊松负似然 | 计数数据 |
| `nn.GaussianNLLLoss` | 高斯负似然（可学方差） | 不确定性估计 |

### 11.3 分布/度量/序列损失

| 类 | 用途 | 知识点 |
| --- | --- | --- |
| `nn.KLDivLoss(log_target=False)` | 分布对齐 | KL 散度；输入是 log 概率 |
| `nn.CosineEmbeddingLoss(margin)` | 相似/不相似对 | 对比学习 |
| `nn.TripletMarginLoss(margin, p=2)` | 三元组 | 度量学习（anchor/pos/neg） |
| `nn.MarginRankingLoss(margin)` | 成对排序 | 排序学习 |
| `nn.HingeEmbeddingLoss` | 成对相似性 | 嵌入学习 |
| `nn.CTCLoss` | 变长序列对齐 | 语音识别（无需帧级标注） |

**KLDivLoss 的 reduction**：默认 `'mean'` 是逐元素平均；
分布级对齐通常用 `'batchmean'`（除以 batch 大小）。

### 11.4 坑

| 坑 | 说明 |
| --- | --- |
| CE 前手动 softmax | 梯度被压缩，收敛慢 |
| BCE 用 logits 却调了 `BCELoss` | 必须先 sigmoid；或者直接用 `BCEWithLogitsLoss` |
| 全部目标都是 `ignore_index` | 返回 nan |
| `reduction='none'` 后忘了加权求和 | 损失形状是 `(N,)`，backward 会报错或行为错 |
| KL 的输入不是 log 概率 | 结果没有意义 |
| 标签 dtype 不是 long | CE 直接报错 |

## 第 12 章 `torch.optim`：优化器

### 12.1 全家福（torch 2.6 实有 15 个）

| 优化器 | 机制 | 知识点 |
| --- | --- | --- |
| `SGD` | `θ ← θ - lr·g` | 最基础的梯度下降 |
| `SGD(momentum=0.9)` | 累积速度 | 动量法 |
| `SGD(nesterov=True)` | 前瞻梯度 | Nesterov 加速 |
| `Adam` | 一阶矩 + 二阶矩 + 偏差修正 | 自适应学习率 |
| `AdamW` | Adam + **解耦**权重衰减 | 大模型默认 |
| `Adamax` | 无穷范数版本 | Adam 变体 |
| `NAdam` | Nesterov + Adam | 变体 |
| `RAdam` | 修正早期方差 | 变体 |
| `Adagrad` | 累积平方梯度 | 稀疏特征 |
| `Adadelta` | 窗口化 Adagrad | 免学习率 |
| `RMSprop` | 指数滑动平方梯度 | RNN 常用 |
| `Rprop` | 只看符号 | 全批量 |
| `ASGD` | 平均随机梯度 | 变体 |
| `LBFGS` | 二阶拟牛顿 | 小模型、需 closure |
| `SparseAdam` | 稀疏梯度版 Adam | 大 Embedding |

### 12.2 关键参数

```python
torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0,
                weight_decay=0, nesterov=False)
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)
torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay=1e-2, amsgrad=False)
```

**Adam 与 AdamW 的区别**：

| | Adam | AdamW |
| --- | --- | --- |
| 权重衰减 | 混进梯度（被自适应缩放） | **解耦**，直接作用于参数 |
| 效果 | 正则强度受 lr 影响 | 衰减与 lr 独立，正则更纯 |

### 12.3 参数组：不同层不同学习率

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(),     "lr": 1e-3},
], weight_decay=0.01)
```

| API | 作用 |
| --- | --- |
| `optimizer.param_groups` | 参数组列表（每步改 lr 就改这里） |
| `optimizer.add_param_group(...)` | 后加参数组 |
| `optimizer.state_dict()` | 保存优化器状态（动量等） |
| `optimizer.zero_grad(set_to_none=True)` | 清梯度（省内存） |
| `optimizer.step()` | 按梯度更新 |

**知识点**：迁移学习常让「底座小 lr、新头大 lr」；参数组就是这个操作的原生支持。

### 12.4 坑

| 坑 | 说明 |
| --- | --- |
| 忘了 `zero_grad` | 梯度累加，学习率等效变大 |
| `weight_decay` 用在 bias/LayerNorm 上 | 常见做法是分开不衰减 |
| LBFGS 不传 closure | 直接报错 |
| 把 `parameters()` 当生成器反复用 | 只可迭代一次，需要时转 list |
| 恢复训练不恢复优化器状态 | 动量/二阶矩丢失，前期抖动 |

> 第 12 章口诀：**Adam 混衰减，AdamW 解耦；参数组管分层学习率。**

---

## 第 13 章 `torch.optim.lr_scheduler`：学习率策略

### 13.1 全家福

| 调度器 | 行为 | 知识点 |
| --- | --- | --- |
| `StepLR(step_size, gamma)` | 每 N 轮乘 gamma | 阶梯衰减 |
| `MultiStepLR(milestones, gamma)` | 指定轮次衰减 | 分阶段 |
| `ExponentialLR(gamma)` | 每轮乘 gamma | 指数衰减 |
| `CosineAnnealingLR(T_max, eta_min)` | 余弦降到地板 | 余弦退火 |
| `CosineAnnealingWarmRestarts(T_0, T_mult)` | 周期性重启 | 热重启（SGDR） |
| `OneCycleLR(max_lr, total_steps)` | 先升后降 | 1-cycle 策略 |
| `CyclicLR(base_lr, max_lr)` | 三角循环 | 循环学习率 |
| `ReduceLROnPlateau(factor, patience)` | 指标不降就降 | 自适应降 lr |
| `LambdaLR(lr_lambda)` | 自定义函数 | 任意曲线 |
| `MultiplicativeLR` | 每轮乘自定义因子 | 同上 |
| `ConstantLR` / `LinearLR` / `PolynomialLR` | 常数/线性/多项式 | warmup 组件 |
| `SequentialLR` / `ChainedScheduler` | 串联多个 | 组合策略 |

### 13.2 正确用法

```python
for step in range(total):
    optimizer.step()
    scheduler.step()          # 顺序不能反
    print(scheduler.get_last_lr())
```

| 注意 | 说明 |
| --- | --- |
| `scheduler.step()` 在 `optimizer.step()` 之后 | 否则第一条曲线错位 |
| `ReduceLROnPlateau` 要传指标 | `scheduler.step(val_loss)` |
| `T_max` 是总步数不是轮数 | 算错会提前归零 |
| 续训要存 `scheduler.state_dict()` | 否则 lr 从头开始 |

### 13.3 Warmup 的标准写法

```python
warmup = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s + 1) / 1000))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[1000])
```

**知识点**：warmup 让训练初期用小 lr 稳定统计量（尤其是 Adam 的二阶矩与 BN 统计）。

> 第 13 章口诀：**先 step 优化器，再 step 调度器；T_max 按总步算。**

---

## 第 14 章 `torch.nn.utils`：梯度与参数的手术刀

### 14.1 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)
torch.nn.utils.clip_grad_value_(parameters, clip_value)
```

| 函数 | 做法 | 知识点 |
| --- | --- | --- |
| `clip_grad_norm_` | 按总范数等比缩放 | 梯度爆炸 |
| `clip_grad_value_` | 逐元素截断 | 同上，更粗暴 |

**必须在 `scaler.unscale_()` 之后**（混合精度时），否则裁的是放大后的梯度。

### 14.2 参数向量化

| API | 作用 |
| --- | --- |
| `parameters_to_vector(params)` | 所有参数拼成一维向量 |
| `vector_to_parameters(vec, params)` | 还原 |

**知识点**：用于实现简单的进化策略、参数平均、模型融合。

### 14.3 权重归一化与谱归一化

| API | 作用 | 知识点 |
| --- | --- | --- |
| `weight_norm(module, 'weight')` | `w = g · v/‖v‖` | 权重归一化 |
| `spectral_norm(module, 'weight')` | 限制最大奇异值 | Lipschitz 约束（GAN） |
| `remove_weight_norm` / `remove_spectral_norm` | 撤销 | — |

### 14.4 剪枝

```python
import torch.nn.utils.prune as prune
prune.l1_unstructured(layer, name="weight", amount=0.3)
prune.random_unstructured(layer, name="weight", amount=0.2)
prune.global_unstructured([(l1,'weight'), (l2,'weight')], pruning_method=..., amount=0.5)
prune.remove(layer, "weight")        # 把 mask 固化进权重
```

| API | 作用 |
| --- | --- |
| `l1_unstructured` | 按 L1 大小剪最小权重 |
| `random_unstructured` | 随机剪 |
| `ln_structured` | 按通道/维度结构化剪枝 |
| `custom_from_mask` | 自定义掩码 |
| `is_pruned` | 检查 |
| `remove` | 固化 |

**知识点**：剪枝 = 给权重加一个 0/1 掩码；结构化剪枝才能真正加速推理。

### 14.5 参数化（parametrizations）

```python
from torch.nn.utils import parametrize
parametrize.register_parametrization(layer, "weight", MyParametrization())
```

**知识点**：把 `weight` 变成「原始参数 + 变换」，
是实现自定义权重约束（正交、正定、低秩）的现代方式。

> 第 14 章口诀：**梯度爆炸先裁剪；归一化改权重，剪枝加掩码。**

---

## 第 15 章 `torch.utils.data`：数据管线

### 15.1 Dataset 家族

| 类 | 作用 |
| --- | --- |
| `Dataset` | 基类，实现 `__len__` + `__getitem__` |
| `IterableDataset` | 流式数据，实现 `__iter__` |
| `TensorDataset(*tensors)` | 按第一维切片打包 |
| `ConcatDataset([...])` | 拼接多个数据集 |
| `Subset(ds, indices)` | 子集 |
| `random_split(ds, lengths)` | 随机划分 |
| `ChainDataset([...])` | 串联流式数据集 |
| `StackDataset(*ds)` | 按字段堆叠 |

**知识点**：Map 式（按下标取）vs Iterable 式（流式）的区别在于
「能否随机访问」——后者不支持 `shuffle`、`sampler`。

### 15.2 Sampler 家族

| Sampler | 顺序 |
| --- | --- |
| `SequentialSampler` | 顺序 |
| `RandomSampler` | 随机（可放回） |
| `SubsetRandomSampler(indices)` | 只在子集内随机 |
| `WeightedRandomSampler(weights, num_samples)` | 按权重采样 |
| `BatchSampler(sampler, batch_size, drop_last)` | 把索引组合成批 |
| `DistributedSampler(ds, num_replicas, rank)` | 多卡切分 |

**知识点**：
- 打乱（shuffle）是 SGD 去相关的关键；
- 类别不平衡用 `WeightedRandomSampler` 过采样稀有类；
- 分布式用 `DistributedSampler`，每轮要 `set_epoch`。

### 15.3 DataLoader 全参数

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, persistent_workers=False,
           prefetch_factor=2, generator=None)
```

| 参数 | 作用 | 坑 |
| --- | --- | --- |
| `shuffle` | 每轮打乱 | 与 `sampler` 互斥 |
| `sampler` | 自定义顺序 | 与 `shuffle` 互斥 |
| `batch_sampler` | 直接给一批索引 | 与 batch_size/shuffle/sampler 互斥 |
| `num_workers` | 子进程数 | Windows 需 `if __name__ == "__main__"` |
| `collate_fn` | 自定义拼批 | 变长序列必须自定义 |
| `pin_memory` | 锁页内存 | 仅 CUDA 有意义 |
| `drop_last` | 丢最后不满的批 | BN 单样本会报错时开 |
| `persistent_workers` | 复用子进程 | 配合 `num_workers>0` |

### 15.4 collate 的两个默认实现

| 函数 | 作用 |
| --- | --- |
| `default_collate(batch)` | 把样本列表拼成批（转张量） |
| `default_convert(data)` | 只转类型不拼批 |

**变长序列的标准做法**：

```python
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True, padding_value=0), torch.stack(labels)
```

> 第 15 章口诀：**Dataset 管单条，Sampler 管顺序，DataLoader 管成批。**

---

## 第 16 章 `torch.amp`：混合精度

### 16.1 两个 API

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    y = model(x)

scaler = torch.amp.GradScaler("cuda", enabled=True,
                              init_scale=65536.0, growth_factor=2.0,
                              backoff_factor=0.5, growth_interval=2000)
```

| API | 作用 | 知识点 |
| --- | --- | --- |
| `autocast` | 自动把部分算子降到低精度 | 混合精度训练 |
| `GradScaler` | 动态缩放损失 | fp16 下溢 |

### 16.2 为什么需要 loss scaling

| 精度 | 指数位 | 最小正规数 | 问题 |
| --- | --- | --- | --- |
| fp32 | 8 | ~1e-38 | 基准 |
| fp16 | 5 | ~6e-5 | 梯度易下溢为 0 |
| bf16 | 8 | ~1e-38 | 范围同 fp32，无需缩放 |

**缩放流程**：

```
loss × S → backward → 梯度 × S → unscale（÷S）→ clip → step
```

`GradScaler` 会自动检测 inf/nan 并调整 S（成功则增大，溢出则减半）。

### 16.3 正确顺序

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.float16):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                 # 裁剪前必须还原
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                     # 溢出时跳过
scaler.update()
```

### 16.4 坑

| 坑 | 说明 |
| --- | --- |
| 用 `optimizer.step()` 代替 `scaler.step()` | 溢出会污染权重 |
| 先 clip 再 unscale | 裁的是放大后的梯度 |
| bf16 也开 scaler | 多余，`enabled=False` 即可 |
| 在 autocast 里做 loss 的数值敏感操作 | 规范在 autocast 外做（如 CE 内部已处理） |

**本机提醒**：GTX 1650（Turing）上 fp16 不一定更快，实测反而更慢；
用不用 AMP 要实测（见《PyTorch 实战》第 7 章）。

> 第 16 章口诀：**autocast 管前向，scaler 管反向；scale 进、unscale 出。**

---

## 第 17 章 `torch.cuda`：设备与显存

### 17.1 设备管理

| API | 作用 |
| --- | --- |
| `torch.cuda.is_available()` | 能否用 GPU |
| `torch.cuda.device_count()` | 卡数 |
| `torch.cuda.current_device()` / `set_device(i)` | 当前卡 |
| `torch.cuda.get_device_name(i)` | 型号 |
| `torch.cuda.get_device_capability(i)` | `(major, minor)` 计算能力 |
| `torch.cuda.get_device_properties(i)` | 显存、SM 数、L2 等 |
| `torch.device("cuda:0")` | 设备对象 |

**知识点**：计算能力（SM 版本）决定支持哪些特性，例如
bf16 需要 Ampere（8.0）及以上、FlashAttention-2 需要 8.0+。

### 17.2 显存监控

| API | 含义 |
| --- | --- |
| `torch.cuda.memory_allocated()` | 张量实际占用 |
| `torch.cuda.memory_reserved()` | 缓存池占用（含碎片） |
| `torch.cuda.max_memory_allocated()` | 峰值 |
| `torch.cuda.reset_peak_memory_stats()` | 重置峰值 |
| `torch.cuda.empty_cache()` | 释放缓存池 |
| `torch.cuda.memory_summary()` | 详细报告 |

**显存的真实构成**：

```
参数 + 梯度 + 优化器状态 + 激活 + 临时缓冲 + CUDA 上下文
```

### 17.3 同步与计时

```python
torch.cuda.synchronize()      # 等待所有核函数完成
```

**为什么计时必须同步**：GPU 调用是异步的，不同步会测出「启动时间」而不是「执行时间」。

### 17.4 流与事件

| API | 作用 |
| --- | --- |
| `torch.cuda.Stream()` | 自定义流（并行任务） |
| `torch.cuda.Event(enable_timing=True)` | 计时 |
| `stream.wait_stream(other)` / `torch.cuda.synchronize()` | 同步 |

> 第 17 章口诀：**先看能力再选精度；计时之前先同步。**

---

## 第 18 章 `torch.distributed` 与 DDP

### 18.1 基础 API

| API | 作用 |
| --- | --- |
| `dist.init_process_group(backend, init_method, world_size, rank)` | 初始化进程组 |
| `dist.destroy_process_group()` | 收尾 |
| `dist.is_available()` / `is_initialized()` | 能力 / 状态 |
| `dist.get_rank()` / `get_world_size()` | 进程号 / 总进程数 |
| `dist.barrier()` | 所有进程同步 |
| `dist.new_group(ranks)` | 子通信组 |

**启动方式**：`torchrun --nproc_per_node=8 train.py` 会自动注入
`RANK / LOCAL_RANK / WORLD_SIZE` 环境变量。

### 18.2 集合通信

| API | 语义 | 知识点 |
| --- | --- | --- |
| `all_reduce(t, op)` | 所有进程归约后人人都有 | 梯度平均 |
| `all_gather(out, t)` | 收集所有进程的张量 | 拼接特征 |
| `reduce(t, dst)` | 只汇聚到某进程 | 汇总统计 |
| `reduce_scatter(out, t)` | 归约后各拿一片 | ZeRO 的通信模式 |
| `broadcast(t, src)` | 从 src 广播 | 同步标志、初始化权重 |
| `gather/scatter` | 一对多/多对一 | — |
| `send/recv` | 点对点 | 流水线并行 |
| `ReduceOp.SUM/AVG/MAX` | 归约算子 | — |

### 18.3 DDP

```python
model = DistributedDataParallel(model,
        device_ids=[local_rank],
        broadcast_buffers=False,        # buffer 各卡一致时关掉
        find_unused_parameters=False,   # 有未用参数时打开
        gradient_as_bucket_view=True)   # 省显存
```

| 参数 | 作用 | 坑 |
| --- | --- | --- |
| `device_ids` | 绑定卡 | 必须与 `set_device` 一致 |
| `broadcast_buffers` | 每步同步 buffer | 默认 True；RoPE 这类确定性 buffer 可关 |
| `find_unused_parameters` | 处理未参与前向的参数 | 打开会变慢；忘记会卡死 |
| `gradient_as_bucket_view` | 梯度直接指向通信桶 | 省显存，但别原地改梯度 |

**知识点**：DDP 的本质是「反向传播时对所有卡的梯度做 all-reduce（求平均）」。
每步通信量约 `2P`（P 为参数量）。

### 18.4 与相关方案的对比

| 方案 | 分片什么 | 通信量 | 适合 |
| --- | --- | --- | --- |
| DDP | 不分片，只同步梯度 | 2P | 模型能单卡放下 |
| ZeRO-1/2/3 | 优化器/梯度/参数 | 2P / 2P / 3P | 显存不足 |
| FSDP | 参数/梯度/优化器 | ≈3P | PyTorch 原生分片 |
| 张量并行 | 层内矩阵 | 高 | 单层放不下 |
| 流水线并行 | 按层切分 | 中 | 超深模型 |

### 18.5 坑

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| 只有部分进程进入分支 | 卡死 | 所有卡必须执行相同的集合通信 |
| 未用参数 | `Expected to have finished reduction` | `find_unused_parameters=True` |
| BN 不同步 | 效果变差 | `SyncBatchNorm` |
| 数据量不能整除卡数 | 各卡步数不同 | `DistributedSampler(drop_last=True)` |
| 每轮不调 `set_epoch` | 每轮切分相同 | 每轮调用 |
| 端口冲突 | 启动失败 | `--master_port` |

> 第 18 章口诀：**通信必须全员参加；能单卡就别上 DDP。**

---

## 第 19 章 保存与加载

### 19.1 两种保存对象

| 方式 | 代码 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 整个模型 | `torch.save(model, path)` | 一行搞定 | 依赖类定义，pickle 风险 |
| **状态字典** | `torch.save(model.state_dict(), path)` | 结构代码与权重分离 | 需要先建结构 |

**生产推荐 `state_dict`**：模型代码改了结构，权重仍可控。

### 19.2 加载

```python
sd = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=True)
```

| 参数 | 作用 | 坑 |
| --- | --- | --- |
| `map_location` | 目标设备 | 不给会去原设备找 |
| `weights_only`（2.6 默认 True） | 只反序列化张量/基础类型 | 存了自定义对象会失败 |
| `strict=True` | 键完全匹配 | `strict=False` 仍会检查形状 |

### 19.3 checkpoint 应该存什么

| 内容 | 为什么 |
| --- | --- |
| `model.state_dict()` | 权重 |
| `optimizer.state_dict()` | 动量、二阶矩 |
| `scheduler.state_dict()` | 学习率进度 |
| `scaler.state_dict()` | 混合精度缩放系数 |
| `epoch` / `step` / `world_size` | 续训位置 |
| RNG 状态（可选） | 完全可复现 |

### 19.4 原子写

```python
tmp = path + ".tmp"
torch.save(obj, tmp)
os.replace(tmp, path)
```

**知识点**：写到一半崩溃时，旧检查点仍然完整。

> 第 19 章口诀：**存 state_dict，加载给 map_location；续训存全套。**

---

## 第 20 章 性能工具

### 20.1 性能分析

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    train_step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

| API | 作用 |
| --- | --- |
| `profile(...)` | 采样算子耗时 |
| `record_function("name")` | 给代码段打标签 |
| `key_averages()` | 按算子聚合 |
| `export_chrome_trace(path)` | 导出时间线 |

### 20.2 编译加速

```python
model = torch.compile(model, mode="default", fullgraph=False)
```

| 参数 | 说明 |
| --- | --- |
| `backend` | 默认 `inductor` |
| `mode` | `default` / `reduce-overhead` / `max-autotune` |
| `dynamic` | 处理变长输入 |
| `fullgraph` | 要求整图可捕获（否则报错） |

**注意**：Windows 上通常不可用（缺 Triton/MSVC），需要 Linux + 编译工具链。

### 20.3 显存换时间：梯度检查点

```python
from torch.utils.checkpoint import checkpoint
y = checkpoint(layer, x, use_reentrant=False)
```

**知识点**：前向不保存中间激活，反向时重算。
显存省 30%~70%，速度慢 20%~30%。

### 20.4 后端开关

| API | 作用 |
| --- | --- |
| `torch.backends.cudnn.benchmark = True` | 自动选最快卷积算法（输入尺寸固定时） |
| `torch.backends.cudnn.deterministic = True` | 确定性（可复现，变慢） |
| `torch.set_float32_matmul_precision("high")` | 允许 TF32（Ampere+） |
| `torch.backends.cuda.matmul.allow_tf32 = True` | 同上，旧接口 |

**知识点**：GPU 上的瓶颈通常是**显存带宽**而不是算力；
融合算子（SDPA、compile）之所以快，是因为减少了显存读写。

> 第 20 章口诀：**先 profile 再优化；显存换时间用 checkpoint。**

---

## 第 21 章 一一对应总表

### 21.1 知识点 → API

| 神经网络知识点 | 对应 API | 一句话 |
| --- | --- | --- |
| 全连接 / 仿射变换 | `nn.Linear` / `F.linear` | `y = xWᵀ+b` |
| 卷积 | `nn.Conv1d/2d/3d` | 局部连接 + 权重共享 |
| 转置卷积（上采样） | `nn.ConvTranspose2d` | 卷积的梯度形式 |
| 池化 | `nn.MaxPool2d` / `AvgPool2d` / `AdaptiveAvgPool2d` | 固定规则下采样 |
| 循环网络 | `nn.RNN` / `LSTM` / `GRU` | 序列状态传递 |
| 门控 | LSTM 三门 / `nn.GLU` | 控制信息流 |
| 批归一化 | `nn.BatchNorm2d` | 用 batch 统计归一化 |
| 层归一化 | `nn.LayerNorm` | 用样本内统计归一化 |
| RMS 归一化 | `nn.RMSNorm` | 只除以均方根 |
| 激活函数 | `nn.ReLU/GELU/SiLU/...` | 引入非线性 |
| 正则化 | `nn.Dropout` / `weight_decay` | 防过拟合 |
| 词嵌入 | `nn.Embedding` | 离散 → 连续 |
| 注意力 | `F.scaled_dot_product_attention` | 内容寻址的加权平均 |
| 多头注意力 | `nn.MultiheadAttention` | 多个子空间并行 |
| Transformer 层 | `nn.TransformerEncoderLayer` | 注意力 + FFN |
| 位置编码 | 正弦 / 可学习 / RoPE | 注入顺序信息 |
| 交叉熵 | `nn.CrossEntropyLoss` | 最大似然 |
| 回归损失 | `nn.MSELoss` / `L1Loss` / `SmoothL1Loss` | 距离度量 |
| 分布对齐 | `nn.KLDivLoss` | 相对熵 |
| 度量学习 | `nn.TripletMarginLoss` | 拉近正样本、推远负样本 |
| 序列对齐 | `nn.CTCLoss` | 免对齐的序列损失 |
| 梯度下降 | `optim.SGD` | `θ ← θ - lr·g` |
| 自适应学习率 | `optim.Adam` / `AdamW` | 一阶/二阶矩 |
| 学习率策略 | `lr_scheduler.*` | warmup / 衰减 |
| 梯度爆炸 | `clip_grad_norm_` | 按范数裁剪 |
| 权重约束 | `spectral_norm` / `weight_norm` | 限制范数 |
| 剪枝 | `nn.utils.prune` | 加 0/1 掩码 |
| 混合精度 | `torch.autocast` + `GradScaler` | 低精度计算 |
| 数据批处理 | `DataLoader` | 小批量 SGD |
| 分布式 | `DDP` + `DistributedSampler` | 数据并行 |
| 序列化 | `torch.save/load` + `state_dict` | 参数与结构分离 |

### 21.2 层形状速查

| 层 | 输入 | 输出 |
| --- | --- | --- |
| `Linear(in,out)` | `(*, in)` | `(*, out)` |
| `Conv2d(Cin,Cout,k)` | `(N,Cin,H,W)` | `(N,Cout,Hout,Wout)` |
| `MaxPool2d(k)` | `(N,C,H,W)` | `(N,C,H/k,W/k)` |
| `AdaptiveAvgPool2d(1)` | `(N,C,H,W)` | `(N,C,1,1)` |
| `LSTM(Hin,Hout)`（默认） | `(L,N,Hin)` | `(L,N,Hout)` |
| `Embedding(V,d)` | `(...,)` long | `(...,d)` |
| `MultiheadAttention(E,h)`（默认） | `(L,N,E)` | `(L,N,E)` |
| `LayerNorm(d)` | `(...,d)` | `(...,d)` |
| `TransformerEncoderLayer(d,h)`（默认） | `(L,N,d)` | `(L,N,d)` |

### 21.3 常见需求 → API

| 我要… | 用 |
| --- | --- |
| 冻结底座 | `p.requires_grad_(False)` |
| 只训一部分 | 参数组 / LoRA |
| 统计参数量 | `sum(p.numel() for p in model.parameters())` |
| 看梯度 | `p.grad` |
| 防 NaN | `clip_grad_norm_` + `set_detect_anomaly` |
| 省显存 | `autocast` / 梯度检查点 / 小 batch |
| 加速推理 | `inference_mode` / `torch.compile` |
| 多卡 | `DDP` + `DistributedSampler` |
| 断点续训 | 保存/恢复 `state_dict` |
| 调学习率 | `param_groups` / `lr_scheduler` |
| 处理变长序列 | `pad_sequence` + `collate_fn` |

---

## 附录 A 四类模型的完整模板

### A.1 MLP

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, d_out))
    def forward(self, x):
        return self.net(x)
```

### A.2 CNN

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, num_classes)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))
```

### A.3 RNN

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab, d=128, hidden=256, classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.rnn = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, classes)
    def forward(self, ids):
        out, _ = self.rnn(self.emb(ids))
        return self.head(out[:, -1])
```

### A.4 Transformer 块

```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab, d=256, heads=8, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(1024, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4*d,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.emb(ids) + self.pos(pos)
        mask = torch.triu(torch.ones(ids.size(1), ids.size(1),
                           dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.encoder(x, mask=mask)
        return self.head(x)
```

---

## 附录 B 常见报错 → 原因 → 修法

| 报错 | 原因 | 修法 |
| --- | --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | 上一层输出维度 ≠ 本层输入 | 打印 shape，改 `in_features` |
| `Expected all tensors to be on the same device` | 张量设备不一致 | 统一 `.to(device)` |
| `expected scalar type Long but found Float` | Embedding/CE 目标不是 long | `.long()` |
| `Target 10 is out of bounds` | 标签越界 | 检查类别数与标签范围 |
| `element 0 of tensors does not require grad` | 前向在 no_grad 里 | 检查上下文 |
| `Trying to backward through the graph a second time` | 同一图反向两次 | 每次前向配一次反向 |
| `one of the variables needed ... modified by an inplace operation` | 原地修改计算图张量 | 去掉 `relu_` 等原地操作 |
| `CUDA out of memory` | 显存不足 | 降 batch / AMP / 检查点 |
| `size mismatch for ...` | state_dict 结构不符 | 检查键名与维度 |
| `Can't convert cuda:0 device type tensor to numpy` | 未先 `.cpu()` | `.detach().cpu().numpy()` |
| `RuntimeError: DataLoader worker ...` | Windows 多进程 | 主入口保护，`num_workers=0` |
| loss 为 nan | lr 过大 / 全 ignore_index / 除零 | 降 lr、检查 mask、加 eps |

---

## 附录 C 术语中英对照

| 中文 | 英文 | 对应 API |
| --- | --- | --- |
| 张量 | tensor | `torch.Tensor` |
| 广播 | broadcasting | 自动规则 |
| 计算图 | computation graph | `autograd` |
| 叶子张量 | leaf tensor | 存 `.grad` |
| 前向 / 反向 | forward / backward | `forward()` / `.backward()` |
| 仿射变换 | affine transform | `nn.Linear` |
| 感受野 | receptive field | 卷积 |
| 权重共享 | weight sharing | 卷积 |
| 门控 | gating | LSTM / GLU |
| 归一化 | normalization | BN / LN |
| 正则化 | regularization | Dropout / weight decay |
| 嵌入 | embedding | `nn.Embedding` |
| 注意力 | attention | `MultiheadAttention` |
| 位置编码 | positional encoding | RoPE / 正弦 |
| 交叉熵 | cross entropy | `CrossEntropyLoss` |
| 似然 | likelihood | CE 的来源 |
| 优化器 | optimizer | `torch.optim` |
| 学习率调度 | learning rate schedule | `lr_scheduler` |
| 梯度裁剪 | gradient clipping | `clip_grad_norm_` |
| 混合精度 | mixed precision | `torch.amp` |
| 数据并行 | data parallel | DDP |
| 检查点 | checkpoint | `torch.save` |

---

## 附录 D 学习路线

```
① 读第 1-5 章      → 能写 MLP、懂张量与求导
② 读第 6-10 章     → 能写 CNN / RNN / Transformer
③ 读第 11-13 章    → 会选损失、优化器、学习率
④ 读第 14-17 章    → 会调显存、混合精度、诊断
⑤ 读第 18-21 章    → 能上多卡、能查表解决问题
⑥ 用附录 A 的四个模板各写一遍 → 手熟了
```

> 全书口诀：**层选对、形状对、mask 对、精度对、通信对——五对俱全，训练不崩。**



