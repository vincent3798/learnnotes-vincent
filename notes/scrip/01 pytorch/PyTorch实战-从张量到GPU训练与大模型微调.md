# PyTorch 实战：从张量到 GPU 训练与大模型微调

> **适用环境（本机实测）**
> conda 环境 `swift_env` ｜ Python 3.10.22 ｜ torch 2.6.0+cu124 ｜ numpy 2.2.6
> GPU：NVIDIA GeForce GTX 1650，显存 4095 MB，计算能力 SM 7.5（Turing）
> cuDNN 9.1.0
>
> 本书里所有「实测」数字都来自这台机器，不是抄来的参考值。硬件不同，数字会变，
> 但判断方法和坑是一样的。

---

## 这本书怎么用

**三遍法**（每章都适用）：

1. **第一遍只读代码，不跑**。看懂输入形状、输出形状、这一行在干什么。
2. **第二遍跑起来**，对着输出验证自己的理解。跑不通就是理解错了，回去改。
3. **第三遍合上书，自己默写**训练循环。默写不出来的地方，才是你真正没学会的地方。

配套脚本在 `code/` 目录，用这个解释器运行：

```bash
D:\conda_envs\swift_env\python.exe code\00_check_env.py
```

**学习路线**（对应你后面要学的包）：

```
PyTorch 张量 / 自动求导 / nn.Module / 训练循环
        ↓  本书
transformers（模型与分词器）→ datasets（数据）→ peft（LoRA）
        ↓
trl / ms-swift（微调）→ accelerate（多卡与显存优化）
```

**一句话总纲**：

> 前向算值，反向求导，优化器改参数，DataLoader 供数据。
> 剩下的全是工程细节。

---

## 目录

- 第 0 章 先跑通：环境与第一条 GPU 命令
- 第 1 章 张量 Tensor：一切的基本单位
- 第 2 章 Autograd：把梯度计算交给机器
- 第 3 章 nn.Module：像搭积木一样搭模型
- 第 4 章 损失、优化器与调度器
- 第 5 章 训练循环的标准模板
- 第 6 章 数据管线：Dataset 与 DataLoader
- 第 7 章 GPU 实战：4GB 显存怎么用
- 第 8 章 调试与报错速查
- 第 9 章 完整项目：字符级语言模型
- 第 10 章 通向大模型：transformers / peft / trl
- 第 11 章 练习与答案
- 附录 A API 速查表
- 附录 B 术语中英对照

---

## 第 0 章 先跑通：环境与第一条 GPU 命令

### 0.1 你实际在用的是什么

| 项目 | 本机实测值 | 说明 |
| --- | --- | --- |
| Python | 3.10.22 | conda 环境 `swift_env` |
| torch | 2.6.0+cu124 | +cu124 表示内置 CUDA 12.4 运行时 |
| numpy | 2.2.6 | torch 2.6 与 numpy 2.x 兼容良好 |
| GPU | GTX 1650 | 计算能力 (7, 5)，即 Turing 架构 |
| 显存 | 4095 MB ≈ 4 GB | **这是你做所有实验的硬上限** |
| cuDNN | 90100（9.1.0） | 卷积/矩阵运算的底层加速库 |

三个必须先记住的硬件事实，它们直接决定第 7 章的写法：

1. **4GB 显存**：7B 参数模型全量微调需要约 112GB 显存（fp32 + Adam），差了两个数量级。
   本地只能做「小模型 + LoRA/QLoRA」或者纯学习实验。
2. **Turing 架构（SM 7.5）**：没有 Ampere 的 Tensor Core 新特性，bf16 没有硬件加速；
   实测 fp16 混合精度反而更慢（见第 7 章数据）。
3. **Windows 平台**：`torch.compile` 用不了（实测报错 `Compiler: cl is not found`，
   且没有 Triton）。不要在 Windows 上折腾它。

### 0.2 环境自检：跑通这三件事再往下学

```python
# code/00_check_env.py 的核心三行
import torch

print(torch.__version__, torch.version.cuda)      # 2.6.0 12.4
print(torch.cuda.is_available())                  # True
print(torch.cuda.get_device_name(0))              # NVIDIA GeForce GTX 1650

x = torch.randn(1024, 1024, device="cuda")
y = x @ x                                          # 第一次 GPU 运算
torch.cuda.synchronize()
print("GPU 运算正常", y.shape, y.device)
```

看到 `True`、显卡名、`cuda:0` 三个输出，环境就通了。任何一步报错，先解决环境，
不要带着坏环境往下学——后面的每个报错你都会怀疑是环境问题。

**实测输出**：

```
2.6.0+cu124 12.4
True
NVIDIA GeForce GTX 1650
GPU 运算正常 torch.Size([1024, 1024]) cuda:0
```

### 0.3 PyTorch 在生态里的位置

| 维度 | NumPy | PyTorch | TensorFlow 2 | JAX |
| --- | --- | --- | --- | --- |
| 主要用途 | 数值计算 | **深度学习训练** | 深度学习部署 | 研究 / 高性能 |
| 自动求导 | 无 | 有（动态图） | 有（静态图为主） | 有（函数式） |
| GPU 支持 | 无（需 CuPy） | 原生 | 原生 | 原生 |
| 调试体验 | 好 | **好（逐行可断点）** | 一般 | 差 |
| 生态重心 | 数据科学 | **LLM / 研究 / 微调** | 工业部署 | 学术研究 |
| 你现在用的包 | pandas 依赖它 | transformers/peft/trl 全依赖它 | — | — |

**结论**：你的整个工具链（transformers、peft、trl、ms-swift、accelerate）都建立在
PyTorch 之上。张量操作和 NumPy 几乎一样，差别只在三件事：**device、autograd、模块化**。

### 0.4 三条铁律

> **口诀：形状对、类型对、设备对——三不对，全白费。**

每次报错，按这个顺序自查：

1. **形状**：这个张量是几维？每一维是什么含义？
2. **类型（dtype）**：`float32` 还是 `int64`？分类标签必须是 `int64`。
3. **设备（device）**：在 `cpu` 还是 `cuda:0`？参与运算的张量必须在同一设备。

这三条能解决新手 80% 的报错。剩下 20% 在第 8 章。

---

## 第 1 章 张量 Tensor：一切的基本单位

### 1.1 张量是什么

张量 = 带设备位置和求导能力的多维数组。

| 概念 | NumPy | PyTorch | 区别 |
| --- | --- | --- | --- |
| 多维数组 | `np.ndarray` | `torch.Tensor` | 多一个 `device` 属性 |
| 数据类型 | `np.float32` | `torch.float32` | 名称不同，语义相同 |
| 求导 | 无 | `requires_grad` | PyTorch 独有 |
| 内存共享 | `np.asarray` | `torch.from_numpy` / `.numpy()` | 双向共享内存 |

### 1.2 创建：四种来源

```python
import torch

# 1) 直接从 Python 数据
a = torch.tensor([[1, 2], [3, 4]])                 # dtype 自动推断为 int64
b = torch.tensor([1.0, 2.0], dtype=torch.float32)  # 显式指定

# 2) 形状 + 初始化方式
z = torch.zeros(2, 3)          # 全 0
o = torch.ones(2, 3)           # 全 1
r = torch.randn(2, 3)          # 标准正态
u = torch.rand(2, 3)           # [0,1) 均匀分布
e = torch.empty(2, 3)          # 不初始化（快，但内容随机，别直接读）

# 3) 从 NumPy 来（共享内存，改一个另一个也变）
import numpy as np
arr = np.arange(6, dtype="float32").reshape(2, 3)
t = torch.from_numpy(arr)

# 4) 形状对齐已有张量
like = torch.zeros_like(a, dtype=torch.float32)    # 形状同 a，类型可覆盖
```

**易错点**：`torch.tensor([1, 2])` 得到的是 **int64**，不是 float。
把它送进 `nn.Linear` 会直接报错。整数除法、标签、索引用 int64 是对的；
模型输入必须是浮点。

> **口诀：整数默认 int64，模型输入必 float。**

### 1.3 三件套：shape / dtype / device

```python
x = torch.randn(2, 3, 4)
print(x.shape)     # torch.Size([2, 3, 4])
print(x.dtype)     # torch.float32
print(x.device)    # cpu
print(x.ndim)      # 3
print(x.numel())   # 24，元素总数

x = x.to("cuda").to(torch.float16)
print(x.device, x.dtype)   # cuda:0 torch.float16
```

`.to()` 返回新张量（除非原地 `x.to_()`）。**不要在每个 forward 里反复 `.to()`**，
在数据进入循环前一次性搬完。

常用的 dtype 对照：

| dtype | 位数 | 用途 | 注意 |
| --- | --- | --- | --- |
| `torch.float32` | 32 | 默认训练精度 | 最稳，显存基准 |
| `torch.float16` | 16 | 混合精度 | 需 GradScaler 防下溢 |
| `torch.bfloat16` | 16 | 大模型训练 | 动态范围大，Turing 无加速 |
| `torch.int64` / `long` | 64 | 标签、索引 | 分类目标必须用这个 |
| `torch.bool` | 8 | 掩码 | `x > 0` 自动产生 |

### 1.4 形状变换：四个高频操作

```python
x = torch.arange(12)                    # shape (12,)

x.view(3, 4)        # 必须是连续内存，形状 (3,4)
x.reshape(3, 4)     # 更安全，不满足连续时自动拷贝
x.view(-1, 6)       # -1 表示"这一维你自己算"

t = torch.randn(2, 3, 4)
t.permute(2, 0, 1).shape    # (4, 2, 3)，换轴顺序
t.transpose(1, 2).shape     # (2, 4, 3)，交换两轴

t.unsqueeze(0).shape        # (1, 2, 3, 4)，插一个新维度
t.squeeze(0).shape          # 去掉长度为 1 的维度（若该维不是 1 则不变）

t.flatten(1).shape          # (2, 12)，从第 1 维起拉平
```

| 操作 | 作用 | 典型场景 |
| --- | --- | --- |
| `view` / `reshape` | 改形状不改数据顺序 | 全连接层前拉平 |
| `permute` / `transpose` | 换轴 | 图像 HWC→CHW、注意力 BHSD |
| `unsqueeze` | 加长度为 1 的维度 | 单样本凑 batch 维 |
| `expand` | 广播复制（不复制内存） | 掩码扩展 |
| `repeat` | 真复制 | 需要独立内存时 |
| `flatten` | 拉平连续几维 | `(B, C, H, W) → (B, C*H*W)` |

> **口诀：view 要连续，reshape 随便用；permute 换轴，unsqueeze 补维。**

### 1.5 索引与筛选

```python
x = torch.randn(4, 5)

x[0]            # 第 0 行，shape (5,)
x[:, 1]         # 第 1 列
x[1:3, 2:]      # 切片
x[x > 0]        # 布尔掩码，返回一维
x.sum(dim=1)    # 按行求和 → (4,)
x.argmax(dim=1) # 每行最大值的下标 → (4,)

idx = torch.tensor([0, 2])
x[idx]          # 花式索引取第 0、2 行
```

**易错点**：`x.sum()` 不带 `dim` 会得到零维标量；要保留维度用 `keepdim=True`。
广播相减时形状对不上，90% 是忘了 `keepdim`。

### 1.6 广播：形状不同也能算

规则只有一条：**从最后一维往左对齐，每一维要么相等，要么有一方为 1（或不存在）。**

| A | B | 结果 | 是否合法 |
| --- | --- | --- | --- |
| (3, 4) | (4,) | (3, 4) | ✅ 最后一维相等 |
| (3, 4) | (3, 1) | (3, 4) | ✅ 维 1 广播 |
| (3, 1, 5) | (1, 4, 5) | (3, 4, 5) | ✅ 中间的 1 被复制 |
| (3, 4) | (3,) | ❌ 报错 | 最后一维 4≠3，且都不是 1 |
| (2, 3) | (3, 2) | ❌ 报错 | 对齐后都不匹配 |

```python
a = torch.randn(3, 4)
b = torch.randn(4)
print((a + b).shape)          # (3, 4)

# 常见需求：标准化 (B, D) 每列减去均值
x = torch.randn(32, 128)
mu = x.mean(dim=0, keepdim=True)     # (1, 128)
x_norm = (x - mu) / (x.std(dim=0, keepdim=True) + 1e-8)
```

> **口诀：右对齐，逐维比；相等或为一，才能算下去。**

### 1.7 矩阵乘法：三种写法

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

A @ B                 # 推荐写法，(3, 5)
torch.matmul(A, B)    # 等价
A.mm(B)               # 仅限二维

# 批量矩阵乘：(B, N, M) @ (B, M, K) → (B, N, K)
X = torch.randn(8, 3, 4)
Y = torch.randn(8, 4, 5)
(X @ Y).shape         # (8, 3, 5)

# 逐元素乘，别和矩阵乘混淆
(A * A).shape         # (3, 4)
```

| 写法 | 语义 | 用途 |
| --- | --- | --- |
| `A * B` | 逐元素乘（Hadamard） | 门控、掩码 |
| `A @ B` | 矩阵乘 | 线性层 |
| `torch.einsum` | 任意缩并 | 注意力、复杂张量网络 |
| `A.sum() * B` | 标量乘 | — |

`einsum` 是进阶但值得早学的工具——一个式子看清所有轴的走向：

```python
q = torch.randn(8, 6, 64)    # (B, H, D)
k = torch.randn(8, 6, 64)
scores = torch.einsum("bhd,bjd->bhj", q, k)   # (8, 6, 6)
print(scores.shape)
```

### 1.8 性能习惯（从第一天就养成）

1. **能向量化就不要写 Python 循环**。循环 1000 次 = 1000 次核函数启动。
2. **避免在循环里 `.to(device)`**。整批搬一次比分批搬快得多。
3. **不要用 `.item()` 频繁取标量**，每次都会同步 CPU 与 GPU。
4. **`torch.no_grad()` 包住推理和评估**，省显存也提速（第 2 章）。
5. **拼接大量小张量用 `torch.stack` / `torch.cat`**，不要在循环里 `+=` 拼。

```python
# 慢：逐条处理
outs = [model(x[i:i+1]) for i in range(len(x))]

# 快：一次成批
outs = model(x)
```

> **第 1 章口诀：形状对、类型对、设备对；右对齐来广播，@ 才是矩阵乘。**

---

## 第 2 章 Autograd：把梯度计算交给机器

### 2.1 它替你解决的问题

神经网络训练 = 反复做一件事：**算出损失对每个参数的偏导，然后按梯度反方向走一步。**
手推几十层网络的梯度不现实，Autograd 就是自动完成这件事的引擎。

### 2.2 最小例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2            # y = x²
y.backward()          # 反向传播，dy/dx = 2x
print(x.grad)         # tensor(4.)  ← 2 * 2
```

三行代码包含了 Autograd 的全部机制：

1. `requires_grad=True`：告诉 PyTorch「这个张量需要被追踪」。
2. 前向运算 `y = x**2`：**同时构建计算图**，记录 `y` 是由哪些操作、哪些张量得到的。
3. `y.backward()`：从 `y` 出发沿图反向走，按链式法则累加梯度，写到叶子张量的 `.grad`。

### 2.3 计算图长什么样

```
x (requires_grad=True)
   │  pow(2)
   ▼
y ── backward() ──►  dy/dy = 1
   │  d(x²)/dx = 2x
   ▼
x.grad = 4.0
```

复杂网络不过是这张图的放大版。**每次 forward 都会新建一张图**，这是理解
「为什么梯度会累积」「为什么要 zero_grad」的钥匙。

### 2.4 叶子张量：只有它才存梯度

```python
a = torch.tensor(3.0, requires_grad=True)   # 叶子
b = a * 2                                   # 非叶子，b = 2a = 6
c = b ** 2                                  # 非叶子，c = (2a)² = 4a² = 36
c.backward()

print(a.grad)   # tensor(24.)  ← dc/da = 8a = 24
print(b.grad)   # None，非叶子默认不保留
```

链式法则验算：`dc/db = 2b = 12`，`db/da = 2`，相乘得 `dc/da = 24`。
和 `c = 4a²` 直接求导 `8a = 24` 一致。

| 概念 | 定义 | 是否自动存 `.grad` |
| --- | --- | --- |
| 叶子张量 | 用户直接创建的、`requires_grad=True` 的张量 | ✅ 是 |
| 非叶子张量 | 运算产生的中间结果 | ❌ 否（除非 `retain_grad()`） |
| 参数 | `nn.Module` 自动注册的 `nn.Parameter` | ✅ 是（本质是叶子） |

模型的参数都是叶子张量，所以优化器能直接从 `.grad` 拿到梯度。

### 2.5 三条必须掌握的规则

**规则一：grad 是累加的，不是覆盖的。**

```python
x = torch.tensor(1.0, requires_grad=True)
(x * 2).backward()
print(x.grad)      # tensor(2.)
(x * 3).backward()
print(x.grad)      # tensor(5.)  ← 2 + 3，累加了
```

这就是为什么训练循环里每次更新前要 `optimizer.zero_grad()`。
（原因：共享子图/多损失求和时累加是正确行为，PyTorch 选择默认累加。）

**规则二：默认反向一次后计算图就释放了。**

```python
y = (x * 2)
y.backward()
# y.backward()   # ❌ RuntimeError: Trying to backward through the graph a second time
```

需要多次反向（如 GAN、高阶梯度）：`y.backward(retain_graph=True)`。
训练中一般用不到，遇到再查。

**规则三：只有标量能直接 backward，非标量要指定梯度种子。**

```python
y = x ** 2                     # y 是标量，OK
z = torch.randn(3, requires_grad=True) * 2
# z.backward()                 # ❌ grad can be implicitly created only for scalar outputs
z.backward(torch.ones(3))      # ✅ 等价于 sum(z).backward()
```

训练里损失函数都返回标量（`mean()` / `sum()`），所以平时不会碰到。

> **口诀：叶子存梯度，梯度会累加，图只走一次。**

### 2.6 `no_grad`、`inference_mode`、`detach`：三兄弟的差别

| 写法 | 作用 | 用在哪 | 开销 |
| --- | --- | --- | --- |
| `torch.no_grad()` | 上下文内不建图 | 评估、推理 | 低 |
| `torch.inference_mode()` | 更强：连版本计数都不追踪 | 纯推理（比 no_grad 更快） | 最低 |
| `x.detach()` | 截断这张量到计算图的连接，返回共享内存的新张量 | 只要数值不要梯度 | 一次性操作 |
| `x.clone()` | 复制数据，仍在图里 | 需要独立副本且要梯度 | 复制内存 |

```python
model.eval()
with torch.inference_mode():          # 评估标准写法
    logits = model(x)
    pred = logits.argmax(-1)
```

**为什么评估必须关梯度**：否则每次 forward 都建图，显存暴涨、速度变慢，
长评估循环直接 OOM。

`detach()` 的典型用途——记录日志时不能把 loss 带进下一轮的计算图：

```python
total_loss += loss.detach()      # ✅ 纯数值累加
# total_loss += loss             # ❌ 会把整个训练图挂住，显存持续上涨
```

> **口诀：训练建图，评估关图；记日志先 detach。**

### 2.7 梯度累积：小显存模拟大 batch

```python
accum_steps = 4
optimizer.zero_grad()

for i, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb), yb) / accum_steps   # 除以累积步数，保持规模一致
    loss.backward()                                  # 梯度自动累加

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

这正是 Autograd「梯度累加」特性的正向利用。第 7 章在 4GB 卡上会反复用到。

### 2.8 Autograd 的三个常见坑

| 现象 | 原因 | 修法 |
| --- | --- | --- |
| `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | 对叶子张量做了 `+=`、`.add_()` 等原地修改 | 改成非原地写法，或用 `with torch.no_grad():` 包住参数更新 |
| 参数 `.grad` 一直是 `None` | 参数没参与前向、或 loss 与参数断开了连接 | 打印 `loss.requires_grad`；检查模型是否真的用到了该层 |
| 显存不断增长 | 日志里累加了带图的 loss；或 list 里存了 loss | `loss.detach().item()` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | 前向中的张量在反向之前被原地改写 | 去掉 `relu_(x)` 这类原地操作 |

> **第 2 章口诀：叶子存梯度，梯度会累加，图走完就散；评估关掉图，日志先 detach。**

---

## 第 3 章 nn.Module：像搭积木一样搭模型

### 3.1 为什么不能只用裸张量

裸张量能算，但要自己管几十组参数和梯度，无法保存、无法搬设备、无法复用。
`nn.Module` 替你做了三件事：

1. **参数注册**：`self.w = nn.Parameter(...)` 或子模块自动登记，`.to(device)`
   时一起搬、`.parameters()` 一次拿全。
2. **前向定义**：子类实现 `forward`，调用时用 `model(x)`（自动触发 hooks）。
3. **状态管理**：`train()` / `eval()` 切换模式，`state_dict()` 保存全部参数。

### 3.2 最小模型

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                          # 必须先调用
        self.fc1 = nn.Linear(in_dim, hidden)        # 子模块自动注册
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):                           # x: (B, in_dim)
        x = self.fc1(x)                             # (B, hidden)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)                          # (B, out_dim)

model = MLP(784, 256, 10)
print(sum(p.numel() for p in model.parameters()))   # 参数量
print(model.training)                               # True，默认训练模式
```

**三条纪律**：

1. 层必须在 `__init__` 里创建并赋给 `self.*`，否则不会被注册成参数。
2. 计算逻辑写在 `forward` 里，不要直接调用 `model.forward(x)`——用 `model(x)`
   （否则 hooks 不触发，某些工具会失效）。
3. 不要在 `forward` 里创建带参数的层（每步都会重新初始化）。

> **口诀：init 里建层，forward 里连线，调用用 model(x)。**

### 3.3 常用层形状约定（最容易记错的地方）

| 层 | 输入形状 | 输出形状 | 记忆点 |
| --- | --- | --- | --- |
| `nn.Linear(in, out)` | `(B, *, in)` | `(B, *, out)` | 只改最后一维 |
| `nn.Conv2d(cin, cout, k)` | `(B, cin, H, W)` | `(B, cout, H', W')` | 通道在前 |
| `nn.Flatten()` | `(B, C, H, W)` | `(B, C*H*W)` | 展平后接 Linear |
| `nn.Embedding(V, d)` | `(B, L)` int64 | `(B, L, d)` | 输入是**整数索引** |
| `nn.LayerNorm(d)` | `(B, L, d)` | 同形状 | 对最后一维归一化 |
| `nn.BatchNorm1d(d)` | `(B, d)` | 同形状 | 对 batch 维统计 |
| `nn.LSTM(d, h)` | `(L, B, d)` | 输出 `(L,B,h)` + `(h,c)` | **默认 batch 不在第一维** |
| `nn.MultiheadAttention(d, H)` | `(L, B, d)` ×3 | `(L, B, d)` | 默认也是 L 在前 |

**最容易踩的两个坑**：

- `nn.Embedding` 的输入必须是 `int64`，传入 float 会报
  `Expected tensor for argument #1 'indices' to have scalar type Long`。
- LSTM / MultiheadAttention 默认 `batch_first=False`，想用 `(B, L, d)` 要显式传
  `batch_first=True`。

> **口诀：Linear 改末维，Conv 通道在第二位，Embedding 要整数，LSTM 记得开 batch_first。**

### 3.4 损失函数也别忘了形状

| 损失 | 预测形状 | 目标形状 | 目标类型 | 是否自带 softmax |
| --- | --- | --- | --- | --- |
| `nn.CrossEntropyLoss()` | `(B, C)` 原始 logits | `(B,)` 类别下标 | int64 | ✅ 自带，**别先 softmax** |
| `nn.BCEWithLogitsLoss()` | `(B,)` 或 `(B,1)` | 同形状 0/1 | float32 | ✅ 自带 sigmoid |
| `nn.MSELoss()` | 任意 | 同形状 | float32 | — |
| `nn.NLLLoss()` | `(B, C)` log 概率 | `(B,)` | int64 | ❌ 需先 log_softmax |

新手最常见的错误：在 `CrossEntropyLoss` 前手动加了 softmax，导致梯度变小、收敛极慢。

> **口诀：CE 吃 logits，别先 softmax；BCE 带 Logits 字样时同理。**

### 3.5 参数管理与设备迁移

```python
model = MLP(784, 256, 10)
model = model.to("cuda")                      # 一次性把所有参数和 buffer 搬到 GPU

print(next(model.parameters()).device)        # cuda:0

for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.requires_grad)
```

保存与加载（torch 2.6 有变化，见第 5 章）：

```python
torch.save(model.state_dict(), "mlp.pt")          # 只存参数，推荐
model.load_state_dict(torch.load("mlp.pt", map_location="cuda"))
```

| 保存方式 | 内容 | 体积 | 适用 |
| --- | --- | --- | --- |
| `state_dict()` | 参数字典 | 小 | **生产环境推荐** |
| 整个 `model` | 结构 + 参数（pickle） | 大 | 快速原型 |
| 检查点 dict | 参数 + 优化器 + epoch + 调度器 | 中 | 断点续训 |

**为什么推荐 state_dict**：模型类的代码变了（重命名层、改结构），整个模型 pickle
可能加载不了；state_dict 只是键值对，可控性强。

### 3.6 train() 与 eval()：不只是给 Dropout 用的

```python
model.train()      # 训练：Dropout 随机丢弃，BatchNorm 用当前 batch 统计
model.eval()       # 评估：Dropout 关闭，BatchNorm 用滑动平均统计

# 标准套路
for epoch in range(epochs):
    model.train()
    train_one_epoch(...)

    model.eval()
    with torch.inference_mode():
        evaluate(...)
```

**忘了 `model.eval()` 的后果**：验证集指标抖动、Dropout 在推理时随机丢神经元
导致预测不稳定。这是「训练时好、测试时差」的常见假象来源。

> **口诀：训练 train，评估 eval，忘了 eval 全白搭。**

### 3.7 打印模型结构

```python
print(model)
# MLP(
#   (fc1): Linear(in_features=784, out_features=256, bias=True)
#   (act): GELU(approximate='none')
#   ...
# )
```

想更细：`torchinfo` 不在当前环境里，直接用 `named_modules()` 或自己遍历参数表。

---

## 第 4 章 损失、优化器与调度器

### 4.1 优化器：三个默认选择

| 优化器 | 一句话 | 显存（每参数） | 何时用 |
| --- | --- | --- | --- |
| `SGD` | 最朴素，沿梯度走 | 1 份（+momentum 1 份） | 需要精调的 CNN，复现论文 |
| `Adam` | 动量 + 自适应学习率 | 2 份状态 | 一般任务 |
| `AdamW` | Adam + 解耦权重衰减 | 2 份状态 | **大模型 / 微调默认** |

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

**Adam 与 AdamW 的关键区别**：Adam 的 weight decay 混进梯度里，被自适应学习率
缩放了，正则效果不纯；AdamW 把衰减独立出来，衰减强度与学习率解耦。
transformers 生态默认 AdamW。

显存估算（重要）：每个参数在 Adam/AdamW 下需要 **4 份 fp32**：

```
参数 4B + 梯度 4B + 一阶动量 4B + 二阶动量 4B = 16 字节/参数
1M 参数 ≈ 16 MB 优化器相关显存
```

这还没算激活值。第 7 章会用它算 4GB 卡到底能装多大模型。

### 4.2 学习率：最重要的超参数

| 学习率 | 现象 |
| --- | --- |
| 太大（如 1e-2 用于 Adam） | loss 震荡或直接 NaN |
| 偏大 | 前期降得快，后期无法收敛 |
| 合适 | 平滑下降 |
| 太小（如 1e-6） | loss 几乎不动，像没训练 |

经验起点：**Adam/AdamW 用 1e-4 ~ 3e-4；微调预训练模型用 1e-5 ~ 2e-5；SGD 用 0.1。**
不确定就先跑 100 步，打印 loss：不动就 ×10，NaN 就 ÷10。

### 4.3 学习率调度器

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR

# 余弦退火：微调最常用
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

# 阶梯衰减：每 10 个 epoch 乘 0.1
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 每个 epoch 结束后调用
for epoch in range(epochs):
    train(...)
    scheduler.step()
    print(optimizer.param_groups[0]["lr"])
```

**关键区别**：`CosineAnnealingLR(..., T_max=total_steps)` 里的 `T_max` 是**总步数**；
而 `StepLR(step_size=10)` 的单位是**调用次数**。如果你按 epoch 调用但按 step 设计，
学习率会提前归零，训练直接停滞。

> **口诀：T_max 是总步数，scheduler.step 要配对。**

### 4.4 梯度裁剪：防止一步走飞

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

放在 `loss.backward()` 之后、`optimizer.step()` 之前。RNN、Transformer 和强化学习
必备；出现 loss 突然跳到 NaN 时，先加裁剪再排查其他原因。

### 4.5 权重衰减与 warmup（大模型微调标配）

```python
# 1) 分离不衰减参数（bias 和 LayerNorm 一般不衰减）
no_decay = ["bias", "LayerNorm.weight"]
groups = [
    {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(groups, lr=2e-5)

# 2) warmup：前 N 步线性升到目标学习率（防止早期梯度冲击）
def warmup_lambda(step, warmup_steps=1000):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
```

这两段代码在 transformers 的 `Trainer` 和 ms-swift 里都是默认行为，理解它才能
看懂训练日志。

> **第 4 章口诀：微调默认 AdamW，学习率先试 3e-4；CE 吃 logits，裁剪在 step 前。**

---

## 第 5 章 训练循环的标准模板

### 5.1 五步循环：背下来

```python
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)   # 0. 数据搬到设备

    optimizer.zero_grad(set_to_none=True)    # 1. 清空上一步的梯度
    logits = model(xb)                       # 2. 前向
    loss = criterion(logits, yb)             # 3. 算损失
    loss.backward()                          # 4. 反向求梯度
    optimizer.step()                         # 5. 更新参数
```

顺序错一步结果就不同。最常见的两个错误：

- **忘了 `zero_grad`**：梯度累加，等于学习率越走越大，几百步后爆炸。
- **`zero_grad` 放在 `backward` 后面**：第一步的梯度被保留，等于第一次更新错了。

> **口诀：清零 → 前向 → 损失 → 反向 → 更新。五步顺序不能动。**

`set_to_none=True` 是 PyTorch 2.0 起的默认值，比把梯度置零更省内存、更快，
不用改回去。

### 5.2 完整例子：用 MLP 拟合一个非线性函数

目标函数：`y = sin(x1) + cos(x2) + 噪声`。这是回归任务，用来把整套流程跑通。

```python
import torch
from torch import nn

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据 ----------
N = 4096
x = torch.rand(N, 2) * 6.283            # [0, 2π)
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1]) + 0.1 * torch.randn(N)
y = y.unsqueeze(1)                      # (N, 1)

# ---------- 模型 ----------
model = nn.Sequential(
    nn.Linear(2, 64), nn.GELU(),
    nn.Linear(64, 64), nn.GELU(),
    nn.Linear(64, 1),
).to(device)

# ---------- 损失与优化器 ----------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

# ---------- 训练 ----------
xb_all, yb_all = x.to(device), y.to(device)
BATCH, EPOCHS = 256, 200

for epoch in range(EPOCHS):
    perm = torch.randperm(N, device=device)          # 打乱
    total = 0.0
    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = xb_all[idx], yb_all[idx]

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total += loss.detach().item()

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {total / (N // BATCH):.4f}")
```

**实测输出（GTX 1650）**：

```
epoch   0  loss 0.8471
epoch  40  loss 0.0642
epoch  80  loss 0.0321
epoch 120  loss 0.0224
epoch 160  loss 0.0189
```

loss 从 0.85 降到 0.02 左右，接近噪声方差 0.01 的下限，说明拟合到位了。
如果 200 个 epoch 后 loss 还停在 0.5，按第 4 章的方法调学习率。

### 5.3 训练集 / 验证集：怎么切、怎么看

```python
n_train = int(N * 0.8)
x_train, x_val = x[:n_train], x[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

# 每轮结束评估
model.eval()
with torch.inference_mode():
    val_loss = criterion(model(x_val.to(device)), y_val.to(device)).item()
model.train()
```

| 现象 | 诊断 | 处理 |
| --- | --- | --- |
| train↓ val↓ | 正常 | 继续 |
| train↓ val↑（差距扩大） | 过拟合 | 加 dropout / 减参数量 / 早停 |
| train 不动 val 不动 | 欠拟合或学习率太小 | 加容量 / 调大 lr |
| train↑ | 学习率太大或代码 bug | 降 lr，检查数据打乱 |

**早停**（patience = 连续多少轮没改善就停）：

```python
best, patience, wait = float("inf"), 20, 0
for epoch in range(EPOCHS):
    train(...)
    if val_loss < best:
        best, wait = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print("early stop")
            break
```

### 5.4 保存与恢复检查点（torch 2.6 有变化）

```python
# 保存：一次存全
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_val": best,
}, "ckpt.pt")

# 恢复
ckpt = torch.load("ckpt.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])
start_epoch = ckpt["epoch"] + 1
```

**torch 2.6 的变化（高频坑）**：`torch.load` 的 `weights_only` 默认值从 `False`
改成了 `True`。它只允许加载「张量 + 基础类型」组成的字典，遇到自定义类、numpy
数组、argparse.Namespace 等会报
`Unsupported global: GLOBAL numpy.core.multiarray._reconstruct`。

| 情况 | 写法 |
| --- | --- |
| 只存了张量/字典（本文写法） | `torch.load(path, weights_only=True)`，默认即可 |
| 检查点里混了 numpy / 自定义对象 | 受信任的文件可显式 `weights_only=False`；不受信任的文件不要这么干（有反序列化风险） |

> **口诀：存 state_dict，载 map_location；2.6 之后 weights_only 默认开。**

### 5.5 日志与进度条

```python
from tqdm import tqdm

pbar = tqdm(loader, desc="train")
for xb, yb in pbar:
    ...
    pbar.set_postfix(loss=f"{loss.item():.4f}")
```

`tqdm` 在你环境里已有。注意 `loss.item()` 每步都会同步 GPU，用于显示可以；
不要在热路径里反复 `.item()` 再把它拼回计算。

> **第 5 章口诀：五步循环记死它；val 看趋势，早停看耐心；checkpoint 存三样——模型、优化器、轮次。**

---

## 第 6 章 数据管线：Dataset 与 DataLoader

### 6.1 两个类各管什么

| 类 | 职责 | 你实现什么 |
| --- | --- | --- |
| `Dataset` | 管**单条**样本：给下标，返回一条 `(x, y)` | `__len__` + `__getitem__` |
| `DataLoader` | 管**成批**：打乱、切 batch、多进程加载、拼张量 | 只传参数，不用继承 |

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
```

这就是全部。`__getitem__` 里做单条样本的读取与预处理，不要做全局操作。

### 6.2 DataLoader 参数表

| 参数 | 默认 | 作用 | 建议 |
| --- | --- | --- | --- |
| `batch_size` | 1 | 每批条数 | 4GB 卡从小开始（8/16/32） |
| `shuffle` | False | 每轮打乱 | 训练集 `True`，验证集 `False` |
| `num_workers` | 0 | 子进程数 | Windows 上先设 0，稳定后再试 2 |
| `pin_memory` | False | 锁页内存，加速 H2D 拷贝 | GPU 训练设 `True` |
| `drop_last` | False | 丢掉不完整的最后一批 | 训练设 `True`（避免 BN 单样本报错） |
| `collate_fn` | 默认拼接 | 自定义成批逻辑 | 变长序列必须自定义 |
| `persistent_workers` | False | 复用 worker | `num_workers>0` 时可开 |

```python
loader = DataLoader(
    MyDataset(x_train, y_train),
    batch_size=32, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True,
)
```

### 6.3 Windows 多进程的三个坑

1. **必须有 `if __name__ == "__main__":` 保护**。Windows 用 `spawn` 启动子进程，
   没有保护会递归导入脚本，报 `RuntimeError: An attempt has been made to start a
   new process before the current process has finished its bootstrapping phase`。
2. **`num_workers > 0` 时，Dataset 里的对象要可 pickle**（lambda 不行）。
3. **调试期用 0**。多进程里断点打不进去，报错信息也被淹没。

```python
if __name__ == "__main__":
    loader = DataLoader(..., num_workers=2)
    for xb, yb in loader:
        ...
```

### 6.4 `collate_fn`：变长序列必须手写

文本任务里每个样本长度不同，默认 collate 拼不起来。标准做法是 padding：

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch: list of (seq_tensor, label)
    seqs, labels = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)  # (B, maxL)
    labels = torch.stack(labels)
    return seqs, labels

loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)
```

| 场景 | 默认 collate 能否用 | 需要做什么 |
| --- | --- | --- |
| 定长向量 | ✅ | 不用管 |
| 变长序列 | ❌ | `pad_sequence` + 掩码 |
| 变长 + 要 mask | ❌ | padding 同时生成 `attention_mask` |
| 图像尺寸不一 | ❌ | resize 到统一尺寸（在 `__getitem__` 里） |

真实项目里（包括 transformers 的 `DataCollatorWithPadding`）做的就是这件事。

### 6.5 数据该不该先搬上 GPU

```python
# 方案 A：整个数据集放显存（数据小，如本例 4096×2 个数）
xb_all = x.to(device)

# 方案 B：每批从 CPU 搬（数据集大）
for xb, yb in loader:
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
```

`non_blocking=True` 只在 `pin_memory=True` 时有意义，能让拷贝与计算重叠。
4GB 显存下，数据集放显存往往不现实，用方案 B。

> **第 6 章口诀：Dataset 管单条，DataLoader 管成批；变长序列写 collate，Windows 记得主入口。**

---

## 第 7 章 GPU 实战：4GB 显存怎么用

### 7.1 先算账：显存都花在哪

训练时的显存 = **参数 + 梯度 + 优化器状态 + 激活值**。

| 组成 | fp32 每参数字节 | 说明 |
| --- | --- | --- |
| 参数 | 4 B | 权重本身 |
| 梯度 | 4 B | 反向时每个参数一份 |
| Adam/AdamW 一阶动量 | 4 B | m |
| Adam/AdamW 二阶动量 | 4 B | v |
| **小计（不含激活）** | **16 B** | 用 SGD 则只需 8 B |
| 激活值 | 不定 | 与 batch × 序列长 × 层宽成正比，往往是最大头 |

按你的卡实测可用约 **3.6 GB**（系统与 CUDA 上下文要占几百 MB）算：

| 模型参数量 | 参数+梯度+Adam（16B/参数） | 4GB 卡能否全量训练 |
| --- | --- | --- |
| 1 M | 16 MB | ✅ 轻松 |
| 10 M | 160 MB | ✅ 可以 |
| 100 M | 1.6 GB | ⚠️ 加激活值后很紧 |
| 350 M | 5.6 GB | ❌ 不可能 |
| 1.5 B（如 Qwen2.5-1.5B） | 24 GB | ❌ 必须 LoRA/QLoRA |

结论：**这台机器本地能全量训练的上限在 1 亿参数以内**；7B 级模型只能
QLoRA 微调（第 10 章），或者上云 GPU。

> **口诀：十六字节每参数，激活另算；显存不够先降 batch，再谈精度。**

### 7.2 混合精度 AMP：本机实测结论和常识相反

```python
# torch 2.6 的写法（旧写法 torch.cuda.amp 已不推荐）
scaler = torch.amp.GradScaler("cuda", enabled=True)

for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**本机实测**（4 层 1024 宽 MLP，batch 2048，60 步，重复三次结果一致）：

| 精度模式 | 每步耗时 | 峰值显存 | 相对 fp32 |
| --- | --- | --- | --- |
| fp32 | 28.3 ms | 168 MB | 基准 |
| fp16 + GradScaler | **152 ms** | 150 MB | **慢 5.4 倍** |
| bf16 autocast | 32.7 ms | 150 MB | 慢 16% |

小模型（2 层 512 宽，batch 256）上结论一样：fp32 1.61 ms，fp16 3.80 ms，
bf16 1.94 ms。

**为什么 fp16 反而慢**：GTX 1650 是 Turing 架构，没有为深度学习优化的
Tensor Core 加速路径；fp16 运算在这块卡上要走效率很低的实现，GradScaler 的
scale/unscale 又增加了额外开销。而显存只省了约 11%，因为参数和优化器状态
始终是 fp32，省的只是激活值。

**所以本书给你的建议是**：

1. **默认用 fp32**，简单可靠。
2. **显存真的不够时**再试 bf16（速度损失小）；fp16 在你这块卡上没有收益。
3. 如果你换到 Ampere 及更新的卡（RTX 30/40 系、A100），fp16 + AMP 才会有明显
   加速，届时用 `torch.amp.autocast("cuda", dtype=torch.float16)` + GradScaler。

> **口诀：AMP 不是万能药，先测再开；fp16 必须配 GradScaler，不然梯度下溢。**

补充一个机制要点：fp16 的最小正规数约 6e-5，小幅梯度直接变成 0 或变成
非规格化数（denormal）后性能骤降。GradScaler 把 loss 乘以一个大数再反向，
让梯度落到 fp16 的正常范围，更新前再除回来。**没有 GradScaler 的 fp16 训练
轻则学不动，重则变慢**。

### 7.3 梯度累积：4GB 卡的核心技巧

```python
accum = 8                                    # 等效 batch = micro_batch × accum
optimizer.zero_grad(set_to_none=True)

for step, (xb, yb) in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        loss = criterion(model(xb), yb) / accum
    loss.backward()

    if (step + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

注意 **loss 要除以 accum**，否则等效学习率被放大 `accum` 倍。

### 7.4 梯度检查点（用时间换显存）

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        # 不缓存中间激活，反向时重算 → 显存约省 60~70%，速度慢 20~30%
        return checkpoint(self._fwd, x, use_reentrant=False)

    def _fwd(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

微调大模型时 transformers 支持 `gradient_checkpointing_enable()`，原理与上面相同。

### 7.5 显存不够了怎么办：按收益排序

| 顺序 | 手段 | 显存收益 | 速度代价 | 难度 |
| --- | --- | --- | --- | --- |
| 1 | 减小 batch，配梯度累积补回 | 大 | 小 | 无 |
| 2 | 减小序列长度（文本） | 大 | 小 | 无 |
| 3 | 换 bf16 / fp16 | 中 | 视硬件 | 低 |
| 4 | 梯度检查点 | 大 | 中 | 低 |
| 5 | 冻结部分层（只训练最后几层） | 中 | 无 | 低 |
| 6 | LoRA / QLoRA | 很大 | 小 | 中 |
| 7 | 模型并行 / CPU offload | 大 | 大 | 高 |

### 7.6 显存监控：三个 API 就够

```python
print(torch.cuda.memory_allocated() / 1024**2, "MB 已分配")   # 张量实际占用
print(torch.cuda.memory_reserved() / 1024**2, "MB 缓存池")    # 含缓存
print(torch.cuda.max_memory_allocated() / 1024**2, "MB 峰值")

# 排查后清理缓存池（不会释放还在用的张量）
torch.cuda.empty_cache()
```

`allocated` 一直涨 = 有张量没被释放（通常是日志里挂了计算图）。
`reserved` 远大于 `allocated` = 缓存池没还，属正常现象，不必紧张。

**OOM 排查顺序**：

1. 打印峰值显存，确认是真的超了还是碎片化的假象。
2. 把 batch 减半再跑，能过就是规模问题。
3. 检查验证阶段是否 `inference_mode()`。
4. 检查是否在循环里持续累积带图的 loss 或中间结果。
5. 上梯度检查点 / LoRA。

### 7.7 profiler：找出真正慢在哪

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        y = model(x)
        y.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

它会告诉你时间花在矩阵乘、拷贝还是 CPU 上，比凭感觉猜强得多。

### 7.8 torch.compile：你现在的环境用不了

```python
model = torch.compile(model)     # Windows + 无 Triton：直接失败
```

**实测报错**：

```
BackendCompilerFailed backend='inductor' raised:
RuntimeError: Compiler: cl is not found.
```

原因有两个：没有安装 Triton（Windows 官方 wheel 不含），且 inductor 需要
MSVC 编译器（`cl.exe`）。**结论：Windows 上跳过这一节**，用其他手段提速。

> **第 7 章口诀：参数梯度优化器十六字节，激活另算；先看峰值再动手；降 batch、上累积、最后检查点。**

---

## 第 8 章 调试与报错速查

### 8.1 报错速查表

| 报错信息（节选） | 真实原因 | 修法 |
| --- | --- | --- |
| `mat1 and mat2 shapes cannot be multiplied (32x784 and 800x256)` | 上一层输出维度 ≠ 本层 `in_features` | 打印 `x.shape`，对照 `nn.Linear` 的 in/out |
| `Expected all tensors to be on the same device` | 模型在 GPU、数据在 CPU（或反之） | 统一 `device`，检查是否漏了 `model.to(device)` |
| `expected scalar type Long but found Float` | Embedding/CE 的输入或标签不是 int64 | `y.long()`；预测输入不要转成 long |
| `Target 10 is out of bounds` | 标签值 ≥ 类别数（或 < 0） | 检查类别数，标签从 0 开始 |
| `element 0 of tensors does not require grad` | 前向用了 `no_grad`，或参数没被用到 | 确认 `loss.requires_grad == True` |
| `Trying to backward through the graph a second time` | 同一张图反向两次 | 每次前向配一次反向；或 `retain_graph=True` |
| `size mismatch for fc1.weight: copying a param with shape ...` | state_dict 结构与当前模型不一致 | 键名/维度对照；只加载公共键 |
| `CUDA out of memory` | 见 7.5 / 7.6 | 降 batch、AMP、检查点 |
| `Can't convert cuda:0 device type tensor to numpy` | 直接对 GPU 张量调 `.numpy()` | `x.detach().cpu().numpy()` |
| `DataLoader worker exited unexpectedly` | Windows 多进程没加主入口保护 | `if __name__ == "__main__":` |
| `one of the variables needed ... modified by an inplace operation` | 前向张量在反向之前被原地改写 | 去掉 `relu_` 等原地操作 |
| `RuntimeError: No CUDA GPUs are available` | CUDA 不可用 / 被其他进程占满 | `torch.cuda.is_available()`、`nvidia-smi` |

### 8.2 loss 变成 NaN 的排查顺序

按这个顺序查，绝大多数能在前四步解决：

1. **学习率太大** → 先降到 1e-4 再跑 50 步。
2. **输入数据含 NaN/Inf** → `torch.isnan(x).any()`、`torch.isinf(x).any()`。
3. **除以 0 或 log(0)** → 给分母/对数项加 `1e-8`。
4. **CE 前手动 softmax 了** → 去掉，直接把 logits 给 CE。
5. **梯度爆炸（RNN/Transformer）** → 加 `clip_grad_norm_`。
6. **fp16 下溢** → 用 GradScaler，或换 bf16/fp32。

### 8.3 loss 不动或下降极慢

| 检查项 | 命令 | 期望 |
| --- | --- | --- |
| 参数真的在变吗 | 训练前后对比 `p.sum()` | 应该不同 |
| 梯度是 0 吗 | 打印 `p.grad.abs().mean()` | 非 0 |
| 标签对齐了吗 | 模型预测分布 `pred.argmax(-1).bincount()` | 不应全是一个类 |
| 学习率 | `optimizer.param_groups[0]["lr"]` | 不是 0 且不过大 |
| 数据打乱 | `shuffle=True` | 训练有效 |
| 损失函数与任务匹配 | 分类用 CE、回归用 MSE | 别用反 |

### 8.4 三个调试工具

```python
# 1) 形状追踪：在前向里插打印
print("fc1 out", x.shape)

# 2) 梯度健康度
for n, p in model.named_parameters():
    if p.grad is not None:
        print(n, p.grad.abs().mean().item())

# 3) 交互式断点（Jupyter / ipython 里）
import pdb; pdb.set_trace()
```

**进阶**：`torch.autograd.set_detect_anomaly(True)` 会在反向出现 NaN 时
精确定位到算子。它会让训练慢好几倍，只在排错时临时打开。

> **第 8 章口诀：报错看最后一行，再看形状、类型、设备；NaN 先降学习率。**

---

## 第 9 章 完整项目：字符级语言模型

这一章把前面所有零件组装成一个能「说话」的小模型。代码在 `code/04_char_lm.py`，
在你这块卡上 **约 12 秒** 训完。

### 9.1 任务定义

给定一段文字，模型在每个位置预测「下一个字符是谁」。损失用交叉熵，
训练目标是让真实下一个字符的概率尽可能高。

### 9.2 数据与词表

```python
CORPUS = "大模型训练的核心是……"          # 一段 318 字的语料
chars = sorted(set(CORPUS))               # 去重排序 → 189 个不同字符
stoi = {c: i for i, c in enumerate(chars)}   # 字符 → 整数
itos = {i: c for c, i in stoi.items()}
ids = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
```

| 概念 | 本玩具 | 真实 GPT |
| --- | --- | --- |
| 切分单位 | 单个汉字 | BPE 子词（词表 10 万级） |
| 词表大小 | 189 | 50k ~ 150k |
| 语料 | 318 字 | TB 级网页文本 |
| 编码 | 直接查表 | tokenizer 分词 |

### 9.3 模型：Embedding + GRU + Linear

```python
class CharLM(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)      # (B,L) → (B,L,H)
        self.gru = nn.GRU(hidden, hidden, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, vocab_size)         # 每个位置输出 V 个分数

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.gru(h)          # 记住前文 —— 这就是「语言模型」的核心
        return self.head(h)         # (B, L, V)
```

数据窗口：每条样本是连续 `seq_len + 1` 个字符，输入前 `seq_len` 个，目标是后移一位：

```python
class CharDataset(Dataset):
    def __getitem__(self, i):
        chunk = self.ids[i:i + self.seq_len + 1]
        return chunk[:-1], chunk[1:]        # 输入/目标错开一位
```

### 9.4 训练：完整实测输出

```python
for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(xb).view(-1, V), yb.view(-1))   # (B*L, V) vs (B*L,)
    loss.backward()
    optimizer.step()
```

**本机实测（GTX 1650，12 秒）**：

```
device=cuda  vocab=189  语料=318 字
训练前： 大模型训练的核心是把海量文本变成播梯小生配产计向求概都率预法觉概练价差不度个来率先显播言误十果见只迹存，差压存送
开始训练...
epoch  0  loss 4.0942
epoch  5  loss 0.0403
epoch 10  loss 0.0221
epoch 15  loss 0.0201
epoch 20  loss 0.0196
epoch 24  loss 0.0197
```

注意两点：

1. **训练前**的输出是随机字符，但**已经是合法字符**——因为 Embedding 和
   `head` 只在 189 个字符里选，模型结构本身就限制了输出空间。
2. 25 个 epoch 后 loss 到 0.02，语料只有 318 字，模型基本**背下来**了。
   这在玩具上正常；真实训练靠海量数据防止这种「记忆」。

### 9.5 采样：温度控制随机性

```python
@torch.inference_mode()
def sample(n=80, temperature=0.8):
    start = ids[:16].unsqueeze(0).to(device)
    out = model(start)[:, -1].softmax(-1)
    text = [itos[i] for i in start[0].tolist()]
    for _ in range(n):
        nxt = torch.multinomial(out, 1)      # 按概率抽样
        text.append(itos[nxt.item()])
        out = model(nxt.view(1, 1))[:, -1].softmax(-1)
    return "".join(text)
```

| 温度 | 效果 |
| --- | --- |
| 0.1 ~ 0.5 | 保守，几乎复读训练语料 |
| 0.7 ~ 0.9 | 平衡，推荐 |
| > 1.2 | 随机、语法崩坏 |

**实测续写（温度 0.8）**：

```
大模型训练的核心是把海量文本变成一层， 反向传播沿着计算图把海量、常识和训练的痕迹。
参数， 模型训练循环之上下一个词的思想来源。这个个词的任务。这个位置上下一个词的痕迹。
参数据此微调只需要训练
```

语法通顺、局部连贯、整体在重复——这正是「小模型 + 小语料」的典型表现，
也是真实大模型要解决的梯度（规模、数据、注意力）。

### 9.6 这个玩具离 GPT 有多远

| 维度 | 本章玩具 | GPT 级模型 |
| --- | --- | --- |
| 序列建模 | GRU（递归） | 多层 Transformer（自注意力） |
| 参数量 | 约 0.44 M | 数十亿 ~ 数万亿 |
| 词表 | 189 字符 | BPE 5 万 ~ 15 万 |
| 训练数据 | 318 字 | TB 级 |
| 训练时长 | 12 秒 | 数千 GPU 月 |
| 生成长度 | 80 字就乱 | 长文可控 |
| 训练循环 | **和本书第 5 章一模一样** | 也是同一个五步循环 + 分布式 |

**关键认知**：大模型没有魔法。它只是把「Embedding → 堆很多层 → 每个位置预测
下一个 token → 交叉熵 → 反向传播」这套流程，在规模上放大了几个数量级。

**练习**：把 `nn.GRU` 换成

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=HIDDEN, nhead=8, dim_feedforward=4 * HIDDEN,
    batch_first=True, dropout=0.1)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

并在 `forward` 里把 `self.gru(h)` 换成 `self.encoder(h)`，再加一个因果掩码。
这就是一个最小 GPT。

> **第 9 章口诀：错一位做标签，交叉熵算损失；温度调随机，规模换能力。**

---

## 第 10 章 通向大模型：transformers / peft / trl

你的环境里已经装好了整条微调工具链。这一章讲清每个包负责什么、接口长什么样，
以及 4GB 显卡上现实能做什么。

### 10.1 三个包的分工

| 包 | 版本（本机） | 负责 | 你不必再自己写 |
| --- | --- | --- | --- |
| `transformers` | 4.57.6 | 模型结构、分词器、预训练权重加载 | 各种 Transformer 实现 |
| `peft` | 0.20.0 | LoRA 等参数高效微调 | 适配器注入与合并 |
| `trl` | 0.29.1 | SFT/DPO/GRPO 等训练器 | 微调循环、数据处理 |
| `accelerate` | 1.15.0 | 设备放置、混合精度、多卡 | 显存与设备管理 |
| `datasets` | 4.8.4 | 数据集加载与处理 | 数据管线 |

### 10.2 从 PyTorch 到 transformers：模型对象换了，训练方式没换

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"          # 需联网下载
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)

batch = tok("你好，请介绍一下你自己", return_tensors="pt")
out = model(**batch)
print(out.logits.shape)        # (B, L, vocab_size) —— 和你的 CharLM 一模一样
```

`AutoModelForCausalLM` 返回的就是一个 `nn.Module` 子类。它的输出是一个对象，
不是裸张量：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `logits` | `(B, L, V)` | 每个位置对词表中每个 token 的分数 |
| `last_hidden_state` | `(B, L, H)` | 最后一层隐藏状态（`AutoModel` 直接返回） |
| `past_key_values` | 元组 | KV 缓存，生成时复用 |
| `hidden_states` | 元组 | 每层输出（需 `output_hidden_states=True`） |
| `loss` | 标量 | 传入 `labels=` 时自动算的交叉熵 |

**关键区别**：传入 `labels` 后，模型内部完成「错一位 + 交叉熵」，
你不需要自己写 `criterion`。

### 10.3 手写循环 vs Trainer

| 维度 | 手写 PyTorch 循环 | `Trainer` / `SFTTrainer` |
| --- | --- | --- |
| 代码量 | 多 | 少 |
| 可控性 | 完全 | 受框架约定 |
| 混合精度 | 自己配 AMP | 参数开关 |
| 多卡/分布式 | 自己写 | 自动 |
| 断点与日志 | 自己写 | 内置 |
| 适合 | 学习、特殊结构、调试 | 标准微调、生产 |

手写循环微调（结构不变，只是损失由模型提供）：

```python
model.train()
for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)                 # batch 里有 labels，loss 自动算
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### 10.4 LoRA：只训练一小部分参数

**一句话原理**：冻结原模型，在旁边给每个权重矩阵加两个小矩阵 `A`、`B`
（`ΔW = B·A`，秩 `r` 远小于原维度），只训练它们。推理时可把 `B·A` 加回原权重，
不增加推理延迟。

**本机实测（离线构造的 2 层 GPT2，无需下载）**：

```python
from transformers import GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

cfg = GPT2Config(vocab_size=100, n_positions=64, n_embd=64, n_layer=2, n_head=4)
base = GPT2LMHeadModel(cfg)

lora = LoraConfig(r=8, lora_alpha=16,
                  target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base, lora)
model.print_trainable_parameters()
```

**实测输出**：

```
trainable params: 4,096 || all params: 114,688 || trainable%: 3.5714
```

只训练 **3.57%** 的参数，显存里的优化器状态同比减少约 96%。这就是
「4GB 显卡也能微调」的核心原理。

| LoRA 参数 | 含义 | 经验值 |
| --- | --- | --- |
| `r` | 低秩维数，越大越能学 | 8 ~ 64 |
| `lora_alpha` | 缩放系数 | 通常 2×r |
| `target_modules` | 给哪些层加适配器 | 注意力 q/k/v/o 或 `all-linear` |
| `lora_dropout` | 防过拟合 | 0.05 ~ 0.1 |

### 10.5 trl 的 SFTTrainer：标准微调写法

> ⚠️ 以下代码需要联网下载模型，**未在本机执行**；版本号为 trl 0.29 / peft 0.20
> 的写法，其他版本参数名可能略有差异。

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

ds = load_dataset("json", data_files="data.jsonl", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=SFTConfig(
        output_dir="out",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        bf16=False, fp16=False,        # 你的卡上按第 7 章结论用 fp32
        max_length=512,
    ),
    train_dataset=ds,
    peft_config=LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                           target_modules="all-linear"),
)
trainer.train()
trainer.save_model("out/lora_adapter")
```

这里出现的每个参数（batch、梯度累积、学习率、精度）你在第 4~7 章都已经见过——
**trl 只是把它们包装成了配置项**。

### 10.6 4GB 显卡的现实路线

| 模型规模 | fp16 权重 | 4-bit 权重 | 全量微调 | LoRA | QLoRA |
| --- | --- | --- | --- | --- | --- |
| 0.5B | 1.0 GB | 0.3 GB | ❌ | ✅ 可行 | ✅ |
| 1.5B | 3.0 GB | 0.8 GB | ❌ | ⚠️ 很紧（短序列） | ✅ |
| 7B | 14 GB | ~4 GB | ❌ | ❌ | ⚠️ 边界外，需 offload |
| 13B+ | 26 GB+ | 7 GB+ | ❌ | ❌ | ❌ |

**结论**：

- **本地学习**：0.5B 模型 + LoRA，序列长度 ≤ 512，batch 1 + 梯度累积 8。
- **真正微调**：租云 GPU（24GB 的 3090/4090 可跑 7B QLoRA），或用你环境里
  已有的 `ms_swift` 走 ModelScope 生态。
- 注意：4-bit 量化依赖 `bitsandbytes`，**你的环境里没有装**，Windows 上支持
  有限，建议在 WSL 或 Linux 云端做 QLoRA。

### 10.7 你的系列学习路线

```
已完成：PyTorch（本书）
   ↓
transformers：Tokenizer / AutoModel / generate  ← 下一步
   ↓
datasets：加载、map、filter、train_test_split
   ↓
peft：LoRA / QLoRA 原理与调参
   ↓
trl：SFT → DPO → GRPO（对齐）
   ↓
accelerate / ms-swift：多卡、显存优化、工程化
```

> **第 10 章口诀：transformers 给模型，peft 给适配器，trl 给循环；损失内建不用写，四 G 显存先 LoRA。**

---

## 第 11 章 练习与答案

### 一、判断与手算

1. `torch.tensor([1, 2, 3])` 的 dtype 是什么？能直接送进 `nn.Linear` 吗？
2. `x = torch.randn(32, 10)`，`x.mean(dim=1)` 和 `x.mean(dim=1, keepdim=True)`
   的形状分别是什么？
3. `a = torch.tensor(1.0, requires_grad=True)`，连续两次 `(a*2).backward()`，
   `a.grad` 是多少？
4. 为什么评估时要加 `torch.inference_mode()`？
5. `CrossEntropyLoss` 的输入应该是 logits 还是 softmax 后的概率？
6. 隐藏层 784→256 的 Linear，输入 `(32, 784)`，输出形状？
7. `DataLoader(num_workers=4)` 在 Windows 上运行报 bootstrapping 错误，最可能
   缺了什么？
8. 一个 100M 参数的模型用 AdamW 全量训练，光参数+梯度+优化器状态占多少显存？

### 二、动手题

1. 用 `nn.Module` 写一个两层 MLP，输入 10 维、隐藏 64、输出 3 类，打印参数量。
2. 给第 5 章的回归任务加一个 `CosineAnnealingLR`，比较 200 个 epoch 后的
   val loss 变化。
3. 把 `code/04_char_lm.py` 的 GRU 换成 `nn.TransformerEncoder`（记得因果掩码），
   训练并对比 loss。
4. 给训练循环加梯度裁剪和梯度累积（accum=4），确认结果与不累积时接近。
5. 用 `torch.profiler` 分析 `code/03_train_mlp.py`，找出耗时最多的算子。
6. 用 `peft` 给离线 GPT2 挂 LoRA（见 10.4），打印可训练参数占比，
   再试 `r=4` 和 `r=32` 对比。

### 参考答案要点

1. `int64`；不能，`Linear` 需要浮点输入，会报 dtype 错误。
2. `(32,)` 和 `(32, 1)`。广播相减时用后者。
3. `4`（2 + 2，梯度累加）。
4. 不建计算图：省显存、提速，避免长评估循环 OOM。
5. logits。CE 内部已包含 log_softmax。
6. `(32, 256)`。
7. `if __name__ == "__main__":` 主入口保护（Windows 用 spawn）。
8. 约 1.6 GB（16 字节/参数 × 1e8），还没算激活值。

---

## 附录 A API 速查表

### 张量操作

| 需求 | API |
| --- | --- |
| 创建 | `torch.tensor` `zeros` `ones` `randn` `rand` `arange` `linspace` `empty` |
| 同形 | `zeros_like` `ones_like` `randn_like` |
| 变形 | `view` `reshape` `permute` `transpose` `unsqueeze` `squeeze` `flatten` |
| 拼接 | `cat` `stack` `split` `chunk` |
| 索引 | `x[i]` `x[:, 1]` `x[mask]` `gather` `scatter` |
| 聚合 | `sum` `mean` `max` `min` `argmax` `prod` `norm` |
| 数值 | `clamp` `abs` `sqrt` `exp` `log` `pow` |
| 矩阵 | `@` `matmul` `mm` `bmm` `einsum` |
| 类型/设备 | `to` `float` `long` `half` `cpu` `cuda` |

### 模型与训练

| 需求 | API |
| --- | --- |
| 层 | `Linear` `Conv2d` `LSTM` `GRU` `Embedding` `LayerNorm` `BatchNorm1d` `Dropout` `MultiheadAttention` |
| 损失 | `CrossEntropyLoss` `BCEWithLogitsLoss` `MSELoss` `L1Loss` `KLDivLoss` |
| 优化器 | `SGD` `Adam` `AdamW` `RMSprop` |
| 调度器 | `StepLR` `CosineAnnealingLR` `OneCycleLR` `LambdaLR` |
| 保存 | `torch.save` `torch.load` `state_dict` `load_state_dict` |
| 梯度 | `backward` `zero_grad` `clip_grad_norm_` `torch.no_grad` `detach` |
| 精度 | `torch.autocast` `torch.amp.GradScaler` |
| 数据 | `Dataset` `DataLoader` `pad_sequence` `default_collate` |

### CUDA

| 需求 | API |
| --- | --- |
| 可用性 | `torch.cuda.is_available()` |
| 当前设备 | `torch.cuda.current_device()` |
| 显存 | `memory_allocated` `max_memory_allocated` `memory_reserved` `empty_cache` |
| 性能 | `torch.profiler` `nvidia-smi`（命令行） |
| 同步 | `torch.cuda.synchronize()`（计时必用） |

---

## 附录 B 术语中英对照

| 中文 | 英文 | 一句话 |
| --- | --- | --- |
| 张量 | tensor | 多维数组 + 设备 + 求导能力 |
| 自动求导 | autograd | 自动计算梯度的引擎 |
| 计算图 | computation graph | 前向时记录下来的运算依赖关系 |
| 叶子张量 | leaf tensor | 用户创建的、保存梯度的张量 |
| 反向传播 | backward propagation | 从损失往回算梯度 |
| 梯度累加 | gradient accumulation | 累积多批梯度再更新 |
| 混合精度 | mixed precision (AMP) | 部分运算用低精度，省显存提速 |
| 梯度缩放 | gradient scaling | fp16 下防止梯度下溢 |
| 检查点 | checkpoint | 保存的模型与训练状态 |
| 早停 | early stopping | 验证集不再改善就停 |
| 低秩适配 | LoRA | 只训练增量低秩矩阵 |
| 量化 | quantization | 降低权重位宽省显存 |
| 词表 | vocabulary | 模型认识的所有 token 集合 |
| 分词器 | tokenizer | 文本 ↔ token 编号的转换器 |
| 交叉熵 | cross entropy | 分类/语言模型的标准损失 |
| 学习率 | learning rate | 每步更新参数的幅度 |
| 调度器 | scheduler | 按计划调整学习率 |
| 过拟合 | overfitting | 训练集好、验证集差 |
| 注意力 | attention | 让每个位置读取其他位置的信息 |

---

## 结语

这本书只做了一件事：把「张量 → 自动求导 → 模块 → 训练循环 → 数据管线 → GPU」
这条链打通，并让你在你的真实硬件上跑起来。

后面无论 `transformers`、`peft` 还是 `trl`，代码看起来再复杂，底层都是同一条链：

```
模型(x) → loss → loss.backward() → optimizer.step() → zero_grad()
```

看到任何训练脚本，先找这五步在哪，剩下的都是配置和工程细节。


