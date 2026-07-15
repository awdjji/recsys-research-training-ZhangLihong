##1. 张量基础操作
### 1.1 创建张量

| 方法 | 说明 | 示例 |
|------|------|------|
| `torch.arange(n)` | 创建 `[0, n)` 的一维张量（左闭右开） | `torch.arange(12)` → `[0..11]` |
| `torch.arange(a, b)` | 创建 `[a, b)` 的一维张量 | `torch.arange(1, 2)` → `[1]` |
| `torch.ones(*shape)` | 全 1 张量 | `torch.ones(2, 1)` |
| `torch.zeros(*shape)` | 全 0 张量 | `torch.zeros(3, 2)` |
| `torch.tensor(list)` | 从列表/嵌套列表构建 | `torch.tensor([[2,1,3],[4,6,1]])` |
| `torch.randn(*shape)` | 标准正态分布随机张量 | `torch.randn(3, 4)` |

### 1.2 查看与变换形状

- **`x.numel()`**：返回张量中元素的总个数
- **`x.shape`**：返回张量的形状（`torch.Size` 对象）
- **`x.reshape(*new_shape)`**：在不改变值的情况下重塑形状
  - `x.reshape(-1)` 可将任意张量展平为一维
  - `-1` 表示由系统自动推断该维度的大小

### 1.3 逐元素运算

张量支持 `+`, `-`, `*`, `/`, `**` 等运算符，**运算时逐元素计算**：

```python
target = from_list + from_list   # 逐元素加
target = from_list * from_list   # 逐元素乘
target = from_list ** from_list  # 逐元素幂
```

### 1.4 广播机制（Broadcasting）

当两个形状不同的张量进行运算时，PyTorch 会自动广播对齐。规则（从**最右边的维度**开始逐维检查）：

每个维度必须满足以下三个条件**之一**：
1. **相等**
2. **其中一个为 1**
3. **其中一个不存在**（维度较少的张量，缺失维度自动视为 1）

```python
test1 = torch.tensor([[2,4,1]])      # shape (1, 3)
test2 = torch.tensor([[1],[2],[3]])   # shape (3, 1)
test1 + test2                         # 广播后 shape (3, 3)
```

### 1.5 张量拼接与堆叠

| 函数 | 行为 | 关键区别 |
|------|------|----------|
| `torch.cat(tensors, dim)` | 沿已有维度拼接 | 不增加维度数，要求非拼接维度形状相同 |
| `torch.stack(tensors, dim)` | 沿**新维度**堆叠 | 输出比输入多一个维度 |

**注意**：
- `cat()` 要求拼接维度以外的维度必须相同
- `stack()` 要求所有张量形状完全相同
- `cat()` 的底层是拼接，`stack()` 的底层是 `stack`（类似 numpy）

```python
torch.cat((A, B), dim=0)    # 行方向拼接
torch.cat((A, B), dim=1)    # 列方向拼接
torch.stack((A, B), dim=0)  # 新增维度 0 进行堆叠
```

### 1.6 矩阵乘法

```python
out = torch.matmul(A, B)   # 等价写法
out = A @ B                # 最简洁写法（推荐）
```

- `torch.dot(a, b)`：**只能用于一维张量**，计算点积（对应元素相乘后求和）

### 1.7 类型转换

| 操作 | 代码 |
|------|------|
| Tensor → NumPy | `A = tensor.numpy()` |
| NumPy → Tensor | `B = torch.tensor(A)` |
| 单元素 Tensor → Python 标量 | `C.item()` 或 `float(C)` |
| 求所有元素之和 | `tensor.sum()` |

---

## 2. 自动求导（Autograd）

### 2.1 核心概念

- **`requires_grad=True`**：启用梯度追踪，**必须是浮点数或复数**才能使用
- **`x.grad`**：存储累积的梯度值
- **`backward()`**：触发反向传播，**只能对标量调用**（非标量需先 `.sum()` 等归约）
- 梯度是**累积**的，每次 `backward()` 后新梯度会**叠加**到旧梯度上

### 2.2 基本用法

```python
x = torch.tensor([[2.,3.,4.],[1.,2.,3.]], requires_grad=True)
y = x ** 2                    # y 会带有 grad_fn=<PowBackward0>
y.sum().backward()            # 反向传播，计算 dy/dx，标量对任意形状的张量求导，结果形状与该张量完全相同
print(x.grad)                 # 结果为 2x
```

**验证**：`z = 2 * torch.dot(a, a)` → `dz/da = 4a`

### 2.3 梯度清零

```python
a.grad.zero_()   # 手动将梯度置零（注意下划线表示原地操作）
```

> **为什么需要清零？** 在训练循环中，每个 batch 的梯度会累积。不清零会导致参数更新方向错误。通常在 `optimizer.zero_grad()` 中完成。

### 2.4 detach() — 脱离计算图

```python
y = a * a
u = y.detach()    # u 与之前的计算图断开
z = u * a
z.sum().backward() # a.grad 不会包含 y=a*a 这条路径的梯度
```

**用途**：
- 冻结某些参数不参与训练
- 在 GAN 等场景中阻止梯度回传
- 将中间结果作为常量使用

---

## 3. 反向传播的链式法则（深入理解）

在 PyTorch 的反向传播中：

1. **上层传入**到本层的是：最终输出 L 对上层输出 y（即本层输入）的求导矩阵（通过链式法则层层传递）
2. **本层计算**雅可比矩阵（本层输出 y 对输入 x 的求导），然后与上层传入的梯度相乘，得到最终输出对本层输入的求导矩阵，再传递给下一层
3. PyTorch **不会显式计算完整的雅可比矩阵**，而是通过矩阵乘法实现：
   - `y = x @ W` 的反向：`grad_x = grad_output @ W.T`，`grad_W = x.T @ grad_output`
4. `.grad` 保存的是**梯度**（L 对 x 的偏导），新梯度与旧梯度**累加**
5. 当使用 `detach()` 将计算移出计算图后，梯度**不会**累加该路径的贡献

### 随机数生成器隔离

PyTorch 有全局默认 RNG（通过 `torch.manual_seed()` 设置）。任何使用随机数的操作都会改变全局状态，导致局部随机结果不一致。

**解决方案**：使用 `torch.Generator().manual_seed(SEED)` 来隔离不应受模型改动影响的关键操作（如数据加载的随机打乱）。

---

## 4. nn.Module 模块系统

### 4.1 核心规则

| 特性 | 说明 |
|------|------|
| 参数自动注册 | 子类中创建的参数自动加入 `.parameters()` |
| 标准层 | 如 `nn.Linear`，参数自动包装 |
| 手动注册参数 | 使用 `nn.Parameter(tensor)` |
| 不可学习参数 | 使用 `register_buffer()` 注册，不会被优化器更新 |
| 核心方法 | 子类重写 `__init__()` 和 `forward()` |
| 状态切换 | `.eval()` / `.train()` 递归作用于所有子模块 |
| 设备/类型迁移 | `.to(device)` / `.float()` 递归改变所有参数 |

### 4.2 nn.Sequential — 快速搭建

```python
net = nn.Sequential(
    nn.Linear(20, 256),  # 第一个维度视作 batch（透明维度），只在第二个维度计算
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

- 支持**索引访问**：`net[0]`、`net[2].bias`
- `net[2].state_dict()` 查看某一层的参数
- `net[2].bias.data` 直接访问偏置的数值

### 4.3 自定义 Module

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()                          # 必须调用父类 init
        self.hidden = nn.Linear(20, 256)            # 标准层 → 参数自动注册
        self.para = nn.Parameter(torch.rand(2, 256)) # 手动注册可学习参数
        self.register_buffer("mask", torch.ones(2, 256)) # 注册不可学习参数（如掩码）

    def forward(self, x):
        x = self.hidden(x)
        x = x * self.para
        x = x ** self.mask
        return x
```

**参数查看**：
```python
for name, p in model.named_parameters():   # 返回 (名称, 张量) 对
    print(name, p.shape, p.requires_grad)

for k, v in model.state_dict().items():    # 包含 buffer
    print(k, v.shape)
```

**区别**：
- `.parameters()` → generator，只含可学习参数
- `.named_parameters()` → generator，含 (名称, 可学习参数)
- `.state_dict()` → OrderedDict，含**所有**参数 + buffer

### 4.4 自定义 Sequential

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        net = nn.Sequential()
        for i, block in enumerate(args):
            net.add_module(f"block{i}", block)  # 带命名的模块注册

    def forward(self, x):
        for block in self._modules.values():    # 按注册顺序前向传播
            x = block(x)
        return x
```

> **`add_module()`** 是推荐的注册子模块方式。直接操作 `_modules` 字典不推荐。
> **`nn.ModuleList`** 可快速自动注册模块列表，但没有命名。

### 4.5 参数共享

将同一个层实例放入网络的不同位置即可实现参数共享：

```python
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    shared,              # 第一次使用
    nn.Linear(8, 8),
    shared,              # 第二次使用，共享同一组参数
    nn.ReLU(),
    nn.Linear(8, 1)
)
# net[2].weight is net[4].weight → True（同一个对象）
```

---

## 5. Dataset 与 DataLoader

### 5.1 Dataset 类

自定义数据集需要继承 `torch.utils.data.Dataset`，实现三个方法：

| 方法 | 职责 |
|------|------|
| `__init__()` | 存储元信息（文件路径、标签、transform 对象） |
| `__len__()` | 返回数据集大小（样本总数） |
| `__getitem__(idx)` | 加载并返回单条数据（含 transform 变换） |

**两种数据加载策略**：

| 策略 | 加载时机 | 内存占用 | 适用场景 |
|------|----------|----------|----------|
| **Eager（急切）** | `__init__()` 中一次性全部加载 | 高（一次性占满内存） | 数据集小，需要快速随机访问 |
| **Lazy（懒加载）** | `__getitem__()` 中按需加载 | 低（仅当前 batch） | 大数据集、磁盘 I/O 读取 |

```python
# Lazy 模式示例
class LazyDataset(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        data = load_from_disk(idx)   # 按需加载
        return data, label
```

### 5.2 DataLoader

数据加载器，封装 Dataset 并提供 batch 迭代功能。

**关键参数**：

| 参数               | 说明                                                                   |
| ---------------- | -------------------------------------------------------------------- |
| `batch_size`     | 每批样本数                                                                |
| `shuffle=True`   | 每个 epoch 打乱加载顺序                                                      |
| `num_workers`    | 数据加载的子进程数（0=主进程加载）                                                   |
| `drop_last=True` | 丢弃不足 batch_size 的尾批（**使用 BatchNorm 时建议开启**，尾批 size=1 会导致方差为 0 → NaN） |
| `collate_fn`     | 自定义 batch 拼装函数                                                       |

**迭代方式**：
```python
loader = DataLoader(dataset, batch_size=64)
for batch in loader:          # 直接迭代
    ...
batch = next(iter(loader))    # 手动取一个 batch
```

### 5.3 default_collate 的行为

默认的 `collate_fn` 底层使用 **`torch.stack`**（非 `cat`），会自动创建 batch 维度：

| 输入类型 | collate 行为 |
|----------|-------------|
| `List[Tuple]` | 转置为 `Tuple[Tensor]`，每个 Tensor 增加 batch 维 |
| `List[Dict]` | 按 key 递归 collate，数值 → Tensor，字符串 → List |

```python
# 元组形式：[(data1, label1), (data2, label2), ...]
# → (stacked_data, stacked_labels)

# 字典形式：[{"img": t1, "lbl": l1}, {"img": t2, "lbl": l2}]
# → {"img": stacked_tensor, "lbl": stacked_labels}
```

**限制**：同一 batch 内所有张量的形状**必须相同**，否则 `stack` 会报错。

### 5.4 自定义 collate_fn — 处理变长序列

当数据长度不一致时（如 NLP 中的变长句子），需要自定义 `collate_fn` 进行 padding：

```python
def pad_collate(batch):
    sentences, labels = zip(*batch)      # 转置：分离数据和标签
    max_len = max(s.size(0) for s in sentences)
    
    padded = torch.zeros(len(sentences), max_len)
    lengths = []
    for i, s in enumerate(sentences):
        padded[i, :s.size(0)] = s        # 右侧补零
        lengths.append(s.size(0))
    
    return padded, labels, lengths       # 返回填充后的数据、标签、原始长度

loader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate)
```

**关键技巧**：
- `zip(*batch)` 将 `[(d1,l1), (d2,l2), ...]` 转置为 `[(d1,d2,...), (l1,l2,...)]`，方便批量操作
- 返回原始长度 `lengths` 以便后续使用 `pack_padded_sequence` 等
- 一维张量的 `size(0)` 是序列长度，二维张量的 `size(0)` 是行数

---

## 6. 速查表

### 张量创建
```python
torch.arange(n)          # [0, 1, ..., n-1]
torch.zeros(3, 4)        # 3x4 全零
torch.ones(2, 3)         # 2x3 全一
torch.randn(5, 5)        # 5x5 标准正态
torch.tensor([[1,2],[3,4]])  # 从列表创建
```

### 形状操作
```python
x.shape                  # 查看形状
x.numel()                # 元素总数
x.reshape(3, 4)          # 重塑
x.reshape(-1)            # 展平
```

### 拼接与堆叠
```python
torch.cat((A, B), dim=0)    # 沿 dim=0 拼接
torch.stack((A, B), dim=0)  # 新增 dim=0 堆叠
```

### 自动求导
```python
x = torch.tensor([1.], requires_grad=True)
y = x ** 2
y.backward()             # 反向传播（仅标量可调用）
x.grad                   # 查看梯度（累积的）
x.grad.zero_()           # 清零梯度
y.detach()               # 脱离计算图
```

### nn.Module
```python
net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
net.parameters()         # 可学习参数 generator
net.named_parameters()   # (名称, 参数) generator
net.state_dict()         # 所有参数 + buffer
net.train() / net.eval() # 训练/评估模式切换
net.to(device)           # 迁移到 GPU
```

### DataLoader
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True,
                    num_workers=4, drop_last=True, collate_fn=pad_collate)
for batch_data, batch_labels in loader:
    ...
```
