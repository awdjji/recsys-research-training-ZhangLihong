# 学习笔记：PyTorch Lightning 训练完成，但最新 checkpoint 没有更新

## 一、问题背景

在复现 FatsMB 论文的过程中，需要训练完整模型 300 个 epoch，以获得扩散模型训练完成后的权重，用于 Figure 6 的潜在偏好可视化，以及后续的潜在空间迁移分析。

训练配置中的关键参数为：

```
model:
  pretrain_epoch: 100
  diffusion_epoch: 200

trainer:
  max_epochs: 300
```

按照该项目的训练逻辑：

- Epoch 0–99：预训练 MBAE。
    
- Epoch 100–199：训练扩散模型。
    
- Epoch 200–299：最终推荐阶段。
    

训练日志已经显示：

```
Epoch 299: 100% ...
`Trainer.fit` stopped: `max_epochs=300` reached.
```

这说明训练循环确实完成了 300 个 epoch。

但检查 checkpoint 时发现：

```
epoch=98-step=57222.ckpt
last.ckpt
```

使用 `torch.load()` 读取后，两者的 `epoch` 都是 98，`global_step` 也相同。之前其他实验中也有类似情况：最佳 checkpoint 停留在 epoch 97 或 99。

这导致两个问题：

1. 没有保存最终阶段的模型权重，无法直接用它做完整模型的潜在表示分析。
    
2. 训练日志虽然记录了指标，但不能仅凭 Recall、NDCG 等标量恢复训练结束时的模型参数。
    

## 二、排错过程与证据

### 1. 先检查 checkpoint 文件，而不是凭训练日志判断

使用命令：

```
find logs -type f \( \
  -name '*.ckpt' -o \
  -name '*.pt' -o \
  -name '*.pth' \
\) -print
```

再用 PyTorch 检查 checkpoint 的内部状态：

```
import torch

path = "logs/retail/full_mcgln_resume/lightning_logs/version_2/checkpoints/last.ckpt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("epoch:", ckpt.get("epoch"))
print("global_step:", ckpt.get("global_step"))
print("callbacks:", ckpt.get("callbacks"))
```

结果表明，checkpoint 实际保存到了 epoch 98 或 99，而不是 epoch 299。

### 2. 检查 ModelCheckpoint 保存状态

在 checkpoint 的 `callbacks` 字段中发现：

```
monitor: Val:Recall@20
best_model_path: .../epoch=99-step=57800.ckpt
last_model_path: .../last.ckpt
best_model_score: 0.21590463817119598
```

这说明原回调监控的是验证集 Recall@20，并记录了 epoch 99 对应的最佳模型。

关键在于：`**save_last=True**` **不能简单理解为“不论其他条件如何，每个 epoch 都必然保存最新权重”。**

在相关 PyTorch Lightning 版本的 `ModelCheckpoint` 文档中，`save_last=True` 的说明是：在 checkpoint 文件被保存时，将相应 checkpoint 复制到 `last.ckpt`。如果监控的最佳模型没有触发新的保存，`last.ckpt` 就可能继续保留较早的状态。

这与本次观察到的现象相符，但仍应以实际安装版本的文档和运行情况为准。

### 3. 查看 CLI 实际解析后的配置

使用：

```
python run.py fit \
  --config src/configs/retail_resume_mcgln.yaml \
  --print_config > /tmp/retail_resume_config.yaml
```

再检查：

```
grep -n -A20 -B2 'callbacks:' /tmp/retail_resume_config.yaml
```

`--print_config` 可以检查 LightningCLI 最终解析出的回调配置，而不是仅依赖 YAML 文件的表面写法。

当需要同时配置多个回调时，使用列表形式：

```
callbacks:
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      ...
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      ...
```

需要注意：`callbacks` 的格式规范化与后续监控指标错误是两个不同的问题，不能把它们混为一个根因。

## 三、第二次错误：为什么 `monitor: current_epoch` 失败？

为了解决最新权重不更新的问题，新增第二个 `ModelCheckpoint`，并尝试：

```
monitor: current_epoch
```

训练时出现错误：

```
MisconfigurationException:
ModelCheckpoint(monitor='current_epoch') could not find
the monitored key in the returned metrics
```

异常中列出了当前可用的指标：

```
'Val:Recall@20', ..., 'epoch', 'step'
```

但没有 `current_epoch`。

这条异常信息就是最直接的调试依据：`ModelCheckpoint` 监控的键必须能从回调收到的指标中找到。`current_epoch` 是 Trainer 的属性名称，并不等同于已出现在该指标集合中的日志键 `epoch`。

因此，应当使用：

```
monitor: epoch
```

而不是：

```
monitor: current_epoch
```

## 四、最终修复方案：两个回调分别负责两件事

最终配置中保留两个 `ModelCheckpoint`。

### 回调 1：保存 Recall@20 最佳模型

```
- class_path: pytorch_lightning.callbacks.ModelCheckpoint
  init_args:
    dirpath: logs/retail/full_mcgln_resume/checkpoints
    filename: best-{epoch:03d}
    monitor: 'Val:Recall@20'
    save_top_k: 1
    save_last: false
    mode: max
```

其作用是保存验证集 Recall@20 最高的模型，供论文指标对比使用。

### 回调 2：保存最新 epoch 的模型

```
- class_path: pytorch_lightning.callbacks.ModelCheckpoint
  init_args:
    dirpath: logs/retail/full_mcgln_resume/checkpoints
    filename: latest-{epoch:03d}
    monitor: epoch
    save_top_k: 1
    save_last: false
    mode: max
    auto_insert_metric_name: false
```

其作用是根据单调递增的 `epoch` 值，保留当前最新的 checkpoint。

由于 `epoch` 随训练递增，在 `mode: max`、`save_top_k: 1` 下，新的 epoch 会取代较早 epoch 对应的该回调 checkpoint。

两个回调的职责是：

|文件类型|用途|
|---|---|
|`best-xxx.ckpt`|验证集 Recall@20 最佳模型|
|`latest-xxx.ckpt`|当前训练进度对应的最新模型|

这样既能保留最佳模型，又能为中断恢复和最终阶段分析保留最新训练状态。

## 五、如何验证修复生效？

### 第一步：检查配置

```
python run.py fit \
  --config src/configs/retail_resume_mcgln.yaml \
  --print_config > /tmp/retail_resume_config.yaml

grep -n -A30 'callbacks:' /tmp/retail_resume_config.yaml
```

确认：

- `callbacks` 解析成列表。
    
- 第一个回调监控 `Val:Recall@20`。
    
- 第二个回调监控 `epoch`。
    
- 保存路径是预期目录。
    

### 第二步：训练时观察 checkpoint

```
watch -n 10 'ls -lt logs/retail/full_mcgln_resume/checkpoints/'
```

随着训练推进，应该能够看到类似以下文件：

```
best-099.ckpt
latest-100.ckpt
latest-101.ckpt
latest-102.ckpt
```

具体文件名取决于保存时的 epoch 编号和实际配置。

### 第三步：读取最新 checkpoint 的元数据

```
import torch
from pathlib import Path

root = Path("logs/retail/full_mcgln_resume/checkpoints")
files = sorted(root.glob("latest-*.ckpt"))

if not files:
    raise RuntimeError("没有找到 latest checkpoint")

path = files[-1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("checkpoint:", path)
print("epoch:", ckpt.get("epoch"))
print("global_step:", ckpt.get("global_step"))
```

完整训练 300 个 epoch 后，预期最终记录的 `epoch` 为 299。

需要注意，之前未保存的 epoch 299 权重无法从标量日志中还原。如果没有其他较晚阶段的 checkpoint，就必须从已有的有效 checkpoint 接续训练。

## 六、问题来源在哪里查？

本次定位主要使用了三类资料。

### 1. 自己的报错信息和 checkpoint 元数据

优先级最高。报错直接列出了可用的指标键，checkpoint 则提供 `epoch`、`global_step`、`best_model_path`、`last_model_path` 等状态。它们帮助将问题缩小到保存回调，而不是数据集或模型结构。

### 2. PyTorch Lightning 官方文档

- [ModelCheckpoint API 文档（PyTorch Lightning 1.7.7）](https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.callbacks.ModelCheckpoint.html)
    
- [ModelCheckpoint 源码文档（PyTorch Lightning 1.7.7）](https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/callbacks/model_checkpoint.html)
    
- [Checkpointing 官方指南](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html)
    

这些资料解释了 `monitor`、`save_top_k`、`save_last` 的语义，以及如何检查回调保存的最佳模型路径。

由于不同 Lightning 版本的实现可能有差异，应先查看本地版本：

```bash
python -c "import pytorch_lightning as pl; print(pl.__version__)"
```

再阅读对应版本的文档。
## 七、这次排错的经验总结
1. 训练日志上有last.ckpt不代表就一定是最新的文件，还要通过其它方式去确定保存的到底是哪个参数文件，我是通过后面做实验感觉图画出来不太对发现的
2. **先用** `**--print_config**` **验证配置解析，再开始长时间训练。** 不要等训练结束后才发现 Callback 配置与预期不符。
3. **训练实验应同时保留最佳 checkpoint 和最新 checkpoint。** 前者用于模型选择与性能报告，后者用于继续训练和分析最终阶段的潜在表示。