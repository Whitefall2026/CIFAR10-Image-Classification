# CIFAR-10 图像分类

一个使用 PyTorch 训练 CIFAR-10 图像分类模型的简单项目。

## 模型结构

- 3 组卷积、BatchNorm 和 ReLU
- 3 次 Max Pooling
- 自适应平均池化与全连接分类器
- 数据增强：随机裁剪、随机水平翻转
- 优化器：Adam
- 学习率调度：Cosine Annealing

默认训练 80 个 epoch，并在训练结束后将模型保存为 `cifar10_cnn.pth`。

## 环境准备

```bash
pip install torch torchvision
```

## 训练

```bash
python cifar10.py
```

首次运行时会自动将 CIFAR-10 数据集下载到 `data/`。脚本会自动选择 CUDA 或 CPU。

## 文件说明

- `cifar10.py`：数据加载、模型定义、训练与测试
- `cifar10_cnn.pth`：已保存的模型权重
- `test_torch.py`：检查 PyTorch、torchvision 和 CUDA 环境
