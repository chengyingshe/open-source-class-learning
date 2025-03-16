# 分布式数据并行

## 简介
本项目是基于 **PyTorch** 实现的 **DeepLab** 模型的分布式数据并行训练脚本。该脚本实现了多 GPU 分布式训练，适用于大规模图像分割任务，例如 **Cityscapes** 数据集。训练过程中，会自动分配数据和同步各个 GPU 之间的梯度，以加速训练过程。

## 使用说明
- 首先，下载 **[Cityscapes](https://www.cityscapes-dataset.com/)** 的 **leftImg8bit** 和 **gtFine** 数据集，并解压到 `data` 目录。
- 运行 `main.py` 开始训练并最终保存模型。
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py
```