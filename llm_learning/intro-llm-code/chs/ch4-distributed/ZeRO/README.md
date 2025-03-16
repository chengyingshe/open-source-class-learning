# 零冗余优化器

## 简介
本项目展示了如何在 **PyTorch** 中通过使用 **ZeroRedundancyOptimizer** 优化计算设备内存的使用。项目通过对比使用和不使用 **ZeRO** 的内存占用情况，展示了如何有效减少内存冗余，从而提升训练过程中的内存效率。

## 使用说明
- 运行 `optimizer.py` 文件，即可看到使用和不使用 **ZeRO** 的内存占用对比。
```
python optimizer.py
```