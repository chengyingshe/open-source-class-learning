# 集合通信实例

## 简介
本项目演示了如何使用 PyTorch 的分布式计算功能进行全局求和（ **All-Reduce** ）。代码使用多进程和分布式计算模块 `torch.distributed` 来初始化分布式环境，并对张量进行全局求和操作。

## 使用说明
- 运行分布式计算。
```
python all_reduce.py
```