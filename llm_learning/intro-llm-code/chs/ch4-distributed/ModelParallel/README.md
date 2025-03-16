# 模型并行

## 简介
此项目展示了如何使用 **PyTorch** 进行 **模型并行**，包括 **流水线并行** 和 **张量并行** 的实现。通过将模型或张量分割成多个部分，并行化每个部分的计算，以此来提升训练效率，特别是在处理超大规模模型时非常有用。
- 流水线并行：将模型的不同模块分布在不同设备上，并行计算每个模块的输出。
- 张量并行：根据模型的具体结构和算子类型，将参数切分到不同设备，并通过集合通信汇聚结果。

## 使用说明
- 运行 `pipeline_parallel.py` 文件进行流水线并行测试。
```
torchrun --nnodes 1 --nproc_per_node=2 pipeline_parallel.py
```
- 运行 `tensor_parallel.py` 文件进行张量并行测试。
```
torchrun --nnodes 1 --nproc_per_node=4 tensor_parallel.py
```