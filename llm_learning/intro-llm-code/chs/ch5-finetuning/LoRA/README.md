# LoRA实践

## 简介
本项目演示了如何使用 **LoRA**（低秩适应）方法对 **Seq2Seq** 模型进行微调，并展示了微调参数量的大小。LoRA 是一种高效的模型微调方法，通过引入低秩矩阵来减少训练参数量，在保持高效性能的同时降低计算资源需求。此代码基于 **Transformers** 和 **PEFT**（Parameter Efficient Fine-Tuning）库，旨在帮助用户快速进行基于 LoRA 的模型微调。

## 使用说明
- 运行 `main.py` 文件，会自动下载并处理模型。
```
python main.py
```