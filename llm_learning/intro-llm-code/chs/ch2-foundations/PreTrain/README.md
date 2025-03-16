# 预训练语言模型实践

## 简介
本项目展示了如何使用 **Hugging Face** 的 **Transformers** 库从头开始训练一个BERT预训练模型。具体来说，项目包括使用 **BERT WordPiece** 进行词元化训练，以及使用该词元化模型训练BERT掩码语言模型（**Masked Language Model, MLM**）。项目基于 **bookcorpus** 和 **wikipedia** 数据集，并使用了多进程处理来加速数据预处理。

## 使用说明
- 运行 `main.py` 文件即可开始训练。
```
python main.py
```