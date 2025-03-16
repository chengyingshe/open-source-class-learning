# Transformer架构用于机器翻译

## 简介
本项目实现了一个基于 **Transformer** 的机器翻译模型，旨在将英语（英文）翻译为法语（法文）。它利用了自注意力机制（Self-Attention）来处理序列数据，并使用了 **PyTorch** 框架进行实现。模型的核心部分包括一个 **编码器** 和 **解码器**，并通过训练语料进行优化。

## 使用说明
- 首先，需要从 [https://github.com/SamLynnEvans/Transformer/tree/master/data](https://github.com/SamLynnEvans/Transformer/tree/master/data) 下载 `english.txt` 和 `french.txt` 数据集，并将其放至 `data` 目录下。
- 然后，运行 `main.py` 文件，它会加载数据集，构建模型，训练模型，并进行翻译测试。
```
python main.py
```