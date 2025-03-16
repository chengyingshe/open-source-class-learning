# LLaMA分布式训练实践

## 简介
本项目是一个基于 **DeepSpeed** 和 **PyTorch** 的大规模分布式训练脚本，旨在通过多机多卡训练加速 **LLaMA** 模型的训练过程。

## 使用说明
- 主要运行逻辑在 `main.py` 文件中。
- 通过shell脚本 `run_llama2_7b.sh` 启动训练。可修改脚本中的参数以调整训练配置。
```
bash run_llama2_7b.sh
```