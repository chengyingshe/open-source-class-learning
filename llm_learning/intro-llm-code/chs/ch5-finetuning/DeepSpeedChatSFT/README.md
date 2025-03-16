# DeepSpeed-Chat 指令微调实践

## 简介
本项目提供了使用 **DeepSpeed** 和 **PyTorch** 进行 **Baichuan 7B** 模型指令微调训练的示例代码。它支持分布式训练、LoRA（低秩适应）、训练数据的动态加载、内存优化等功能，旨在帮助用户高效地进行大规模语言模型的训练。

## 使用说明
- 下载指令微调用数据集至 `data/MyDataset` 目录下，并修改`dschat/utils/data/raw_datasets.py` 文件中的加载数据集逻辑函数 `my_load` 。
- 下载 **Baichuan-7B** 预训练模型至 `models` 目录下。
- 运行第一阶段训练脚本 `run_baichuan_7b.sh` ，训练指令微调用模型。
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash training/step1_supervised_finetuing/training_scripts/baichuan/run_baichuan_7b.sh
```