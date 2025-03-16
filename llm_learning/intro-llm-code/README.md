# intro-llm-code

# 大语言模型：从理论到实践

本代码库为《大语言模型：从理论到实践》一书的配套实践项目，涵盖从基础理论到前沿应用的全流程技术实现。通过模块化代码和分阶段实践，帮助读者深入理解大语言模型的核心技术。

## 核心内容

| 章节 | 主题         | 关键技术            | 代码目录              |
|------|--------------|---------------------|-----------------------|
| 2    | 大语言模型基础 | Transformer，预训练 | ch2-foundations/       |
| 3    | 大语言模型预训练数据 | 数据清洗，语言处理   | ch3-pretrain-data/     |
| 4    | 分布式训练     | 数据并行，流水线并行  | ch4-distributed/       |
| 5    | 指令微调       | LoRA，Prompt Tuning   | ch5-finetuning/        |
| 6    | 强化学习       | PPO，RLHF        | ch6-rl/                |
| 7    | 多模态大语言模型     | MiniGPT-4，视觉-语言对齐   | ch7-multimodal/        |
| 8    | 大模型智能体   | LangChain，智能体       | ch8-agents/            |
| 9    | 检索增强生成   | RAG，知识库集成       | ch9-rag/               |
| 10   | 大语言模型效率优化       | 高效推理        | ch10-optimization/     |
| 11   | 大语言模型评估       | 对比评估，评估指标    | ch11-evaluation/       |

  
## 🛠️ 环境配置

### 最低硬件要求
- **GPU**: NVIDIA A100 40GB * 1（单卡推理） / * 8（全量预训练）
- **内存**: 64GB RAM
- **存储**: 1TB NVMe SSD

### 软件依赖
- **操作系统**: Linux (Ubuntu 22.04+ 推荐) / Windows 11 WSL2
- **Python**: 3.10+ (Anaconda 推荐)

### 深度学习框架
- **PyTorch**: 2.3+ with CUDA 11.8
- **DeepSpeed**: 0.14+

## 📂 目录结构

```
.
└── ch{2-11}-*/             # 各章节代码
    ├── project1/             # 项目1
    │   ├── data/               # 所需数据
    │   ├── main.py/            # 脚本入口
    │   ├── README.md           # 章节说明
    │   └── requirements.txt    # 环境依赖 
    │
    ├── project2/           # 项目2
    └── .../                # ...
```


## 🚀 运行方式

- 克隆或下载代码
- 安装依赖库
```
pip install -r requirements.txt
```
- 运行相应的python文件、ipynb文件或shell脚本
```
python main.py

torchrun --nnodes 1 --nproc_per_node=4 tensor_parallel.py
```