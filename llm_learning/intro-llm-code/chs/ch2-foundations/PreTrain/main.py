'''1.2.3 预训练语言模型实践'''

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from datasets import concatenate_datasets, load_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline

from itertools import chain


bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")

# 仅保留 'text' 列
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

dataset = concatenate_datasets([bookcorpus, wiki])

# 将数据集切分为 90% 训练集，10% 测试集
d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
    """将数据集文本保存到磁盘的通用函数"""
    with open(output_filename, "w", encoding="utf-8") as f:
        for t in dataset["text"]:
            print(t, file=f)

# 将训练集保存为 train.txt
dataset_to_text(d["train"], "train.txt")

# 将测试集保存为 test.txt
dataset_to_text(d["test"], "test.txt")


special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# 如果根据训练和测试两个集合训练词元分析器，则需要修改files
# files = ["train.txt", "test.txt"]
# 仅根据训练集合训练词元分析器
files = ["train.txt"]
# BERT中采用的默认词表大小为30522，可以随意修改
vocab_size = 30_522
# 最大序列长度，该值越小，训练速度越快
max_length = 512
# 是否将长样本截断
truncate_longer_samples = False

# 初始化WordPiece词元分析器
tokenizer = BertWordPieceTokenizer()
# 训练词元分析器
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# 允许截断达到最大512个词元
tokenizer.enable_truncation(max_length=max_length)

model_path = "pretrained-bert"

# 如果文件夹不存在，则先创建文件夹
if not os.path.isdir(model_path):
  os.mkdir(model_path)
# 保存词元分析器模型
tokenizer.save_model(model_path)
# 将一些词元分析器中的配置保存到配置文件，包括特殊词元、转换为小写、最大序列长度等
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

# 当词元分析器进行训练和配置时，将其装载到BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)

def encode_with_truncation(examples):
    """使用词元分析对句子进行处理并截断的映射函数（Mapping function）"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
    """使用词元分析对句子进行处理且不截断的映射函数（Mapping function）"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

# 编码函数将依赖于 truncate_longer_samples 变量
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# 对训练数据集进行分词处理
train_dataset = d["train"].map(encode, batched=True)
# 对测试数据集进行分词处理
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    # 移除其他列，将 input_ids 和 attention_mask 设置为 PyTorch 张量
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    # 移除其他列，将它们保留为 Python 列表
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


# 主要数据处理函数，拼接数据集中的所有文本并生成最大序列长度的块
def group_texts(examples):
    # 拼接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 舍弃了剩余部分，如果模型支持填充而不是舍弃，则可以根据需要自定义这部分
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # 按照最大长度分割成块
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# 请注意，使用 batched=True，此映射一次处理 1000 个文本
# 因此，group_texts 会为这 1000 个文本组抛弃不足的部分
# 可以在这里调整 batch_size，但较高的值可能会使预处理速度变慢
#
# 为了加速这一部分，使用了多进程处理
if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True,
                                      desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
    # 将它们从列表转换为 PyTorch 张量
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

# 使用配置文件初始化模型
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

# 初始化数据整理器，随机屏蔽 20%（默认为 15%）的标记
# 用于掩盖语言建模（MLM）任务
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir=model_path,          # 输出目录，用于保存模型检查点
    evaluation_strategy="steps",    # 每隔 `logging_steps` 步进行一次评估
    overwrite_output_dir=True,
    num_train_epochs=10,            # 训练时的轮数，可以根据需要进行调整
    per_device_train_batch_size=10, # 训练批量大小，可以根据 GPU 内存容量将其设置得尽可能大
    gradient_accumulation_steps=8,  # 在更新权重之前累积梯度
    per_device_eval_batch_size=64,  # 评估批量大小
    logging_steps=1000,             # 每隔 1000 步进行一次评估，记录并保存模型检查点
    save_steps=1000,
    # load_best_model_at_end=True,  # 是否在训练结束时加载最佳模型（根据损失）
    # save_total_limit=3,           # 如果磁盘空间有限，则可以限制只保存 3 个模型权重
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 训练模型
trainer.train()

# 加载模型检查点
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))
# 加载词元分析器
tokenizer = BertTokenizerFast.from_pretrained(model_path)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# 进行预测
examples = [
  "Today's most trending hashtags on [MASK] is Donald Trump",
  "The [MASK] was cloudy yesterday, but today it's rainy.",
]
for example in examples:
  for prediction in fill_mask(example):
    print(f"{prediction['sequence']}, confidence: {prediction['score']}")
  print("="*50)
