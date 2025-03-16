'''1.4.2 LLaMA分布式训练实践'''

import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import convert_lora_to_linear_layer
from dschat.utils.model.model_utils import create_llama_model
from dschat.utils.perf import print_throughput


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add `eot_token` as additional special token to tokenizer")
    parser.add_argument(
        "--eot_token",
        type=str,
        default="<|endoftext|>",
        help="Specify the format of the `eot_token`",
    )
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    '''DeepSpeed 初始化'''
    # 根据 local_rank 设置设备，local_rank 用于分布式训练
    if args.local_rank == -1:
        # 如果没有使用分布式训练，使用默认的 GPU
        device = torch.device("cuda")  
    else:
        # 如果在分布式训练中，设置当前进程的 GPU 设备
        torch.cuda.set_device(args.local_rank)  # 设置当前 GPU 设备
        device = torch.device("cuda", args.local_rank)  # 使用指定的 GPU
    
    # 初始化分布式后端，处理节点/GPU 的同步
    # torch.distributed.init_process_group(backend='nccl')  # 注释掉的初始化代码
    deepspeed.init_distributed()  # 使用 DeepSpeed 初始化分布式环境
    
    # 获取当前进程的全局排名
    args.global_rank = torch.distributed.get_rank()
    
    # 获取训练的 DeepSpeed 配置
    ds_config = get_train_ds_config(
        offload=args.offload,  # 是否卸载参数到 CPU
        dtype=args.dtype,  # 训练数据类型
        stage=args.zero_stage,  # ZeRO 优化的阶段
        enable_tensorboard=args.enable_tensorboard,  # 是否启用 TensorBoard
        tb_path=args.tensorboard_path,  # TensorBoard 日志路径
        tb_name="step1_model"  # TensorBoard 日志名称
    )
    
    # 设置每个 GPU 的微批次大小
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    
    # 设置全局批次大小，计算方法为每个设备的批次大小乘以 GPU 数量乘以梯度累积步数
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    
    # 如果传入了种子，设置训练种子
    set_random_seed(args.seed)
    
    # 在所有进程中同步
    torch.distributed.barrier()

    '''模型载入'''
    # 加载 tokenizer和模型
    tokenizer, model = create_llama_model(args.model_name_or_path)

    '''训练数据配置'''
    # 准备数据集
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len
    )
    
    # 选择采样器
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size
    )

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        return perplexity, losses.item()

    '''优化器设置'''
    # 将模型的权重参数分为两组：一组应用权重衰减，另一组不应用权重衰减。
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,                # 输入模型
        args.weight_decay,   # 权重衰减参数
        args.learning_rate    # 学习率
    )
    
    # 根据是否进行模型参数卸载，选择合适的优化器
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    
    # 初始化优化器，设置学习率和动量参数（betas）
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters,  # 之前定义的参数组
        lr=args.learning_rate,          # 学习率
        betas=(0.9, 0.95)               # Adam优化器的动量参数
    )
    
    # 计算每个epoch的更新步骤数
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    
    # 创建学习率调度器，动态调整学习率
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,            # 指定学习率调度器的类型
        optimizer=optimizer,                    # 之前定义的优化器
        num_warmup_steps=args.num_warmup_steps, # 预热步数，用于逐渐增加学习率
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,  # 总训练步数
    )

    # 使用 DeepSpeed 初始化模型、优化器、学习率调度器等
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,  # 要训练的模型
        optimizer=optimizer,  # 优化器
        args=args,  # 传入的参数
        config=ds_config,  # DeepSpeed 配置
        lr_scheduler=lr_scheduler,  # 学习率调度器
        dist_init_required=True  # 需要初始化分布式环境
    )
    
    # 如果启用了梯度检查点，启用模型的梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # 启用梯度检查点

    '''模型训练'''
    # 训练!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    
    # 评估模型的困惑度
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)
    
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, \
    Total Micro Batches {len(train_dataloader)}",
            args.global_rank
        )
        model.train()  # 切换模型到训练模式
        import time
    
        for step, batch in enumerate(train_dataloader):
            start = time.time()  # 记录开始时间
            batch = to_device(batch, device)  # 将数据加载到设备上
            outputs = model(**batch, use_cache=False)  # 执行前向传播
            loss = outputs.loss  # 获取损失
    
            # 打印损失
            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, \
    Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
    
            model.backward(loss)  # 反向传播
            model.step()  # 更新模型参数
            end = time.time()  # 记录结束时间
    
            # 计算吞吐量
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start, args.global_rank)
    
        # 保存模型
        if args.output_dir is not None:
            print_rank_0('saving the final model ...', args.global_rank)
            model = convert_lora_to_linear_layer(model)  # 转换模型层
    
            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args)  # 保存 Hugging Face 格式的模型
    
            if args.zero_stage == 3:
                # 对于 zero stage 3，每个 GPU 只拥有模型的一部分，需要特殊的保存函数
                save_zero_three_model(model, args.global_rank, args.output_dir, zero_stage=args.zero_stage)

if __name__ == "__main__":
    main()
