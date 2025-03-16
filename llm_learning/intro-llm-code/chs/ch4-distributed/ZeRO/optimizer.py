'''1.2.4 计算设备内存优化'''

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB")

def example(rank, world_size, use_zero):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # 创建默认进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建本地模型
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
    print_peak_memory("Max memory allocated after creating local model", rank)

    # 构建 DDP 模型
    ddp_model = DDP(model, device_ids=[rank])
    print_peak_memory("Max memory allocated after creating DDP", rank)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(  # 这里使用了 ZeroRedundancyOptimizer
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam,  # 包装了 Adam
            lr=0.01
        )
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

    # 前向传播
    outputs = ddp_model(torch.randn(20, 2000).to(rank))
    labels = torch.randn(20, 2000).to(rank)
    # 反向传播
    loss_fn(outputs, labels).backward()

    # 更新参数
    print_peak_memory("Max memory allocated before optimizer step()", rank)
    optimizer.step()
    print_peak_memory("Max memory allocated after optimizer step()", rank)

    print(f"params sum is: {sum(model.parameters()).sum()}")

def main():
    world_size = 2
    print("=== Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
        args=(world_size, True),
        nprocs=world_size,
        join=True)

    print("=== Not Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
        args=(world_size, False),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    main()
