'''1.3.3 去中心化架构'''

import os
from typing import Callable

import torch
import torch.distributed as dist

def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    """初始化分布式环境"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


import torch.multiprocessing as mp

def do_all_reduce(rank: int, size: int):
    # 创建包含所有处理器的群组
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # 可以是dist.ReduceOp.PRODUCT，dist.ReduceOp.MAX，dist.ReduceOp.MIN
    # 将输出所有秩为4的结果
    print(f"[{rank}] data = {tensor[0]}")

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_all_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

