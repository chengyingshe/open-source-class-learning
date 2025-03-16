'''1.2.1 数据并行'''

import argparse
import os
import shutil
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from models import DeepLab
from dataset import Cityscaples

# 参数设置
parser = argparse.ArgumentParser(description='DeepLab')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=3, type=int,
                    metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--local_rank', type=int,
                    help='node rank for distributed training')

args = parser.parse_args()
torch.distributed.init_process_group(backend="nccl")  # 初始化
args.local_rank = int(os.environ['LOCAL_RANK'])  # 进程号
print("Use GPU:{} for training".format(args.local_rank))

# 创建模型
model = DeepLab()

torch.cuda.set_device(args.local_rank)  # 当前显卡
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    output_device=args.local_rank, find_unused_parameters=True)  # 数据并行

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

image_transform  = transforms.Compose([
    transforms.Resize((512, 1024)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

label_transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # 调整标签大小
    transforms.ToTensor()  # 转换为 Tensor
])

train_dataset = Cityscaples(image_transform=image_transform, label_transform=label_transform)
train_sampler = DistributedSampler(train_dataset)  # 分配数据

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)

# 训练
for epoch in range(args.start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")

# 保存模型
if args.local_rank == 0:
    torch.save(model.state_dict(), "deeplab.pth")
