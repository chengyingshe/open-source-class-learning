'''1.2.2 模型并行的张量并行'''

import torch
from torch.distributed._tensor import DeviceMesh, Replicate, Shard, distribute_tensor, distribute_module
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))

mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1], [2, 3]])

# def shard_params(mod_name, mod, mesh):
#     rowwise_placement = [Shard(0), Replicate()]
#     def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)
#     mod._apply(to_dist_tensor)

# sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)

def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0), Replicate()]
    if mod_name == "fc1":
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_fc)

input = torch.randn(16, 8)
sharded_input = distribute_tensor(input, mesh, [Shard(0), Replicate()])
output = sharded_module(sharded_input)
print(output)
