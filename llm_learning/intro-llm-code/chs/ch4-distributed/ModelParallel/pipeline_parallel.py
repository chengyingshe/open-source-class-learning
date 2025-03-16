'''1.2.2 模型并行的流水线并行'''

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()

   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc = nn.ModuleList([self.fc1, self.fc2])

    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
        return x

def manual_model_split(model) -> PipelineStage:
    if stage_index == 0:
        del model.fc[1]

    elif stage_index == 1:
        del model.fc[0]

    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
        )
    return stage


if __name__ == "__main__":
    init_distributed()
    num_microbatches = 2

    model = ToyModel()

    x = torch.rand(16, 16, dtype=torch.float)
    y = torch.randint(0, 4, (16,), dtype=torch.long)
    example_input_microbatch = x.chunk(num_microbatches)[0]

    # Option 1: Manual model splitting
    stage = manual_model_split(model)

    # Option 2: Tracer model splitting
    # stage = tracer_model_split(model, example_input_microbatch)

    model.to(device)
    x = x.to(device)
    y = y.to(device)

    def loss_fn(outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fn)

    if rank == 0:
        schedule.step(x)
    elif rank == 1:
        losses = []
        output = schedule.step(target=y, losses=losses)
        print(f"losses: {losses}")
    dist.destroy_process_group()

