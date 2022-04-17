import os

import cubework
import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce
from cubework.utils import get_current_device, get_logger
from tqdm import trange

cubework.initialize_distributed()

logger = get_logger()

dim = int(os.environ["DIM"])
num_steps = 20

total_vol = 0
total_time = 0.0

progress = trange(num_steps) if pm.GLOBAL.rank == 0 else range(num_steps)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in progress:
    x = torch.randn(dim, dim * 4).to(get_current_device())
    start.record()
    x = all_reduce(x, pm.GLOBAL)
    end.record()
    start.synchronize()
    end.synchronize()
    total_vol += x.element_size() * x.numel() * 2 * (pm.GLOBAL.world_size - 1) / pm.GLOBAL.world_size
    used_time = start.elapsed_time(end)
    logger.info(f"Step {i}: used time = {used_time/ 1e3:.3f} s | all completed = {end.query()}")
    total_time += used_time

logger.info(
    f"Vol = {total_vol / 1024**3 :.3f} GB | time = {total_time/ 1e3:.3f} s | bandwidth = {total_vol * 1e3 / (total_time * 1024 ** 3):.3f} GB/s"
)
