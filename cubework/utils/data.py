import random

import numpy as np
import torch
from cubework.distributed import ParallelManager as pm
from torch.utils.data import DataLoader, DistributedSampler


def get_dataloader(dataset, batch_size, shuffle=False, seed=1024, **kwargs):
    world_size = pm.DATA.world_size
    sampler = DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None

    def seed_worker(_):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size // world_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        worker_init_fn=seed_worker,
        **kwargs
    )
