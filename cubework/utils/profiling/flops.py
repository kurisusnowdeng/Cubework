import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce
from cubework.global_vars import NUM_PARTITIONS

from ..common import get_current_device


def calc_model_size(model: torch.nn.Module):
    tensor_parallel_size = 0
    if pm.TENSOR.is_initialized():
        tensor_parallel_size = pm.TENSOR.world_size
    numel = 0
    numel_per_device = 0
    for p in model.parameters():
        num_partitions = getattr(p, NUM_PARTITIONS, 0)
        if tensor_parallel_size > 1 and num_partitions > 1:
            numel += p.numel() * num_partitions
        else:
            numel += p.numel()
        numel_per_device += p.numel()

    if tensor_parallel_size > 1:
        numel = torch.tensor(numel).to(get_current_device())
        numel = all_reduce(numel, pm.TENSOR) / tensor_parallel_size
        numel = numel.item()
    return numel, numel_per_device


def calc_tflops(numel: int, num_tokens: int, iter_time: float, with_backward=True, checkpoint=False) -> float:
    flops = numel * num_tokens * 2
    multiple = 1
    if with_backward:
        multiple += 2
    if checkpoint:
        multiple += 1
    return (flops * multiple / (1e12 * pm.GLOBAL.world_size)) / (iter_time + 1e-12)
