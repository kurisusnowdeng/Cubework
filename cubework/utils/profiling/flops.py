import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce

from ..common import get_current_device


def calc_model_size(model: torch.nn.Module):
    numel = sum(p.numel() for p in model.parameters())
    if pm.TENSOR.is_initialized() and pm.TENSOR.world_size > 1:
        numel = torch.tensor(numel).to(get_current_device())
        numel = all_reduce(numel, pm.TENSOR)
        numel = numel.item()
    return numel


def calc_tflops(numel: int, num_tokens: int, iter_time: float, with_backward=True, checkpoint=False) -> float:
    flops = numel * num_tokens * 2
    multiple = 1
    if with_backward:
        multiple += 2
    if checkpoint:
        multiple += 1
    return (flops / 1e12) / (iter_time + 1e-12)
