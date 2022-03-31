"""Adapted from
https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
and https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/optimizer/clip_grads.py
"""

from typing import Iterable, Union

import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce
from cubework.global_vars import NUM_PARTITIONS
from torch import Tensor
from torch._six import inf


def clip_grad_norm(
    parameters: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        if pm.TENSOR.is_initialized() and pm.TENSOR.world_size > 1:
            total_norm = all_reduce(total_norm, pm.TENSOR, op=torch.distributed.ReduceOp.MAX)
    else:
        std_norms = list()
        tp_norms = list()
        for p in parameters:
            norm = torch.norm(p.grad.detach(), norm_type) ** norm_type
            norm = norm.to(device)
            num_partitions = getattr(p, NUM_PARTITIONS, 0)
            if num_partitions > 0:
                tp_norms.append(norm * num_partitions)
            else:
                std_norms.append(norm)
        std_norm = (
            torch.sum(torch.stack(std_norms)) if len(std_norms) > 0 else torch.zeros(()).to(torch.float).to(device)
        )
        if len(tp_norms) > 0:
            tp_norm = torch.sum(torch.stack(tp_norms))
            tp_norm = all_reduce(tp_norm, pm.TENSOR) / pm.TENSOR.world_size
        else:
            tp_norm = torch.zeros(()).to(torch.float).to(device)
        total_norm = (std_norm + tp_norm) ** (1.0 / norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm
