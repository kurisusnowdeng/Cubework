from typing import Any, Tuple

import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_gather, all_reduce, reduce_scatter
from cubework.distributed.utils import ParallelMode
from cubework.global_vars import env
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


def get_summa_dim_from_env() -> int:
    return env.summa_dim


def assert_summa_initialization():
    assert (
        pm.PARALLEL_2D_COL.is_initialized() and pm.PARALLEL_2D_ROW.is_initialized()
    ), "Both TWO_DIMENSION_COL and TWO_DIMENSION_ROW must be initialized by the process group initializer"


class _AllGatherTensor2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode

        outputs = all_gather(inputs, dim, parallel_mode)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad = reduce_scatter(output_grad, ctx.dim, ctx.parallel_mode)
        return grad.contiguous(), None, None


def all_gather_tensor_2d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    return _AllGatherTensor2D.apply(tensor, dim, parallel_mode)


def split_batch_2d(input_: Tensor, dim: int = 0) -> Tensor:
    if input_.size(dim) <= 1:
        return input_
    return torch.chunk(input_, pm.PARALLEL_2D_COL.world_size, dim=dim)[pm.PARALLEL_2D_COL.local_rank].contiguous()


class _ReduceTensor2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_2d(input_: Tensor, parallel_mode: ParallelMode) -> Tensor:
    return _ReduceTensor2D.apply(input_, parallel_mode)


class _ReduceScatterTensor2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        return reduce_scatter(input_, dim, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return all_gather(output_grad, ctx.dim, ctx.parallel_mode), None, None


def reduce_scatter_tensor_2d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    return _ReduceScatterTensor2D.apply(tensor, dim, parallel_mode)


class _ReduceByBatch2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_, reduce_mean: bool = False):
        output = all_reduce(input_, pm.PARALLEL_2D_COL)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = pm.PARALLEL_2D_COL.world_size
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None
        else:
            return output_grad, None


def reduce_by_batch_2d(input_, reduce_mean: bool = False) -> Tensor:
    return _ReduceByBatch2D.apply(input_, reduce_mean)
