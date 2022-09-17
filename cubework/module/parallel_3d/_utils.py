from typing import Tuple

import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_gather, all_reduce, broadcast, reduce, reduce_scatter
from cubework.distributed.utils import ParallelMode
from cubework.global_vars import env
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from ..utils import push_async_grad


def get_depth_from_env() -> int:
    return env.depth_3d


def get_input_parallel_mode() -> ParallelMode:
    return getattr(pm, env.input_group_3d)


def get_weight_parallel_mode() -> ParallelMode:
    return getattr(pm, env.weight_group_3d)


def get_output_parallel_mode() -> ParallelMode:
    return getattr(pm, env.output_group_3d)


def get_input_x_weight_parallel_mode() -> ParallelMode:
    return getattr(pm, env.input_x_weight_group_3d)


def get_output_x_weight_parallel_mode() -> ParallelMode:
    return getattr(pm, env.output_x_weight_group_3d)


def swap_in_out_group():
    env.input_group_3d, env.output_group_3d = env.output_group_3d, env.input_group_3d
    env.input_x_weight_group_3d, env.output_x_weight_group_3d = (
        env.output_x_weight_group_3d,
        env.input_x_weight_group_3d,
    )


def split_batch_3d(
    input_: Tensor,
    dim: int = 0,
    input_parallel_mode: ParallelMode = pm.PARALLEL_3D_INPUT,
    weight_parallel_mode: ParallelMode = pm.PARALLEL_3D_WEIGHT,
) -> Tensor:
    if input_.size(dim) <= 1:
        return input_
    weight_parallel_mode = get_weight_parallel_mode()
    input_parallel_mode = get_input_parallel_mode()
    output = torch.chunk(input_, weight_parallel_mode.world_size, dim=dim)[weight_parallel_mode.local_rank].contiguous()
    output = torch.chunk(output, input_parallel_mode.world_size, dim=dim)[input_parallel_mode.local_rank].contiguous()
    return output


class _ReduceTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_3d(tensor: Tensor, parallel_mode: ParallelMode) -> Tensor:
    return _ReduceTensor3D.apply(tensor, parallel_mode)


class _AllGatherWeight3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        output = all_gather(weight, dim, parallel_mode)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        grad = reduce_scatter(output_grad, ctx.dim, ctx.parallel_mode)
        return grad, None, None


def all_gather_weight_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    return _AllGatherWeight3D.apply(tensor, dim, parallel_mode)


class _ReduceScatterTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        return reduce_scatter(input_, dim, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        input_grad = all_gather(output_grad, ctx.dim, ctx.parallel_mode)
        return input_grad, None, None


def reduce_scatter_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    return _ReduceScatterTensor3D.apply(tensor, dim, parallel_mode)


class _ReduceByBatch3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        input_: Tensor,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        reduce_mean: bool = False,
    ) -> Tensor:
        output = all_reduce(input_, input_parallel_mode)
        output = all_reduce(output, weight_parallel_mode)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = input_parallel_mode.world_size * weight_parallel_mode.world_size
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None, None
        else:
            return output_grad, None, None, None


def reduce_by_batch_3d(tensor: Tensor,
                       input_parallel_mode: ParallelMode,
                       weight_parallel_mode: ParallelMode,
                       reduce_mean: bool = False) -> Tensor:
    return _ReduceByBatch3D.apply(tensor, input_parallel_mode, weight_parallel_mode, reduce_mean)


class _BroadcastWeight3D_FromDiagonal(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        src_rank = input_parallel_mode.ranks_in_group[output_parallel_mode.local_rank]
        output = broadcast(input_, src_rank, input_parallel_mode)
        ctx.src_rank = src_rank
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_grad = reduce(output_grad, ctx.src_rank, ctx.input_parallel_mode)
        if ctx.input_parallel_mode.local_rank == ctx.output_parallel_mode.local_rank:
            input_grad = all_reduce(input_grad, ctx.weight_parallel_mode)
        else:
            input_grad = None
        return input_grad, None, None, None


def broadcast_weight_3d_from_diagonal(
    tensor: Tensor,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    return _BroadcastWeight3D_FromDiagonal.apply(tensor, input_parallel_mode, weight_parallel_mode,
                                                 output_parallel_mode)
