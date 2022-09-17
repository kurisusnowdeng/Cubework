from typing import Optional, Tuple

import torch
from cubework.distributed import all_gather, all_reduce, broadcast, reduce, reduce_scatter
from cubework.distributed.utils import ParallelMode
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from ..utils import push_async_grad


class _Linear3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        input_dim: int = 0,
        weight_dim: int = -1,
        output_dim: int = 0,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.weight_id = id(weight)

        input_ = all_gather(input_, input_dim, input_parallel_mode)
        weight = all_gather(weight, weight_dim, weight_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, output_dim, output_parallel_mode)

        if bias is not None:
            ctx.bias_id = id(bias)
            output += bias

        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_dim = input_dim
        ctx.weight_dim = weight_dim
        ctx.output_dim = output_dim
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        with torch.no_grad():
            output_grad = all_gather(output_grad, ctx.output_dim, ctx.output_parallel_mode)

            weight_grad = torch.matmul(
                input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
            weight_grad, op = reduce_scatter(weight_grad, ctx.weight_dim, ctx.weight_parallel_mode, async_op=True)
            weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
            else:
                bias_grad = None

            input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
            input_grad = reduce_scatter(input_grad, ctx.input_dim, ctx.input_parallel_mode)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def linear_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    input_dim: int = 0,
    weight_dim: int = -1,
    output_dim: int = 0,
) -> Tensor:
    return _Linear3D.apply(
        input_,
        weight,
        bias,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        input_dim,
        weight_dim,
        output_dim,
    )


class _Classifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.weight_id = id(weight)

        src_rank = input_parallel_mode.ranks_in_group[output_parallel_mode.local_rank]
        weight = broadcast(weight, src_rank, input_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight.transpose(0, 1))
        output = all_reduce(output, output_parallel_mode)

        if bias is not None:
            ctx.bias_id = id(bias)
            output += bias

        ctx.src_rank = src_rank
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        with torch.no_grad():
            weight_grad = torch.matmul(
                output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), input_.reshape(-1, input_.shape[-1]))
            weight_grad = reduce(weight_grad, ctx.src_rank, ctx.input_parallel_mode)
            if ctx.input_parallel_mode.local_rank == ctx.output_parallel_mode.local_rank:
                weight_grad, op = all_reduce(weight_grad, ctx.weight_parallel_mode, async_op=True)
                weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)
            else:
                weight_grad = None

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad = all_reduce(bias_grad, ctx.input_parallel_mode)
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
            else:
                bias_grad = None

            input_grad = torch.matmul(output_grad, weight)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    return _Classifier3D.apply(input_, weight, bias, input_parallel_mode, weight_parallel_mode, output_parallel_mode)


class _VocabParallelClassifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        input_dim: int = 0,
        weight_dim: int = -1,
        output_dim: int = 0,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.weight_id = id(weight)

        input_ = all_gather(input_, input_dim, input_parallel_mode)
        weight = all_gather(weight.transpose(0, 1), weight_dim, weight_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, output_dim, output_parallel_mode)

        if bias is not None:
            ctx.bias_id = id(bias)
            output += bias

        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_dim = input_dim
        ctx.weight_dim = (weight_dim + 1) % 2
        ctx.output_dim = output_dim
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        with torch.no_grad():
            output_grad = all_gather(output_grad, ctx.output_dim, ctx.output_parallel_mode)

            weight_grad = torch.matmul(
                input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
            weight_grad, op = reduce_scatter(weight_grad.transpose(0, 1),
                                             ctx.weight_dim,
                                             ctx.weight_parallel_mode,
                                             async_op=True)
            weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
            else:
                bias_grad = None

            input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
            input_grad = reduce_scatter(input_grad, ctx.input_dim, ctx.input_parallel_mode)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def vocab_parallel_classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    input_dim: int = 0,
    weight_dim: int = -1,
    output_dim: int = 0,
) -> Tensor:
    return _VocabParallelClassifier3D.apply(
        input_,
        weight,
        bias,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        input_dim,
        weight_dim,
        output_dim,
    )


class _Layernorm3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: int,
        eps: float,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        input_x_weight_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.weight_id = id(weight)
        ctx.bias_id = id(bias)

        mean = all_reduce(torch.sum(input_, dim=-1, keepdim=True), output_parallel_mode) / normalized_shape
        mu = input_ - mean
        var = all_reduce(torch.sum(mu**2, dim=-1, keepdim=True), output_parallel_mode) / normalized_shape
        sigma = torch.sqrt(var + eps)

        ctx.save_for_backward(mu, sigma, weight)

        z = mu / sigma
        output = weight * z + bias

        ctx.normalized_shape = normalized_shape
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_x_weight_parallel_mode = input_x_weight_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        mu, sigma, weight = ctx.saved_tensors
        with torch.no_grad():

            bias_grad, weight_grad = output_grad, output_grad * mu / sigma
            bias_grad = torch.sum(bias_grad, dim=tuple(range(len(bias_grad.shape))[:-1]))
            bias_grad, op = all_reduce(bias_grad, ctx.input_x_weight_parallel_mode, async_op=True)
            bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
            weight_grad = torch.sum(weight_grad, dim=tuple(range(len(weight_grad.shape))[:-1]))
            weight_grad, op = all_reduce(weight_grad, ctx.input_x_weight_parallel_mode, async_op=True)
            weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

            dz = output_grad * weight
            dvar = dz * mu * (-0.5) * sigma**(-3)
            dvar = all_reduce(torch.sum(dvar, dim=-1, keepdim=True), ctx.output_parallel_mode)
            dmean = dz * (-1 / sigma) + dvar * -2 * mu / ctx.normalized_shape
            dmean = all_reduce(torch.sum(dmean, dim=-1, keepdim=True), ctx.output_parallel_mode)

            input_grad = dz / sigma + dvar * 2 * mu / ctx.normalized_shape + dmean / ctx.normalized_shape

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def layernorm_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor,
    normalized_shape: int,
    eps: float,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    input_x_weight_parallel_mode: ParallelMode,
) -> Tensor:
    return _Layernorm3D.apply(
        input_,
        weight,
        bias,
        normalized_shape,
        eps,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        input_x_weight_parallel_mode,
    )
