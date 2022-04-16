"""Adapted from Optimus:
https://github.com/xuqifan897/Optimus/blob/main/summa/mpu/layers.py
"""

from typing import Any, Optional, Tuple

import torch
from cubework.distributed import all_gather, all_reduce, reduce_scatter, broadcast, reduce
from cubework.distributed.utils import ParallelMode
from cubework.utils import get_current_device
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


class _SUMMA_AB(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        # A: [b / q, s, h / q] -> [(b * s) / q, h / q]
        # B: [h / q, s / q]
        # C: [b / q, s, s / q] -> [(b * s) / q, s / q]

        assert A.shape[-1] == B.shape[-2], "Invalid shapes: A={}, B={} for AB.".format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(summa_dim):
            A_temp = broadcast(A.clone().detach(), row_parallel_mode.rank_by_idx(i), row_parallel_mode)
            B_temp = broadcast(B.clone().detach(), col_parallel_mode.rank_by_idx(i), col_parallel_mode)

            C = torch.addmm(C, A_temp, B_temp)

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = summa_ABT(
                output_grad,
                B,
                ctx.summa_dim,
                ctx.A_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = summa_ATB(
                A,
                output_grad,
                ctx.summa_dim,
                ctx.B_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return A_grad, B_grad, None, None, None, None


def summa_AB(
    A: Tensor,
    B: Tensor,
    summa_dim: int,
    out_shape: Tuple[int, ...],
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _SUMMA_AB.apply(A, B, summa_dim, out_shape, row_parallel_mode, col_parallel_mode)


class _SUMMA_ABT(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], "Invalid shapes: A={}, B={} for ABT.".format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(summa_dim):
            B_temp = broadcast(B.clone(), col_parallel_mode.rank_by_idx(i), col_parallel_mode)

            C_temp = torch.matmul(A, B_temp.transpose(0, 1))
            C_temp = reduce(C_temp, row_parallel_mode.rank_by_idx(i), row_parallel_mode)
            if row_parallel_mode.local_rank == i:
                C.copy_(C_temp)

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = summa_AB(
                output_grad,
                B,
                ctx.summa_dim,
                ctx.A_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = summa_ATB(
                output_grad,
                A,
                ctx.summa_dim,
                ctx.B_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return A_grad, B_grad, None, None, None, None


def summa_ABT(
    A: Tensor,
    B: Tensor,
    summa_dim: int,
    out_shape: Tuple[int, ...],
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _SUMMA_ABT.apply(A, B, summa_dim, out_shape, row_parallel_mode, col_parallel_mode)


class _SUMMA_ATB(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:

        assert A.shape[-2] == B.shape[-2], "Invalid shapes: A={}, B={} for ATB.".format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(summa_dim):
            A_temp = broadcast(A.clone(), row_parallel_mode.rank_by_idx(i), row_parallel_mode)
            C_temp = torch.matmul(A_temp.transpose(0, 1), B)
            C_temp = reduce(C_temp, col_parallel_mode.rank_by_idx(i), col_parallel_mode)
            if col_parallel_mode.local_rank == i:
                C.copy_(C_temp)

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = summa_ABT(
                B,
                output_grad,
                ctx.summa_dim,
                ctx.A_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = summa_AB(
                A,
                output_grad,
                ctx.summa_dim,
                ctx.B_shape,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return A_grad, B_grad, None, None, None, None


def summa_ATB(
    A: Tensor,
    B: Tensor,
    summa_dim: int,
    out_shape: Tuple[int, ...],
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _SUMMA_ATB.apply(A, B, summa_dim, out_shape, row_parallel_mode, col_parallel_mode)


class _SUMMAbias(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        input_: Tensor,
        bias: Tensor,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        skip_bias_add: bool,
    ) -> Tensor:
        bias_temp = all_gather(bias, -1, col_parallel_mode)

        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.bias = skip_bias_add

        if skip_bias_add:
            return bias_temp
        else:
            output = input_ + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        col_parallel_mode = ctx.col_parallel_mode

        if ctx.bias:
            grad = reduce_scatter(output_grad, -1, col_parallel_mode)
            return None, grad, None, None, None
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            grad = reduce_scatter(reduce, -1, col_parallel_mode)
            return output_grad, grad, None, None, None


def add_bias_2d(
    input_: Tensor,
    bias: Tensor,
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
    skip_bias_add: bool,
) -> Tensor:
    return _SUMMAbias.apply(
        input_,
        bias,
        row_parallel_mode,
        col_parallel_mode,
        skip_bias_add,
    )


class _Layernorm_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        input_: Tensor,
        E_x: Tensor,
        Var_x: Tensor,
        hidden_size: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        input_ = input_ - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.normalized_shape = hidden_size
        output = input_ * Var_x
        ctx.save_for_backward(output, Var_x)
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        row_parallel_mode = ctx.row_parallel_mode
        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_sum, group=row_parallel_mode.group)
        output_grad_sum /= ctx.normalized_shape

        output_grad_mul_x_sum = torch.sum(output_grad * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_mul_x_sum, group=row_parallel_mode.group)
        output_grad_mul_x_sum /= ctx.normalized_shape

        input_grad = output_grad.clone()
        input_grad -= x * output_grad_mul_x_sum
        input_grad -= output_grad_sum
        input_grad *= Var_x

        return input_grad, None, None, None, None, None


def layernorm_2d(
    input_: Tensor,
    E_x: Tensor,
    Var_x: Tensor,
    hidden_size: int,
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _Layernorm_2D.apply(input_, E_x, Var_x, hidden_size, row_parallel_mode, col_parallel_mode)


class _Classifier2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        bias: Optional[Tensor],
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        B_temp = all_gather(B, -1, col_parallel_mode)
        if ctx:
            ctx.save_for_backward(A, B_temp)

        C = torch.matmul(A, B_temp.transpose(0, 1))

        C = all_reduce(C, row_parallel_mode)

        ctx.use_bias = bias is not None
        if bias is not None:
            C = C + bias

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = torch.matmul(output_grad, B)
            A_grad = A_grad.reshape(ctx.A_shape)
            B_grad = torch.matmul(output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), A)
            B_grad = reduce_scatter(B_grad, -1, ctx.col_parallel_mode)
            B_grad = B_grad.reshape(ctx.B_shape)
            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(output_grad.ndim - 1)))
                bias_grad = all_reduce(bias_grad, ctx.col_parallel_mode)
            else:
                bias_grad = None

        return A_grad, B_grad, bias_grad, None, None, None, None


def classifier_2d(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor],
    summa_dim: int,
    out_shape: Tuple[int, ...],
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _Classifier2D.apply(
        A,
        B,
        bias,
        summa_dim,
        out_shape,
        row_parallel_mode,
        col_parallel_mode,
    )
