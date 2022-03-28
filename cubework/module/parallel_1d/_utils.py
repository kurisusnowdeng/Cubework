# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapted from Megatron-LM:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/mappings.py
"""

import torch
from cubework.distributed import all_gather, all_reduce
from cubework.global_vars import env

from ..utils import split_tensor


def set_parallel_input(input_parallel: bool):
    env.parallel_input_1d = input_parallel


def get_parallel_input():
    return env.parallel_input_1d


def _reduce(input_, parallel_mode):
    output = all_reduce(input_, parallel_mode)

    return output


def _split(input_, parallel_mode, dim=-1):
    output = split_tensor(input_, dim, parallel_mode)

    return output


def _gather(input_, parallel_mode, dim=-1):
    output = all_gather(input_, dim, parallel_mode)

    return output


class _ReduceGrad(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        ctx.mode = parallel_mode
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.mode), None


class _ReduceInput(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return _reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _split(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _gather(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def reduce_grad(input_, parallel_mode):
    return _ReduceGrad.apply(input_, parallel_mode)


def reduce_input(input_, parallel_mode):
    return _ReduceInput.apply(input_, parallel_mode)


def split_forward_gather_backward(input_, parallel_mode, dim):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim)


def gather_forward_split_backward(input_, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim)
