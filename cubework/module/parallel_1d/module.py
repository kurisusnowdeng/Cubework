"""Adapted from Megatron-LM:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/layers.py
"""

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from cubework.distributed import ParallelManager as pm
from cubework.distributed import broadcast
from cubework.global_vars import env
from cubework.utils import get_current_device, seed
from torch import Tensor
from torch.nn.parameter import Parameter

from .. import init
from .._entry_module import CubeModule
from ..module_std import PatchEmbeddingSTD
from ..utils import set_tensor_parallel_attribute_by_partition
from ._utils import (
    gather_forward_split_backward,
    get_parallel_input,
    reduce_grad,
    reduce_input,
    set_parallel_input,
    split_forward_gather_backward,
)


class Linear1D(CubeModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        parallel_input = get_parallel_input()
        if not parallel_input:
            layer = Linear1D_Col(
                in_features,
                out_features,
                bias=bias,
                dtype=dtype,
                gather_output=gather_output,
                skip_bias_add=skip_bias_add,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        else:
            layer = Linear1D_Row(
                in_features,
                out_features,
                bias=bias,
                dtype=dtype,
                parallel_input=parallel_input,
                skip_bias_add=skip_bias_add,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        super().__init__(layer)


class LayerNorm1D(CubeModule):
    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None):
        norm = nn.LayerNorm(normalized_shape, eps=eps).to(dtype).to(get_current_device())
        super().__init__(norm)


class Classifier1D(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        weight: Parameter = None,
        bias: bool = True,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.parallel_input = get_parallel_input()

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = in_features // pm.PARALLEL_1D.world_size

        # Parameters.
        # Initialize weight.
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(torch.empty(self.num_classes, self.input_size_per_partition, **factory_kwargs))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes, **factory_kwargs))
        else:
            self.bias = None
        with seed(pm.PARALLEL_1D):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = False

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            broadcast(self.bias, pm.PARALLEL_1D.rank_by_idx(0), pm.PARALLEL_1D)

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            num_partition = pm.TENSOR.world_size
            set_tensor_parallel_attribute_by_partition(self.weight, num_partition)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            input_ = input_
        else:
            input_ = split_forward_gather_backward(input_, pm.PARALLEL_1D, dim=-1)

        output_parallel = F.linear(input_, self.weight)
        output = reduce_input(output_parallel, pm.PARALLEL_1D)
        if self.bias is not None:
            output = output + self.bias
        return output


class VocabParallelClassifier1D(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        weight: Parameter = None,
        bias: bool = True,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.parallel_input = get_parallel_input()

        # Divide the weight matrix along the last dimension.
        self.num_classes_per_partition = num_classes // pm.PARALLEL_1D.world_size

        # Parameters.
        # Initialize weight.
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(torch.empty(self.num_classes_per_partition, self.in_features, **factory_kwargs))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes_per_partition, **factory_kwargs))
        else:
            self.bias = None
        with seed(pm.PARALLEL_1D):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = True

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _set_tensor_parallel_attributes(self):
        num_partition = pm.PARALLEL_1D.world_size
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, pm.PARALLEL_1D)
        # Matrix multiply.
        output = F.linear(input_parallel, self.weight, self.bias)
        return output


class Linear1D_Col(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        if skip_bias_add and not bias:
            raise ValueError("cannot skip bias addition if bias is None")

        self.out_features_per_partition = out_features // pm.PARALLEL_1D.world_size

        # Parameters.
        # Initialize weight.
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        self.weight = Parameter(torch.empty(self.out_features_per_partition, self.in_features, **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(self.out_features_per_partition, **factory_kwargs))
        else:
            self.bias = None
        with seed(pm.PARALLEL_1D):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(True)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _set_tensor_parallel_attributes(self):
        num_partition = pm.PARALLEL_1D.world_size
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, pm.PARALLEL_1D)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, pm.PARALLEL_1D, dim=-1)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output


class Linear1D_Row(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        parallel_input: bool = True,
        skip_bias_add: bool = False,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_input = parallel_input
        self.skip_bias_add = skip_bias_add

        if skip_bias_add and not bias:
            raise ValueError("cannot skip bias addition if bias is None")

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = in_features // pm.PARALLEL_1D.world_size

        # Parameters.
        # Initialize weight.
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        self.weight = Parameter(torch.empty(self.out_features, self.input_size_per_partition, **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.bias = None
        with seed(pm.PARALLEL_1D):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            broadcast(self.bias, pm.PARALLEL_1D.rank_by_idx(0), pm.PARALLEL_1D)

    def _set_tensor_parallel_attributes(self):
        num_partition = pm.PARALLEL_1D.world_size
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            input_ = input_
        else:
            input_ = split_forward_gather_backward(input_, pm.PARALLEL_1D, dim=-1)

        output_parallel = F.linear(input_, self.weight)
        output = reduce_input(output_parallel, pm.PARALLEL_1D)

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


class Embedding1D(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.normal_(),
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // pm.PARALLEL_1D.world_size

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype)
        )

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, pm.PARALLEL_1D.world_size)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.PARALLEL_1D):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:

        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output = gather_forward_split_backward(output_parallel, pm.PARALLEL_1D, dim=-1)

        return output


class VocabParallelEmbedding1D(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.normal_(),
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        tensor_parallel_size = pm.PARALLEL_1D.world_size
        tensor_parallel_rank = pm.PARALLEL_1D.local_rank
        self.num_embeddings_per_partition = num_embeddings // tensor_parallel_size
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim), device=get_current_device(), dtype=dtype)
        )

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, pm.PARALLEL_1D.world_size)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.PARALLEL_1D):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if (
            self.padding_idx is not None
            and self.padding_idx >= self.vocab_start_index
            and self.padding_idx < self.vocab_end_index
        ):
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output_parallel = F.embedding(
            masked_input, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs
        )

        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_input(output_parallel, pm.PARALLEL_1D)
        return output


class Dropout1D(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.parallel_input = get_parallel_input()
        self.p = p
        self.inplace = inplace

    def forward(self, input_: Tensor) -> Tensor:
        if self.parallel_input:
            with seed(pm.TENSOR):
                output = F.dropout(input_, self.p, self.training, self.inplace)
        else:
            output = F.dropout(input_, self.p, self.training, self.inplace)
        return output


class PatchEmbedding1D(CubeModule):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_size: int,
        dtype: torch.dtype = None,
        flatten: bool = True,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        position_embed_initializer: Callable = init.zeros_(),
    ):
        embed = PatchEmbeddingSTD(
            img_size,
            patch_size,
            in_chans,
            embed_size,
            dtype=dtype,
            flatten=flatten,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            position_embed_initializer=position_embed_initializer,
        )
        super().__init__(embed)
