"""Adapted from Optimus:
https://github.com/xuqifan897/Optimus/blob/main/summa/mpu/layers.py
"""
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from cubework.distributed import ParallelManager as pm
from cubework.distributed import broadcast
from cubework.distributed.collective import all_reduce
from cubework.global_vars import env
from cubework.utils import get_current_device, seed
from torch import Tensor
from torch.nn import Parameter

from .. import init
from ..utils import set_tensor_parallel_attribute_by_partition, to_2tuple
from ._operation import summa_AB, summa_ABT, add_bias_2d, classifier_2d, layernorm_2d
from ._utils import (
    all_gather_tensor_2d,
    assert_summa_initialization,
    get_summa_dim_from_env,
    reduce_scatter_tensor_2d,
    split_batch_2d,
)


class Linear2D(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        skip_bias_add: bool = False,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        # parallel settings
        assert_summa_initialization()
        self.row_rank = pm.PARALLEL_2D_COL.local_rank
        self.col_rank = pm.PARALLEL_2D_ROW.local_rank
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.input_size_per_partition = self.in_features // self.summa_dim
        self.hidden_size_per_partition = self.out_features // self.summa_dim

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        self.weight = Parameter(
            torch.empty(self.input_size_per_partition, self.hidden_size_per_partition, **factory_kwargs)
        )

        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(self.out_features // self.summa_dim**2, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # initialize parameters
        with seed(pm.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition,)

        output = summa_AB(
            x,
            self.weight,
            self.summa_dim,
            out_shape,
            pm.PARALLEL_2D_ROW,
            pm.PARALLEL_2D_COL,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2d(
                    None,
                    self.bias,
                    pm.PARALLEL_2D_ROW,
                    pm.PARALLEL_2D_COL,
                    True,
                )
                return output, bias
            else:
                output = add_bias_2d(
                    output,
                    self.bias,
                    pm.PARALLEL_2D_ROW,
                    pm.PARALLEL_2D_COL,
                    False,
                )
                return output
        else:
            return output


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-05, dtype=None):
        super().__init__()

        # layer norm config
        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        # parallel setting
        assert_summa_initialization()
        self.row_rank = pm.PARALLEL_2D_COL.local_rank
        self.col_rank = pm.PARALLEL_2D_ROW.local_rank
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.partitioned_partition = normalized_shape // self.summa_dim**2

        # create parameters
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}

        self.weight = Parameter(torch.ones(self.partitioned_partition, **factory_kwargs))
        self.bias = Parameter(torch.zeros(self.partitioned_partition, **factory_kwargs))

        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            E_x = torch.sum(x, dim=-1, keepdim=True)  # [b/q, s, 1]
            E_x = all_reduce(E_x, pm.PARALLEL_2D_ROW) / self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(x * x, dim=-1, keepdim=True)  # [b/q, s, 1]
            Var_x = all_reduce(Var_x, pm.PARALLEL_2D_ROW) / self.normalized_shape

            Var_x = Var_x - E_x * E_x  # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = layernorm_2d(x, E_x, Var_x, self.normalized_shape, pm.PARALLEL_2D_ROW, pm.PARALLEL_2D_COL)
        bias = add_bias_2d(
            None,
            self.bias,
            pm.PARALLEL_2D_ROW,
            pm.PARALLEL_2D_COL,
            True,
        )
        scale = add_bias_2d(
            None,
            self.weight,
            pm.PARALLEL_2D_ROW,
            pm.PARALLEL_2D_COL,
            True,
        )
        output = torch.addcmul(bias, scale, output)
        return output


class PatchEmbedding2D(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_size: int,
        flatten: bool = True,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        position_embed_initializer: Callable = init.zeros_(),
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_size = embed_size
        self.embed_size_per_partition = embed_size // (self.summa_dim**2)

        self.weight = Parameter(
            torch.empty(
                (self.embed_size_per_partition, in_chans, *self.patch_size), device=get_current_device(), dtype=dtype
            )
        )
        self.bias = Parameter(torch.empty(self.embed_size_per_partition, device=get_current_device(), dtype=dtype))

        self.cls_token = Parameter(
            torch.zeros((1, 1, self.embed_size_per_partition), device=get_current_device(), dtype=dtype)
        )
        self.pos_embed = Parameter(
            torch.zeros(
                (1, self.num_patches + 1, self.embed_size_per_partition), device=get_current_device(), dtype=dtype
            )
        )

        self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.cls_token, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.pos_embed, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer):
        with seed(pm.TENSOR):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_out = self.embed_size
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            bias_initializer(self.bias, fan_in=fan_in)
            position_embed_initializer(self.pos_embed)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_2d(input_)

        B, C, H, W = input_.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        weight = all_gather_tensor_2d(self.weight, 0, pm.PARALLEL_2D_COL)
        bias = all_gather_tensor_2d(self.bias, 0, pm.PARALLEL_2D_COL)

        output = F.conv2d(input_, weight, bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)  # BCHW -> BNC

        cls_token = all_gather_tensor_2d(self.cls_token, -1, pm.PARALLEL_2D_COL)
        pos_embed = all_gather_tensor_2d(self.pos_embed, -1, pm.PARALLEL_2D_COL)
        cls_token = cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + pos_embed

        return output


class Embedding2D(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.normal_(),
        *args,
        **kwargs,
    ):
        super().__init__()

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // self.summa_dim**2

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype)
        )

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_2d(input_)

        weight = all_gather_tensor_2d(self.weight, -1, pm.PARALLEL_2D_COL)
        output = F.embedding(input_, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        return output


class VocabParallelEmbedding2D(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        weight_initializer: Callable = init.normal_(),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.num_embeddings_per_partition = self.num_embeddings // self.summa_dim
        self.embed_dim_per_partition = self.embed_dim // self.summa_dim
        tensor_parallel_rank = pm.PARALLEL_2D_COL.local_rank
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty(
                (self.num_embeddings_per_partition, self.embed_dim_per_partition),
                device=get_current_device(),
                dtype=dtype,
            )
        )

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.TENSOR):
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
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output_parallel = F.embedding(
            masked_input, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs
        )

        output_parallel[input_mask, :] = 0.0
        output = reduce_scatter_tensor_2d(output_parallel, 0, pm.PARALLEL_2D_COL)
        return output


class Classifier2D(nn.Module):
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
        assert_summa_initialization()
        self.row_rank = pm.PARALLEL_2D_COL.local_rank
        self.col_rank = pm.PARALLEL_2D_ROW.local_rank
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.input_size_per_partition = self.in_features // self.summa_dim**2

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.num_classes, self.input_size_per_partition, device=get_current_device(), dtype=dtype)
            )
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.zeros(self.num_classes, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes
            col_src_rank = pm.PARALLEL_2D_COL.rank_by_idx(0)
            row_src_rank = pm.PARALLEL_2D_ROW.rank_by_idx(0)

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, col_src_rank, pm.PARALLEL_2D_COL)
                broadcast(self.bias, row_src_rank, pm.PARALLEL_2D_ROW)

    def forward(self, input_: Tensor) -> Tensor:
        out_shape = input_.shape[:-1] + (self.num_classes,)

        return classifier_2d(
            input_,
            self.weight,
            self.bias,
            self.summa_dim,
            out_shape,
            pm.PARALLEL_2D_ROW,
            pm.PARALLEL_2D_COL,
        )


class VocabParallelClassifier2D(nn.Module):
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

        # parallel setting
        assert_summa_initialization()
        self.row_rank = pm.PARALLEL_2D_COL.local_rank
        self.col_rank = pm.PARALLEL_2D_ROW.local_rank
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.input_size_per_partition = in_features // self.summa_dim
        self.output_size_per_partition = num_classes // self.summa_dim

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {"device": get_current_device(), "dtype": dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.output_size_per_partition, self.input_size_per_partition, **factory_kwargs)
            )
            self.has_weight = True
        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes // self.summa_dim**2, **factory_kwargs))
        else:
            self.bias = None

        # initialize parameters
        with seed(pm.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = x.shape[:-1] + (self.output_size_per_partition,)

        output = summa_ABT(
            x,
            self.weight,
            self.summa_dim,
            out_shape,
            pm.PARALLEL_2D_ROW,
            pm.PARALLEL_2D_COL,
        )

        if self.bias is not None:
            output = add_bias_2d(
                output,
                self.bias,
                pm.PARALLEL_2D_ROW,
                pm.PARALLEL_2D_COL,
                False,
            )
        return output
