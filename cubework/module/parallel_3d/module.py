import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce, broadcast
from cubework.global_vars import env
from cubework.utils import get_current_device, seed
from torch import Tensor
from torch.nn import Parameter

from .. import init
from ..utils import set_tensor_parallel_attribute_by_partition, split_tensor, to_2tuple
from ._operation import classifier_3d, layernorm_3d, linear_3d, vocab_parallel_classifier_3d
from ._utils import (all_gather_weight_3d, get_depth_from_env, get_input_parallel_mode,
                     get_input_x_weight_parallel_mode, get_output_parallel_mode, get_output_x_weight_parallel_mode,
                     get_weight_parallel_mode, reduce_scatter_tensor_3d, split_batch_3d, swap_in_out_group)


class LayerNorm3D(nn.Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-12, dtype=None):

        super().__init__()
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.input_x_weight_parallel_mode = get_input_x_weight_parallel_mode()
        self.depth = get_depth_from_env()
        self.normalized_shape = normalized_shape
        self.normalized_shape_per_partition = normalized_shape // self.depth

        self.weight = Parameter(
            torch.ones(self.normalized_shape_per_partition, device=get_current_device(), dtype=dtype))
        self.bias = Parameter(torch.zeros(self.normalized_shape_per_partition, device=get_current_device(),
                                          dtype=dtype))
        self.variance_epsilon = eps
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)
        set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def reset_parameters(self) -> None:
        init.zeros_()(self.bias)
        init.ones_()(self.weight)

    def forward(self, input_: Tensor) -> Tensor:
        return layernorm_3d(
            input_,
            self.weight,
            self.bias,
            self.normalized_shape,
            self.variance_epsilon,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
            self.input_x_weight_parallel_mode,
        )


class Linear3D(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            dtype: torch.dtype = None,
            weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.output_x_weight_parallel_mode = get_output_x_weight_parallel_mode()
        self.depth = get_depth_from_env()
        self.in_features_per_partition = in_features // self.depth
        self.out_features_per_partition = out_features // self.depth**2
        self.bias_features_per_partition = out_features // self.depth

        self.weight = Parameter(
            torch.empty(
                self.in_features_per_partition,
                self.out_features_per_partition,
                device=get_current_device(),
                dtype=dtype,
            ))
        if bias:
            self.bias = Parameter(
                torch.zeros(self.bias_features_per_partition, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        swap_in_out_group()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.in_features, self.out_features

            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, self.output_x_weight_parallel_mode.rank_by_idx(0),
                          self.output_x_weight_parallel_mode)

    def forward(self, input_: Tensor) -> Tensor:
        return linear_3d(
            input_,
            self.weight,
            self.bias,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )


class Classifier3D(nn.Module):

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
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.depth = get_depth_from_env()
        self.in_features_per_partition = in_features // self.depth

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.num_classes, self.in_features_per_partition, device=get_current_device(), dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.zeros(self.num_classes, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.depth)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
                broadcast(self.weight, self.weight_parallel_mode.rank_by_idx(0), self.weight_parallel_mode)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, pm.TENSOR.rank_by_idx(0), pm.TENSOR)

    def forward(self, input_: Tensor) -> Tensor:
        return classifier_3d(
            input_,
            self.weight,
            self.bias,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )


class VocabParallelClassifier3D(nn.Module):

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
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.output_x_weight_parallel_mode = get_output_x_weight_parallel_mode()
        self.depth = get_depth_from_env()
        self.in_features_per_partition = in_features // self.depth
        self.out_features_per_partition = num_classes // self.depth**2
        self.bias_features_per_partition = num_classes // self.depth

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.out_features_per_partition,
                            self.in_features_per_partition,
                            device=get_current_device(),
                            dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = Parameter(
                torch.zeros(self.bias_features_per_partition, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        swap_in_out_group()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self) -> None:
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, self.output_x_weight_parallel_mode.rank_by_idx(0),
                          self.output_x_weight_parallel_mode)

    def forward(self, input_: Tensor) -> Tensor:
        return vocab_parallel_classifier_3d(
            input_,
            self.weight,
            self.bias,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )


class PatchEmbedding3D(nn.Module):

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
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.input_x_weight_parallel_mode = get_input_x_weight_parallel_mode()
        self.patch_size = to_2tuple(patch_size)
        grid_size = to_2tuple(img_size // patch_size)
        num_patches = grid_size[0] * grid_size[1]
        self.embed_size = embed_size
        embed_size_per_partition = embed_size // self.depth
        self.flatten = flatten

        self.weight = nn.Parameter(
            torch.empty((embed_size_per_partition, in_chans, *self.patch_size),
                        device=get_current_device(),
                        dtype=dtype))
        self.bias = nn.Parameter(torch.empty(embed_size_per_partition, device=get_current_device(), dtype=dtype))

        self.cls_token = nn.Parameter(
            torch.zeros((1, 1, embed_size_per_partition), device=get_current_device(), dtype=dtype))
        self.pos_embed = nn.Parameter(
            torch.zeros((1, num_patches + 1, embed_size_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)
        set_tensor_parallel_attribute_by_partition(self.bias, self.depth)
        set_tensor_parallel_attribute_by_partition(self.cls_token, self.depth)
        set_tensor_parallel_attribute_by_partition(self.pos_embed, self.depth)

    def _sync_grad_hook(self, grad) -> Tensor:
        grad = all_reduce(grad.clone(), self.input_x_weight_parallel_mode)
        return grad

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_out = self.embed_size
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            bias_initializer(self.bias, fan_in=fan_in)
            position_embed_initializer(self.pos_embed)

        broadcast(self.weight, self.input_x_weight_parallel_mode.rank_by_idx(0), self.input_x_weight_parallel_mode)
        broadcast(self.bias, self.input_x_weight_parallel_mode.rank_by_idx(0), self.input_x_weight_parallel_mode)
        broadcast(self.pos_embed, self.input_x_weight_parallel_mode.rank_by_idx(0), self.input_x_weight_parallel_mode)

        self.weight.register_hook(self._sync_grad_hook)
        self.bias.register_hook(self._sync_grad_hook)
        self.cls_token.register_hook(self._sync_grad_hook)
        self.pos_embed.register_hook(self._sync_grad_hook)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_3d(input_,
                                input_parallel_mode=self.input_parallel_mode,
                                weight_parallel_mode=self.weight_parallel_mode)
        output = F.conv2d(input_, self.weight, self.bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)    # BCHW -> BNC

        cls_token = self.cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + self.pos_embed

        return output


class Embedding3D(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.input_x_weight_parallel_mode = get_input_x_weight_parallel_mode()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // self.depth
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)

    def _sync_grad_hook(self, grad) -> Tensor:
        grad = all_reduce(grad.clone(), self.input_x_weight_parallel_mode)
        return grad

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()
        broadcast(self.weight, self.input_x_weight_parallel_mode.rank_by_idx(0), self.input_x_weight_parallel_mode)
        self.weight.register_hook(self._sync_grad_hook)

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_3d(input_,
                                input_parallel_mode=self.input_parallel_mode,
                                weight_parallel_mode=self.weight_parallel_mode)
        output = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        return output


class VocabParallelEmbedding3D(torch.nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()
        self.output_parallel_mode = get_output_parallel_mode()
        self.num_embeddings_per_partition = self.num_embeddings // self.depth**2
        self.embed_dim_per_partition = self.embed_dim // self.depth
        vocab_parallel_rank = self.input_parallel_mode.local_rank
        self.vocab_start_index = vocab_parallel_rank * self.num_embeddings_per_partition * self.depth
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition * self.depth

        self.weight = Parameter(
            torch.empty(
                (self.num_embeddings_per_partition, self.embed_dim_per_partition),
                device=get_current_device(),
                dtype=dtype,
            ))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(pm.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if (self.padding_idx is not None and self.padding_idx >= self.vocab_start_index
                and self.padding_idx < self.vocab_end_index):
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_tensor(input_, 0, self.weight_parallel_mode)

        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        weight = all_gather_weight_3d(self.weight, 0, self.weight_parallel_mode)

        output_parallel = F.embedding(masked_input, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output_parallel[input_mask, :] = 0.0
        output = reduce_scatter_tensor_3d(output_parallel, 0, self.input_parallel_mode)

        return output
