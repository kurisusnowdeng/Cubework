import math
from typing import Callable

from cubework.distributed import ParallelManager as pm
from cubework.global_vars import VOCAB_PARALLEL
from cubework.utils import get_current_device, seed
from torch import dtype, nn

from . import init as init
from ._entry_module import CubeModule
from .module_std import ClassifierSTD, DropPath, PatchEmbeddingSTD
from .parallel_1d import (Classifier1D, Dropout1D, Embedding1D, LayerNorm1D, Linear1D, PatchEmbedding1D,
                          VocabParallelClassifier1D, VocabParallelEmbedding1D)
from .parallel_2d import (Classifier2D, Embedding2D, LayerNorm2D, Linear2D, PatchEmbedding2D, VocabParallelClassifier2D,
                          VocabParallelEmbedding2D, split_batch_2d)
from .parallel_3d import (Classifier3D, Embedding3D, LayerNorm3D, Linear3D, PatchEmbedding3D, VocabParallelClassifier3D,
                          VocabParallelEmbedding3D, split_batch_3d)
from .utils import get_tensor_parallel_mode

_parallel_layernorm = {
    "1d": LayerNorm1D,
    "2d": LayerNorm2D,
    "3d": LayerNorm3D,
}

_parallel_linear = {
    "1d": Linear1D,
    "2d": Linear2D,
    "3d": Linear3D,
}

_parallel_classifier = {
    None: ClassifierSTD,
    "1d": Classifier1D,
    "2d": Classifier2D,
    "3d": Classifier3D,
}

_vocab_parallel_classifier = {
    "1d": VocabParallelClassifier1D,
    "2d": VocabParallelClassifier2D,
    "3d": VocabParallelClassifier3D,
}
_parallel_embedding = {
    "1d": Embedding1D,
    "2d": Embedding2D,
    "3d": Embedding3D,
}

_vocab_parallel_embedding = {
    "1d": VocabParallelEmbedding1D,
    "2d": VocabParallelEmbedding2D,
    "3d": VocabParallelEmbedding3D,
}

_parallel_patchembedding = {
    None: PatchEmbeddingSTD,
    "1d": PatchEmbedding1D,
    "2d": PatchEmbedding2D,
    "3d": PatchEmbedding3D,
}

_parallel_split_batch = {
    "2d": split_batch_2d,
    "3d": split_batch_3d,
}


def partition_batch(input_):
    tensor_parallel_mode = get_tensor_parallel_mode()
    if tensor_parallel_mode in _parallel_split_batch:
        if isinstance(input_, dict):
            return {k: _parallel_split_batch[tensor_parallel_mode](v) for k, v in input_.items()}
        elif isinstance(input_, (tuple, list)):
            return type(input_)(_parallel_split_batch[tensor_parallel_mode](x) for x in input_)
        else:
            return _parallel_split_batch[tensor_parallel_mode](input_)
    else:
        return input_


class LayerNorm(CubeModule):

    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            norm = nn.LayerNorm(normalized_shape, eps=eps).to(dtype).to(get_current_device())
        else:
            norm = _parallel_layernorm[tensor_parallel](normalized_shape, eps=eps, dtype=dtype)
        super().__init__(norm)


class Linear(CubeModule):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 **kwargs) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            layer = nn.Linear(in_features, out_features, bias=bias).to(dtype).to(get_current_device())
            weight_initializer(layer.weight, fan_in=in_features, fan_out=out_features)
            if layer.bias is not None:
                bias_initializer(layer.bias, fan_in=in_features)
        else:
            layer = _parallel_linear[tensor_parallel](
                in_features,
                out_features,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                **kwargs,
            )
        super().__init__(layer)


class Classifier(CubeModule):

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 vocab_parallel: bool = False,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        tensor_parallel = get_tensor_parallel_mode()
        vocab_parallel = getattr(weight, VOCAB_PARALLEL, vocab_parallel)
        if vocab_parallel:
            layer = _vocab_parallel_classifier[tensor_parallel](
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        else:
            layer = _parallel_classifier[tensor_parallel](
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        super().__init__(layer)


class Embedding(CubeModule):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 vocab_parallel: bool = False,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, *args,
                                 **kwargs).to(dtype).to(get_current_device())
            weight_initializer(embed.weight, fan_in=num_embeddings, fan_out=embedding_dim)
        elif vocab_parallel:
            embed = _vocab_parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )
        else:
            embed = _parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )
        super().__init__(embed)


class PatchEmbedding(CubeModule):

    def __init__(
            self,
            img_size: int,
            patch_size: int,
            in_chans: int,
            embed_size: int,
            dtype: dtype = None,
            flatten: bool = True,
            weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
            position_embed_initializer: Callable = init.zeros_(),
    ):
        tensor_parallel = get_tensor_parallel_mode()
        embed = _parallel_patchembedding[tensor_parallel](
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


class Dropout(CubeModule):

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel == "1d":
            drop = Dropout1D(p, inplace)
        else:
            drop = nn.Dropout(p, inplace)
        super().__init__(drop, tensor_parallel=tensor_parallel)

    def forward(self, *args):
        if self.tensor_parallel in [None, "1d"]:
            return self._forward_func(*args)
        else:
            with seed(pm.TENSOR):
                return self._forward_func(*args)
