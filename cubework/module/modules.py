import math
from typing import Callable

from cubework.global_vars import VOCAB_PARALLEL
from torch import dtype, nn

from . import init as init
from ._entry_module import CubeModule
from .module_std import ClassifierSTD, PatchEmbeddingSTD
from .utils import get_tensor_parallel_mode

_parallel_classifier = {
    None: ClassifierSTD,
}

_vocab_parallel_classifier = {}

_parallel_patchembedding = {
    None: PatchEmbeddingSTD,
}


class Classifier(CubeModule):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 vocab_parallel_limit=128):
        tensor_parallel = get_tensor_parallel_mode()
        vocab_parallel = tensor_parallel is not None and num_classes > vocab_parallel_limit
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


class PatchEmbedding(CubeModule):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 dtype: dtype = None,
                 flatten: bool = True,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_()):
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