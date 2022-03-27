import torch.nn as nn
from cubework.module import CubeModule, get_tensor_parallel_mode

_parallel_cross_entropy = {}

_vocab_parallel_cross_entropy = {}


class CrossEntropyLoss(CubeModule):
    def __init__(self, reduction: bool = True, *args, **kwargs):
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is not None and env.vocab_parallel:
            loss = _vocab_parallel_cross_entropy[tensor_parallel](reduction=reduction, *args, **kwargs)
        elif tensor_parallel is None or tensor_parallel == '1d':
            reduction = 'mean' if reduction else 'none'
            loss = nn.CrossEntropyLoss(reduction=reduction, *args, **kwargs)
        else:
            loss = _parallel_cross_entropy[tensor_parallel](reduction=reduction, *args, **kwargs)
        super().__init__(loss)
