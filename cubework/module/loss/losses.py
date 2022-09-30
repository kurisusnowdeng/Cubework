import torch.nn as nn
from cubework.global_vars import env

from ..utils import get_tensor_parallel_mode
from .loss_1d import VocabParallelCrossEntropyLoss1D
from .loss_2d import CrossEntropyLoss2D, VocabParallelCrossEntropyLoss2D
from .loss_3d import CrossEntropyLoss3D, VocabParallelCrossEntropyLoss3D

_parallel_cross_entropy = {
    "2d": CrossEntropyLoss2D,
    "3d": CrossEntropyLoss3D,
}

_vocab_parallel_cross_entropy = {
    "1d": VocabParallelCrossEntropyLoss1D,
    "2d": VocabParallelCrossEntropyLoss2D,
    "3d": VocabParallelCrossEntropyLoss3D,
}


class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction: bool = True, *args, **kwargs):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is not None and env.vocab_parallel:
            self.loss = _vocab_parallel_cross_entropy[tensor_parallel](reduction=reduction, *args, **kwargs)
        elif tensor_parallel is None or tensor_parallel == "1d":
            reduction = "mean" if reduction else "none"
            self.loss = nn.CrossEntropyLoss(reduction=reduction, *args, **kwargs)
        else:
            self.loss = _parallel_cross_entropy[tensor_parallel](reduction=reduction, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
