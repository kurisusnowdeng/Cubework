import torch
import torch.nn as nn

from ..utils import get_tensor_parallel_mode
from .metric_2d import calc_acc_2d
from .metric_3d import calc_acc_3d
from .metric_std import calc_acc

_parallel_accuracy = {
    None: calc_acc,
    '1d': calc_acc,
    '2d': calc_acc_2d,
    '3d': calc_acc_3d,
}


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        self.acc = _parallel_accuracy[tensor_parallel]

    def forward(self, *args):
        with torch.no_grad():
            correct = self.acc(*args)
        return correct


class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss):
        with torch.no_grad():
            return torch.exp(loss)
