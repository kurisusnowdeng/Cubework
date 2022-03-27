import torch.nn as nn
from cubework.module import get_tensor_parallel_mode
import torch

from .metric_std import calc_acc

_parallel_accuracy = {
    None: calc_acc,
    '1d': calc_acc,
}


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        self.acc = _parallel_accuracy[tensor_parallel]

    def forward(self, *args):
        return self.acc(*args)


class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, loss):
        return torch.exp(loss)
