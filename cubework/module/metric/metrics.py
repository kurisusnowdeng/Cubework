import torch
import torch.nn as nn

from ..utils import get_tensor_parallel_mode
from .metric_2d import calc_acc_2d
from .metric_3d import calc_acc_3d
from .metric_std import calc_acc
from cubework.utils import get_current_device
from cubework.distributed import all_reduce
from cubework.distributed import ParallelManager as pm

_parallel_accuracy = {
    None: calc_acc,
    "1d": calc_acc,
    "2d": calc_acc_2d,
    "3d": calc_acc_3d,
}


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        self.acc = _parallel_accuracy[tensor_parallel]

    def forward(self, logits, targets, loss):
        with torch.no_grad():
            batch_size = torch.LongTensor(targets.size(0)).to(get_current_device())
            correct = self.acc(logits, targets)
            reduced_values = all_reduce(torch.stack([correct, batch_size]), pm.DATA)
            return reduced_values[0] / reduced_values[1]


class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, loss):
        with torch.no_grad():
            reduced_loss = all_reduce(loss, pm.DATA) / pm.DATA.world_size
            return torch.exp(reduced_loss)
