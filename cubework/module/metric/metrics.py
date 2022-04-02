from abc import abstractmethod

import torch
import torch.nn as nn
from cubework.distributed import ParallelManager as pm
from cubework.distributed import all_reduce
from cubework.utils import get_current_device

from ..utils import get_tensor_parallel_mode
from .metric_2d import Accuracy2D
from .metric_3d import Accuracy3D
from .metric_std import AccuracySTD

_parallel_accuracy = {
    None: AccuracySTD,
    "1d": AccuracySTD,
    "2d": Accuracy2D,
    "3d": Accuracy3D,
}


class Metric(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def forward(self, logits, targets, loss):
        ...

    @abstractmethod
    def value(self):
        ...


class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy")
        tensor_parallel = get_tensor_parallel_mode()
        self.acc = _parallel_accuracy[tensor_parallel]()
        self.reset()

    def reset(self):
        self.total_correct = torch.zeros(()).to(torch.int).to(get_current_device())
        self.total_samples = torch.zeros(()).to(torch.int).to(get_current_device())

    def forward(self, logits, targets, loss):
        with torch.no_grad():
            batch_size = targets.size(0)
            correct = self.acc(logits, targets)
            self.total_samples += batch_size
            self.total_correct += correct
            return correct / batch_size

    def value(self):
        with torch.no_grad():
            reduced_values = all_reduce(torch.stack([self.total_correct, self.total_samples]), pm.DATA)
            return reduced_values[0] / reduced_values[1]

    def to_str(self):
        return f"{self.value().item()*100:.2f} %"


class Perplexity(Metric):
    def __init__(self):
        super().__init__("Perplexity")
        self.reset()

    def reset(self):
        self.cnt = 0
        self.total_loss = torch.zeros(()).to(torch.float).to(get_current_device())

    def forward(self, logits, targets, loss):
        with torch.no_grad():
            self.cnt += 1
            self.total_loss += loss
            return torch.exp(loss)

    def value(self):
        with torch.no_grad():
            reduced_loss = all_reduce(self.total_loss, pm.DATA) / (self.cnt * pm.DATA.world_size)
            return torch.exp(reduced_loss)

    def to_str(self):
        return f"{self.value().item():.5g}"
