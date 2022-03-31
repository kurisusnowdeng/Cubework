import torch
import torch.nn as nn

from ..parallel_3d import reduce_by_batch_3d, split_batch_3d
from ..parallel_3d._utils import get_input_parallel_mode, get_weight_parallel_mode


class Accuracy3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inupt_parallel_mode = get_input_parallel_mode()
        self.weight_parallel_mode = get_weight_parallel_mode()

    def forward(self, logits, targets):
        targets = split_batch_3d(targets, 0, self.inupt_parallel_mode, self.weight_parallel_mode)
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        correct = reduce_by_batch_3d(correct, self.inupt_parallel_mode, self.weight_parallel_mode)
        return correct
