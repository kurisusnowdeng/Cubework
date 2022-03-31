import torch
import torch.nn as nn

from ..parallel_2d import reduce_by_batch_2d, split_batch_2d


class Accuracy2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        targets = split_batch_2d(targets)
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        correct = reduce_by_batch_2d(correct)
        return correct
