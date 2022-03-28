import torch

from ..parallel_2d import reduce_by_batch_2d, split_batch_2d


def calc_acc_2d(logits, targets):
    targets = split_batch_2d(targets)
    preds = torch.argmax(logits, dim=-1)
    correct = torch.sum(targets == preds)
    correct = reduce_by_batch_2d(correct)
    return correct
