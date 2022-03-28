import torch

from ..parallel_3d import reduce_by_batch_3d, split_batch_3d


def calc_acc_3d(logits, targets):
    targets = split_batch_3d(targets)
    preds = torch.argmax(logits, dim=-1)
    correct = torch.sum(targets == preds)
    correct = reduce_by_batch_3d(correct)
    return correct
