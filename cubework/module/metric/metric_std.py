import torch
import torch.nn as nn


class AccuracySTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        return correct
