import torch


def calc_model_size(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def calc_tflops(numel: int, num_tokens: int, iter_time: float) -> float:
    flops = numel * num_tokens * 2.0 * 4.0
    return (flops / 1e12) / (iter_time + 1e-12)
