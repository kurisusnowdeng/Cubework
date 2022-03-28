import torch.nn as nn


class CubeModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        # copy values
        self.__dict__ = module.__dict__.copy()
        # copy methods
        for name, attr in module.__class__.__dict__.items():
            if name not in ["__init__", "forward"] and callable(attr):
                setattr(self, name, getattr(module, name))
        self._forward_func = module.forward

    def forward(self, *args):
        return self._forward_func(*args)
