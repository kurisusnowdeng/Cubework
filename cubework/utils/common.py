from contextlib import contextmanager

import torch


@contextmanager
def seed(paralle_mode):
    try:
        cur_rng_state = torch.cuda.get_rng_state()
        yield torch.cuda.set_rng_state(paralle_mode.rng_state)
    finally:
        torch.cuda.set_rng_state(cur_rng_state)

def set_device(rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def get_current_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device('cpu')
