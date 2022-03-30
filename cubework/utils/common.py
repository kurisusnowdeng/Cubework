import random
import socket
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def seed(paralle_mode):
    try:
        cur_rng_state = torch.get_rng_state()
        cur_cuda_rng_state = torch.cuda.get_rng_state()
        torch.set_rng_state(paralle_mode.rng_state)
        yield torch.cuda.set_rng_state(paralle_mode.cuda_rng_state)
    finally:
        torch.set_rng_state(cur_rng_state)
        torch.cuda.set_rng_state(cur_cuda_rng_state)


def set_device(rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_current_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def free_port():
    while True:
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = random.randint(20000, 65000)
            sock.bind(("localhost", port))
            sock.close()
            return port
        except Exception:
            continue
