import collections.abc
from collections import deque
from itertools import repeat

import torch
from cubework.global_vars import NUM_PARTITIONS, env


def set_tensor_parallel_attribute_by_partition(param, num_partitions):
    setattr(param, NUM_PARTITIONS, num_partitions)


def get_tensor_parallel_mode():
    return env.mode


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def split_tensor(tensor, dim, parallel_mode):
    if tensor.size(dim) <= 1:
        return tensor
    output = torch.chunk(tensor, parallel_mode.world_size, dim=dim)[parallel_mode.local_rank].contiguous()
    return output


async_comm_bucket = deque()


def synchronize():
    while len(async_comm_bucket) > 0:
        op = async_comm_bucket.pop()
        if op is not None:
            op.wait()
    torch.cuda.default_stream().synchronize()
