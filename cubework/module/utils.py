import collections.abc
from itertools import repeat

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
