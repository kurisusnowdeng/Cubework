import collections.abc
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


class AsyncGradientBucket(object):

    def __init__(self):
        self.bucket = dict()

    def push(self, async_op, grad_tensor, param_id):
        self.bucket[param_id] = tuple((async_op, grad_tensor))
        return None

    def synchronize(self, params):
        for p in params:
            i = id(p)
            if i in self.bucket:
                op, grad = self.bucket.pop(i)
                op.wait()
                if p.grad is None:
                    p.grad = grad
                else:
                    p.grad.add_(grad)
        torch.cuda.default_stream().synchronize()


_async_grad_bucket = AsyncGradientBucket()


def push_async_grad(op, grad, param_id):
    return _async_grad_bucket.push(op, grad, param_id)


def synchronize(params):
    return _async_grad_bucket.synchronize(params)
