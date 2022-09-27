from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from itertools import repeat

import torch
from cubework.global_vars import NUM_PARTITIONS, env


def set_tensor_parallel_attribute_by_partition(param, num_partitions):
    setattr(param, NUM_PARTITIONS, num_partitions)


def get_tensor_parallel_mode():
    return env.mode


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable):
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
        self.bucket = OrderedDict()

    def __len__(self):
        return len(self.bucket)

    def push(self, async_op, grad_tensor, param_id):
        self.bucket[param_id] = tuple((async_op, grad_tensor))
        return torch.zeros_like(grad_tensor, dtype=grad_tensor.dtype, device=grad_tensor.device)

    def pop(self, param_id):
        grad = None
        if param_id in self.bucket:
            op, grad = self.bucket.pop(param_id)
            op.wait()
        return grad

    def synchronize(self, params):
        for p in params:
            i = id(p)
            if i in self.bucket:
                op, grad = self.bucket.pop(i)
                op.wait()
                p.grad.add_(grad)


_async_grad_bucket = AsyncGradientBucket()


def push_async_grad(op, grad, param_id):
    return _async_grad_bucket.push(op, grad, param_id)


def pop_async_grad(param_id):
    return _async_grad_bucket.pop(param_id)


def _async_grad_hook(grad, param_id):
    grad.add_(pop_async_grad(param_id))
    return grad


def register_async_grad_hook(param):
    param.register_hook(partial(_async_grad_hook, param_id=id(param)))


def synchronize(params=list()):
    _async_grad_bucket.synchronize(params)
    torch.cuda.default_stream().synchronize()
    if len(_async_grad_bucket) > 0:
        raise RuntimeError(f"{len(_async_grad_bucket)} asynchronous gradient(s) not collected.")
