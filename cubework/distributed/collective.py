import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def all_gather(tensor, dim, parallel_mode, async_op=False):
    depth = parallel_mode.world_size
    if depth == 1:
        out = tensor
        work = None
    else:
        shape = list(tensor.shape)
        shape[0], shape[dim] = shape[dim], shape[0]
        shape[0] *= depth
        out = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        temp = list(torch.chunk(out, depth, dim=0))
        work = dist.all_gather(
            tensor_list=temp, tensor=tensor.transpose(0, dim).contiguous(), group=parallel_mode.group, async_op=async_op
        )
        out = torch.transpose(out, 0, dim)
    if async_op:
        return out, work
    else:
        return out


def reduce_scatter(tensor, dim, parallel_mode, op=ReduceOp.SUM, async_op=False):
    depth = parallel_mode.world_size
    if depth == 1:
        out = tensor
        work = None
    else:
        temp = list(map(lambda x: x.contiguous(), torch.chunk(tensor, depth, dim=dim)))
        out = torch.empty(temp[0].shape, dtype=tensor.dtype, device=tensor.device)
        work = dist.reduce_scatter(output=out, input_list=temp, op=op, group=parallel_mode.group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def all_reduce(tensor, parallel_mode, op=ReduceOp.SUM, async_op=False):
    depth = parallel_mode.world_size
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.all_reduce(out, op=op, group=parallel_mode.group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def broadcast(tensor, src, parallel_mode, async_op=False):
    depth = parallel_mode.world_size
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.broadcast(out, src=src, group=parallel_mode.group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def reduce(tensor, dst, parallel_mode, op=ReduceOp.SUM, async_op=False):
    depth = parallel_mode.world_size
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.reduce(out, dst=dst, op=op, group=parallel_mode.group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out
