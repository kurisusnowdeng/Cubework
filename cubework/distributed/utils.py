import math

import torch
import torch.distributed as dist
from cubework.global_vars import (DATA, GLOBAL, PARALLEL_1D, PARALLEL_2D_COL, PARALLEL_2D_ROW, PARALLEL_3D_INPUT,
                                  PARALLEL_3D_OUTPUT, PARALLEL_3D_WEIGHT, TENSOR, env)

_INITIALIZE = {
    '1d': (is_1d_parallel_initialized, init_1d_parallel),
    '2d': (is_2d_parallel_initialized, init_2d_parallel),
    '3d': (is_3d_parallel_initialized, init_3d_parallel),
}

parallel_modes = dict()


class ParallelMode(object):
    def __init__(self, rank, local_rank, world_size, process_group, ranks_in_group, seed=None):
        self._global_rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._group = process_group
        self._ranks_in_group = ranks_in_group

        if seed is not None:
            current_state = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(seed)
            self._rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(current_state)
        else:
            self._rng_state = torch.cuda.get_rng_state()

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def group(self):
        return self._group

    @property
    def ranks_in_group(self):
        return self._ranks_in_group

    def global_rank_by_idx(self, idx):
        return self._ranks_in_group[idx]

    @property
    def rng_state(self):
        return self._rng_state


def is_initialized():
    return GLOBAL in parallel_modes


def init_global():
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parallel_modes[GLOBAL] = ParallelMode(rank, rank, world_size, None, list(range(world_size)))


def is_data_parallel_initialized():
    return DATA in parallel_modes


def init_data_parallel(data_parallel_size):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_data_parallel_group = world_size // data_parallel_size

    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    for i in range(num_data_parallel_group):
        ranks = [i + j * num_data_parallel_group for j in range(data_parallel_size)]
        group = dist.new_group(ranks)

        if rank in ranks:
            local_rank = ranks.index(self.rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks

    parallel_modes[DATA] = ParallelMode(rank, local_rank, group_world_size, process_group, ranks_in_group)


def is_tensor_parallel_initialized():
    is_nd_initialized, _ = _INITIALIZE[env.mode]
    return TENSOR in parallel_modes and is_nd_initialized()


def init_tensor_parallel(tensor_parallel_size, seed):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_tensor_parallel_group = world_size // tensor_parallel_size

    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    for i in range(num_tensor_parallel_group):
        ranks = [i * tensor_parallel_size + j for j in range(tensor_parallel_size)]
        group = dist.new_group(ranks)

        if self.rank in ranks:
            local_rank = ranks.index(self.rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks
    offset = seed + 1024
    tensor_parallel_seed = offset + local_rank

    parallel_modes[TENSOR] = ParallelMode(rank,
                                          local_rank,
                                          group_world_size,
                                          process_group,
                                          ranks_in_group,
                                          seed=tensor_parallel_seed)

    _, init_nd_parallel = _INITIALIZE[env.mode]
    init_nd_parallel(tensor_parallel_size, tensor_parallel_seed)


def is_1d_parallel_initialized():
    return PARALLEL_1D in parallel_modes


def init_1d_parallel(tensor_parallel_size, seed):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_1d_group = world_size // tensor_parallel_size

    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.parallel_input_1d = False

    for i in range(num_1d_group):
        ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
        group = dist.new_group(ranks)

        if self.rank in ranks:
            local_rank = ranks.index(self.rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks

    parallel_modes[PARALLEL_1D] = ParallelMode(rank,
                                               local_rank,
                                               group_world_size,
                                               process_group,
                                               ranks_in_group,
                                               seed=seed)


def is_2d_parallel_initialized():
    return PARALLEL_2D_COL in parallel_modes and PARALLEL_2D_ROW in parallel_modes


def init_2d_parallel(tensor_parallel_size, seed):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_2d_group = world_size // tensor_parallel_size
    summa_dim = int(math.sqrt(self.tensor_parallel_size))
    env.summa_dim = summa_dim

    # column group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    for i in range(num_2d_group):
        for j in range(summa_dim):
            ranks = [i * tensor_parallel_size + j + k * summa_dim for k in range(summa_dim)]
            group = dist.new_group(ranks)

            if rank in ranks:
                local_rank = ranks.index(rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

    parallel_modes[PARALLEL_2D_COL] = ParallelMode(rank,
                                                   local_rank,
                                                   group_world_size,
                                                   process_group,
                                                   ranks_in_group,
                                                   seed=seed)

    # row group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    for i in range(num_2d_group):
        for j in range(summa_dim):
            ranks = [i * tensor_parallel_size + j * summa_dim + k for k in range(summa_dim)]
            group = dist.new_group(ranks)

            if rank in ranks:
                local_rank = ranks.index(rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

    parallel_modes[PARALLEL_2D_ROW] = ParallelMode(rank,
                                                   local_rank,
                                                   group_world_size,
                                                   process_group,
                                                   ranks_in_group,
                                                   seed=seed)


def is_3d_parallel_initialized():
    return PARALLEL_3D_INPUT in parallel_modes \
        and PARALLEL_3D_WEIGHT in parallel_modes \
        and PARALLEL_3D_OUTPUT in parallel_modes


def init_3d_parallel(tensor_parallel_size, seed):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_3d_group = world_size // tensor_parallel_size
    depth = round(math.pow(tensor_parallel_size, 1 / 3))
    env.depth_3d = depth

    # input group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.input_group_3d = PARALLEL_3D_INPUT
    for h in range(num_3d_group):
        for i in range(depth):
            for k in range(depth):
                ranks = [h * depth**3 + i + depth * (j + depth * k) for j in range(depth)]
                group = dist.new_group(ranks)

                if rank in ranks:
                    local_rank = ranks.index(rank)
                    group_world_size = len(ranks)
                    process_group = group
                    ranks_in_group = ranks

    parallel_modes[PARALLEL_3D_INPUT] = ParallelMode(rank,
                                                     local_rank,
                                                     group_world_size,
                                                     process_group,
                                                     ranks_in_group,
                                                     seed=seed)

    # weight group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.weight_group_3d = PARALLEL_3D_WEIGHT
    for h in range(num_3d_group):
        for k in range(depth):
            for j in range(depth):
                ranks = [h * depth**3 + i + depth * (j + depth * k) for i in range(depth)]
                group = dist.new_group(ranks)

                if rank in ranks:
                    local_rank = ranks.index(rank)
                    group_world_size = len(ranks)
                    process_group = group
                    ranks_in_group = ranks

    parallel_modes[PARALLEL_3D_WEIGHT] = ParallelMode(rank,
                                                      local_rank,
                                                      group_world_size,
                                                      process_group,
                                                      ranks_in_group,
                                                      seed=seed)

    # output group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.output_group_3d = PARALLEL_3D_OUTPUT
    for h in range(num_3d_group):
        for i in range(depth):
            for j in range(depth):
                ranks = [h * depth**3 + i + depth * (j + depth * k) for k in range(depth)]
                group = dist.new_group(ranks)

                if rank in ranks:
                    local_rank = ranks.index(rank)
                    group_world_size = len(ranks)
                    process_group = group
                    ranks_in_group = ranks

    parallel_modes[PARALLEL_3D_OUTPUT] = ParallelMode(rank,
                                                      local_rank,
                                                      group_world_size,
                                                      process_group,
                                                      ranks_in_group,
                                                      seed=seed)
