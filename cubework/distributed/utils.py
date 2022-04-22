import math

import torch
import torch.distributed as dist
from cubework.global_vars import (
    DATA,
    GLOBAL,
    PARALLEL_1D,
    PARALLEL_2D_COL,
    PARALLEL_2D_ROW,
    PARALLEL_3D_INPUT,
    PARALLEL_3D_OUTPUT,
    PARALLEL_3D_WEIGHT,
    PARALLEL_3D_INPUT_X_WEIGHT,
    PARALLEL_3D_OUTPUT_X_WEIGHT,
    TENSOR,
    env,
)


class ParallelMode(object):
    _name = None
    _initialized = None
    _rank = None
    _local_rank = None
    _world_size = None
    _group = None
    _ranks_in_group = None
    _rng_state = None
    _cuda_rng_state = None

    def __init__(self, name):
        self._name = name
        self._initialized = False

    @property
    def name(self):
        return self._name

    @property
    def rank(self):
        return self._rank

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

    def rank_by_idx(self, idx):
        return self._ranks_in_group[idx]

    @property
    def rng_state(self):
        return self._rng_state

    @property
    def cuda_rng_state(self):
        return self._cuda_rng_state

    def is_initialized(self):
        return self._initialized

    def init(self, rank, local_rank, world_size, process_group, ranks_in_group, seed=None):
        self._initialized = True
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._group = process_group
        self._ranks_in_group = ranks_in_group

        if seed is not None:
            cur_state = torch.get_rng_state()
            cur_cuda_state = torch.cuda.get_rng_state()
            torch.manual_seed(seed)
            self._rng_state = torch.get_rng_state()
            self._cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(cur_state)
            torch.cuda.set_rng_state(cur_cuda_state)
        else:
            self._rng_state = torch.get_rng_state()
            self._cuda_rng_state = torch.cuda.get_rng_state()

    def destroy(self):
        dist.destroy_process_group(self._group)


class ParallelManager(object):
    DATA = ParallelMode(DATA)
    GLOBAL = ParallelMode(GLOBAL)
    TENSOR = ParallelMode(TENSOR)
    PARALLEL_1D = ParallelMode(PARALLEL_1D)
    PARALLEL_2D_COL = ParallelMode(PARALLEL_2D_COL)
    PARALLEL_2D_ROW = ParallelMode(PARALLEL_2D_ROW)
    PARALLEL_3D_INPUT = ParallelMode(PARALLEL_3D_INPUT)
    PARALLEL_3D_WEIGHT = ParallelMode(PARALLEL_3D_WEIGHT)
    PARALLEL_3D_OUTPUT = ParallelMode(PARALLEL_3D_OUTPUT)
    PARALLEL_3D_INPUT_X_WEIGHT = ParallelMode(PARALLEL_3D_INPUT_X_WEIGHT)
    PARALLEL_3D_OUTPUT_X_WEIGHT = ParallelMode(PARALLEL_3D_OUTPUT_X_WEIGHT)


def init_global():
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ParallelManager.GLOBAL.init(rank, rank, world_size, None, list(range(world_size)))


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
            local_rank = ranks.index(rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks

    ParallelManager.DATA.init(rank, local_rank, group_world_size, process_group, ranks_in_group)


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
        ranks = [i * tensor_parallel_size + j for j in range(tensor_parallel_size)]
        group = dist.new_group(ranks)

        if rank in ranks:
            local_rank = ranks.index(rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks

    ParallelManager.PARALLEL_1D.init(rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed)


def init_2d_parallel(tensor_parallel_size, seed):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_2d_group = world_size // tensor_parallel_size
    summa_dim = int(math.sqrt(tensor_parallel_size))
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

    ParallelManager.PARALLEL_2D_COL.init(rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed)

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

    ParallelManager.PARALLEL_2D_ROW.init(rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed)


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

    ParallelManager.PARALLEL_3D_INPUT.init(rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed)

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

    ParallelManager.PARALLEL_3D_WEIGHT.init(
        rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed
    )

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

    ParallelManager.PARALLEL_3D_OUTPUT.init(
        rank, local_rank, group_world_size, process_group, ranks_in_group, seed=seed
    )

    # input x weight group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.input_x_weight_group_3d = PARALLEL_3D_INPUT_X_WEIGHT
    for h in range(num_3d_group):
        for k in range(depth):
            ranks = [h * depth**3 + i + depth * (j + depth * k) for j in range(depth) for i in range(depth)]
            group = dist.new_group(ranks)

            if rank in ranks:
                local_rank = ranks.index(rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

    ParallelManager.PARALLEL_3D_INPUT_X_WEIGHT.init(
        rank,
        local_rank,
        group_world_size,
        process_group,
        ranks_in_group,
        seed=seed,
    )

    # output x weight group
    local_rank = None
    ranks_in_group = None
    process_group = None
    group_world_size = None
    env.output_x_weight_group_3d = PARALLEL_3D_OUTPUT_X_WEIGHT
    for h in range(num_3d_group):
        for j in range(depth):
            ranks = [h * depth**3 + i + depth * (j + depth * k) for k in range(depth) for i in range(depth)]
            group = dist.new_group(ranks)

            if rank in ranks:
                local_rank = ranks.index(rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

    ParallelManager.PARALLEL_3D_OUTPUT_X_WEIGHT.init(
        rank,
        local_rank,
        group_world_size,
        process_group,
        ranks_in_group,
        seed=seed,
    )


_TENSOR_PARALLEL_INIT_FUNCS = {
    "1d": init_1d_parallel,
    "2d": init_2d_parallel,
    "3d": init_3d_parallel,
}


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

        if rank in ranks:
            local_rank = ranks.index(rank)
            group_world_size = len(ranks)
            process_group = group
            ranks_in_group = ranks
    offset = seed + 1024
    tensor_parallel_seed = offset + local_rank

    ParallelManager.TENSOR.init(
        rank, local_rank, group_world_size, process_group, ranks_in_group, seed=tensor_parallel_seed
    )

    _TENSOR_PARALLEL_INIT_FUNCS[env.mode](tensor_parallel_size, tensor_parallel_seed)


def destroy_distributed():
    for name, mode in vars(ParallelManager).items():
        if isinstance(mode, ParallelMode) and name != "GLOBAL":
            if mode.is_initialized():
                mode.destroy()
    if dist.is_initialized():
        dist.destroy_process_group()
