import cubework
import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import broadcast
from functools import partial
import torch.multiprocessing as mp
import os
from cubework.utils import seed


def run(rank, world_size, port):

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    cubework.initialize_distributed()
    args = cubework.parse_args()
    print(
        f'Rank {pm.GLOBAL.rank} - dp rank {pm.DATA.local_rank} - tp rank {pm.TENSOR.local_rank if args.tensor_parallel else None}'
    )
    with seed(pm.TENSOR):
        x = torch.randn((4, ))
    print(f'Rank {pm.GLOBAL.rank} before: {x}')
    x = broadcast(x, pm.GLOBAL.rank_by_idx(0), pm.GLOBAL)
    print(f'Rank {pm.GLOBAL.rank} after: {x}')


def main():
    world_size = 8
    run_func = partial(run, world_size=world_size, port=23333)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    main()