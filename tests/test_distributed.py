import cubework
import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed import broadcast
from functools import partial
import torch.multiprocessing as mp
import os
from cubework.utils import seed, get_current_device, free_port
import sys


def run(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    cubework.initialize_distributed()
    args = cubework.parse_args()
    tp_rank = pm.TENSOR.local_rank if args.tensor_parallel else None
    print(f"Rank {pm.GLOBAL.rank} - dp rank {pm.DATA.local_rank} - tp rank {tp_rank}")
    with seed(pm.TENSOR):
        x = torch.randn((4,)).to(get_current_device())
    print(f"Rank {pm.GLOBAL.rank} before: {x}")
    x = broadcast(x, pm.GLOBAL.rank_by_idx(0), pm.GLOBAL)
    print(f"Rank {pm.GLOBAL.rank} after: {x}")


def test_distributed():
    world_size = 4
    tensor_parallel = "1d"
    tensor_parallel_size = 4
    sys.argv.append(f"--tp={tensor_parallel}")
    sys.argv.append(f"--tp_size={tensor_parallel_size}")

    run_func = partial(run, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_distributed()
