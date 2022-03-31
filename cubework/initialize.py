import os

import torch.distributed as dist

import cubework.distributed as cube_dist
from cubework.arguments import parse_args
from cubework.global_vars import ALLOWED_MODES, env
from cubework.utils import get_logger, init_logger, set_device, set_seed

_DEFAULT_SEED = 1024


def _get_version():
    version_file = os.path.join(os.path.dirname(__file__), "../version.txt")
    if os.path.isfile(version_file):
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0"

    return version


def initialize_distributed(parser=None):
    args = parse_args(parser)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    addr = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    init_method = f"tcp://{addr}:{port}"
    backend = "nccl" if args.backend is None else args.backend
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)

    init_logger()
    logger = get_logger()
    logger.info(f"Cubework v{_get_version()}")

    set_device(local_rank)

    cube_dist.init_global()

    data_parallel_size = world_size if args.tensor_parallel_size is None else world_size // args.tensor_parallel_size
    cube_dist.init_data_parallel(data_parallel_size)

    seed = args.seed if args.seed is not None else _DEFAULT_SEED
    set_seed(seed)

    env.mode = args.tensor_parallel
    assert env.mode in ALLOWED_MODES
    if args.tensor_parallel is not None:
        cube_dist.init_tensor_parallel(args.tensor_parallel_size, seed)

    if env.mode is None:
        logger.info("Using data parallelism")
    else:
        logger.info(f"Using {env.mode.upper()} tensor parallelism")
