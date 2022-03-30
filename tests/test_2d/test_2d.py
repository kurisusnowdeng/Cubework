import os
import sys
from functools import partial

import cubework
import torch.multiprocessing as mp
from cubework.utils import free_port

from check_2d_modules import (
    check_classifier_given_embed_weight,
    check_classifier_no_given_weight,
    check_embed,
    check_layernorm,
    check_linear,
    check_loss,
    check_patch_embed,
    check_vocab_parallel_classifier_given_embed_weight,
    check_vocab_parallel_classifier_no_given_weight,
    check_vocab_parallel_embed,
    check_vocab_parallel_loss,
)


def run(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    cubework.initialize_distributed()
    check_layernorm()
    check_linear()
    check_embed()
    check_classifier_no_given_weight()
    check_classifier_given_embed_weight()
    check_vocab_parallel_classifier_no_given_weight()
    check_vocab_parallel_classifier_given_embed_weight()
    check_patch_embed(),
    check_vocab_parallel_embed()
    check_loss(),
    check_vocab_parallel_loss()


def test_2d():
    world_size = 4
    tensor_parallel = "2d"
    tensor_parallel_size = 4
    sys.argv.append(f"--tp={tensor_parallel}")
    sys.argv.append(f"--tp_size={tensor_parallel_size}")

    run_func = partial(run, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_2d()
