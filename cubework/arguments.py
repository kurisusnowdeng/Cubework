import argparse

_ARGS = None


def get_args():
    return _ARGS


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--tensor_parallel", "--tp", type=str)
    parser.add_argument("--tensor_parallel_size", "--tp_size", type=int)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int)

    global _ARGS
    _ARGS = parser.parse_args()
    return _ARGS
