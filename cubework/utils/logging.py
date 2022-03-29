import logging
import os

import torch.distributed as dist
from rich.logging import RichHandler


def get_logger(file=None):
    logger = logging.getLogger("cubework")

    level = logging.INFO
    if dist.is_initialized() and dist.get_rank() > 0:
        level = logging.ERROR
    stream_handler = RichHandler(level=level)
    logger.addHandler(stream_handler)

    if file is not None:
        path = os.path.dirname(file)
        os.makedirs(path, exist_ok=True)

        name, ext = os.path.splitext(file)
        suffix = ""
        if dist.is_initialized():
            suffix = "_rank_" + str(dist.get_rank())
        file = name + suffix + ext

        file_handler = logging.FileHandler(file, mode="a")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s > %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
