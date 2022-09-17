import logging
import os

import torch.distributed as dist
from rich.logging import RichHandler


def _add_file_handler(file, logger):
    path = os.path.dirname(file)
    os.makedirs(path, exist_ok=True)

    name, ext = os.path.splitext(file)
    suffix = ""
    if dist.is_initialized():
        suffix = "_rank_" + str(dist.get_rank())
    file = name + suffix + ext

    file_handler = logging.FileHandler(file, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s > %(message)s", datefmt="[%Y/%m/%d %H:%M:%S.%f]")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


_default_logger = None


def init_logger():
    global _default_logger
    _default_logger = logging.getLogger("cubework")

    level = logging.INFO
    if dist.is_initialized() and dist.get_rank() > 0:
        level = logging.ERROR

    _default_logger.setLevel(level)

    handler = RichHandler()
    formatter = logging.Formatter("%(message)s", datefmt="[%Y/%m/%d %H:%M:%S.%f]")
    handler.setFormatter(formatter)
    _default_logger.addHandler(handler)


def write_logger_to_file(file, logger=None):
    if logger is None:
        logger = _default_logger
    _add_file_handler(file, _default_logger)


def get_logger():
    if _default_logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.NOTSET)
        return logger
    else:
        return _default_logger
