"""Adapted from PatrickStar
https://github.com/Tencent/PatrickStar/blob/master/patrickstar/core/memtracer/memtracer.py.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from ..common import get_current_device


class MemoryTracker(object):
    def __init__(self, file=None, interval=1e-3):
        self.device = get_current_device()
        self.executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.cuda.set_device(self.device))
        self.interval = interval
        self.file = None
        if file is not None:
            name, ext = os.path.splitext(file)
            suffix = "_memory"
            if dist.is_initialized():
                suffix += "_rank_" + str(dist.get_rank())
            self.file = name + suffix + ext

        self.reset()

    def start(self):
        self.keep_measuring = True
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.time()

        self.monitor_thread = self.executor.submit(self._measure_usage)

    def stop(self):
        if self.keep_measuring:
            self.keep_measuring = False
            records = self.monitor_thread.result()
            self.accumulated_time += time.time() - self.start_time
            _, gpu_usage = zip(*records)

            self.monitor_thread = None
            self.start_time = None

            if self.file is not None:
                rank = dist.get_rank() if dist.is_initialized else 0
                with open(self.file, "a") as f:
                    f.writelines(
                        list(map(lambda record: f"[ {record[0]} ] rank {rank} : {record[1]:.3f} MB" + "\n", records))
                    )

            return max(gpu_usage)

    def reset(self):
        self.keep_measuring = False
        self.accumulated_time = 0
        self.start_time = None
        self.monitor_thread = None

    def _measure_usage(self):
        records = list()
        while self.keep_measuring:
            gpu_usage = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)  # MB
            record_time = time.time() - self.start_time + self.accumulated_time
            records.append((record_time, gpu_usage))

            torch.cuda.reset_peak_memory_stats(self.device)
            time.sleep(self.interval)

        return records
