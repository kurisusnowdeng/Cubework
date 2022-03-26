from typing import Optional

ALLOWED_MODES = [None, '1d', '2d', '3d']

GLOBAL = 'global'

# data parallel
DATA = 'data'

# tensor parallel
TENSOR = 'tensor'
NUM_PARTITIONS = 'num_partitions'

# 1D Parallel
PARALLEL_1D = '1d'

# 2D parallel
PARALLEL_2D_ROW = '2d_row'
PARALLEL_2D_COL = '2d_col'

# 3D parallel
PARALLEL_3D_INPUT = '3d_input'
PARALLEL_3D_WEIGHT = '3d_weight'
PARALLEL_3D_OUTPUT = '3d_output'


class TensorParallelEnv(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, *args, **kwargs):
        self.load(*args, **kwargs)

    def load(self,
             mode: Optional[str] = None,
             vocab_parallel: bool = False,
             parallel_input_1d: bool = False,
             summa_dim: int = None,
             depth_3d: int = None,
             input_group_3d=None,
             weight_group_3d=None,
             output_group_3d=None):
        self.mode = mode
        self.vocab_parallel = vocab_parallel
        self.parallel_input_1d = parallel_input_1d
        self.summa_dim = summa_dim
        self.depth_3d = depth_3d
        self.input_group_3d = input_group_3d
        self.weight_group_3d = weight_group_3d
        self.output_group_3d = output_group_3d

    def save(self):
        return dict(mode=self.mode,
                    vocab_parallel=self.vocab_parallel,
                    parallel_input_1d=self.parallel_input_1d,
                    summa_dim=self.summa_dim,
                    depth_3d=self.depth_3d,
                    input_group_3d=self.input_group_3d,
                    weight_group_3d=self.weight_group_3d,
                    output_group_3d=self.output_group_3d)


env = TensorParallelEnv()
