from train import get_parser
import cubework
from cubework.utils import get_logger
from vit import build_data

parser = get_parser()
cubework.initialize_distributed(parser)
args = cubework.get_args()

logger = get_logger()

train_data, test_data = build_data(args)

logger.info(len(train_data))
data = next(iter(test_data))
logger.info(data["pixel_values"].shape)
logger.info(data["labels"])
