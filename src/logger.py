from torch.utils.tensorboard import SummaryWriter
import logging

_writer = None

def init_writer(log_dir=None):
    global _writer
    _writer = SummaryWriter(log_dir=log_dir)

def get_writer():
    if _writer is None:
        raise RuntimeError("TensorBoard writer has not been initialized. Call init_writer(log_dir) first.")
    return _writer

formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s][%(filename)s(%(lineno)d)]  -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S")

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False