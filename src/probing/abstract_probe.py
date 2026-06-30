from torch import nn
from torch.utils.data import Dataset

class AbstractProbe():
    def __init__(self) -> None:
        pass

    def probe(self,
              encoder: nn,
              tr_dataset: Dataset,
              test_dataset: Dataset,
              val_dataset: Dataset = None,
              exp_idx: int = None, # Task index on which probing is executed, if None, we are in joint or upto probing
              tr_samples_ratio: float = 1.0,
              save_file: str = None
              ):
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError