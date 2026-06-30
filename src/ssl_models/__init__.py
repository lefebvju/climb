import os
import torch

from .barlow_twins import BarlowTwins
from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .mae import MAE
from .abstract_ssl_model import AbstractSSLModel
from .stl import STL

def recover_ssl_model(ssl_model: AbstractSSLModel, path: str) -> AbstractSSLModel:
    """
     If the path exists, it loads the SSL model from the path and returns it.

    Args:
        path (str): The path to the SSL model.
        ssl_model (AbstractSSLModel): The SSL model to be recovered.
    Returns:
        AbstractSSLModel: The recovered SSL model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'SSL model not found at {path}')
    
    # Recover SSL model
    ssl_model.load_state_dict(torch.load(path, map_location='cpu'))
    print('Correctly initialized SSL model from {}'.format(path))
    return ssl_model