from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy


class NoStrategy(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 device = 'cpu',
                 save_pth: str  = None,
               ):
                   
        super().__init__()
        self.ssl_model = ssl_model
        self.device = device
        self.save_pth = save_pth

        self.strategy_name = 'no_strategy'

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')