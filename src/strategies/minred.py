import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class MinRed(AbstractStrategy):
    """
    Continual strategy based on training only on buffer samples, eliminating most correlated samples from buffer (to be paired with MinRed buffer).
    From the article of Purushwalkam et al. https://arxiv.org/abs/2203.12710 .
    """

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                ):           

        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size

        self.strategy_name = 'minred'

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')

    def before_mb_passes(self, stream_mbatch):
        """Add the stream mbatch to the buffer, with mbatch features obtained 
        with an additional encoder pass."""
        # Skip if mb size == 1 (problems with batchnorm)
        if not len(stream_mbatch) == 1:
            with torch.no_grad():
                e_mbatch = self.ssl_model.get_encoder()(stream_mbatch.detach())
                z_mbatch = self.ssl_model.get_projector()(e_mbatch)

            # Add stream minibatch and features to buffer
            self.buffer.add(stream_mbatch.detach(), z_mbatch.detach())

        return stream_mbatch

    def before_forward(self, stream_mbatch):
        """Sample from buffer, disregard stream mbatch"""

        # Sample from buffer (indices needed for buffer features update)
        replay_batch_size = min(self.replay_mb_size, len(self.buffer.buffer))
        replay_batch, _, replay_indices = self.buffer.sample(replay_batch_size)
        replay_batch = replay_batch.to(self.device)
        self.replay_indices = replay_indices

        return replay_batch
    

    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features."""
        # Update samples with avg of last extracted features
        avg_replayed_z = sum(z_list)/len(z_list)
        self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
        return loss