import numpy as np

import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class LUMP(AbstractStrategy):
    """Continual SSL strategy that does mixup of stream samples with buffer
    samples. Intorduced by D. Madaan et al. https://arxiv.org/abs/2110.06976"""

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 alpha_lump: float = 0.4,
    ):        

        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.alpha_lump = alpha_lump

        self.strategy_name = 'lump'


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'alpha_lump: {self.alpha_lump}\n')


    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""
        self.stream_mbatch = stream_mbatch
        self.stream_mbatch_size = stream_mbatch.shape[0]

        if len(self.buffer.buffer) > self.stream_mbatch_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.stream_mbatch_size)
            replay_batch = replay_batch.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch
    

    def after_transforms(self, x_views_list):
        """Mixup stream and replay samples"""
        if self.use_replay:
            # Separate replay sample views and stream sample views
            x_views_list_replay = [x[:self.stream_mbatch_size] for x in x_views_list]
            x_views_list_stream = [x[self.stream_mbatch_size:] for x in x_views_list]

            # Apply mixup
            lambd = np.random.beta(self.alpha_lump, self.alpha_lump)

            x_views_list_mixed = []
            for i, x_stream in enumerate(x_views_list_stream):
                x_views_list_mixed.append(lambd * x_stream + (1 - lambd) * x_views_list_replay[i])
            return x_views_list_mixed

        else:
            # No mixup
            return x_views_list
        
    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features for replayed samples"""
        self.z_list = z_list
        if self.use_replay:
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.stream_mbatch_size] for z in z_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
        return loss
        

    def after_mb_passes(self):
        """Update buffer with new samples after all mb pass with streaming mbatch."""

        # Get features only of the streaming mbatch and their avg across views
        z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
        z_stream_avg = sum(z_list_stream)/len(z_list_stream)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach())

