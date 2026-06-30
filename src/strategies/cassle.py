import copy

import torch
from torch import nn

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class CaSSLe(AbstractStrategy):
    """Continual SSL strategy that aligns current representations to past task frozen network,
     introduced by Fini et al. https://openaccess.thecvf.com/content/CVPR2022/papers/Fini_Self-Supervised_Models_Are_Continual_Learners_CVPR_2022_paper.pdf
     It needs task boundaries. """

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 omega: float = 1.0,
                 align_criterion: str = 'ssl',
                 use_aligner: bool = True,
                 align_after_proj: bool = True,
                 aligner_dim: int = 512
                ):

        super().__init__()
        self.ssl_model = ssl_model
        self.device = device
        self.save_pth = save_pth
        self.omega = omega
        self.align_criterion_name = align_criterion
        self.use_aligner = use_aligner
        self.align_after_proj = align_after_proj
        self.aligner_dim = aligner_dim

        self.strategy_name = 'cassle'

        # Set up feature alignment criterion
        if self.align_criterion_name == 'ssl':
            criterion, is_binary = self.ssl_model.get_criterion()
            if is_binary:
                self.align_criterion = criterion
            else:
                raise Exception(f"Needs a binary criterion for alignment, cannot use {self.ssl_model.get_name()} as alignment loss.")
        elif self.align_criterion_name == 'mse':
            self.align_criterion = nn.MSELoss()
        elif self.align_criterion_name == 'cosine':
            self.align_criterion = lambda x,y: -nn.CosineSimilarity(dim=1)(x,y)
        else:
            raise Exception(f"Invalid alignment criterion: {self.align_criterion_name}")

        # Set up alignment projector
        if self.align_after_proj:
            dim_proj = self.ssl_model.get_projector_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_proj, self.aligner_dim, bias=False),
                                                nn.BatchNorm1d(self.aligner_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.aligner_dim, dim_proj)).to(self.device)
        else:
            dim_encoder_embed = self.ssl_model.get_embedding_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_encoder_embed, self.aligner_dim, bias=False),
                                                nn.BatchNorm1d(self.aligner_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.aligner_dim, dim_encoder_embed)).to(self.device)
        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'omega: {self.omega}\n')
                f.write(f'align_criterion: {self.align_criterion_name}\n')
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'align_after_proj: {self.align_after_proj}\n')
                f.write(f'aligner_dim: {self.aligner_dim}\n')


    def get_params(self):
        """Get trainable parameters of the strategy.
        
        Returns:
            alignment_projector (nn.Module): The alignment projector module.
        """
        return list(self.alignment_projector.parameters())
    

    def before_experience(self):
        """Save frozen model of past experience before training on current experience."""
        # Save frozen model of past experience
        self.frozen_encoder = copy.deepcopy(self.ssl_model.get_encoder())
        self.frozen_projector = copy.deepcopy(self.ssl_model.get_projector())
        # Stop gradient in frozen model
        self.frozen_encoder.requires_grad_(False)
        self.frozen_projector.requires_grad_(False)


    def after_forward(self, x_views_list, loss, z_list, e_list):
        """Calculate alignment loss and update replayed samples with new encoder features
            z_list: a list of minibatches, each minibatch corresponds to the one view of the samples
        """
        if not self.align_after_proj:
            # Use encoder features instead projector features
            z_list = e_list
        # Concatenate the features from all views
        z = torch.cat(z_list, dim=0)

        if self.use_aligner:
            # Align features after aligner
            aligned_features = self.alignment_projector(z)
        else:
            # Do not use aligner
            aligned_features = z

        # Frozen model pass
        with torch.no_grad():
            frozen_e = self.frozen_encoder(torch.cat(x_views_list, dim=0))
            if self.align_after_proj:
                frozen_z = self.frozen_projector(frozen_e)
            else:
                # Directly use encoder features as alignment targets
                frozen_z = frozen_e

        # Compute alignment loss between aligned features and EMA features
        loss_align = self.align_criterion(aligned_features, frozen_z)
        loss += self.omega * loss_align.mean()
        
        return loss