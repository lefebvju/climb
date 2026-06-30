import torch
import torch.nn as nn
import torch.nn.functional as F


from ..ssl_models import AbstractSSLModel
from ..strategies.abstract_strategy import AbstractStrategy

class OsirisR(AbstractStrategy, AbstractSSLModel):

    def __init__(self,
                 base_encoder: nn.Module,
                 dim_backbone_features: int,
                 dim_proj: int = 2048,
                 buffer = None,
                 device: str = 'cpu',
                 replay_mb_size: int = 32,
                 save_pth: str = None):
            
        super().__init__()
        self.encoder = base_encoder
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.save_pth = save_pth
        self.model_name = 'simsiam'
        self.dim_projector = dim_proj

        self.strategy_name = 'osiris_r'
        self.model_name = 'osiris_r' 
        
        self.criterion_curr = NT_Xent()
        self.criterion_cross = Cross_NT_Xent()
        self.criterion_replay = NT_Xent()

         # Build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_proj),
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.predictor = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_proj),
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        for param in self.predictor.parameters():
            param.requires_grad = False




        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL AND STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch

        if len(self.buffer.buffer) > self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch = replay_batch.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch
    
    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        e1, e2 = self.encoder(x1), self.encoder(x2)

        if not self.use_replay:
            z1_s1, z2_s1 = self.projector(e1), self.projector(e2)
            loss = self.criterion_curr(z1_s1, z2_s1)
            self.z_list_stream = [z1_s1, z2_s1]
            return loss, [z1_s1, z2_s1], [e1, e2]

        else:
            z1, z2 = e1[self.replay_mb_size:], e2[self.replay_mb_size:]
            u1, u2 = e1[:self.replay_mb_size], e2[:self.replay_mb_size]

            # current task loss
            # on space 1 (i.e., with g o f)
            z1_s1 = self.projector(z1)
            z2_s1 = self.projector(z2)
            loss1 = self.criterion_curr(z1_s1, z2_s1)

            # cross-task loss
            # on space 2 (i.e., with h o f)
            z1_s2 = self.predictor(z1)
            z2_s2 = self.predictor(z2)
            u1_s2 = self.predictor(u1)
            u2_s2 = self.predictor(u2)
            loss2 = self.criterion_cross(z1_s2, z2_s2, u1_s2, u2_s2)

            # past-task loss
            # also on space 2 (i.e., with h o f)
            loss3 = self.criterion_replay(u1_s2, u2_s2)

            self.z_list_stream = [z1_s2, z2_s2]

            loss = loss1 + 0.5 * (loss2 + loss3)    # overall loss

            return loss, [torch.cat((u1_s2, z1_s1)), torch.cat((u2_s2, z2_s1))], [e1, e2]

    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

        # Get features only of the streaming mbatch and their avg across views
        z_stream_avg = sum(self.z_list_stream)/len(self.z_list_stream)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach())




    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_projector
    
    def get_criterion(self):
        return None, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())

#  Utility functions and classes for Osiris-R
def _mask_correlated_samples(batch_size):
    """
    Generate a boolean mask which masks out the similarity between views of the same example in the similarity matrix
    e.g., a mask for batch size = 2 is a 4x4 matrix (due to two augmented views)
        0  1  0  1
        1  0  1  0
        0  1  0  1  
        1  0  1  0 
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    mask[:, batch_size:].fill_diagonal_(0)
    mask[batch_size:, :].fill_diagonal_(0)
    return mask

class NT_Xent(nn.Module):
    """
    https://arxiv.org/abs/2002.05709
    Modified from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, z_i, z_j):
        """
        Standard contrastive loss on [z_i, z_j]

        param z_i (bsz, d): the stacked g(f(x)) for one augmented view x
        param z_j (bsz, d): the stacked g(f(x')) for the other view x'
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        batch_size = z_i.size(0)
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = z @ z.t()

        # positives are the similarity between different views of the same example 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # negatives are the similarity between different examples
        mask = _mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class Cross_NT_Xent(nn.Module):
    """
    Cross-task loss in Osiris
    """
    def __init__(self, temperature=0.1):
        super(Cross_NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, u_i, u_j):
        """
        Contrastive loss for discriminating z and u
        No comparison between examples within z or u

        param z_i (bsz, d): the stacked h(f(x)) for one augmented view x from the current task
        param z_j (bsz, d): the stacked h(f(x')) for the other view x' from the current task
        param u_i (p*bsz, d): the stacked h(f(y)) for one augmented view y from the memory
        param u_j (p*bsz, d): the stacked h(f(y')) for the other view y' from the memory
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        u_i = F.normalize(u_i, p=2, dim=1)
        u_j = F.normalize(u_j, p=2, dim=1)

        batch_size = z_i.size(0)
        N = batch_size * 2

        # positives are the similarity between different views of the same example within z
        positive_samples = torch.sum(z_i*z_j, dim=-1).repeat(2).reshape(N, 1)

        # negatives are comparisons between the examples in z and the ones in u
        z = torch.cat([z_i, z_j], dim=0)
        u = torch.cat([u_i, u_j], dim=0)
        negative_samples = z @ u.t()

        # loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1) / self.temperature
        loss_zu = self.criterion(logits, labels)
        loss_zu /= N
        
        # for a symmetric loss, switch z and u
        # we do not need to recompute the similarity matrix between z and u
        # simply use the columns rather than the rows of the matrix as negatives
        batch_size = u_i.size(0)
        N = batch_size * 2
        positive_samples = torch.sum(u_i*u_j, dim=-1).repeat(2).reshape(N, 1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples.t()], dim=1) / self.temperature
        loss_uz =  self.criterion(logits, labels)
        loss_uz /= N

        # final cross-task loss
        loss = 0.5 * (loss_zu + loss_uz)

        return loss