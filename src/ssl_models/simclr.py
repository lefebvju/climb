import torch
from torch import nn
import torch.nn.functional as F

from .abstract_ssl_model import AbstractSSLModel

class SimCLR(nn.Module, AbstractSSLModel):
    """
    Build a SimCLR model.
    """
    def __init__(self, base_encoder, dim_backbone_features, dim_proj=2048, save_pth=None, temperature=0.5):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.save_pth = save_pth
        self.model_name = 'simclr'
        self.dim_projector = dim_proj
        self.temperature = temperature

        # Set up criterion
        self.criterion = nn.CosineSimilarity(dim=1)

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


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'SimCLR temperature: {self.temperature}\n')

    def simclr_loss(self, z1, z2):
        batch_size = z1.size(0)
        z1_norm, z2_norm = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

        out = torch.cat([z1_norm, z2_norm], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(z1_norm * z2_norm, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        # Compute features for both views
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        z1 = self.projector(e1) # NxC
        z2 = self.projector(e2) # NxC        
        loss = self.simclr_loss(z1, z2)
        

        return loss, [z1, z2], [e1, e2]
        
    def embed(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            return z

    def embed_proj(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            h = self.projector(z)
            return h
            
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
        return self.simclr_loss, True
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())

    def configure_optimizers(self, lr, wd, epochs, num_iters):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        
        return optimizer, None