import torch
from torch import nn
from .abstract_ssl_model import AbstractSSLModel

class SimSiam(nn.Module, AbstractSSLModel):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim_backbone_features, dim_proj=2048, dim_pred=512, save_pth=None):
        super(SimSiam, self).__init__()
        self.encoder = base_encoder
        self.save_pth = save_pth
        self.model_name = 'simsiam'
        self.dim_projector = dim_proj
        self.dim_predictor = dim_pred

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


        # Build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'dim_predictor: {dim_pred}\n')

    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        # Compute features for both views
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        z1 = self.projector(e1) # NxC
        z2 = self.projector(e2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

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
        return lambda x, y: -self.criterion(x,y), True
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())