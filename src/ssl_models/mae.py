import torch

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from .abstract_ssl_model import AbstractSSLModel

from ..backbones import ViT


class MAE(torch.nn.Module, AbstractSSLModel):
    def __init__(self,
                 vit_encoder = ViT,
                 image_size: int = 32,
                 patch_size: int = 2,
                 emb_dim: int = 192, 
                 decoder_layer: int = 4,
                 decoder_head: int = 3,
                 mask_ratio: float = 0.75,
                 save_pth: str = None,
                 ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.encoder = vit_encoder
        self.encoder.init_mask_ratio(mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

        self.mask_ratio = mask_ratio
        self.save_pth = save_pth

        self.num_patches = (image_size // patch_size) ** 2

        self.model_name = 'mae'

        def mae_loss(predicted_img, img, mask):
            return torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
        
        self.criterion = mae_loss

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'image_size: {image_size}\n')
                f.write(f'patch_size: {patch_size}\n')
                f.write(f'emb_dim: {emb_dim}\n')
                f.write(f'decoder_layer: {decoder_layer}\n')
                f.write(f'decoder_head: {decoder_head}\n')
                f.write(f'mask_ratio: {mask_ratio}\n')

    def forward(self, x_views_list):
        img = x_views_list[0]

        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        loss = self.criterion(predicted_img, img, mask)
        
        clf_features = features[0]
        return loss, [clf_features],  [clf_features]

    def embed(self, x):
        with torch.no_grad():
            features, _ = self.encoder(x)
            return features[0]

    def embed_proj(self, x):
        with torch.no_grad():
            features, _ = self.encoder(x)
            return features[0]

    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):           
        return self.encoder.return_features_wrapper()
    
    def get_projector(self):
        # No projector head
        return torch.nn.Identity()
    
    def get_embedding_dim(self):
        return self.emb_dim
    
    def get_projector_dim(self):
        # No projector head
        return self.get_embedding_dim()
    
    def get_criterion(self):
        return self.criterion, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))