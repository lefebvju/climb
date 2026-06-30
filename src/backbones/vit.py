import numpy as np
import torch
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops import repeat, rearrange

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class ViT(torch.nn.Module):
    # Default ViT encoder is ViT-Tiny
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 return_avg_pooling=False,
                 save_pth: str = None,
                 ) -> None:
        super(ViT, self).__init__()

        self.emb_dim = emb_dim
        self.return_average_pooling = return_avg_pooling
        self.save_pth = save_pth

        self.backbone_name = 'ViT'

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(ratio=0)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

        if self.save_pth is not None:
            # Save backbone configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- BACKBONE MODEL CONFIG ----\n')
                f.write(f'BACKBONE: {self.backbone_name}\n')
                f.write(f'image_size: {image_size}\n')
                f.write(f'patch_size: {patch_size}\n')
                f.write(f'emb_dim: {emb_dim}\n')
                f.write(f'num_layer: {num_layer}\n')
                f.write(f'num_head: {num_head}\n')
                f.write(f'ViT Average Pooling: {self.return_average_pooling}\n')


    def init_mask_ratio(self, mask_ratio=0.75):
        self.shuffle = PatchShuffle(mask_ratio)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes
    
    def return_features_wrapper(self):
        return ViTFeaturesWrapper(self, self.return_average_pooling)
    

class ViTFeaturesWrapper(torch.nn.Module):
    def __init__(self, encoder, return_avg_pooling=False):
        super().__init__()
        self.encoder = encoder
        # Return avg pooling of transformer token outputs, otherwise only return clf token
        self.return_avg_pooling = return_avg_pooling
    def forward(self, x):
        features, _ = self.encoder(x)
        features = rearrange(features, 't b c -> b t c')
        if self.return_avg_pooling:
            return features.mean(dim=1)
        else:
            return features[:,0]