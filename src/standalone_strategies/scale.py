import os
from tqdm import tqdm
import copy

import torch
from torch.utils.data import DataLoader
from torch.functional import F
import torch.nn as nn

from ..utils import UnsupervisedDataset
from ..transforms import get_transforms
from ..optims import init_optim

class SCALE():

    def __init__(self,
                 encoder: nn.Module = None,
                 dim_backbone_features: int = 512,
                 buffer = None,
                 buffer_type: str = 'scale',
                 optim: str = 'SGD',
                 lr: float = 5e-4,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 train_mb_size: int = 32,
                 train_epochs: int = 1,
                 mb_passes: int = 3,
                 device = 'cpu',
                 dataset_name: str = 'cifar100',
                 save_pth: str  = None,
                 save_model: bool = False,
                 common_transforms: bool = True,
                 mem_size: int = 2000,
                 replay_mb_size: int = 32,

                 temperature_cont: float = 0.1,
                 temperature_past: float = 0.01,
                 temperature_curr: float = 0.1,
                 distill_power: float = 0.15,
                 temp_tsne: float = 0.1,
                 tsne_thresh_ratio: float = 0.1,
                 dim_features: int = 128,
    ):           
        
        if encoder is None:
            raise Exception(f'This strategy requires an encoder.')
        if buffer is None:
            raise Exception(f'This strategy requires a buffer')
        
        self.encoder = encoder.to(device)

        self.lr = lr
        self.buffer = buffer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.common_transforms = common_transforms
        self.mem_size = mem_size
        self.replay_mb_size = replay_mb_size

        self.temperature_cont = temperature_cont
        self.temperature_past = temperature_past
        self.temperature_curr = temperature_curr
        self.distill_power = distill_power
        self.temp_tsne = temp_tsne
        self.tsne_thresh_ratio = tsne_thresh_ratio
        self.features_dim = dim_features
        self.past_encoder = None

        if buffer_type == 'scale':
            self.use_scale_buffer = True
        else:
            self.use_scale_buffer = False

        self.strategy_name = 'SCALE'

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common')
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.strategy_name)

        self.tr_distill_power = 0.0

        prev_dim = dim_backbone_features
        print('prev_dim:', prev_dim)
        self.proj_dim = prev_dim
        self.encoder.fc = nn.Identity() # Remove cls output layer

        self.projector = nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.ReLU(inplace=True),
                nn.Linear(prev_dim, self.features_dim)
            ).to(self.device)
        
        self.criterion = SupConLoss(stream_bsz=self.train_mb_size,
                                projector=self.projector,
                                temperature=self.temperature_cont,
                                device=self.device).to(self.device)
        
        self.criterion_reg = IRDLoss(projector=self.projector,
                            current_temperature=self.temperature_curr,
                            past_temperature=self.temperature_past, device=self.device).to(self.device)
        
        # Set up optimizer
        all_parameters = [{
            'name': 'backbone',
            'params': [param for name, param in self.encoder.named_parameters()],
        }, {
            'name': 'heads',
            'params': [param for name, param in self.criterion.named_parameters()],
        }]
        self.optimizer = init_optim(optim, all_parameters, lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay, lars_eta=0.005)           

        self.losses_contrast = AverageMeter()
        self.losses_distill = AverageMeter()


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'optim: {optim}\n') 
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'optim-momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')
                f.write(f'train_epochs: {self.train_epochs}\n')
                f.write(f'mb_passes: {self.mb_passes}\n')
                f.write(f'replay_mb_size: {self.replay_mb_size}\n')

                f.write(f'temperature_cont: {self.temperature_cont}\n')
                f.write(f'temperature_past: {self.temperature_past}\n')
                f.write(f'temperature_curr: {self.temperature_curr}\n')
                f.write(f'distill_power: {self.distill_power}\n')
                f.write(f'temp_tsne: {self.temp_tsne}\n')
                f.write(f'tsne_thresh_ratio: {self.tsne_thresh_ratio}\n')
                f.write(f'dim_features: {self.features_dim}\n')
                f.write(f'use_scale_buffer: {self.use_scale_buffer}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')

                with open(os.path.join(self.save_pth, 'tr_distill_power.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')


    def train_experience(self, 
                         dataset,
                         exp_idx: int
                         ):
        # Prepare data
        exp_data = UnsupervisedDataset(dataset)  
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True, num_workers=12)

        self.encoder.train()
        
        for epoch in range(self.train_epochs):
            for mb_idx, mbatch in tqdm(enumerate(data_loader)):
                mbatch = mbatch.to(self.device)
                new_mbatch = mbatch

                for k in range(self.mb_passes):
                    if self.past_encoder is not None:
                        del self.past_encoder
                        torch.cuda.empty_cache()
                    self.past_encoder = copy.deepcopy(self.encoder) # CHECKED! ONLY THE ENCODERS ARE COPIED, NOT THE PROJECTION HEADS!
                    self.past_encoder.eval().to(self.device)

                    if self.use_scale_buffer:
                        # Try sampling from scale buffer
                        replay_batch = self.buffer.sample(self.replay_mb_size)
                        if replay_batch is None:
                            # Not enough elements in buffer
                            combined_batch = mbatch
                        else:
                            # Concat buffer with stream samples
                            combined_batch = torch.cat((replay_batch.to(self.device), mbatch), dim=0)
                            
                    else:
                        # Try sampling from default buffer
                        if len(self.buffer.buffer) > self.replay_mb_size:
                            use_replay = True
                            # Sample from buffer and concat
                            replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
                            replay_batch = replay_batch.to(self.device)
                            combined_batch = torch.cat((replay_batch, mbatch), dim=0)
                        else:
                            use_replay = False
                            # Do not sample buffer if not enough elements in it
                            combined_batch = mbatch
                   
                    # Apply transforms
                    x1, x2 = self.transforms(combined_batch)

                    all_x = torch.cat((x1, x2), dim=0)

                    combined_batch_size = combined_batch.shape[0]
                    loss_distill = .0

                    x1_logits, loss_distill = self.criterion_reg(self.encoder, self.past_encoder, x1)
                    self.losses_distill.update(loss_distill.item(), combined_batch_size)
                    
                    features_all = self.encoder(all_x)
                    contrast_mask = similarity_mask_old(features_all, combined_batch_size,
                                                        self.device, self.temp_tsne, self.tsne_thresh_ratio, self.train_mb_size)
                    loss_contrast = self.criterion(self.encoder, self.encoder, x1, x2,
                                            mask=contrast_mask)
                    
                    self.losses_contrast.update(loss_contrast.item(), combined_batch_size)

                    if self.tr_distill_power <= 0.0 and loss_distill > 0.0:
                        self.tr_distill_power = self.losses_contrast.avg * self.distill_power / self.losses_distill.avg

                    loss = loss_contrast + self.tr_distill_power * loss_distill

                    if not self.use_scale_buffer and use_replay:
                        replay_z_new_1 = features_all[:replay_batch.shape[0]]
                        replay_z_new_2 = features_all[combined_batch_size:combined_batch_size+replay_batch.shape[0]]

                        # Update replayed samples with avg of last extracted features
                        self.buffer.update_features(((replay_z_new_1+replay_z_new_2)/2).detach(), replay_indices)


                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                        # Save distill power
                        with open(os.path.join(self.save_pth, 'tr_distill_power.csv'), 'a') as f:
                            f.write(f'{self.tr_distill_power},{exp_idx},{epoch},{mb_idx},{k}\n')


                # Update buffer with new samples
                if self.use_scale_buffer:
                    all_embeddings, select_indexes = self.buffer.update_wo_labels(new_mbatch.detach().cpu(), self.encoder)
                else:
                    if use_replay:
                        start_idx = replay_batch.shape[0]
                    else:
                        start_idx = 0
                    self.buffer.add(new_mbatch.detach(), features_all[start_idx:combined_batch_size].detach())

            
            
        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'proj_state_dict': self.criterion.projector.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self
    
    def get_encoder(self):
        return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder 



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,
                 stream_bsz,
                 projector,
                 temperature=0.07,
                 base_temperature=0.07,
                 device="cpu"):
        super(SupConLoss, self).__init__()
        self.stream_bsz = stream_bsz
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.projector = projector
        self.device = device

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        The arguments format is designed to align with other losses.
        In SimCLR, the two backbones should be the same
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        z_stu = F.normalize(self.projector(backbone_stu(x_stu)), dim=1)
        z_tch = F.normalize(self.projector(backbone_tch(x_tch)), dim=1)

        batch_size = x_stu.shape[0]

        all_features = torch.cat((z_stu, z_tch), dim=0)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        # print(mean_log_prob_pos.shape, mean_log_prob_pos.max().item(), mean_log_prob_pos.mean().item(), mean_log_prob_pos.min().item())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batch_size)
        stream_mask = torch.zeros_like(loss).float().to(self.device)
        stream_mask[:, :self.stream_bsz] = 1
        loss = (stream_mask * loss).sum() / stream_mask.sum()
        return loss
    

class IRDLoss(nn.Module):
    """Instance-wise Relation Distillation (IRD) Loss for Contrastive Continual Learning
        https://arxiv.org/pdf/2106.14413.pdf
    """
    def __init__(self, projector, current_temperature=0.2,
                past_temperature=0.01, device="cpu"):
        super(IRDLoss, self).__init__()
        self.projector = projector
        self.curr_temp = current_temperature
        self.past_temp = past_temperature
        self.device = device

    def forward(self, backbone, past_backbone, x):
        """Compute loss for model.
        Args:
            backbone: current backbone
            past_backbone: past backbone
            x: raw input of shape [bsz * n_views, ...]
        Returns:
            A loss scalar.
        """

        cur_features = F.normalize(self.projector(backbone(x)), dim=1)
        past_features = F.normalize(self.projector(past_backbone(x)), dim=1)

        cur_features_sim = torch.div(torch.matmul(cur_features, cur_features.T),
                                    self.curr_temp)
        logits_mask = torch.scatter(
            torch.ones_like(cur_features_sim),
            1,
            torch.arange(cur_features_sim.size(0)).view(-1, 1).to(self.device),
            0
        )
        cur_logits_max, _ = torch.max(cur_features_sim * logits_mask, dim=1, keepdim=True)
        cur_features_sim = cur_features_sim - cur_logits_max.detach()
        row_size =cur_features_sim.size(0)
        cur_logits = torch.exp(cur_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            cur_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        # print('cur_logits', cur_logits * 1e4)

        past_features_sim = torch.div(torch.matmul(past_features, past_features.T), self.past_temp)
        past_logits_max, _ = torch.max(past_features_sim * logits_mask, dim=1, keepdim=True)
        past_features_sim = past_features_sim - past_logits_max.detach()
        past_logits = torch.exp(past_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            past_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        loss_distill = (- past_logits * torch.log(cur_logits)).sum(1).mean()
        #return loss_distill

        return cur_logits, loss_distill
    

def similarity_mask_old(feat_all, bsz, device, temp_tsne, tsne_thresh_ratio, batch_size):
    """Calculate the pairwise similarity and the mask for contrastive learning
    Args:
        feat_all: all hidden features of shape [n_views * bsz, ...].
        bsz: int, batch size of input data (stacked streaming and memory samples)
        opt: arguments
    Returns:
        contrast_mask: mask of shape [bsz, bsz]
    """
    #print(feat_all[0])
    #print(feat_all[1])
    feat_size = feat_all.size(0)
    n_views = int(feat_size / bsz)
    assert (n_views * bsz == feat_size), "Unmatch feature sizes and batch size!"

    # Compute the pairwise distance and similarity between each view
    # and add the similarity together for average
    simil_mat_avg = torch.zeros(bsz, bsz).to(device)
    mat_cnt = 0
    for i in range(n_views):
        for j in range(n_views):
            # feat_row and feat_col should be of size [bsz^2, bsz^2]
            #feat_row, feat_col = PairEnum(feat_all[i*bsz: (i+1)*bsz],
            #                              feat_all[j*bsz: (j+1)*bsz])
            #tmp_distance = -(((feat_row - feat_col) / temperature) ** 2.).sum(1)  # Euclidean distance
            # Note, all features are normalized
            # tSNE similarity
            # compute euclidean distance pairs
            simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                            feat_all[j*bsz: (j+1)*bsz].T)
            #print('\teuc dist', simil_mat * 1e4)
            tmp_distance = - torch.div(simil_mat, temp_tsne)
            tmp_distance = tmp_distance - 1000 * torch.eye(bsz).to(device)
            #print('\ttemp dist', tmp_distance * 1e4)
            simil_mat = 0.5 * torch.softmax(tmp_distance, 1) + 0.5 * torch.softmax(tmp_distance, 0)
            #print(torch.softmax(tmp_distance, 1))
            #print('simil_mat', simil_mat)

            # Add the new probability to the average probability
            simil_mat_avg = (mat_cnt * simil_mat_avg + simil_mat) / (mat_cnt + 1)
            mat_cnt += 1
    #print('simil_mat_avg', simil_mat_avg * 1e4)
    logits_mask = torch.scatter(
        torch.ones_like(simil_mat_avg),
        1,
        torch.arange(simil_mat_avg.size(0)).view(-1, 1).to(device),
        0
    )
    simil_max = simil_mat_avg[logits_mask.bool()].max()
    simil_mean = simil_mat_avg[logits_mask.bool()].mean()
    simil_min = simil_mat_avg[logits_mask.bool()].min()
    #print('prob_simil_avg: dim {}\tmax {}\tavg {}\tmin {}'.format(
    #    simil_mat_avg.shape[0], simil_max, simil_mean, simil_min))
    # Set diagonal of similarity matrix to ones
    masks = torch.eye(bsz).to(device)
    simil_mat_avg = simil_mat_avg * (1 - masks) + masks

    # mask out memory elements
    stream_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    stream_mask[:batch_size, :batch_size] = 1
    simil_mat_avg = simil_mat_avg * stream_mask

    contrast_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    tsne_simil_thres = simil_mean + tsne_thresh_ratio * (simil_max - simil_mean)
    # print(simil_thres)
    contrast_mask[simil_mat_avg > tsne_simil_thres] = 1

    return contrast_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count