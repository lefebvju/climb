# climb.py
import math, time
from collections import deque, defaultdict
import copy
import random

from src.buffers.climb_buffer import ClimbBuffer
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils

from torchmetrics.functional import pairwise_cosine_similarity


from src import  AbstractStrategy
from src.utils import update_ema_params

from src.logger import get_writer
from src.logger import logger

writer = None


def perc(array, percent):
    """Compute the percentile of a sorted array."""
    k = (len(array) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return array[int(k)]
    d0 = array[int(f)] * (c - k)
    d1 = array[int(c)] * (k - f)
    return d0 + d1


class CLIMB(AbstractStrategy):
    """Continual self-supervised learning strategy using a two-level centroid memory (STM/LTM) with consolidation and replay."""

    def __init__(self,
                 ssl_model: nn.Module = None,
                 dataset_name: str = "imagenet100",
                 dim_backbone_features: int = 512,
                 patch_size: int = 224,
                 bs = 138,
                 stride: int = 1,
                 alpha: float = 0.1,
                 novelty_percentile: float = 0.95,
                 stm_to_ltm_threshold: int = 30,
                 stm_size: int = 100,
                 lr: float = 0.6,
                 wd: float = 1.0e-05,
                 n_workers: int = 12,
                 distill_alpha: float = 0.1,
                 mem_update='no_reset',
                 update_use=1,
                 update_use_all=True,
                 max_examples_per_centroid=30,
                 init_M=17,
                 M_min=5,
                 ltm_max: int = 60,
                 ema=False,
                 tau_ema=0.999,
                 file_path=None,
                 device="cpu",
                 ltm_merge_strategy="random",
                 update_ltm_bool=False,
                 eta_min=0.0001,
                 ratio_ltm=0.5,
                 replay_mb_size=54,
                 buffer=None,
                 align_after_proj=True,
                 use_aligner=True,
                 aligner_dim=512,
                 window_size=1000,
                 ltm_replace_mode='random',
                 centroid_alpha=0.0,
                 trim_instead_of_consolidation=False,
                 mem_size: int = 2500,
                 num_views: int = 2
                 ):
        super().__init__()
        global writer
        writer = get_writer()

        # Basic config
        self.strategy_name = "climb"
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

        # Model setup
        self.model = ssl_model
        if align_after_proj:
            self.feat_size = self.model.get_projector_dim()
        else:
            self.feat_size = dim_backbone_features

        # Memory parameters
        self.alpha = alpha
        self.novelty_percentile = novelty_percentile
        self.stm_to_ltm_threshold = stm_to_ltm_threshold
        self.stm_size = int(stm_size)
        self.max_examples_per_centroid = max_examples_per_centroid
        self.M_min = M_min
        self.init_M = init_M
        self.ltm_max = ltm_max
        self.ltm_merge_strategy = ltm_merge_strategy
        self.window_size=window_size

        # Initialize Memory Manager
        self.memory = ClimbBuffer(
            embedding_dim=self.feat_size,
            stm_size=self.stm_size,
            ltm_max=self.ltm_max,
            alpha=self.alpha,
            novelty_percentile=self.novelty_percentile,
            stm_to_ltm_threshold=self.stm_to_ltm_threshold,
            max_examples_per_centroid=self.max_examples_per_centroid,
            window_size=self.window_size,
            device=self.device
        )
        self.memory.ltm_replace_mode = ltm_replace_mode
        self.memory.example_selection_mode = ltm_merge_strategy

        # Training parameters
        self.lr = lr
        self.wd = wd
        self.bs = bs
        self.n_workers = n_workers
        self.eta_min = eta_min

        self.consolidation_cycles = 1

        # Replay parameters
        self.use_replay = False
        self.stream_batch = None
        self.replay_indices_ltm = []
        self.replay_indices_stm = []
        self.buffer = buffer
        self.replay_mb_size = replay_mb_size
        self.ratio_ltm = ratio_ltm

        # Alignment parameters
        self.align_after_proj = align_after_proj
        self.use_aligner = use_aligner
        self.aligner_dim = aligner_dim
        self.align_criterion = lambda x, y: -nn.CosineSimilarity(dim=1)(x, y)

        # Distillation/EMA
        self.ema = ema
        self.tau_ema = tau_ema
        self.distill_alpha = distill_alpha
        self.centroid_alpha = centroid_alpha
        self.trim_instead_of_consolidation = trim_instead_of_consolidation
        self.mem_size = mem_size
        self.train_start_time = time.time()


        # Memory update config
        self.mem_update = mem_update
        self.update_use = update_use
        self.update_use_all = update_use_all
        self.update_ltm_bool = update_ltm_bool

        # Transforms
        self.num_views = num_views

        # Logging and tracking
        self.save_pth = file_path
        self.dataset = dataset_name
        self.nb_gradient = 0
        self.nb_merge = 0

        # Visualization
        self.distance_threshold_history = []
        self.ltm_size_history = []
        self.ltm_mem_size = []
        self.stm_mem_size = []
        self.all_embeddings = []
        self.nd_history = deque(maxlen=100)


        # EMA models for alignment
        self.ema_encoder = copy.deepcopy(self.model.get_encoder())
        self.ema_projector = copy.deepcopy(self.model.get_projector())
        self.ema_encoder.requires_grad_(False)
        self.ema_projector.requires_grad_(False)

        # Setup embedding function and alignment projector
        if self.align_after_proj:
            self.embed = self.model.embed_proj
            self.alignment_projector = nn.Sequential(
                nn.Linear(self.feat_size, self.aligner_dim, bias=False),
                nn.BatchNorm1d(self.aligner_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.aligner_dim, self.feat_size)
            ).to(self.device)
        else:
            self.embed = self.model.embed
            self.alignment_projector = nn.Sequential(
                nn.Linear(self.feat_size, self.aligner_dim, bias=False),
                nn.BatchNorm1d(self.aligner_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.aligner_dim, self.feat_size)
            ).to(self.device)

        # Save configuration
        if self.save_pth is not None:
            self._save_config(ssl_model)


    def _save_config(self, ssl_model):
        """Save model configuration to file."""
        with open(self.save_pth + '/config.txt', 'a') as f:
            f.write('\n')
            f.write('---- STRATEGY CONFIG ----\n')
            f.write(f'STRATEGY: CLIMB\n')
            f.write(f'ssl_model: {ssl_model.model_name}\n')
            f.write(f'patch_size: {self.patch_size}\n')
            f.write(f'feat_size: {self.feat_size}\n')
            f.write(f'LR: {self.lr}\n')
            f.write(f'WD: {self.wd}\n')
            f.write(f'eta_min: {self.eta_min}\n')
            f.write('\n')
            f.write('---- STM CONFIG ----\n')
            f.write(f'stm_size: {self.stm_size}\n')
            f.write(f'mem_size: {self.mem_size}\n')
            f.write(f'alpha: {self.alpha}\n')
            f.write(f'novelty_percentile: {self.novelty_percentile}\n')
            f.write(f'stm_to_ltm_threshold: {self.stm_to_ltm_threshold}\n')
            f.write(f'M: {self.max_examples_per_centroid}\n')
            f.write(f'M_min: {self.M_min}\n')
            f.write(f'init_M: {self.init_M}\n')
            f.write(f'window_size: {self.window_size}\n')
            f.write('\n')
            f.write('---- LTM CONFIG ----\n')
            f.write(f'ltm_max: {self.ltm_max}\n')
            f.write(f'ltm_merge_strategy: {self.ltm_merge_strategy}\n')
            f.write(f'update_ltm_bool: {self.update_ltm_bool}\n')
            f.write(f'ratio_ltm: {self.ratio_ltm}\n')
            f.write(f'ltm_replace_mode: {self.memory.ltm_replace_mode}\n')
            f.write(f'centroid_alpha: {self.centroid_alpha}\n')
            f.write(f'trim_instead_of_consolidation: {self.trim_instead_of_consolidation}\n')
            f.write('\n')
            f.write('---- DISTILLATION / EMA CONFIG ----\n')
            f.write(f'ema: {self.ema}\n')
            f.write(f'tau_ema: {self.tau_ema}\n')
            f.write(f'distill_alpha: {self.distill_alpha}\n')
            f.write('\n')
            f.write('---- ALIGNMENT CONFIG ----\n')
            f.write(f'use_aligner: {self.use_aligner}\n')
            f.write(f'align_after_proj: {self.align_after_proj}\n')
            f.write(f'aligner_dim: {self.aligner_dim}\n')
            f.write('\n')
            f.write('---- REPLAY CONFIG ----\n')
            f.write(f'replay_mb_size: {self.replay_mb_size}\n')
            f.write(f'mem_update: {self.mem_update}\n')
            f.write('-------------------------\n')

    def after_transforms(self, stream_mbatch):
        return stream_mbatch

    def before_forward(self, batch):
        """Sample replay batch and combine with stream batch."""
        self.stream_batch = batch.cpu().detach()
        replay_bs = self.bs - batch.size(0)

        if self.buffer is not None:
            if len(self.buffer.buffer) > replay_bs:
                self.use_replay = True
                replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
                extra = replay_batch.to(self.device)
                self.replay_indices_ltm = replay_indices
                self.replay_indices_stm = []
            else:
                self.use_replay = False
                return batch
        else:
            # Sample from memory manager
            extra, ltm_indices, stm_indices = self.memory.sample_replay(replay_bs, self.ratio_ltm)

            if len(extra) == 0:
                self.use_replay = False
                return batch

            self.replay_indices_ltm = ltm_indices
            self.replay_indices_stm = stm_indices
            extra = torch.stack(extra, dim=0).to(batch.device)

        self.use_replay = True
        self.nb_gradient += 1
        writer.add_scalar('Gradient/CBP', (self.nb_gradient * self.bs) * self.num_views, self.memory.step)

        return torch.cat([extra, batch], dim=0)

    def store(self, x, z,grad=None):
        """Store samples and update memory."""
        # Process batch through memory manager
        stats = self.memory.process_batch(x, z, self.consolidation_cycles, grad)

        # Log statistics
        writer.add_scalar('CentroidSTM/num_novelties', stats['num_novelties'], self.memory.step)
        writer.add_scalar('STM/nb_promotions', stats['nb_promotions'], self.memory.step)
        writer.add_scalar('InputDistance/threshold', stats['distance_threshold'], self.memory.step)
        writer.add_scalar('InputDistance/Min', stats['close_val_z'].mean().item(), self.memory.step)
        writer.add_scalar('InputDistance/Max', torch.amax(stats['distances'], dim=1).mean().item(), self.memory.step)
        writer.add_scalar('InputDistance/Mean', torch.mean(stats['distances'], dim=1).mean().item(), self.memory.step)
        writer.add_scalar('InputDistance/Std', torch.std(stats['distances'], dim=1).mean().item(), self.memory.step)
        writer.add_scalar('CentroidLTM/nb_match', stats['num_ltm_matches'], self.memory.step)

        # Log merge info if available
        self.nb_merge += 1 if stats['merge_info'] is not None else 0

        # Update history
        self.distance_threshold_history.append(stats['distance_threshold'])
        self.nd_history.append(stats['num_novelties'])
        self.ltm_size_history.append(len(self.memory.ltm))


        # Pairwise cosine on all centroids per sample — disabled by default
        self._log_memory_stats(z.detach().cpu())



    def _log_merge_visualization(self, merge_info):
        """Log visualization of merged centroids."""
        c_1, c_2 = merge_info['c_1'], merge_info['c_2']

        img_1 = torch.stack(self.memory.ltm[c_1].get_data_list())
        img_2 = torch.stack(self.memory.ltm[c_2].get_data_list())

        grid_1 = vutils.make_grid(img_1, normalize=True, scale_each=True)
        grid_2 = vutils.make_grid(img_2, normalize=True, scale_each=True)

        writer.add_image(f"centroid_merge/centroid_{c_1}", grid_1, self.memory.step)
        writer.add_image(f"centroid_merge/centroid_{c_2}", grid_2, self.memory.step)

    def _log_memory_stats(self, z):
        """Log memory statistics."""
        stm_cpu = self.memory.stm.get_embeddings(device='cpu')
        ltm_cpu = self.memory.ltm.get_embeddings(device='cpu')
        centroids = torch.cat([stm_cpu, ltm_cpu], dim=0)

        writer.add_scalar("CentroidSTM/nb_centroids", stm_cpu.size(0), global_step=self.memory.step)
        writer.add_scalar("CentroidLTM/nb_centroids", ltm_cpu.size(0), global_step=self.memory.step)
        writer.add_scalar("CentroidAll/nb_centroids", centroids.size(0), global_step=self.memory.step)

        if stm_cpu.numel() > 1:
            writer.add_scalar("CentroidSTM/Variance", torch.var(stm_cpu).item(), global_step=self.memory.step)
        if ltm_cpu.numel() > 1:
            writer.add_scalar("CentroidLTM/Variance", torch.var(ltm_cpu).item(), global_step=self.memory.step)
        if centroids.numel() > 1:
            writer.add_scalar("CentroidAll/Variance", torch.var(centroids).item(), global_step=self.memory.step)

        def pairwise_cosine_sim(a):
            if a.size(0) == 0:
                return torch.tensor([])
            a_norm = torch.nn.functional.normalize(a, dim=1)
            return torch.matmul(a_norm, a_norm.T)

        sim_stm = pairwise_cosine_sim(stm_cpu)
        sim_ltm = pairwise_cosine_sim(ltm_cpu)
        sim_all = pairwise_cosine_sim(centroids)

        if sim_stm.numel() > 1:
            mean_sim_stm = sim_stm[~torch.eye(sim_stm.size(0), dtype=bool)].mean()
            writer.add_scalar("CentroidSTM/IntraSimilarity_Mean", mean_sim_stm.item(), global_step=self.memory.step)

        if sim_ltm.numel() > 1:
            mean_sim_ltm = sim_ltm[~torch.eye(sim_ltm.size(0), dtype=bool)].mean()
            writer.add_scalar("CentroidLTM/IntraSimilarity_Mean", mean_sim_ltm.item(), global_step=self.memory.step)

        if sim_all.numel() > 1:
            mean_sim_all = sim_all[~torch.eye(sim_all.size(0), dtype=bool)].mean()
            writer.add_scalar("CentroidAll/IntraSimilarity_Mean", mean_sim_all.item(), global_step=self.memory.step)

        stm_counts = torch.tensor([len(c.examples) for c in self.memory.stm.centroids], dtype=torch.float)
        ltm_counts = torch.tensor([len(c.examples) for c in self.memory.ltm.centroids], dtype=torch.float)
        if stm_counts.numel() > 0:
            writer.add_histogram("CentroidSTM/examples_per_centroid", stm_counts, global_step=self.memory.step)
        if ltm_counts.numel() > 0:
            writer.add_histogram("CentroidLTM/examples_per_centroid", ltm_counts, global_step=self.memory.step)

        writer.add_scalar("Time/elapsed_seconds", time.time() - self.train_start_time, global_step=self.memory.step)

    def after_mb_passes(self):
        """Process after minibatch passes."""
        if self.buffer is not None:
            z_list_stream = [z[-len(self.stream_batch):] for z in self.z_list]
            z_stream_avg = sum(z_list_stream) / len(z_list_stream)
            self.buffer.add(self.stream_batch.detach(), z_stream_avg.detach())
        else:
            batch = self.stream_batch.detach()
            with torch.no_grad():
                features = self.embed(batch.to(self.device))

            for x, z in zip(batch, features):
                self.store(x.unsqueeze(0), z.unsqueeze(0))
            del features
            stats = self.memory.get_statistics()
            writer.add_scalar('CentroidSTM/nb_images', stats['total_stm_examples'], self.memory.step)
            writer.add_scalar('CentroidLTM/nb_images', stats['total_ltm_examples'], self.memory.step)
            writer.add_scalar('CentroidAll/nb_images', stats['total_stm_examples'] + stats['total_ltm_examples'],
                              self.memory.step)

            if stats['total_stm_examples'] + stats['total_ltm_examples'] >= self.mem_size:
                writer.add_scalar('Consolidation/step', self.memory.step, self.memory.step)

                self.model.train()
                self.model.get_encoder().train()

                # Get memory statistics

                self.ltm_mem_size.append(stats['total_ltm_examples'])
                self.stm_mem_size.append(stats['total_stm_examples'])


                if self.trim_instead_of_consolidation:
                    self.memory.trim_to_size(self.mem_size)
                elif stats['total_stm_examples'] + stats['total_ltm_examples'] >= self.mem_size:
                    self.consolidation()

    def consolidation(self):
        """Perform consolidation."""
        torch.cuda.empty_cache()
        

        # Prepare dataset
        stm_examples_lists = self.memory.stm.get_all_examples()
        ltm_examples_lists = self.memory.ltm.get_all_examples()

        stm_sizes = np.array([len(ex) for ex in stm_examples_lists])
        stm_ex = torch.cat(
            [torch.stack(stm_examples_lists[i])[:self.stm_to_ltm_threshold] for i in np.argwhere(stm_sizes >= 1).flatten()])
        ltm_ex = torch.cat([torch.stack(ex) for ex in ltm_examples_lists])

        mem_ex = torch.cat((stm_ex, ltm_ex))

        self.model.eval()
        self.model.get_encoder().eval()

        # Update memories after consolidation
        if self.update_ltm_bool:
            self._update_ltm(ltm_ex)

        self._update_memory(mem_ex)

        if self.ltm_max > 0:
            ltm_embeddings = self.memory.ltm.get_embeddings(device=self.device)
            self.memory.ltm_simil = pairwise_cosine_similarity(ltm_embeddings).cpu()
            del ltm_embeddings

        self.consolidation_cycles += 1

        self.model.train()
        self.model.get_encoder().train()

        # Cleanup
        del stm_ex, ltm_ex, mem_ex
        del stm_examples_lists, ltm_examples_lists

        torch.cuda.empty_cache()

    def _update_memory(self, mem_exs):
        """Update memory embeddings after consolidation."""
        if self.mem_update == 'reset':
            self.memory.update_embeddings_from_model(
                self.embed,
                self.update_use,
                self.update_use_all,
                reset_stm=False
            )
            self._reset_novelty_detection(mem_exs)
        elif self.mem_update == 'reset_stm':
            self.memory.update_embeddings_from_model(
                self.embed,
                self.update_use,
                self.update_use_all,
                reset_stm=True
            )
        else:
            self.memory.update_embeddings_from_model(
                self.embed,
                self.update_use,
                self.update_use_all,
                reset_stm=False
            )

    def _reset_novelty_detection(self, mem_exs):
        """Reset novelty detection window."""
        self.memory.window = deque(maxlen=self.memory.window_size)

        with torch.no_grad():
            embeddings = []
            for ex in mem_exs:
                embeddings.append(self.embed(ex[None, :].to(self.device)))
            logger.info('Resetting Novelty')
            embeddings = torch.cat(embeddings)
            inds = torch.Tensor(np.random.choice(len(embeddings), min(2000, len(embeddings)), replace=False)).to(
                torch.int64).to(self.device)
            dists = 1 - pairwise_cosine_similarity(embeddings[inds]).cpu()
            del embeddings, inds
        dists += 2 * torch.eye(len(dists))
        dists = torch.amin(dists, dim=1)
        self.memory.window.extend(dists.detach().cpu().numpy().tolist())
        self.memory.distance_threshold = perc(np.sort(np.array(self.memory.window)), self.novelty_percentile)

    def _update_ltm(self, ltm_mem):
        """Update LTM by re-encoding and reassigning examples."""
        with torch.no_grad():
            z = self.embed(ltm_mem.to(self.device))
            ltm_embeddings = self.memory.ltm.get_embeddings(device=self.device)
            distances = 1 - pairwise_cosine_similarity(z, ltm_embeddings)
            del z, ltm_embeddings

        # Get all arrival times
        arrival_times = [t for sublist in self.memory.ltm.get_all_arrival_times() for t in sublist]

        # Reset examples
        for i in range(len(self.memory.ltm)):
            self.memory.ltm[i].examples = []

        # Reassign examples
        for i, image in enumerate(ltm_mem):
            ind = torch.argmin(distances[i]).item()
            self.memory.ltm[ind].add_example(image, arrival_times[i])

        # Filter and limit examples per centroid
        indices_to_remove = []
        for i in range(len(self.memory.ltm)):
            if len(self.memory.ltm[i].examples) == 0:
                indices_to_remove.append(i)
            elif len(self.memory.ltm[i].examples) > self.max_examples_per_centroid:
                if "arrival_time" in self.ltm_merge_strategy:
                    selected_indices = self._uniform_sample_indices_from_centroid(i)
                    self.memory.ltm[i].examples = [self.memory.ltm[i].examples[j] for j in selected_indices]
                elif self.ltm_merge_strategy == "gradient":
                    examples = self.memory.ltm[i].examples
                    gradients = [ex.gradient for ex in examples]
                    selected_indices = sorted(
                        range(len(gradients)),
                        key=lambda j: gradients[j],
                        reverse=True
                    )[:self.max_examples_per_centroid]
                    self.memory.ltm[i].examples = [self.memory.ltm[i].examples[j] for j in selected_indices]
                else:
                    selected_indices = random.sample(range(len(self.memory.ltm[i].examples)), self.max_examples_per_centroid)
                    self.memory.ltm[i].examples = [self.memory.ltm[i].examples[j] for j in selected_indices]

        # Remove empty centroids
        for idx in reversed(indices_to_remove):
            self.memory.ltm.remove_centroid(idx)

        # Recompute embeddings
        for i in range(len(self.memory.ltm)):
            if len(self.memory.ltm[i].examples) > 0:
                images_tensor = torch.stack([ex.data.to(self.device) for ex in self.memory.ltm[i].examples])
                with torch.no_grad():
                    updated_embedding = self.embed(images_tensor).mean(dim=0)
                del images_tensor
                self.memory.ltm[i].embedding = updated_embedding.detach().cpu()
                del updated_embedding

    def _uniform_sample_indices_from_centroid(self, centroid_idx):
        """Sample indices uniformly from a centroid's examples."""
        all_arr = self.memory.ltm[centroid_idx].get_arrival_times()
        all_examples = self.memory.ltm[centroid_idx].examples

        value_to_indices = defaultdict(list)
        for i, val in enumerate(all_arr):
            value_to_indices[val].append(i)

        unique_vals = list(value_to_indices.keys())
        n_classes = len(unique_vals)
        per_class = self.max_examples_per_centroid // n_classes

        selected_indices = []
        for val in unique_vals:
            indices = value_to_indices[val]
            if "random" in self.ltm_merge_strategy:
                random.shuffle(indices)
                take = min(len(indices), per_class)
                selected_indices.extend(indices[:take])
            elif "minred" in self.ltm_merge_strategy:
                z = self.embed(torch.stack([all_examples[i].data for i in indices]).to(self.device))
                dist = 1 - pairwise_cosine_similarity(z)
                for _ in range(z.shape[0] - per_class):
                    d_nneig, _ = dist.min(dim=1)
                    i_redundant = d_nneig.argmin(dim=0)
                    indices.pop(i_redundant)
                    dist = torch.cat([dist[:i_redundant], dist[i_redundant + 1:]], dim=0)
                    dist = torch.cat([dist[:, :i_redundant], dist[:, i_redundant + 1:]], dim=1)
                selected_indices.extend(indices)

        remaining = self.max_examples_per_centroid - len(selected_indices)
        if remaining > 0:
            extra_pool = [i for v in unique_vals for i in value_to_indices[v] if i not in selected_indices]
            random.shuffle(extra_pool)
            selected_indices.extend(extra_pool[:remaining])

        selected_indices.sort()
        return selected_indices

    def after_forward(self, x_views_list, loss, z_list, e_list):
        """Calculate alignment loss and update replayed samples."""
        if not self.align_after_proj:
            z_list = e_list

        self.z_list = z_list

        writer.add_scalar('Loss/ssl_loss', loss.mean().item(), self.memory.step)


        if self.use_replay:
            # Restrict to the first num_views entries so that z_replay and ema_z have
            # matching batch dimensions (models like SwAV produce more views in z_list
            # than extract_image_batch returns).
            e_list_replay = [e[:self.replay_mb_size] for e in e_list[:self.num_views]]
            z_list_replay = [z[:self.replay_mb_size] for z in z_list[:self.num_views]]
            z_replay = torch.cat(z_list_replay, dim=0)

            if self.use_aligner:
                aligned_features = self.alignment_projector(z_replay)
            else:
                aligned_features = z_replay

            # EMA model pass
            with torch.no_grad():
                x_replay_list = [x[:self.replay_mb_size] for x in self.model.extract_image_batch(x_views_list)]
                ema_e = self.ema_encoder(torch.cat(x_replay_list, dim=0))
                if self.align_after_proj:
                    ema_z = self.ema_projector(ema_e)
                else:
                    ema_z = ema_e

            # Alignment loss
            loss_align = self.align_criterion(aligned_features, ema_z)
            writer.add_scalar('Loss/alignment_loss', loss_align.mean().item(), self.memory.step)
            loss += self.distill_alpha * loss_align.mean()

            # Centroid loss: pull ema_z toward its associated centroid embedding
            if self.centroid_alpha > 0.0 and self.buffer is None:
                ltm_centroids = [self.memory.ltm[c].embedding for c, _ in self.replay_indices_ltm]
                stm_centroids = [self.memory.stm[c].embedding for c, _ in self.replay_indices_stm]
                centroid_targets = torch.stack(ltm_centroids + stm_centroids).to(ema_z.device)  # (replay_mb_size, D)
                centroid_targets = torch.cat([centroid_targets] * self.num_views, dim=0)         # (num_views * replay_mb_size, D)
                loss_centroid = self.align_criterion(ema_z, centroid_targets)
                writer.add_scalar('Loss/centroid_loss', loss_centroid.mean().item(), self.memory.step)
                loss += self.centroid_alpha * loss_centroid.mean()

        if self.use_replay:
            # Update features
            if self.align_after_proj:
                avg_replayed = sum(z_list_replay) / len(z_list_replay)
            else:
                avg_replayed = sum(e_list_replay) / len(e_list_replay)

            if self.buffer is not None:
                self.buffer.update_features(avg_replayed.detach(), self.replay_indices_ltm)
            else:
                self.memory.update_centroid_embeddings(
                    avg_replayed.detach(),
                    self.replay_indices_stm,
                    self.replay_indices_ltm
                )
        return loss

    def after_backward(self):
        """Update EMA model after backward pass."""
        update_ema_params(self.model.get_encoder().parameters(), self.ema_encoder.parameters(), self.tau_ema)
        update_ema_params(self.model.get_projector().parameters(), self.ema_projector.parameters(), self.tau_ema)