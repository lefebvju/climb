import random
from collections import deque

import torch
import numpy as np
from typing import List, Optional
import random as rnd
from collections import Counter

from torchmetrics.functional import pairwise_cosine_similarity


class CentroidExample:
    """
    Represents a stored example with its metadata.
    """

    def __init__(self, data: torch.Tensor, arrival_time: int, gradient: Optional[torch.Tensor] = None,
                 embedding: Optional[torch.Tensor] = None):
        """
        Args:
            data: Example data (image, vector, etc.)
            arrival_time: Arrival time (sleep cycle)
            gradient: Optional gradient information.
            embedding: Optional embedding vector at insertion time, used for similarity-based replacement.
        """
        self.data = data.detach().cpu()
        self.arrival_time = arrival_time
        self.gradient = gradient
        self.embedding = embedding.detach().cpu() if embedding is not None else None

    def to(self, device):
        """Move data to device."""
        return self.data.to(device)

    def __repr__(self):
        return f"Example(arrival_time={self.arrival_time}, shape={self.data.shape}, gradient={self.gradient})"


class Centroid:
    """
    Represents a centroid with its embedding vector and associated examples.
    """

    def __init__(self,
                 embedding: torch.Tensor,
                 max_examples: int = 30,
                 initial_data: Optional[torch.Tensor] = None,
                 initial_arrival_time: int = 0,
                 initial_gradient: Optional[torch.Tensor] = None):
        """
        Args:
            embedding: Centroid embedding vector (stored on CPU)
            max_examples: Maximum number of examples to store
            initial_data: First associated example
            initial_arrival_time: Arrival time of first example
        """
        self.embedding = embedding.detach().cpu().clone()
        self.max_examples = max_examples
        self.examples = []
        self.match_count = 0
        self.age = 0

        if initial_data is not None:
            self.examples.append(CentroidExample(initial_data, initial_arrival_time, initial_gradient))

    def update_embedding(self, new_embedding: torch.Tensor, alpha: float = 0.1):
        """
        Update embedding with EMA.

        Args:
            new_embedding: New embedding to integrate (will be moved to CPU)
            alpha: Update coefficient (learning rate)
        """
        new_embedding = new_embedding.detach().cpu()
        self.embedding = (1 - alpha) * self.embedding + alpha * new_embedding
        self.match_count += 1

    def add_example(self, data: torch.Tensor, arrival_time: int, gradient: Optional[torch.Tensor] = 0,
                    embedding: Optional[torch.Tensor] = None) -> bool:
        """
        Add an example if capacity is not reached.

        Args:
            data: Example data
            arrival_time: Arrival time
            embedding: Optional embedding vector at insertion time.

        Returns:
            True if added, False otherwise
        """
        if len(self.examples) < self.max_examples:
            self.examples.append(CentroidExample(data, arrival_time, gradient, embedding))
            return True
        return False

    def replace_random_example(self, data: torch.Tensor, arrival_time: int,
                                gradient: Optional[torch.Tensor] = None,
                                embedding: Optional[torch.Tensor] = None):
        """Replace a uniformly random example with a new one."""
        if len(self.examples) > 0:
            idx = np.random.randint(0, len(self.examples))
            self.examples[idx] = CentroidExample(data, arrival_time, gradient, embedding)

    def replace_oldest_example(self, data: torch.Tensor, arrival_time: int,
                                gradient: Optional[torch.Tensor] = None,
                                embedding: Optional[torch.Tensor] = None):
        """
        Replace the example with the smallest arrival_time (oldest).
        Keeps the memory fresh by evicting the most outdated example.
        """
        if len(self.examples) > 0:
            idx = min(range(len(self.examples)), key=lambda i: self.examples[i].arrival_time)
            self.examples[idx] = CentroidExample(data, arrival_time, gradient, embedding)

    def replace_most_similar_example(self, data: torch.Tensor, arrival_time: int,
                                      new_embedding: torch.Tensor,
                                      gradient: Optional[torch.Tensor] = None):
        """
        Replace the stored example whose embedding is most similar to new_embedding.
        Maximises diversity by removing the most redundant example.
        Falls back to random replacement for examples without stored embeddings.

        Args:
            new_embedding: Embedding of the incoming example (used for similarity computation).
        """
        if len(self.examples) == 0:
            return
        scored = []
        for i, ex in enumerate(self.examples):
            if ex.embedding is not None:
                sim = torch.nn.functional.cosine_similarity(
                    new_embedding.cpu().unsqueeze(0), ex.embedding.unsqueeze(0)
                ).item()
            else:
                sim = -float('inf')  # fallback: never preferred over a real embedding
            scored.append((i, sim))
        # All embeddings are None → fall back to random
        if all(s == -float('inf') for _, s in scored):
            idx = np.random.randint(0, len(self.examples))
        else:
            idx = max(scored, key=lambda t: t[1])[0]
        self.examples[idx] = CentroidExample(data, arrival_time, gradient, new_embedding)

    def replace_balanced_time_example(self, data: torch.Tensor, arrival_time: int,
                                       gradient: Optional[torch.Tensor] = None,
                                       embedding: Optional[torch.Tensor] = None):
        """
        Replace a random example from the most over-represented arrival_time group.
        Maintains a balanced distribution of arrival times across stored examples,
        preventing any single time period from dominating the centroid's memory.
        """
        if len(self.examples) == 0:
            return
        # Count examples per arrival_time
        counts: dict = {}
        for ex in self.examples:
            counts[ex.arrival_time] = counts.get(ex.arrival_time, 0) + 1
        # Pick the most represented group
        busiest_time = max(counts, key=lambda t: counts[t])
        candidates = [i for i, ex in enumerate(self.examples) if ex.arrival_time == busiest_time]
        idx = random.choice(candidates)
        self.examples[idx] = CentroidExample(data, arrival_time, gradient, embedding)

    def get_data_list(self) -> List[torch.Tensor]:
        """Return list of example data."""
        return [ex.data for ex in self.examples]

    def get_arrival_times(self) -> List[int]:
        """Return list of arrival times."""
        return [ex.arrival_time for ex in self.examples]

    def get_sample_examples(self, n: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the first n examples.

        Args:
            n: Number of examples to retrieve

        Returns:
            List of example data
        """
        return [(ex.data, ex.gradient) for ex in self.examples[:min(n, len(self.examples))]]

    def increment_age(self):
        """Increment centroid age."""
        self.age += 1

    def reset_age(self):
        """Reset age to zero."""
        self.age = 0

    def is_mature(self, threshold: int) -> bool:
        """Check if centroid is mature (ready for LTM)."""
        return self.match_count >= threshold

    def mark_as_transferred(self):
        """Mark centroid as transferred to LTM."""
        self.match_count = -1

    def is_transferred(self) -> bool:
        """Check if centroid has been transferred to LTM."""
        return self.match_count == -1

    def __len__(self):
        """Return number of examples."""
        return len(self.examples)

    def __repr__(self):
        return (f"Centroid(matches={self.match_count}, age={self.age}, "
                f"examples={len(self.examples)}/{self.max_examples})")


class CentroidMemory:
    """
    Manages a collection of centroids (STM or LTM).
    All embeddings are stored on CPU and moved to device only when needed.
    """

    def __init__(self, embedding_dim: int, max_centroids: int):
        """
        Args:
            embedding_dim: Dimension of embeddings
            max_centroids: Maximum number of centroids (-1 for unlimited)
        """
        self.embedding_dim = embedding_dim
        self.max_centroids = max_centroids
        self.centroids: List[Centroid] = []

    def initialize_empty(self, max_examples: int = 30):
        """Initialize with empty centroids.

        Args:
            max_examples: Maximum number of examples per centroid.
        """
        self.centroids = []
        for _ in range(self.max_centroids):
            zero_embedding = torch.zeros(self.embedding_dim)
            centroid = Centroid(zero_embedding, max_examples=max_examples)
            self.centroids.append(centroid)

    def add_centroid(self, centroid: Centroid):
        """Add a centroid to the memory."""
        self.centroids.append(centroid)

    def remove_centroid(self, index: int):
        """Remove a centroid at given index."""
        if 0 <= index < len(self.centroids):
            self.centroids.pop(index)

    def get_embeddings(self, device: str = 'cpu') -> torch.Tensor:
        """
        Return all embeddings as a tensor.

        Args:
            device: Device to move embeddings to (default: 'cpu')

        Returns:
            Tensor of shape [n_centroids, embedding_dim] on specified device
        """
        if len(self.centroids) == 0:
            return torch.empty(0, self.embedding_dim).to(device)
        embeddings = torch.stack([c.embedding for c in self.centroids])
        return embeddings.to(device)

    def set_embeddings(self, embeddings: torch.Tensor):
        """Set embeddings from a tensor (will be moved to CPU)."""
        embeddings = embeddings.detach().cpu()
        assert embeddings.shape[0] == len(self.centroids)
        for i, emb in enumerate(embeddings):
            self.centroids[i].embedding = emb.clone()

    def get_matches(self, device: str = 'cpu') -> torch.Tensor:
        """Return match counts as tensor."""
        return torch.tensor([c.match_count for c in self.centroids],
                            dtype=torch.float32).to(device)

    def set_matches(self, matches: torch.Tensor):
        """Set match counts from tensor."""
        matches = matches.detach().cpu()
        for i, match in enumerate(matches):
            self.centroids[i].match_count = match.item()

    def get_ages(self, device: str = 'cpu') -> torch.Tensor:
        """Return ages as tensor."""
        return torch.tensor([c.age for c in self.centroids],
                            dtype=torch.float32).to(device)

    def set_ages(self, ages: torch.Tensor):
        """Set ages from tensor."""
        ages = ages.detach().cpu()
        for i, age in enumerate(ages):
            self.centroids[i].age = age.item()

    def increment_all_ages(self):
        """Increment age of all centroids."""
        for c in self.centroids:
            c.increment_age()

    def get_all_examples(self) -> List[List[torch.Tensor]]:
        """Return all examples as list of lists."""
        return [c.get_data_list() for c in self.centroids]

    def get_all_arrival_times(self) -> List[List[int]]:
        """Return all arrival times as list of lists."""
        return [c.get_arrival_times() for c in self.centroids]

    def __len__(self):
        """Return number of centroids."""
        return len(self.centroids)

    def __getitem__(self, index):
        """Get centroid at index."""
        return self.centroids[index]

    def __setitem__(self, index, value):
        """Set centroid at index."""
        self.centroids[index] = value


class ClimbBuffer:
    """
    Manages STM and LTM memories with all operations.
    Separates memory logic from the main CLIMB strategy.
    """

    def __init__(self,
                 embedding_dim: int,
                 stm_size: int,
                 ltm_max: int,
                 alpha: float,
                 novelty_percentile: float,
                 stm_to_ltm_threshold: int,
                 max_examples_per_centroid: int,
                 window_size:int,
                 device: str = 'cpu'):
        """
        Args:
            embedding_dim: Dimension of embeddings
            stm_size: Size of short-term memory (delta)
            ltm_max: Maximum size of long-term memory
            alpha: Centroid learning rate
            novelty_percentile: Percentile for novelty threshold
            stm_to_ltm_threshold: STM to LTM maturity threshold
            max_examples_per_centroid: Maximum examples per LTM centroid
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.stm_size = stm_size
        self.ltm_max = ltm_max
        self.alpha = alpha
        self.novelty_percentile = novelty_percentile
        self.stm_to_ltm_threshold = stm_to_ltm_threshold
        self.max_examples_per_centroid = max_examples_per_centroid

        self.device = device

        # Initialize memories
        self.stm = CentroidMemory(embedding_dim, stm_size)
        #self.stm.initialize_empty()
        self.ltm = CentroidMemory(embedding_dim, max_centroids=-1)

        # Novelty detection
        self.window_size = window_size
        self.window = deque(maxlen=self.window_size)
        self.num_distance_samples = 10
        self.distance_threshold = -1

        # Tracking
        self.step = 0
        self.nb_novelties = 0
        self.nb_ltm_match = 0
        self.no_update_steps = -1

        # LTM similarity matrix for merging
        self.ltm_simil = torch.zeros((0, 0))
        self.example_selection_mode = 'balanced'  # ou 'gradient', 'random'
        self.ltm_replace_mode = 'random'  # 'random' | 'oldest' | 'reservoir' | 'similar' | 'balanced_time'
                                          # | 'reservoir_oldest' | 'reservoir_similar' | 'reservoir_balanced_time'

    def initialize_memory(self):
        """Reset all memory structures."""
        self.stm = CentroidMemory(self.embedding_dim, self.stm_size)
        #self.stm.initialize_empty()
        self.ltm = CentroidMemory(self.embedding_dim, max_centroids=-1)
        self.step = 0
        self.window = deque(maxlen=self.window_size)
        self.distance_threshold = -1
        self.ltm_simil = torch.zeros((0, 0))

    def process_batch(self,
                      x: torch.Tensor,
                      z: torch.Tensor,
                      sleep_cycles: int,
                      gradient: Optional[torch.Tensor] = None) -> dict:
        """
        Process a batch of data and update memories.

        Args:
            x: Input data (images)
            z: Embeddings
            sleep_cycles: Current sleep cycle number

        Returns:
            Dictionary with statistics and flags
        """

        # Get embeddings from STM and LTM
        stm_embeddings = self.stm.get_embeddings(device=self.device)
        ltm_embeddings = self.ltm.get_embeddings(device=self.device)

        # Compute distances
        distances = 1 - pairwise_cosine_similarity(z, torch.cat((stm_embeddings, ltm_embeddings)))
        if stm_embeddings.shape[0] != self.stm.max_centroids:
            distances = torch.cat([distances,torch.tensor([[torch.nan]*(self.stm.max_centroids - stm_embeddings.shape[0])],device=self.device)], dim=1)
        del stm_embeddings, ltm_embeddings

        # Find closest centroids
        close_ind_z = torch.argmin(distances, dim=1)
        close_val_z = torch.amin(distances, dim=1)


        # Handle novelties
        novel_inds = torch.argwhere(
            ((close_val_z > self.distance_threshold) | torch.isnan(close_val_z))
        ).flatten().tolist()

        num_novelties = len(novel_inds)
        self.nb_novelties += num_novelties

        if num_novelties > 0:
            self._handle_novelties(novel_inds, x, z, sleep_cycles, gradient)

        # Handle STM matches
        stm_matches = self.stm.get_matches(device=self.device)
        z_match_inds = set(torch.argwhere(close_val_z <= self.distance_threshold).flatten().tolist())
        z_match_inds_stm = list(
            z_match_inds.intersection(
                set(torch.argwhere(close_ind_z < self.stm_size).flatten().tolist())
            ).intersection(
                set(torch.argwhere(stm_matches >= 0).flatten().tolist())
            )
        )

        if len(z_match_inds_stm) > 0:
            self._update_stm_matches(z_match_inds_stm, close_ind_z, x, z, sleep_cycles, gradient)

        # Handle LTM matches
        z_match_inds_ltm = torch.argwhere(
            (close_ind_z >= len(self.stm)) &
            (close_val_z <= self.distance_threshold)
        ).flatten().tolist()

        self.nb_ltm_match += len(z_match_inds_ltm)

        if len(z_match_inds_ltm) > 0:
            self._update_ltm_matches(z_match_inds_ltm, close_ind_z, x, z, sleep_cycles,gradient)

        # Increment ages
        self.stm.increment_all_ages()

        # Update novelty threshold
        sampled_d = np.random.choice(close_val_z.cpu().numpy(), self.num_distance_samples)
        self.window.extend(sampled_d)
        self.distance_threshold = self._percentile(np.sort(np.array(self.window)), self.novelty_percentile)

        self.step += len(x)

        # Check for mature centroids and transfer to LTM
        need_merge, n_promotions = self._transfer_mature_to_ltm(sleep_cycles)

        # Merge LTM if needed
        merge_info = None
        if need_merge and self.ltm_max > 0 and len(self.ltm) > self.ltm_max:
            merge_info = self._merge_ltm_centroids()

        return {
            'num_novelties': num_novelties,
            'num_ltm_matches': self.nb_ltm_match,
            'num_stm_matches': len(z_match_inds_stm),
            'distance_threshold': self.distance_threshold,
            'merge_info': merge_info,
            'close_val_z': close_val_z,
            'distances': distances,
            'nb_promotions': n_promotions
        }

    def _handle_novelties(self, novel_inds, x, z, sleep_cycles, gradient: Optional[torch.Tensor] = None):
        """Handle novel samples by evicting oldest STM centroids."""
        if len(self.stm) < self.stm.max_centroids:
            # STM pas pleine — ajout direct sans éviction
            for idx in novel_inds:
                self.stm.add_centroid(Centroid(embedding=z[idx].detach().cpu(), max_examples=self.max_examples_per_centroid,
                                                  initial_data=x[idx].detach().cpu(), initial_arrival_time= sleep_cycles, initial_gradient=gradient))
        else:

            stm_ages = self.stm.get_ages(device=self.device)
            evict_inds = torch.argsort(stm_ages)[-len(novel_inds):]

            # Update embeddings
            stm_embeddings = self.stm.get_embeddings(device=self.device)
            stm_embeddings[evict_inds] = z[novel_inds]
            self.stm.set_embeddings(stm_embeddings)

            # Update matches
            stm_matches = self.stm.get_matches(device=self.device)
            stm_matches[evict_inds] = 1
            self.stm.set_matches(stm_matches)

            # Reset ages
            stm_ages[evict_inds] = 0
            self.stm.set_ages(stm_ages)

            # Update examples
            for i in range(len(evict_inds)):
                evict_idx = evict_inds[i].item()
                novel_idx = novel_inds[i]
                self.stm[evict_idx].examples = [CentroidExample(x[novel_idx], sleep_cycles, gradient)]

    def _update_stm_matches(self, z_match_inds_stm, close_ind_z, x, z, sleep_cycles,gradient: Optional[torch.Tensor] = None):
        """Update STM centroids that matched."""
        stm_match_inds = close_ind_z[z_match_inds_stm]

        # Update embeddings with EMA
        stm_embeddings = self.stm.get_embeddings(device=self.device)
        stm_embeddings[stm_match_inds] = (1 - self.alpha) * stm_embeddings[stm_match_inds] \
                                         + self.alpha * z[z_match_inds_stm]
        self.stm.set_embeddings(stm_embeddings)

        # Add examples
        for i in range(len(stm_match_inds)):
            stm_idx = stm_match_inds[i].item()
            z_idx = z_match_inds_stm[i]

            if len(self.stm[stm_idx]) < self.stm_to_ltm_threshold:
                self.stm[stm_idx].add_example(x[z_idx], sleep_cycles, gradient)

        # Increment matches
        stm_matches = self.stm.get_matches(device=self.device)
        stm_matches[stm_match_inds] += 1
        self.stm.set_matches(stm_matches)

        # Reset ages
        stm_ages = self.stm.get_ages()
        stm_ages[stm_match_inds] = 0
        self.stm.set_ages(stm_ages)

    def _update_ltm_matches(self, z_match_inds_ltm, close_ind_z, x, z, sleep_cycles, gradient: Optional[torch.Tensor] = None):
        """
        Update LTM centroids that matched an incoming sample.

        For each matched LTM centroid:
        - Increment its match count.
        - If the centroid is not full, add the new example directly.
        - If the centroid is full, delegate to _replace_ltm_example which applies
          the replacement strategy selected by self.ltm_replace_mode.

        Args:
            z_match_inds_ltm: Indices (in the batch) of samples that matched an LTM centroid.
            close_ind_z: Index of the closest centroid (STM + LTM combined) for each sample.
            x: Input images for the current batch.
            z: Embeddings for the current batch.
            sleep_cycles: Current sleep cycle number.
            gradient: Optional gradient information associated with the samples.
        """
        raw_ltm_inds = close_ind_z[z_match_inds_ltm]
        ltm_match_inds = raw_ltm_inds[raw_ltm_inds >= len(self.stm)] - len(self.stm)

        for i in range(len(ltm_match_inds)):
            ltm_idx = ltm_match_inds[i].item()
            z_idx = z_match_inds_ltm[i]

            self.ltm[ltm_idx].match_count += 1

            img = x[z_idx].detach().cpu()
            emb = z[z_idx].detach().cpu()

            if len(self.ltm[ltm_idx]) < self.max_examples_per_centroid:
                self.ltm[ltm_idx].add_example(img, sleep_cycles, gradient, emb)
            elif self.ltm_replace_mode != 'None':
                self._replace_ltm_example(self.ltm[ltm_idx], img, emb, sleep_cycles, gradient)

    def _replace_ltm_example(self, centroid, img: torch.Tensor, emb: torch.Tensor,
                              arrival_time: int, gradient: Optional[torch.Tensor] = None):
        """
        Replace an existing example in a full LTM centroid according to self.ltm_replace_mode.

        Modes:
            'random'                : Binary draw (p=0.5) then replace a uniformly random example.
            'oldest'                : Always replace the example with the smallest arrival_time.
            'reservoir'             : Accept with probability k/n (k=capacity, n=match_count) then
                                      replace a uniformly random example. Guarantees a uniform
                                      distribution over all examples seen by the centroid.
            'similar'               : Always replace the stored example most similar to the new one
                                      (maximises diversity). Requires stored embeddings; falls back
                                      to random when none are available.
            'balanced_time'         : Always replace an example from the most over-represented
                                      arrival_time group (maintains a balanced time distribution).
            'reservoir_oldest'      : Accept with reservoir probability k/n, then replace oldest.
            'reservoir_similar'     : Accept with reservoir probability k/n, then replace most similar.
            'reservoir_balanced_time': Accept with reservoir probability k/n, then replace from
                                      most over-represented time group.

        Args:
            centroid: The LTM Centroid object to update.
            img: New example image (CPU tensor).
            emb: Embedding of the new example (CPU tensor).
            arrival_time: Arrival time (sleep cycle) of the new example.
            gradient: Optional gradient information.
        """
        mode = self.ltm_replace_mode

        # Reservoir gate: compute acceptance probability for reservoir-prefixed modes
        if mode.startswith('reservoir'):
            k = self.max_examples_per_centroid
            n = centroid.match_count
            if n == 0 or random.random() >= k / n:
                return

        if mode == 'random':
            if random.random() < 0.5:
                centroid.replace_random_example(img, arrival_time, gradient, emb)

        elif mode in ('oldest', 'reservoir_oldest'):
            centroid.replace_oldest_example(img, arrival_time, gradient, emb)

        elif mode == 'reservoir':
            centroid.replace_random_example(img, arrival_time, gradient, emb)

        elif mode in ('similar', 'reservoir_similar'):
            centroid.replace_most_similar_example(img, arrival_time, emb, gradient)

        elif mode in ('balanced_time', 'reservoir_balanced_time'):
            centroid.replace_balanced_time_example(img, arrival_time, gradient, emb)

        else:
            raise ValueError(f"Unknown ltm_replace_mode: '{mode}'. "
                             f"Choose from 'random', 'oldest', 'reservoir', 'similar', 'balanced_time', "
                             f"'reservoir_oldest', 'reservoir_similar', 'reservoir_balanced_time'.")

    def _transfer_mature_to_ltm(self, sleep_cycles):
        """Transfer mature STM centroids to LTM. Returns (need_merge, n_promotions)."""
        from torchmetrics.functional import pairwise_cosine_similarity

        stm_matches = self.stm.get_matches(device='cpu')
        mature_cent_inds = torch.argwhere(stm_matches >= self.stm_to_ltm_threshold).flatten().detach().cpu().numpy()

        if len(mature_cent_inds) > 0:
            self.no_update_steps = 0

            # Mark as transferred
            stm_matches[mature_cent_inds] = -1
            self.stm.set_matches(stm_matches)

            # Get embeddings
            stm_embeddings = self.stm.get_embeddings(device=self.device)
            new_items = stm_embeddings[mature_cent_inds]
            n_new = new_items.shape[0]

            # Update similarity matrix
            if self.ltm_max > 0:
                ltm_embeddings = self.ltm.get_embeddings(device=self.device)
                if ltm_embeddings.shape[0] == 0:
                    self.ltm_simil = torch.zeros(n_new, n_new)
                else:
                    n_existing = ltm_embeddings.shape[0]
                    new_to_existing = pairwise_cosine_similarity(new_items, ltm_embeddings).cpu()
                    new_to_new = pairwise_cosine_similarity(new_items, new_items).cpu()

                    top_row = torch.cat([self.ltm_simil, new_to_existing.T], dim=1)
                    bottom_row = torch.cat([new_to_existing, new_to_new], dim=1)
                    self.ltm_simil = torch.cat([top_row, bottom_row], dim=0)

            # Transfer centroids
            for cent_idx in mature_cent_inds:
                new_centroid = Centroid(
                    embedding=self.stm[cent_idx].embedding,
                    max_examples=self.max_examples_per_centroid,

                )

                stm_examples = self.stm[cent_idx].get_sample_examples(min(self.max_examples_per_centroid, self.stm_to_ltm_threshold))
                for ex,grad in stm_examples:
                    new_centroid.add_example(ex, sleep_cycles, grad)

                self.ltm.add_centroid(new_centroid)

                # Reset STM centroid
                if len(self.stm[cent_idx].examples) > 0:
                    first_example = self.stm[cent_idx].examples[0]
                    self.stm[cent_idx].examples = [first_example]

            return True, len(mature_cent_inds)

        elif self.no_update_steps != -1:
            self.no_update_steps += 1

        return False, 0

    def _select_examples(self, examples: list) -> list:
        """
        Select examples to keep when merging centroids.

        Args:
            examples: list of Example objects
            max_count: maximum number of examples to keep
            mode: 'gradient' | 'balanced' | 'random'

        Returns:
            Selected list of examples (len <= max_count)
        """
        max_count = self.max_examples_per_centroid
        mode = self.example_selection_mode  # 'gradient' | 'balanced' | 'random'
        if len(examples) <= max_count:
            return examples

        if mode == 'random':
            return rnd.sample(examples, max_count)

        elif mode == 'gradient':
            return sorted(examples, key=lambda ex: ex.gradient, reverse=True)[:max_count]

        elif mode == 'balanced':
            # Group examples by arrival_time
            groups = {}
            for ex in examples:
                groups.setdefault(ex.arrival_time, []).append(ex)

            # Each group gets a quota proportional to its size
            # Start by giving each group floor(quota), then distribute remainders
            total = len(examples)
            keys = list(groups.keys())

            # Compute exact floating quota per group
            quotas = {k: (len(groups[k]) / total) * max_count for k in keys}

            # Floor quotas, track remainders
            floor_quotas = {k: int(quotas[k]) for k in keys}
            remainders = {k: quotas[k] - floor_quotas[k] for k in keys}

            # Distribute remaining slots to groups with highest remainder
            allocated = sum(floor_quotas.values())
            leftover = max_count - allocated
            sorted_by_remainder = sorted(keys, key=lambda k: remainders[k], reverse=True)
            for i in range(leftover):
                floor_quotas[sorted_by_remainder[i]] += 1

            # Select from each group up to its quota
            selected = []
            for k in keys:
                quota = floor_quotas[k]
                if quota > 0:
                    selected.extend(groups[k][:quota])

            return selected

        else:
            raise ValueError(f"Unknown selection mode: '{mode}'. Use 'gradient', 'balanced', or 'random'.")
    def _merge_ltm_centroids(self) -> dict:
        """Merge most similar LTM centroids."""
        from torchmetrics.functional import pairwise_cosine_similarity
        import torchvision.utils as vutils

        ltm_simil_no_diag = self.ltm_simil.clone()
        ltm_simil_no_diag.fill_diagonal_(-float('inf'))

        c_1, c_2 = torch.unravel_index(
            torch.argmax(ltm_simil_no_diag),
            ltm_simil_no_diag.shape
        )
        c_1, c_2 = c_1.item(), c_2.item()

        if c_1 > c_2:
            c_1, c_2 = c_2, c_1

        # Create merged centroid
        new_embedding = (self.ltm[c_1].embedding + self.ltm[c_2].embedding) / 2
        merged_centroid = Centroid(new_embedding, max_examples=self.max_examples_per_centroid)

        # Merge examples
        all_examples = self.ltm[c_1].examples + self.ltm[c_2].examples
        if len(all_examples) <= self.max_examples_per_centroid:
            merged_centroid.examples = all_examples
        else:
            merged_centroid.examples = self._select_examples(
                all_examples
            )

        # Update c_1
        self.ltm[c_1] = merged_centroid

        # Remove c_2
        self.ltm.remove_centroid(c_2)

        # Update similarity matrix
        self.ltm_simil = torch.cat([
            torch.cat([self.ltm_simil[:c_2, :c_2], self.ltm_simil[:c_2, c_2 + 1:]], dim=1),
            torch.cat([self.ltm_simil[c_2 + 1:, :c_2], self.ltm_simil[c_2 + 1:, c_2 + 1:]], dim=1)
        ], dim=0)

        # Recompute similarities for merged centroid
        ltm_embeddings = self.ltm.get_embeddings(device=self.device)
        updated_simil = pairwise_cosine_similarity(
            ltm_embeddings[c_1:c_1 + 1], ltm_embeddings
        ).cpu().squeeze()
        self.ltm_simil[c_1, :] = updated_simil
        self.ltm_simil[:, c_1] = updated_simil

        return {'c_1': c_1, 'c_2': c_2}

    def sample_replay(self, batch_size: int, ratio_ltm: float = 0.5) :
        """
        Sample examples for replay from STM and LTM.

        Args:
            batch_size: Number of samples to retrieve
            ratio_ltm: Ratio of LTM samples (0 to 1)

        Returns:
            Tuple of (samples, ltm_indices, stm_indices)
        """
        ltm_examples_lists = self.ltm.get_all_examples()
        stm_examples_lists = self.stm.get_all_examples()

        total_ltm = sum(len(s) for s in ltm_examples_lists)
        total_stm = sum(len(s) for s in stm_examples_lists)

        if total_ltm + total_stm < batch_size:
            return [], [], []

        if ratio_ltm == -1:
            # Random global sampling
            all_examples = ltm_examples_lists + stm_examples_lists
            n_ltm_lists = len(ltm_examples_lists)

            extra_all, all_centroids, all_indices = self._random_batch_from_lists(all_examples, batch_size)

            replay_indices_ltm = [(c,i) for c,i in zip(all_centroids, all_indices) if c < n_ltm_lists]
            replay_indices_stm = [(c - n_ltm_lists,i) for c,i in zip(all_centroids, all_indices) if c >= n_ltm_lists]

            return extra_all, replay_indices_ltm, replay_indices_stm
        else:
            # Fixed ratio sampling
            ltm_bs = int(ratio_ltm * batch_size)
            stm_bs = batch_size - ltm_bs

            # Adjust if not enough
            if ltm_bs > total_ltm:
                stm_bs += ltm_bs - total_ltm
                ltm_bs = total_ltm

            if stm_bs > total_stm:
                ltm_bs += stm_bs - total_stm
                stm_bs = total_stm

            ltm_bs = min(ltm_bs, total_ltm)
            stm_bs = min(stm_bs, total_stm)

            extra = []
            replay_indices_ltm = []
            replay_indices_stm = []

            if ltm_bs > 0:
                extra_ltm, ltm_centroids, all_indices = self._random_batch_from_lists(ltm_examples_lists, ltm_bs)
                extra.extend(extra_ltm)
                replay_indices_ltm = list(zip(ltm_centroids, all_indices))

            if stm_bs > 0:
                extra_stm, stm_centroids, all_indices = self._random_batch_from_lists(stm_examples_lists, stm_bs)
                extra.extend(extra_stm)
                replay_indices_stm = list(zip(stm_centroids, all_indices))

            return extra, replay_indices_ltm, replay_indices_stm

    def update_centroid_embeddings(self, avg_features, stm_indices, ltm_indices, gradient: Optional[torch.Tensor] = None):
        """Update centroid embeddings with new features from replay."""
        # Update STM
        if len(stm_indices) > 0:
            stm_embeddings = self.stm.get_embeddings(device=self.device)
            for replay_id, (stm_id, example_id) in enumerate(stm_indices):
                alpha = 0.5 * (1 / len(self.stm[stm_id]))
                stm_embeddings[stm_id] = (1 - alpha) * stm_embeddings[stm_id] + alpha * avg_features[replay_id]
            self.stm.set_embeddings(stm_embeddings)

        # Update LTM
        if len(ltm_indices) > 0:
            ltm_embeddings = self.ltm.get_embeddings(device=self.device)
            for replay_id, (ltm_id, example_id) in enumerate(ltm_indices):
                alpha = 0.5 * (1 / len(self.ltm[ltm_id]))
                ltm_embeddings[ltm_id] = (1 - alpha) * ltm_embeddings[ltm_id] + alpha * avg_features[
                    len(stm_indices) + replay_id]
            self.ltm.set_embeddings(ltm_embeddings)

    def _random_batch_from_lists(self, list_of_lists, batch_size):
        """Sample batch from list of lists."""
        total = sum(len(s) for s in list_of_lists)
        if batch_size > total:
            raise ValueError("batch_size too large")

        indices = random.sample(range(total), k=batch_size)

        batch = []
        centroid_indices = []
        example_indices = []
        for idx in indices:
            r = idx
            for centroid_id, s in enumerate(list_of_lists):
                if r < len(s):
                    batch.append(s[r])
                    centroid_indices.append(centroid_id)
                    example_indices.append(r)
                    break
                r -= len(s)

        return batch, centroid_indices, example_indices

    def update_embeddings_from_model(self, embed_fn, update_use: int, update_use_all: bool, reset_stm: bool = False):
        """
        Update embeddings by re-encoding examples.

        Args:
            embed_fn: Function to embed examples
            update_use: Number of examples to use per centroid
            update_use_all: Whether to use all examples
            reset_stm: Whether to reset STM examples
        """
        stm_examples_lists = self.stm.get_all_examples()
        ltm_examples_lists = self.ltm.get_all_examples()

        stm_sizes = np.array([len(ex) for ex in stm_examples_lists])

        # Update STM
        if not reset_stm:
            with torch.no_grad():
                if update_use_all:
                    new_stm_embeddings = torch.stack(
                        [torch.mean(embed_fn(torch.stack(stm_examples_lists[i]).to(self.device)), dim=0)
                         for i in np.argwhere(stm_sizes >= update_use).flatten()])
                else:
                    new_stm_embeddings = torch.stack(
                        [torch.mean(embed_fn(torch.stack(stm_examples_lists[i])[:update_use].to(self.device)), dim=0)
                         for i in np.argwhere(stm_sizes >= update_use).flatten()])
            self.stm.set_embeddings(new_stm_embeddings)
            del new_stm_embeddings

        # Update LTM
        with torch.no_grad():
            if update_use_all:
                new_ltm_embeddings = torch.stack(
                    [torch.mean(embed_fn(torch.stack(ex).to(self.device)), dim=0)
                     for ex in ltm_examples_lists])
            else:
                new_ltm_embeddings = torch.stack(
                    [torch.mean(embed_fn(torch.stack(ex)[:update_use].to(self.device)), dim=0)
                     for ex in ltm_examples_lists])
        self.ltm.set_embeddings(new_ltm_embeddings)
        del new_ltm_embeddings

        # Reset examples
        if reset_stm:
            for i in range(len(self.stm)):
                self.stm[i].examples = []
        else:
            for i in range(len(self.stm)):
                if len(self.stm[i].examples) > 0:
                    first_example = self.stm[i].examples[0]
                    self.stm[i].examples = [first_example]

        # Reset matches
        stm_matches = self.stm.get_matches(device='cpu')
        stm_matches[:] = 0
        self.stm.set_matches(stm_matches)

    def get_statistics(self) -> dict:
        """Get memory statistics."""
        stm_examples_lists = self.stm.get_all_examples()
        ltm_examples_lists = self.ltm.get_all_examples()

        return {
            'stm_size': len(self.stm),
            'ltm_size': len(self.ltm),
            'stm_examples': [len(ex) for ex in stm_examples_lists],
            'ltm_examples': [len(ex) for ex in ltm_examples_lists],
            'total_stm_examples': sum(len(ex) for ex in stm_examples_lists),
            'total_ltm_examples': sum(len(ex) for ex in ltm_examples_lists),
            'distance_threshold': self.distance_threshold,
            'nb_novelties': self.nb_novelties,
            'nb_ltm_match': self.nb_ltm_match,
            'step': self.step
        }

    def trim_to_size(self, max_size: int):
        """Trim memory to max_size by removing examples from the oldest centroids.

        For each iteration, finds the centroid (STM or LTM) with the oldest
        first example that still has at least 2 examples, then removes all
        examples except the first. Repeats until total size <= max_size.
        """
        while True:
            total = sum(len(c.examples) for c in self.stm.centroids) + \
                    sum(len(c.examples) for c in self.ltm.centroids)
            if total <= max_size:
                break

            # Collect all centroids with >= 2 examples and their oldest arrival_time
            candidates = []
            for c in self.stm.centroids:
                if len(c.examples) >= 2:
                    candidates.append((c.age, c))


            if not candidates:
                break

            # Pick the oldest centroid
            oldest_centroid = max(candidates, key=lambda x: x[0])[1]
            oldest_centroid.examples = oldest_centroid.examples[:-min(len(oldest_centroid.examples)-1,  total-max_size)]

    @staticmethod
    def _percentile(array, percent):
        """Calculate percentile."""
        import math
        k = (len(array) - 1) * percent
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return array[int(k)]
        d0 = array[int(f)] * (c - k)
        d1 = array[int(c)] * (k - f)
        return d0 + d1