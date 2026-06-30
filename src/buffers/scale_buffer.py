import numpy as np
from sklearn.metrics import pairwise_distances

import torch


import diversipy


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1


class Memory(object):
    def __init__(self,
                 mem_update_type='mo_rdn',
                 mem_size=2000,
                 mem_max_classes=10,
                 mem_max_new_ratio=0.1,
                 device = "cpu"
                 ):
        """
        Initialize memory.
        
        Args:
            mem_max_classes (int): Maximum number of (pseudo-)classes to store in memory.
            size_per_class (int): Number of samples to store per class in memory. 
            mem_update_type (str): Memory update strategy. Can be one of:
                - 'rdn': Random selection
                - 'mo_rdn': Momentum random selection
                - 'reservoir': Reservoir sampling
                - 'simil': Similarity-based selection
            mem_update_class_based (bool): Whether to cluster and update memory separately per class.
            mem_max_new_ratio (float): Maximum ratio of new samples if 'mo_rdn' update type is used. 
    """

        self.max_classes = mem_max_classes
        self.max_size = mem_size
        self.size_per_class = self.max_size // self.max_classes
        self.mem_update_type = mem_update_type
        self.max_new_ratio = mem_max_new_ratio
        self.device = device

        self.images = []  # A list of numpy arrays
        self.labels_set = []  # Pseud labels assisting memory update
        self.true_labels = []  # Same organization as self.images for true labels record
        self.update_cnt = 0
        self.num_seen_examples = 0
        self.replay_batch_size=0

    def sampling(self, lb, old_sz, new_sz, sz_per_lb):
        """
        Implementation of various sampling methods.

        Args:
            lb: int, ground-truth label or pseudo label of the class
            old_sz: int, size of old data samples
            new_sz: int, size of new data samples
            sz_per_lb: int, upperbound on size of samples per label/class,
                take self.size_per_class with class-based sampling,
                take self.max_size without class-based sampling

        Return:
            select_ind: numpy array of the list of indices that are selected in
                the ind th memory bin
        """
        ind = self.labels_set.index(lb)
        select_ind = np.arange(old_sz + new_sz)
        # Memory Update - sample selection
        if old_sz + new_sz > sz_per_lb:
            if self.mem_update_type == 'rdn':
                select_ind = np.random.choice(old_sz + new_sz, sz_per_lb,
                                              replace=False)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.mem_update_type == 'mo_rdn':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                    sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)
                select_ind_old = np.random.choice(old_sz, num_old_samples,
                                                  replace=False)
                select_ind_new = old_sz + np.random.choice(new_sz, num_new_samples,
                                                           replace=False)
                select_ind = np.concatenate((select_ind_old, select_ind_new), axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.mem_update_type == 'reservoir':
                select_ind = list(np.arange(sz_per_lb))
                cur_ind = np.arange(sz_per_lb)  # Use to record the original index
                for i in range(sz_per_lb, old_sz + new_sz):
                    # i corresponds to the extra portion
                    index = reservoir(self.num_seen_examples, sz_per_lb)
                    if index >= 0:
                        self.images[ind][index] = self.images[ind][i]
                        self.true_labels[ind][index] = self.true_labels[ind][i]
                        select_ind.remove(cur_ind[index])
                        cur_ind[index] = i
                        select_ind.append(i)

                self.images[ind] = self.images[ind][:sz_per_lb]
                self.true_labels[ind] = self.true_labels[ind][:sz_per_lb]
                select_ind = np.array(select_ind)
            elif self.mem_update_type == 'simil':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                                      sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)

                simil_sum = np.sum(self.similarity_matrix[ind], axis=1)
                select_ind_old = (-simil_sum[:old_sz]).argsort()[:num_old_samples]
                select_ind_new = old_sz + (-simil_sum[old_sz:]).argsort()[:num_new_samples]

                select_ind = np.concatenate((select_ind_old, select_ind_new),
                                            axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            else:
                raise ValueError(
                    'memory update policy not supported: {}'.format(self.mem_update_type))

        return select_ind

    def update_w_labels(self, new_images, new_labels):
        """
        Update memory samples.
        No need to check the number of classes all_embeddings = []
for i in range(0, len(feed_images), batch_size):
    with torch.no_grad():
        batch = feed_images[i:i+batch_size].to(self.device)
        all_embeddings.append(model(batch).detach().cpu().numpy())
all_embeddings = np.concatenate(all_embeddings, axis=0)if labels are provided.
        Args:
            new_images: torch array, new incoming images
            new_labels: torch array, new ground-truth labels
        """
        new_images = new_images.detach().numpy()
        new_labels = new_labels.detach().numpy()
        new_labels_set = set(new_labels)
        self.num_seen_examples += new_images.shape[0]

        for lb in new_labels_set:
            new_ind = (np.array(new_labels) == lb)
            new_sz = np.sum(new_ind)
            if lb in self.labels_set:  # already seen
                ind = self.labels_set.index(lb)
                old_sz = self.images[ind].shape[0]
                self.images[ind] = np.concatenate(
                    (self.images[ind], new_images[new_ind]),
                    axis=0)
                self.true_labels[ind] = np.concatenate(
                    (self.true_labels[ind], new_labels[new_ind]),
                    axis=0)
            else:  # first-time seen labels
                self.labels_set.append(lb)
                old_sz = 0
                self.images.append(new_images[new_ind])
                self.true_labels.append(new_labels[new_ind])

            # Memory update - sample selection
            # The key is transfer lb - the ground-truth label,
            # and sz_per_lb - size upperbound for each class
            self.sampling(lb, old_sz, new_sz, self.size_per_class)

    def update_wo_labels(self, new_images, model=None):
        """
        Update memory samples.
        Args:
            new_images: torch array, new incoming images
            model: network model being trained, used in kmeans and spectral cluster type

        Return:
            select_indices: numpy array of selected indices in all_images
        """
        new_images = new_images.detach().numpy()
        self.num_seen_examples += new_images.shape[0]

        if len(self.images) > 0:  # Not first-time insertion
            old_images = np.concatenate(self.images)
            old_sz = old_images.shape[0]
            #old_images = torch.from_numpy(old_images)

            all_images = np.concatenate((old_images, new_images), axis=0)
        else:  # first-time insertion
            old_sz = 0
            all_images = new_images

        # Create a binary indicator of whether the image is an old or new sample
        old_ind = np.zeros(all_images.shape[0], dtype=bool)
        old_ind[:old_sz] = 1

        # Get latent embeddings
        feed_images = torch.from_numpy(all_images).to(self.device, non_blocking=True)
        # ATTENTION! ADDITIONAL FORWARD PASS FOR ALL MEMORY SAMPLES! IS THIS TRICK ILLEGAL?
        all_embeddings = []
        batch_size = min(self.replay_batch_size+new_images.shape[0],len(feed_images))
        with torch.no_grad():
            for i in range(0, len(all_images), batch_size):
                batch = torch.from_numpy(all_images[i:i+batch_size]).to(self.device, non_blocking=True)
        
                z = model(batch)            
                all_embeddings.append(z.cpu())  
        
                del batch, z
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        # all_embeddings_mean = np.mean(all_embeddings, axis=0, keepdims=True)
        # all_embeddings = (all_embeddings - all_embeddings_mean) * 1e4

        # PSA clustering
        # Clustering
        simil_matrix = tsne_simil(all_embeddings, metric='cosine')

        # Init selected indices as all indices
        select_indices = np.arange(all_embeddings.shape[0])

        if all_embeddings.shape[0] > self.max_size:  # needs subset selection
            selected_embeddings = diversipy.subset.psa_select(all_embeddings, self.max_size) # psa_select() returns already selected embeddings, not indices
            select_indices = np.where(np.all(all_embeddings[:, None, :] == selected_embeddings[None, :, :], axis=-1).any(axis=1))[0] # convert embeddings to indices
            select_indices.sort()

        self.images = [all_images[select_indices]]
        self.labels_set = [0]


        return all_embeddings, select_indices


    def get_mem_samples(self):
        """
        Combine all stored samples and pseudo labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all pseudo labels, (sample #, pseudo label)
        If updated with update_w_labels, the returned labels are the ground-truth labels.
        If updated with update_wo_labels, the returned labels are the pseudo labels.
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = np.repeat(lb, self.images[0].shape[0])
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(labels)

    def get_mem_samples_w_true_labels(self):
        """
        Combine all stored samples and true labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all true labels, (sample #, true label)
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = self.true_labels[ind]
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)
                labels = np.concatenate((labels, self.true_labels[ind]), axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(np.array(labels))
        
    # New Sample method
    # Returns None when void buffer, otherwise returns samples
    def sample(self, replay_batch_size):
        if self.replay_batch_size==0:
            self.replay_batch_size=replay_batch_size
        if len(self.images) > 0:
            mem_images = np.concatenate(self.images, axis=0)
            mem_len = mem_images.shape[0]
            sample_cnt = min(mem_len, replay_batch_size)
            select_ind = np.random.choice(range(mem_len), sample_cnt, replace=False)

            return torch.from_numpy(mem_images[select_ind])
        else:
            return None
        

def tsne_simil(x, metric='euclidean', sigma=1.0):
    dist_matrix = pairwise_distances(x, metric=metric)
    cur_sim = np.divide(- dist_matrix, 2 * sigma ** 2)
    # print(np.sum(cur_sim, axis=1, keepdims=True))

    # mask-out self-contrast cases
    # the diagonal elements of exp_logits should be zero
    logits_mask = np.ones((x.shape[0], x.shape[0]))
    np.fill_diagonal(logits_mask, 0)
    # print(logits_mask)
    exp_logits = np.exp(cur_sim) * logits_mask
    # print(exp_logits.shape)
    # print(np.sum(exp_logits, axis=1, keepdims=True))

    p = np.divide(exp_logits, np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    p = p + p.T
    p /= 2 * x.shape[0]
    return p