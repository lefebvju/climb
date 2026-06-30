import random
import torch



class MinRedBuffer:
    """
    MinRed buffer class for batches of samples without labels.
    """
    def __init__(self, buffer_size, alpha_ema=0.5, device='cpu'):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.alpha_ema = alpha_ema
        self.buffer = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.device = device

    # Add a batch of samples to the buffer
    def add(self, batch_x, batch_features):
        assert batch_x.size(0) == batch_features.size(0)

        batch_x, batch_features = batch_x.to(self.device), batch_features.to(self.device)

        # Initialize empty buffers
        if self.buffer.size(0) == 0:
            # Extend buffer to have same dim of batch_x
            buffer_shape = list(batch_x.size())
            buffer_shape[0] = 0
            self.buffer = torch.empty(buffer_shape).to(self.device)

            # Extend buffer_features to have same dim of batch_features
            buffer_shape = list(batch_features.size())
            buffer_shape[0] = 0
            self.buffer_features = torch.empty(buffer_shape).to(self.device)

        batch_size = batch_x.size(0)
        n_excess = len(self.buffer) + batch_size - self.buffer_size

        # Remove n_excess samples
        if n_excess > 0:
            # Buffer is full
            for _ in range(n_excess):
                # Cosine distance = 1 - cosine similarity
                tensor_normalized = torch.nn.functional.normalize(self.buffer_features, p=2, dim=1)
                d = 1- torch.mm(tensor_normalized, tensor_normalized.t())
                # Set d diagonal to 1 (maximum distance for cosine distance)
                d = d.fill_diagonal_(1.0)

                # Nearest neighbor for each sample
                nearneigh, _ = torch.min(d, dim=1)
                # Minimum distance in d matrix
                _, min_indices = torch.min(nearneigh, dim=0)
                
                # Get index of sample to remove
                idx_to_remove = min_indices.item()
                self.buffer = torch.cat((self.buffer[:idx_to_remove], self.buffer[idx_to_remove + 1:]), dim=0)
                self.buffer_features = torch.cat((self.buffer_features[:idx_to_remove], self.buffer_features[idx_to_remove + 1:]), dim=0)

        # Add samples to buffer
        self.buffer = torch.cat((self.buffer, batch_x), dim=0)        
        self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)


    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)

        # Get sample batch from indices
        batch_x = self.buffer[indices]
        batch_features = self.buffer_features[indices]

        return batch_x, batch_features, indices
    
    def state_dict(self) -> dict:
        return {
            'buffer': self.buffer.cpu(),
            'buffer_features': self.buffer_features.cpu(),
        }

    def load_state_dict(self, state: dict):
        self.buffer = state['buffer'].to(self.device)
        self.buffer_features = state['buffer_features'].to(self.device)

    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices):
        assert batch_features.size(0) == len(indices)

        batch_features = batch_features.to(self.device)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]

