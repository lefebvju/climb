import random
import torch


class FIFOBuffer:
    """
    Custom FIFO buffer class for batches of samples without labels, but with encoder features.
    FIFO means that oldest samples are replaced with new ones on each add call, sampling remains random.
    """
    def __init__(self, buffer_size, alpha_ema=1.0):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = [] # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # Buffer for corresponding sample features
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features


    # Add a batch of samples and features to the buffer
    def add(self, batch_x, batch_features):
        assert batch_x.size(0) == batch_features.size(0)
        # Adds batch with a FIFO strategy

        self.buffer.extend(batch_x)
        self.buffer_features.extend(batch_features)

        if len(self.buffer) > self.buffer_size:
            # Remove oldest samples
            self.buffer = self.buffer[-self.buffer_size:]
            self.buffer_features = self.buffer_features[-self.buffer_size:]

    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]
        features = [self.buffer_features[i] for i in indices]

        # Reconstruct batch from samples
        batch_x = torch.stack([sample for sample in samples])
        batch_features = torch.stack([feature for feature in features])

        return batch_x, batch_features, indices
    
    def state_dict(self) -> dict:
        return {
            'buffer': [x.cpu() if isinstance(x, torch.Tensor) else x for x in self.buffer],
            'buffer_features': [x.cpu() if isinstance(x, torch.Tensor) else x for x in self.buffer_features],
        }

    def load_state_dict(self, state: dict):
        self.buffer = state['buffer']
        self.buffer_features = state['buffer_features']

    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices):
        assert batch_features.size(0) == len(indices)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]
