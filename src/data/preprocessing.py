import torch
import numpy as np

class ZScoreNormalizer:
    """
    Fits and stores mean and standard deviation for Z-score normalization of IoT flow data.
    """
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.mean = torch.zeros(num_features)
        self.std = torch.ones(num_features)
        self.is_fitted = False

    def fit(self, X: torch.Tensor):
        # X: (N, seq_len, num_features) or (N, num_features)
        if len(X.shape) == 3:
            flat_X = X.view(-1, self.num_features)
        else:
            flat_X = X
        self.mean = torch.mean(flat_X, dim=0)
        self.std = torch.std(flat_X, dim=0)
        # Avoid division by zero
        self.std[self.std < 1e-6] = 1.0
        self.is_fitted = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, seq_len, num_features) or (N, num_features)
        # Standardize using stored mean/std
        mean = self.mean.to(X.device)
        std = self.std.to(X.device)
        return (X - mean) / std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def state_dict(self):
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'is_fitted': self.is_fitted
        }

    def load_state_dict(self, state):
        self.mean = torch.tensor(state['mean'])
        self.std = torch.tensor(state['std'])
        self.is_fitted = state['is_fitted']


def log_compact(X: torch.Tensor, indices: list = [0, 2]) -> torch.Tensor:
    """
    Applies f(x) = log(1 + x) compaction to specific features (e.g., packet length and IAT).
    Default indices 0 (length) and 2 (IAT).
    """
    X_compacted = X.clone()
    for idx in indices:
        if len(X.shape) == 3:
            X_compacted[:, :, idx] = torch.log1p(torch.clamp(X[:, :, idx], min=0.0))
        elif len(X.shape) == 2:
            X_compacted[:, idx] = torch.log1p(torch.clamp(X[:, idx], min=0.0))
    return X_compacted


def mask_sequence(X: torch.Tensor, mask_ratio: float = 0.15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the masking operator M(X) for pretraining.
    Masks ~15% of packet tokens (across the sequence length dimension).
    
    Args:
        X: original tensor of shape (batch_size, seq_len, num_features)
        mask_ratio: ratio of tokens to mask
        
    Returns:
        masked_X: tensor of shape (batch_size, seq_len, num_features) with masked positions set to 0.0
        X: original input tensor
        mask: boolean mask of shape (batch_size, seq_len) where True indicates masked positions
    """
    batch_size, seq_len, num_features = X.shape
    
    # Generate random masking indices
    rand = torch.rand(batch_size, seq_len, device=X.device)
    mask = rand < mask_ratio
    
    # Ensure at least one token is masked per sequence to prevent trivial loss
    # (if none are masked, force mask the first element)
    no_masked = torch.sum(mask, dim=1) == 0
    if torch.any(no_masked):
        mask[no_masked, 0] = True
        
    masked_X = X.clone()
    # Apply masking by zeroing out the features at masked positions
    masked_X[mask] = 0.0
    
    return masked_X, X, mask
