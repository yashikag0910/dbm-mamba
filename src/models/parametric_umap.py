import torch
import torch.nn as nn
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class ParametricUMAPLoss(nn.Module):
    """
    Differentiable Parametric UMAP Loss module.
    Computes k-NN graph on input batch, calculates fuzzy membership v_ij,
    projects to latent space, calculates latent membership w_ij, and
    minimizes the fuzzy cross-entropy.
    """
    def __init__(self, k: int = 15, a: float = 1.58, b: float = 0.895, eps: float = 1e-4):
        super().__init__()
        self.k = k
        self.a = a
        self.b = b
        self.eps = eps

    def compute_pairwise_distances(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise Euclidean distance matrix for a batch X.
        X shape: (B, D_feat)
        Returns: (B, B)
        """
        # Using torch.cdist which is natively stable and optimized in PyTorch
        return torch.cdist(X, X, p=2.0)

    def find_knn(self, D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Finds the k nearest neighbors for each sample in the batch.
        D shape: (B, B)
        Returns:
            knn_indices: (B, k)
            knn_dists: (B, k)
        """
        B = D.shape[0]
        actual_k = min(self.k + 1, B) # +1 because we exclude self
        
        # If FAISS is available and batch is very large, we can use it, 
        # but for PyTorch autograd and GPU/MPS batch training, standard torch.topk is faster.
        # We will use torch.topk on the distance matrix directly.
        knn_dists, knn_indices = torch.topk(D, k=actual_k, largest=False, dim=1)
        
        # Exclude self-distance (which is the first column since distance to self is 0)
        return knn_indices[:, 1:], knn_dists[:, 1:]

    def solve_sigmas(self, knn_dists: torch.Tensor, rho: torch.Tensor, target_val: float, num_iters: int = 15) -> torch.Tensor:
        """
        Solves for sigma_i for each row using binary search such that:
        sum_j exp(-(d(x_i, x_j) - rho_i) / sigma_i) = target_val
        """
        B, k_dim = knn_dists.shape
        device = knn_dists.device
        
        # Initialize binary search limits
        low = torch.zeros(B, device=device)
        high = torch.ones(B, device=device) * 100.0
        
        # Find a suitable initial upper bound
        for _ in range(3):
            # Compute current sum
            sigmas = high.unsqueeze(-1)
            diffs = knn_dists - rho.unsqueeze(-1)
            val = torch.sum(torch.exp(-diffs / (sigmas + 1e-8)), dim=1)
            under = val < target_val
            high[under] = high[under] * 2.0
            
        # Binary search
        for _ in range(num_iters):
            sigmas = (low + high) / 2.0
            diffs = knn_dists - rho.unsqueeze(-1)
            val = torch.sum(torch.exp(-diffs / (sigmas.unsqueeze(-1) + 1e-8)), dim=1)
            
            over = val > target_val
            low[over] = sigmas[over]
            high[~over] = sigmas[~over]
            
        return (low + high) / 2.0

    def compute_fuzzy_membership(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the fuzzy membership matrix V in the input space.
        V[i, j] = exp(-(d(x_i, x_j) - rho_i) / sigma_i) for neighbors.
        Returns: (B, B)
        """
        B = X.shape[0]
        device = X.device
        
        # Flatten sequence input if necessary
        if len(X.shape) == 3:
            X_flat = X.view(B, -1)
        else:
            X_flat = X
            
        D = self.compute_pairwise_distances(X_flat)
        knn_indices, knn_dists = self.find_knn(D)
        
        # rho_i is the distance to the nearest neighbor (1st neighbor in the sorted list)
        rho = knn_dists[:, 0]
        
        target_val = np.log2(self.k)
        sigmas = self.solve_sigmas(knn_dists, rho, target_val)
        
        # Compute fuzzy membership matrix V
        V = torch.zeros(B, B, device=device)
        
        for i in range(B):
            idx = knn_indices[i]
            dists = knn_dists[i]
            # v_ij = exp(-(d_ij - rho_i) / sigma_i)
            v_val = torch.exp(-(dists - rho[i]) / (sigmas[i] + 1e-8))
            V[i, idx] = v_val
            
        # Symmetrize the membership matrix: V = V + V^T - V * V^T
        V_sym = V + V.t() - V * V.t()
        return V_sym

    def forward(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: input tensor of shape (B, seq_len, num_features) or (B, feat_dim)
            z: latent embeddings from encoder of shape (B, d_model)
        Returns:
            loss: fuzzy cross-entropy UMAP loss
        """
        B = X.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
            
        # 1. Compute input fuzzy memberships V (frozen targets, no grads needed)
        with torch.no_grad():
            V = self.compute_fuzzy_membership(X)
            
        # 2. Compute latent pairwise distances (differentiable!)
        D_z = self.compute_pairwise_distances(z)
        
        # Apply Diagonal Gradient Fix: set diagonal to 1.0 to avoid zero-distance gradient NaN
        eye = torch.eye(B, device=z.device)
        D_z_fixed = D_z * (1.0 - eye) + eye * 1.0
        
        # 3. Compute latent membership w_ij
        # w_ij = 1 / (1 + a * d_z_ij^(2b))
        w = 1.0 / (1.0 + self.a * (torch.clamp(D_z_fixed, min=1e-4) ** (2.0 * self.b)) + 1e-8)
        
        # 4. Fuzzy Cross Entropy Loss
        # Clamp to avoid log(0)
        w_clipped = torch.clamp(w, min=self.eps, max=1.0 - self.eps)
        
        # Loss: -V * log(W) - (1-V) * log(1-W)
        loss = - V * torch.log(w_clipped) - (1.0 - V) * torch.log(1.0 - w_clipped)
        
        # Mask out diagonal elements in loss computation
        mask = ~eye.bool()
        loss = loss[mask]
        
        # Return average over all off-diagonal pairs
        return loss.mean()
