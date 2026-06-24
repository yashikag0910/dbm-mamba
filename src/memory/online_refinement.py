import torch
import torch.nn as nn
import torch.optim as optim

class OnlineMemoryRefiner:
    """
    Stage 6: Online Triplet Loss adaptation for memory banks.
    Computes adaptive margin, performs semi-hard negative mining,
    and updates prototypes using EMA and Triplet Loss gradients.
    """
    def __init__(self, beta: float = 0.01, lr: float = 0.01):
        self.beta = beta
        self.lr = lr

    def compute_adaptive_margin(self, centroids: torch.Tensor, default_margin: float = 0.5) -> float:
        """
        Computes alpha_d = MeanInterCentroidDistance(C_d).
        If K=1, returns default_margin.
        """
        K = centroids.shape[0]
        if K <= 1:
            return default_margin
            
        # Compute all pairwise distances between centroids
        # centroids shape: (K, d_model)
        r = torch.sum(centroids * centroids, dim=1, keepdim=True)
        D_sq = r - 2.0 * torch.matmul(centroids, centroids.t()) + r.t()
        D_sq = torch.clamp(D_sq, min=0.0)
        dists = torch.sqrt(D_sq + 1e-8)
        
        # Mean of non-zero entries (off-diagonal elements)
        sum_dists = torch.sum(dists)
        num_pairs = K * (K - 1)
        mean_dist = sum_dists / num_pairs
        return float(mean_dist.item())

    def mine_semi_hard_negative(self, z: torch.Tensor, centroids: torch.Tensor, 
                                 same_idx: int, alpha: float) -> int:
        """
        Selects a semi-hard negative centroid c_other from the centroids.
        Semi-hard negative condition:
        ||z - c_same||^2 < ||z - c_other||^2 < ||z - c_same||^2 + alpha
        
        If no semi-hard negative is found, falls back to the nearest negative.
        """
        K = centroids.shape[0]
        if K <= 1:
            return -1
            
        c_same = centroids[same_idx]
        d_same_sq = torch.sum((z - c_same) ** 2).item()
        
        candidates = []
        fallback_idx = -1
        min_fallback_diff = float('inf')
        
        for idx in range(K):
            if idx == same_idx:
                continue
                
            c_other = centroids[idx]
            d_other_sq = torch.sum((z - c_other) ** 2).item()
            
            # Condition check
            if d_same_sq < d_other_sq < (d_same_sq + alpha):
                candidates.append((idx, d_other_sq))
            else:
                # Track nearest negative for fallback
                diff = abs(d_other_sq - (d_same_sq + alpha))
                if diff < min_fallback_diff:
                    min_fallback_diff = diff
                    fallback_idx = idx
                    
        if candidates:
            # Pick the one closest to the upper bound (semi-hard negative mining)
            candidates.sort(key=lambda val: val[1], reverse=True)
            return candidates[0][0]
            
        return fallback_idx

    def refine_step(self, z: torch.Tensor, memory_bank, other_banks: list = None) -> tuple[float, int, int]:
        """
        Performs online refinement for a single benign sample z.
        1. Find c_same (closest centroid in memory_bank.centroids).
        2. Mine c_other (semi-hard negative).
        3. Compute triplet loss and update centroids via gradient descent.
        4. Apply EMA update to c_same.
        """
        if memory_bank.centroids is None:
            return 0.0, -1, -1
            
        z = z.detach() # Ensure no encoder gradients
        centroids = memory_bank.centroids
        K = centroids.shape[0]
        
        # 1. Find c_same
        dists = torch.norm(z.unsqueeze(0) - centroids, dim=1)
        same_idx = int(torch.argmin(dists).item())
        
        # 2. Get negative centroids (other_centroids)
        # If K=1, search other_banks for negative centroids
        if K <= 1 and other_banks is not None:
            neg_candidates = []
            for ob in other_banks:
                if ob.centroids is not None and ob.device_class_id != memory_bank.device_class_id:
                    neg_candidates.append(ob.centroids)
            if neg_candidates:
                neg_centroids = torch.cat(neg_candidates, dim=0)
            else:
                neg_centroids = centroids # self fallback
        else:
            neg_centroids = centroids
            
        # Compute adaptive margin
        alpha = self.compute_adaptive_margin(neg_centroids)
        
        # 3. Mine semi-hard negative index
        other_idx = self.mine_semi_hard_negative(z, neg_centroids, same_idx if K > 1 else -1, alpha)
        
        loss_val = 0.0
        # 4. Triplet loss update
        if other_idx != -1:
            # Create copies with gradient tracking
            c_same_param = centroids[same_idx].clone().detach().requires_grad_(True)
            c_other_param = neg_centroids[other_idx].clone().detach().requires_grad_(True)
            
            d_same_sq = torch.sum((z - c_same_param) ** 2)
            d_other_sq = torch.sum((z - c_other_param) ** 2)
            
            loss = torch.clamp(d_same_sq - d_other_sq + alpha, min=0.0)
            loss_val = float(loss.item())
            
            if loss.item() > 0:
                loss.backward()
                
                # Apply gradient step manually
                with torch.no_grad():
                    centroids[same_idx] -= self.lr * c_same_param.grad
                    if K > 1:
                        centroids[other_idx] -= self.lr * c_other_param.grad
                        
        # 5. EMA update to c_same
        with torch.no_grad():
            memory_bank.centroids[same_idx] = (1.0 - self.beta) * memory_bank.centroids[same_idx] + self.beta * z
            
        return loss_val, same_idx, other_idx
