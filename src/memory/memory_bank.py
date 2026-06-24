import numpy as np
import torch
from sklearn.cluster import KMeans

class DeviceMemoryBank:
    """
    Per-device Behavioral Memory Bank.
    Learns benign cluster prototypes (centroids) for a specific device class,
    selecting the optimal number of clusters K in {2, 4, 8} using the Elbow method.
    """
    def __init__(self, device_class_id: int, k_candidates: list[int] = [2, 4, 8]):
        self.device_class_id = device_class_id
        self.k_candidates = sorted(k_candidates)
        self.optimal_k = self.k_candidates[0]
        self.centroids = None # PyTorch tensor of shape (optimal_k, d_model)
        self.wcss_values = {}

    def fit(self, z: torch.Tensor):
        """
        Fits per-device clusters on frozen encoder embeddings of benign traffic.
        z shape: (N, d_model)
        """
        N = z.shape[0]
        device = z.device
        z_np = z.cpu().detach().numpy()
        
        # Handle small data edge cases
        if N < min(self.k_candidates):
            # If we don't have enough samples, just set centroids to the average
            self.optimal_k = 1
            self.centroids = torch.mean(z, dim=0, keepdim=True)
            return

        wcss = []
        models = {}
        
        for k in self.k_candidates:
            if k > N:
                continue
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(z_np)
            wcss.append(kmeans.inertia_)
            models[k] = kmeans
            self.wcss_values[k] = kmeans.inertia_
            
        if not wcss:
            self.optimal_k = 1
            self.centroids = torch.mean(z, dim=0, keepdim=True)
            return

        # Determine optimal K using Chord-Distance Elbow heuristic
        if len(wcss) >= 3:
            # We have wcss for k=2, 4, 8 (or others)
            # Find the point that maximizes the distance to the line connecting first and last points.
            k_vals = list(models.keys())
            w1, wn = wcss[0], wcss[-1]
            k1, kn = k_vals[0], k_vals[-1]
            
            # Line equation coefficients: y = m*x + c -> m*x - y + c = 0
            m = (wn - w1) / (kn - k1 + 1e-8)
            c = w1 - m * k1
            
            max_dist = -1.0
            best_k = k_vals[0]
            
            for idx, k in enumerate(k_vals):
                w_k = wcss[idx]
                # Distance of (k, w_k) to line m*x - y + c = 0
                dist = abs(m * k - w_k + c) / np.sqrt(m**2 + 1.0)
                if dist > max_dist:
                    max_dist = dist
                    best_k = k
            self.optimal_k = best_k
        else:
            # Fallback for fewer than 3 Candidates: select the first one
            self.optimal_k = list(models.keys())[0]
            
        # Extract centroids of the optimal model
        best_model = models[self.optimal_k]
        self.centroids = torch.tensor(best_model.cluster_centers_, dtype=torch.float32, device=device)

    def compute_anomaly_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the anomaly score: S(z) = min ||z - c||_2.
        z shape: (B, d_model)
        Returns: (B,)
        """
        if self.centroids is None:
            # If not fitted yet, return zero anomaly score
            return torch.zeros(z.shape[0], device=z.device)
            
        # Compute distances from each z to each centroid
        # z: (B, d_model) -> (B, 1, d_model)
        # centroids: (K, d_model) -> (1, K, d_model)
        z_expanded = z.unsqueeze(1)
        centroids_expanded = self.centroids.unsqueeze(0)
        
        # Distances: (B, K)
        dists = torch.norm(z_expanded - centroids_expanded, dim=2)
        
        # S(z) is the minimum distance to any centroid
        min_dists, _ = torch.min(dists, dim=1)
        return min_dists
