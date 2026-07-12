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
        self.centroids = None  # PyTorch tensor of shape (optimal_k, d_model)
        self.wcss_values = {}

    def fit(self, z: torch.Tensor):
        """
        Fits per-device clusters on frozen encoder embeddings of benign traffic.
        z shape: (N, d_model)
        """
        N = z.shape[0]
        device = z.device
        z_np = z.cpu().detach().numpy()

        if N < min(self.k_candidates):
            self.optimal_k = 1
            self.centroids = torch.mean(z, dim=0, keepdim=True)
            return

        wcss = []
        models = {}
        for k in self.k_candidates:
            if k > N:
                continue
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
            k_vals = list(models.keys())
            w1, wn = wcss[0], wcss[-1]
            k1, kn = k_vals[0], k_vals[-1]

            m = (wn - w1) / (kn - k1 + 1e-8)
            c = w1 - m * k1
            max_dist = -1.0
            best_k = k_vals[0]

            for idx, k in enumerate(k_vals):
                w_k = wcss[idx]
                dist = abs(m * k - w_k + c) / np.sqrt(m**2 + 1.0)
                if dist > max_dist:
                    max_dist = dist
                    best_k = k

            self.optimal_k = best_k
        else:
            self.optimal_k = list(models.keys())[0]

        best_model = models[self.optimal_k]
        self.centroids = torch.tensor(best_model.cluster_centers_, dtype=torch.float32, device=device)

    def compute_anomaly_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the anomaly score: S(z) = min ||z - c||_2.
        z shape: (B, d_model)
        Returns: (B,)
        """
        if self.centroids is None:
            # Bank not fitted -- no benign samples were seen for this device
            # class during training. Returning zero scores means nothing will
            # ever be flagged as anomalous for this device, which silently
            # hurts recall. Check that all device classes have benign samples
            # in your training split.
            print(f"[WARNING] DeviceMemoryBank for device_class_id={self.device_class_id} "
                  f"not fitted. Returning zero anomaly scores -- this device class "
                  f"will never trigger anomaly detection.")
            return torch.zeros(z.shape[0], device=z.device)

        z_expanded = z.unsqueeze(1)  # (B, 1, d_model)

        # Move centroids to the same device as z at inference time.
        # Necessary when training runs on CPU but inference runs on MPS/CUDA.
        centroids_expanded = self.centroids.to(z.device).unsqueeze(0)  # (1, K, d_model)

        dists = torch.norm(z_expanded - centroids_expanded, dim=2)  # (B, K)
        min_dists, _ = torch.min(dists, dim=1)  # (B,)
        return min_dists