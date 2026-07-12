import torch
import torch.nn as nn
import torch.optim as optim


class OnlineMemoryRefiner:
    """
    Stage 6: Online Triplet Loss adaptation for memory banks.
    Computes adaptive margin, performs semi-hard negative mining,
    and updates prototypes using EMA and Triplet Loss gradients.

    Update order per refine_step call:
      1. Gradient step on c_same and c_other (triplet loss).
      2. EMA pull of c_same toward the new benign embedding z.
    The EMA partially overwrites the gradient step on c_same by design --
    EMA is the dominant, stabilizing update and the gradient step provides
    the margin-shaping signal. Both target memory_bank.centroids in-place.
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

        r = torch.sum(centroids * centroids, dim=1, keepdim=True)
        D_sq = r - 2.0 * torch.matmul(centroids, centroids.t()) + r.t()
        D_sq = torch.clamp(D_sq, min=0.0)
        dists = torch.sqrt(D_sq + 1e-8)

        sum_dists = torch.sum(dists)
        num_pairs = K * (K - 1)
        mean_dist = sum_dists / num_pairs
        return float(mean_dist.item())

    def mine_semi_hard_negative(self, z: torch.Tensor, neg_centroids: torch.Tensor,
                                 anchor_idx: int, alpha: float) -> int:
        """
        Selects a semi-hard negative centroid from neg_centroids.
        Semi-hard negative condition:
          ||z - c_anchor||^2 < ||z - c_other||^2 < ||z - c_anchor||^2 + alpha

        Args:
            z: embedding of the current sample (d_model,)
            neg_centroids: candidate negative centroids (K_neg, d_model)
            anchor_idx: index in neg_centroids to treat as the anchor/positive
                        (skipped during search). Pass -1 if neg_centroids is a
                        separate bank with no anchor to skip.
            alpha: adaptive margin

        Returns:
            index into neg_centroids of the selected negative, or -1 if K_neg <= 1
            and no valid negative exists.
        """
        K_neg = neg_centroids.shape[0]
        if K_neg <= 1 and anchor_idx != -1:
            # Single centroid and it IS the anchor -- no valid negative exists.
            return -1

        if anchor_idx >= 0:
            c_anchor = neg_centroids[anchor_idx]
        else:
            # anchor lives in a different bank; compute distance to nearest centroid
            # in neg_centroids as a proxy for d_same.
            dists_to_neg = torch.norm(z.unsqueeze(0) - neg_centroids, dim=1)
            anchor_idx = int(torch.argmin(dists_to_neg).item())
            c_anchor = neg_centroids[anchor_idx]

        d_same_sq = torch.sum((z - c_anchor) ** 2).item()

        candidates = []
        fallback_idx = -1
        min_fallback_diff = float('inf')

        for idx in range(K_neg):
            if idx == anchor_idx:
                continue

            c_other = neg_centroids[idx]
            d_other_sq = torch.sum((z - c_other) ** 2).item()

            if d_same_sq < d_other_sq < (d_same_sq + alpha):
                candidates.append((idx, d_other_sq))
            else:
                diff = abs(d_other_sq - (d_same_sq + alpha))
                if diff < min_fallback_diff:
                    min_fallback_diff = diff
                    fallback_idx = idx

        if candidates:
            candidates.sort(key=lambda val: val[1], reverse=True)
            return candidates[0][0]

        return fallback_idx

    def refine_step(self, z: torch.Tensor, memory_bank, other_banks: list = None) -> tuple[float, int, int]:
        """
        Performs online refinement for a single confirmed-benign embedding z.

        1. Find c_same (closest centroid in memory_bank.centroids).
        2. Mine c_other (semi-hard negative) from within the bank (K > 1)
           or from other_banks (K == 1).
        3. Compute triplet loss and update memory_bank.centroids in-place
           via manual gradient step.
        4. Apply EMA update to c_same, pulling it toward z.

        NOTE: z must be a confirmed-benign embedding -- the calling pipeline
        is responsible for only invoking this on flows that passed the
        anomaly threshold check (S(z) <= tau_d).
        """
        if memory_bank.centroids is None:
            return 0.0, -1, -1

        z = z.detach()
        # Keep a direct reference to memory_bank.centroids so all in-place
        # ops below modify the bank's actual tensor, not a detached copy.
        centroids = memory_bank.centroids
        K = centroids.shape[0]

        # 1. Find c_same
        dists = torch.norm(z.unsqueeze(0) - centroids, dim=1)
        same_idx = int(torch.argmin(dists).item())

        # 2. Determine the negative centroid pool
        if K <= 1 and other_banks is not None:
            # Single-centroid bank: mine negatives from other device banks
            neg_candidates = []
            for ob in other_banks:
                if ob.centroids is not None and ob.device_class_id != memory_bank.device_class_id:
                    neg_candidates.append(ob.centroids.to(z.device))
            if neg_candidates:
                neg_centroids = torch.cat(neg_candidates, dim=0)
                # anchor_idx = -1 because the anchor (c_same) is not in neg_centroids
                neg_anchor_idx = -1
            else:
                # No other banks available and K=1 -- no valid negative, skip triplet step.
                neg_centroids = None
                neg_anchor_idx = -1
        else:
            # Multi-centroid bank: mine negatives from within the same bank
            neg_centroids = centroids
            neg_anchor_idx = same_idx

        loss_val = 0.0
        other_idx = -1

        # 3. Triplet loss update
        if neg_centroids is not None:
            alpha = self.compute_adaptive_margin(neg_centroids)
            other_idx = self.mine_semi_hard_negative(z, neg_centroids, neg_anchor_idx, alpha)

            if other_idx != -1:
                c_same_param = centroids[same_idx].clone().detach().requires_grad_(True)
                c_other_param = neg_centroids[other_idx].clone().detach().requires_grad_(True)

                d_same_sq = torch.sum((z - c_same_param) ** 2)
                d_other_sq = torch.sum((z - c_other_param) ** 2)

                loss = torch.clamp(d_same_sq - d_other_sq + alpha, min=0.0)
                loss_val = float(loss.item())

                if loss.item() > 0:
                    loss.backward()

                    with torch.no_grad():
                        # Update directly on memory_bank.centroids (via the
                        # `centroids` reference) to ensure changes persist.
                        memory_bank.centroids[same_idx] -= self.lr * c_same_param.grad
                        if K > 1:
                            # only update neg centroid if it lives in this bank
                            memory_bank.centroids[other_idx] -= self.lr * c_other_param.grad

        # 4. EMA update on c_same (dominant stabilizing update, applied after triplet)
        with torch.no_grad():
            memory_bank.centroids[same_idx] = (
                (1.0 - self.beta) * memory_bank.centroids[same_idx] + self.beta * z
            )

        return loss_val, same_idx, other_idx