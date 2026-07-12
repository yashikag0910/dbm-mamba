import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.preprocessing import log_compact
from src.models.focal_loss import FocalLoss
from src.models.parametric_umap import ParametricUMAPLoss


class JointFinetuningModel(nn.Module):
    """
    Ties the MambaEncoder and ClassifierHead together for fine-tuning.
    """
    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: preprocessed input flow tokens, shape (B, L, num_features)
        Returns:
            logits: classification logits, shape (B, n_classes)
            z: latent sequence representation, shape (B, d_model)
        """
        _, z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


def get_lambda_umap(epoch: int, total_epochs: int, lambda_init: float,
                    lambda_final: float, schedule: str = 'linear') -> float:
    """
    Computes lambda_umap for the current epoch based on the chosen annealing schedule.

    This MUST be called by the training loop once per epoch and the result
    passed into train_finetune_epoch. If a constant float is passed instead,
    the annealing schedule is bypassed -- verify your training loop calls
    this per epoch.

    Schedules:
      'linear':      lambda increases linearly from lambda_init to lambda_final
      'exponential': lambda increases geometrically
      other:         constant at lambda_init (fallback)
    """
    if total_epochs <= 1:
        return lambda_final

    if schedule == 'linear':
        frac = epoch / (total_epochs - 1)
        return lambda_init + frac * (lambda_final - lambda_init)

    elif schedule == 'exponential':
        frac = epoch / (total_epochs - 1)
        ratio = max(lambda_final, 1e-8) / max(lambda_init, 1e-8)
        return lambda_init * (ratio ** frac)

    else:
        return lambda_init  # constant fallback


def train_finetune_epoch(model: nn.Module, dataloader: DataLoader,
                         optimizer: optim.Optimizer, normalizer,
                         focal_loss_fn: FocalLoss, umap_loss_fn: ParametricUMAPLoss,
                         lambda_umap: float, device: str) -> tuple[float, float, float]:
    """
    Trains one fine-tuning epoch optimizing:
        L_total = L_focal + lambda_umap * L_UMAP

    where:
      - L_focal: multi-class Focal Loss on classifier logits vs. ground-truth labels
      - L_UMAP:  Parametric UMAP fuzzy cross-entropy computed between input-space
                 topology (frozen, computed from x_processed) and latent-space
                 pairwise structure (differentiable, computed from z).

    UMAP loss arguments: umap_loss_fn(X=x_processed, z=z)
      X is used to build the frozen fuzzy membership graph V (input topology target).
      z is the differentiable latent embedding whose pairwise structure is optimized
      to match V. This is the correct parametric UMAP formulation.

    lambda_umap should be computed via get_lambda_umap(epoch, ...) in the
    calling training loop -- passing a constant bypasses annealing.

    Args:
        model: JointFinetuningModel (encoder + classifier)
        dataloader: yields (x_batch, y_batch, device_classes, macs)
        optimizer: optimizer over model parameters
        normalizer: fitted ZScoreNormalizer
        focal_loss_fn: FocalLoss instance
        umap_loss_fn: ParametricUMAPLoss instance
        lambda_umap: current epoch's UMAP loss weight (from get_lambda_umap)
        device: torch device string

    Returns:
        avg_total_loss, avg_focal_loss, avg_umap_loss
    """
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_umap = 0.0
    num_batches = 0

    for x_batch, y_batch, _, _ in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Preprocessing: log compaction + z-score normalization
        x_processed = log_compact(x_batch)
        if normalizer.is_fitted:
            x_processed = normalizer.transform(x_processed)

        optimizer.zero_grad()

        # Forward pass
        logits, z = model(x_processed)

        # Focal loss on classifier output
        loss_focal = focal_loss_fn(logits, y_batch)

        # Parametric UMAP loss:
        #   X = x_processed  (input space, used to build frozen topology target V)
        #   z = z             (latent space, differentiable, structure is optimized)
        loss_umap = umap_loss_fn(x_processed, z)

        loss_total = loss_focal + lambda_umap * loss_umap

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()
        total_focal += loss_focal.item()
        total_umap += loss_umap.item()
        num_batches += 1

    avg_total = total_loss / max(1, num_batches)
    avg_focal = total_focal / max(1, num_batches)
    avg_umap = total_umap / max(1, num_batches)

    return avg_total, avg_focal, avg_umap