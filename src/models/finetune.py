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


def get_lambda_umap(epoch: int, total_epochs: int, lambda_init: float, lambda_final: float, schedule: str = 'linear') -> float:
    """
    Computes lambda value based on the chosen annealing schedule.
    """
    if total_epochs <= 1:
        return lambda_final
        
    if schedule == 'linear':
        frac = epoch / (total_epochs - 1)
        return lambda_init + frac * (lambda_final - lambda_init)
    elif schedule == 'exponential':
        frac = epoch / (total_epochs - 1)
        # Avoid zero/log division issues
        ratio = max(lambda_final, 1e-8) / max(lambda_init, 1e-8)
        return lambda_init * (ratio ** frac)
    else:
        return lambda_init # constant fallback


def train_finetune_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                         normalizer, focal_loss_fn: FocalLoss, umap_loss_fn: ParametricUMAPLoss,
                         lambda_umap: float, device: str) -> tuple[float, float, float]:
    """
    Trains one fine-tuning epoch optimizing Focal Loss and Parametric UMAP Loss.
    """
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_umap = 0.0
    num_batches = 0
    
    for x_batch, y_batch, _, _ in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Preprocessing: log compaction + transform
        x_processed = log_compact(x_batch)
        if normalizer.is_fitted:
            x_processed = normalizer.transform(x_processed)
            
        optimizer.zero_grad()
        
        # Forward pass
        logits, z = model(x_processed)
        
        # Compute losses
        loss_focal = focal_loss_fn(logits, y_batch)
        loss_umap = umap_loss_fn(x_processed, z)
        
        loss_total = loss_focal + lambda_umap * loss_umap
        
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss_total.item()
        total_focal += loss_focal.item()
        total_umap += loss_umap.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches), total_focal / max(1, num_batches), total_umap / max(1, num_batches)
