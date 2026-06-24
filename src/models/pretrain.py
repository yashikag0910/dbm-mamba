import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.preprocessing import mask_sequence, log_compact

class PretrainReconstructionHead(nn.Module):
    """
    Decodes Mamba sequence representations back into packet attribute space.
    """
    def __init__(self, d_model: int = 64, num_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_features)
        )

    def forward(self, seq_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeddings: shape (B, L, d_model)
        Returns:
            reconstructed_features: shape (B, L, num_features)
        """
        return self.net(seq_embeddings)


class PretrainModel(nn.Module):
    """
    Stage 1 model tying the MambaEncoder and the PretrainReconstructionHead together.
    """
    def __init__(self, encoder: nn.Module, num_features: int = 4):
        super().__init__()
        self.encoder = encoder
        self.head = PretrainReconstructionHead(encoder.d_model, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is masked input, shape (B, L, num_features)
        seq_embeddings, _ = self.encoder(x)
        return self.head(seq_embeddings)


def train_pretrain_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                         normalizer, mask_ratio: float, device: str) -> float:
    """
    Trains one pretraining epoch using masked packet reconstruction.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, _, _, _ in dataloader:
        # x_batch shape: (B, L, num_features)
        x_batch = x_batch.to(device)
        
        # Apply preprocessing: log compaction and normalization
        x_processed = log_compact(x_batch)
        if normalizer.is_fitted:
            x_processed = normalizer.transform(x_processed)
            
        # Apply token masking M(X)
        masked_x, target_x, mask = mask_sequence(x_processed, mask_ratio=mask_ratio)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_x = model(masked_x) # (B, L, num_features)
        
        # Compute loss on masked tokens only
        # mask shape: (B, L). We expand it to (B, L, num_features)
        mask_expanded = mask.unsqueeze(-1).expand_as(target_x)
        
        # MSE loss on masked tokens
        loss = F_mse_loss(pred_x, target_x, mask_expanded)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)


def F_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error only for masked elements.
    """
    masked_pred = pred[mask]
    masked_target = target[mask]
    if masked_pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.mse_loss(masked_pred, masked_target)
