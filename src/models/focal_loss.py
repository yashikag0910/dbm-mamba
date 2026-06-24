import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address dataset class imbalance.
    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: model output before softmax, shape (B, C)
            targets: target labels, shape (B,)
        Returns:
            loss: focal loss value
        """
        # Compute standard cross-entropy loss (negative log probability)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # p_t is the model's estimated probability for the correct class
        p_t = torch.exp(-ce_loss)
        
        # Focal loss formula
        loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
network_loss = FocalLoss
