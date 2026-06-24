import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    Supervised classification head for projecting latent embeddings to attack categories.
    """
    def __init__(self, d_model: int = 64, n_classes: int = 34, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent embedding tensor of shape (B, d_model)
        Returns:
            logits: Predicted class logits of shape (B, n_classes)
        """
        return self.net(z)
