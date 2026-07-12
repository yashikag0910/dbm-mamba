import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveInferenceGate(nn.Module):
    """
    Adaptive Inference Gating Network.
    Computes packet length entropy H and IAT variance, evaluates a logistic gate
    to decide between fast (linear/MLP) and slow (Mamba) paths, and handles TCP escalation.
    """
    def __init__(self, entropy_threshold: float = 1.2, var_iat_threshold: float = 0.5,
                 mamba_confidence_th: float = 0.95, anomaly_distill_multiplier: float = 0.5,
                 tcp_flag_escalation_th: float = 0.5):
        super().__init__()
        # 2-input logistic gate parameters: P_high = sigmoid(w_entropy * H + w_var * Var(IAT) + bias)
        self.w_entropy = nn.Parameter(torch.tensor(1.0))
        self.w_var = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

        self.entropy_threshold = entropy_threshold
        self.var_iat_threshold = var_iat_threshold
        self.mamba_confidence_th = mamba_confidence_th

        # Fixed multiplier from the proposal's self-distillation rule
        # (Section 4.6): a flow is labeled low-complexity only if
        # S(z) < 0.5 * tau_d. This is a spec-defined constant, not a
        # general "confidence threshold" -- kept as a constructor arg
        # for testability, but the name now reflects what it actually
        # controls, and it should not be casually changed during
        # hyperparameter tuning without re-checking the proposal.
        self.anomaly_distill_multiplier = anomaly_distill_multiplier

        # Threshold on the combined/weighted TCP flag feature used for
        # hard escalation. NOTE: because `flags` in the current data
        # pipeline is a single scalar (weighted sum of individual flag
        # booleans / 32, see dataset loader), this threshold cannot
        # distinguish specific flag *combinations* (e.g. SYN+RST vs a
        # normal SYN-ACK) the way the proposal's Section 4.6 escalation
        # rule describes. It currently escalates whenever the combined
        # weighted flag value exceeds this threshold, which is a coarser
        # proxy for "protocol-inconsistent flag pattern," not an exact
        # implementation of it. To implement the precise rule, flags
        # would need to be carried as a multi-bit/one-hot feature
        # upstream rather than collapsed into one scalar.
        self.tcp_flag_escalation_th = tcp_flag_escalation_th

    def extract_features(self, seq_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts gating features from sequence:
        1. H: Entropy of packet lengths over 4 bins:
           Bin 1: [0, 100]
           Bin 2: (100, 500]
           Bin 3: (500, 1000]
           Bin 4: (1000, inf)
        2. Var(IAT): Variance of Inter-Arrival Times

        Args:
            seq_x: tensor of shape (B, L, 4) where L is seq length
                   [0] = length, [2] = IAT
        Returns:
            H: (B,) tensor of entropy values
            var_iat: (B,) tensor of IAT variances
        """
        B, L, _ = seq_x.shape
        device = seq_x.device

        lengths = seq_x[:, :, 0]  # (B, L)
        iats = seq_x[:, :, 2]  # (B, L)

        # 1. Packet length entropy
        b1 = (lengths <= 100).float()
        b2 = ((lengths > 100) & (lengths <= 500)).float()
        b3 = ((lengths > 500) & (lengths <= 1000)).float()
        b4 = (lengths > 1000).float()

        counts = torch.stack([
            b1.sum(dim=1),
            b2.sum(dim=1),
            b3.sum(dim=1),
            b4.sum(dim=1)
        ], dim=-1)

        probs = counts / (L + 1e-8)

        eps = 1e-8
        H = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # (B,)

        # 2. IAT variance
        var_iat = torch.var(iats, dim=1)  # (B,)
        var_iat = torch.nan_to_num(var_iat, nan=0.0)

        return H, var_iat

    def check_tcp_escalation(self, seq_x: torch.Tensor) -> torch.Tensor:
        """
        TCP escalation rule (coarse proxy -- see tcp_flag_escalation_th docstring
        above for the gap between this and the proposal's exact
        SYN+RST / unexpected-URG / PSH-without-session rule).

        Escalates if any packet's combined weighted flag feature exceeds
        tcp_flag_escalation_th.
        """
        flags = seq_x[:, :, 3]  # (B, L)
        escalate = torch.any(flags > self.tcp_flag_escalation_th, dim=1)  # (B,)
        return escalate

    def forward(self, seq_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_x: tensor of shape (B, L, 4)
        Returns:
            P_high: probability of selecting the high-complexity path, shape (B,)
            escalate: boolean tensor of shape (B,) indicating hard TCP escalation
        """
        H, var_iat = self.extract_features(seq_x)

        logits = self.w_entropy * H + self.w_var * var_iat + self.bias
        P_high = torch.sigmoid(logits)

        escalate = self.check_tcp_escalation(seq_x)
        P_high = torch.where(escalate, torch.ones_like(P_high), P_high)

        return P_high, escalate

    def self_distill_labels(self, seq_x: torch.Tensor, mamba_logits: torch.Tensor,
                             anomaly_scores: torch.Tensor, tau_d: torch.Tensor) -> torch.Tensor:
        """
        Generates self-distilled targets for training the gate.
        Low-complexity path (y=0) preferred if:
          - Mamba classifier confidence > 0.95
          AND
          - Anomaly score S(z) < 0.5 * tau_d   (anomaly_distill_multiplier)
        Otherwise, requires high-complexity path (y=1).

        Returns:
            targets: (B,) float tensor of 0.0 or 1.0
        """
        probs = F.softmax(mamba_logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)

        confidence_cond = max_probs > self.mamba_confidence_th
        anomaly_cond = anomaly_scores < (self.anomaly_distill_multiplier * tau_d)

        low_complexity_pref = confidence_cond & anomaly_cond

        targets = torch.where(low_complexity_pref, torch.zeros_like(max_probs), torch.ones_like(max_probs))
        return targets

    def fit_gate(self, features: torch.Tensor, targets: torch.Tensor, lr: float = 0.05, epochs: int = 100):
        """
        Fits gate weights using PyTorch binary cross-entropy optimization.
        features shape: (N, 2) where column 0 is H, column 1 is Var(IAT)
        targets shape: (N,)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            H, var_iat = features[:, 0], features[:, 1]
            logits = self.w_entropy * H + self.w_var * var_iat + self.bias
            P_high = torch.sigmoid(logits)

            loss = F.binary_cross_entropy(P_high, targets)
            loss.backward()
            optimizer.step()