import shap
import torch
import numpy as np
from src.data.preprocessing import log_compact


class AnomalySHAPExplainer:
    """
    Computes SHAP Shapley value attributions over packet features for flagged anomalies.
    Explains the mapping from input sequence features to the anomaly score S(z).

    NOTE: This explainer should only be invoked on flows already flagged as
    anomalous (S(z) > tau_d) by the calling pipeline code -- it does not
    enforce that threshold itself, since it is a general-purpose explanation
    tool. Running it on every flow (rather than only flagged anomalies)
    would be a significant and unnecessary performance cost.
    """
    def __init__(self, encoder, memory_bank, normalizer, background_samples: torch.Tensor,
                 device: str = "cpu", use_gradient_explainer: bool = False):
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.normalizer = normalizer
        self.device = device

        # Move/eval the encoder once here rather than on every SHAP query.
        self.encoder.eval()
        self.encoder.to(self.device)

        self.seq_len = background_samples.shape[1]
        self.num_features = background_samples.shape[2]

        self.use_gradient_explainer = use_gradient_explainer

        if use_gradient_explainer:
            # GradientExplainer uses backprop instead of perturbation sampling,
            # which is substantially faster on continuous, low-dimensional
            # inputs like ours (seq_len x num_features). Requires a
            # differentiable wrapper module rather than a numpy function.
            self._bg_tensor = background_samples.to(self.device)
            self._wrapper = _AnomalyScoreModule(self.encoder, self.memory_bank, self.normalizer)
            self.explainer = shap.GradientExplainer(self._wrapper, self._bg_tensor)
        else:
            # KernelExplainer (perturbation-based, model-agnostic, slower).
            bg_flat = background_samples.cpu().detach().numpy().reshape(background_samples.shape[0], -1)
            self.explainer = shap.KernelExplainer(self.predict_anomaly_score_np, bg_flat)

    def predict_anomaly_score_np(self, x_flat_np: np.ndarray) -> np.ndarray:
        """
        Wrapper function mapping flattened numpy input sequences to anomaly scores.
        Used only by KernelExplainer.
        """
        B = x_flat_np.shape[0]
        x_tensor = torch.tensor(x_flat_np, dtype=torch.float32, device=self.device)
        x_tensor = x_tensor.view(B, self.seq_len, self.num_features)
        with torch.no_grad():
            x_processed = log_compact(x_tensor)
            if self.normalizer.is_fitted:
                x_processed = self.normalizer.transform(x_processed)
            _, z = self.encoder(x_processed)
            scores = self.memory_bank.compute_anomaly_score(z)
        return scores.cpu().numpy()

    def explain_anomaly(self, target_sample: torch.Tensor, nsamples: int = 100) -> np.ndarray:
        """
        Computes SHAP values for a single target sequence sample.
        Args:
            target_sample: Tensor of shape (1, seq_len, num_features) or (seq_len, num_features)
            nsamples: number of perturbation samples (KernelExplainer only; ignored for GradientExplainer)
        Returns:
            shap_values: Array of shape (seq_len, num_features) representing feature importance.
        """
        if len(target_sample.shape) == 2:
            target_sample = target_sample.unsqueeze(0)

        result = self.explain_anomalies_batch(target_sample, nsamples=nsamples)
        return result[0]

    def explain_anomalies_batch(self, target_samples: torch.Tensor, nsamples: int = 100) -> np.ndarray:
        """
        Computes SHAP values for a batch of target sequences in one explainer call,
        avoiding the overhead of re-running the kernel approximation per-sample.

        Args:
            target_samples: Tensor of shape (B, seq_len, num_features)
            nsamples: number of perturbation samples (KernelExplainer only; ignored for GradientExplainer)
        Returns:
            shap_values: Array of shape (B, seq_len, num_features)
        """
        B = target_samples.shape[0]

        if self.use_gradient_explainer:
            target_tensor = target_samples.to(self.device)
            shap_values = self.explainer.shap_values(target_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if torch.is_tensor(shap_values):
                shap_values = shap_values.cpu().detach().numpy()
            return shap_values.reshape(B, self.seq_len, self.num_features)

        target_flat = target_samples.cpu().detach().numpy().reshape(B, -1)
        shap_values_flat = self.explainer.shap_values(target_flat, nsamples=nsamples)
        if isinstance(shap_values_flat, list):
            shap_values_flat = shap_values_flat[0]
        return shap_values_flat.reshape(B, self.seq_len, self.num_features)


class _AnomalyScoreModule(torch.nn.Module):
    """
    Differentiable wrapper exposing anomaly-score computation as a single
    forward pass, required for shap.GradientExplainer (which needs a
    torch.nn.Module rather than an arbitrary numpy function).

    NOTE: requires that self.encoder and self.memory_bank.compute_anomaly_score
    are both differentiable end-to-end. If memory_bank scoring involves a
    non-differentiable operation (e.g. exact nearest-neighbor argmin without
    a soft relaxation), GradientExplainer's attributions may not behave as
    expected at the discontinuity -- this is a known SHAP/Integrated-Gradients
    caveat for nearest-centroid style scoring functions, not a bug in this
    wrapper. Use use_gradient_explainer=False (KernelExplainer) if this matters
    for your case study.
    """
    def __init__(self, encoder, memory_bank, normalizer):
        super().__init__()
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.normalizer = normalizer

    def forward(self, x):
        x_processed = log_compact(x)
        if self.normalizer.is_fitted:
            x_processed = self.normalizer.transform(x_processed)
        _, z = self.encoder(x_processed)
        scores = self.memory_bank.compute_anomaly_score(z)
        return scores