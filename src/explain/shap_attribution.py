import shap
import torch
import numpy as np

class AnomalySHAPExplainer:
    """
    Computes SHAP Shapley value attributions over packet features for flagged anomalies.
    Explains the mapping from input sequence features to the anomaly score S(z).
    """
    def __init__(self, encoder, memory_bank, normalizer, background_samples: torch.Tensor, device: str = "cpu"):
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.normalizer = normalizer
        self.device = device
        
        # Prepare background data for SHAP (KernelExplainer expects numpy arrays)
        # Background samples shape: (num_bg_samples, seq_len, num_features)
        self.seq_len = background_samples.shape[1]
        self.num_features = background_samples.shape[2]
        
        # Flatten background samples for KernelExplainer
        bg_flat = background_samples.cpu().detach().numpy().reshape(background_samples.shape[0], -1)
        
        # Create the explainer
        self.explainer = shap.KernelExplainer(self.predict_anomaly_score_np, bg_flat)

    def predict_anomaly_score_np(self, x_flat_np: np.ndarray) -> np.ndarray:
        """
        Wrapper function mapping flattened numpy input sequences to anomaly scores.
        """
        # Convert back to tensor
        B = x_flat_np.shape[0]
        x_tensor = torch.tensor(x_flat_np, dtype=torch.float32, device=self.device)
        x_tensor = x_tensor.view(B, self.seq_len, self.num_features)
        
        self.encoder.eval()
        self.encoder.to(self.device)
        
        with torch.no_grad():
            # Apply preprocessing inside the explainer to keep it end-to-end
            from src.data.preprocessing import log_compact
            x_processed = log_compact(x_tensor)
            if self.normalizer.is_fitted:
                x_processed = self.normalizer.transform(x_processed)
                
            _, z = self.encoder(x_processed)
            scores = self.memory_bank.compute_anomaly_score(z)
            
        return scores.cpu().numpy()

    def explain_anomaly(self, target_sample: torch.Tensor) -> np.ndarray:
        """
        Computes SHAP values for a single target sequence sample.
        Args:
            target_sample: Tensor of shape (1, seq_len, num_features) or (seq_len, num_features)
        Returns:
            shap_values: Array of shape (seq_len, num_features) representing feature importance.
        """
        if len(target_sample.shape) == 2:
            target_sample = target_sample.unsqueeze(0)
            
        target_flat = target_sample.cpu().detach().numpy().reshape(1, -1)
        
        # Compute shap values (approximate using nsamples)
        shap_values_flat = self.explainer.shap_values(target_flat, nsamples=100)
        
        # Reshape to (seq_len, num_features)
        if isinstance(shap_values_flat, list):
            shap_values_flat = shap_values_flat[0]
            
        return shap_values_flat.reshape(self.seq_len, self.num_features)
