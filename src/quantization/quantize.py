import torch
import torch.nn as nn
import numpy as np
import copy

def log_quantize_dequantize(W: torch.Tensor, bits: int = 8, outlier_sigma: float = 3.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Outlier-aware logarithmic quantization.
    1. Identifies outliers outside [-outlier_sigma * std, outlier_sigma * std].
    2. Keeps outliers in FP32.
    3. Quantizes inliers to log-scale integers.
    """
    device = W.device
    W_flat = W.detach()
    
    # 1. Identify outliers
    std = torch.std(W_flat)
    mean = torch.mean(W_flat)
    outlier_mask = torch.abs(W_flat - mean) > (outlier_sigma * std)
    
    # Separate outliers and inliers
    outliers = torch.where(outlier_mask, W_flat, torch.zeros_like(W_flat))
    inliers = torch.where(~outlier_mask, W_flat, torch.zeros_like(W_flat))
    
    # 2. Log-scale quantize inliers
    signs = torch.sign(inliers)
    abs_inliers = torch.abs(inliers)
    
    # Log compaction: log(|x| + eps)
    # Clamp to avoid log(0) on zero elements
    log_inliers = torch.log(torch.clamp(abs_inliers, min=eps))
    
    # Determine quantization scale
    min_log = torch.min(log_inliers)
    max_log = torch.max(log_inliers)
    
    num_buckets = (2 ** bits) - 1
    scale = (max_log - min_log) / (num_buckets + 1e-8)
    
    # Quantize to bucket indices
    q_indices = torch.round((log_inliers - min_log) / (scale + 1e-8))
    q_indices = torch.clamp(q_indices, 0, num_buckets)
    
    # Dequantize inliers
    dequant_inliers = signs * torch.exp(q_indices * scale + min_log)
    # Zero out elements that were originally zero
    dequant_inliers = torch.where(abs_inliers < eps, torch.zeros_like(dequant_inliers), dequant_inliers)
    
    # Combine outliers (FP32) + quantized inliers
    W_quant = outliers + dequant_inliers
    return W_quant


def linear_quantize_dequantize(W: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Standard symmetric INT8 quantization.
    """
    # Scale calculation
    max_val = torch.max(torch.abs(W))
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    
    scale = max_val / qmax
    if scale == 0:
        return W
        
    q_W = torch.round(W / scale)
    q_W = torch.clamp(q_W, qmin, qmax)
    dequant_W = q_W * scale
    return dequant_W


class QuantizedLinear(nn.Module):
    """
    Mock quantized Linear layer that performs forward pass with fake quantized weights.
    """
    def __init__(self, original_linear: nn.Linear, mode: str = "int8"):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = nn.Parameter(original_linear.weight.clone())
        self.bias = nn.Parameter(original_linear.bias.clone()) if original_linear.bias is not None else None
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "int8":
            q_weight = linear_quantize_dequantize(self.weight)
        elif self.mode == "log":
            q_weight = log_quantize_dequantize(self.weight)
        else:
            q_weight = self.weight
            
        return nn.functional.linear(x, q_weight, self.bias)


def quantize_model(model: nn.Module, mode: str = "hybrid") -> nn.Module:
    """
    Applies quantization to a copy of the model based on the selected mode:
      - 'full': original FP32 precision
      - 'int8': standard INT8 for both linear layers and SSM parameters
      - 'hybrid': log-quantization for SSM params (A, dt_proj.weight), INT8 for linear layers
      - 'log': log-quantization for both linear layers and SSM parameters
    """
    model_copy = copy.deepcopy(model)
    
    if mode == "full":
        return model_copy
        
    # Helper to traverse and replace linear layers
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Determine quantization mode for linear layers
                linear_mode = "int8" if mode in ["int8", "hybrid"] else "log"
                setattr(module, name, QuantizedLinear(child, mode=linear_mode))
            else:
                replace_layers(child)
                
    replace_layers(model_copy)
    
    # Apply quantization to SSM parameters (A and selective projections)
    # Search for parameters in Mamba blocks
    for name, param in model_copy.named_parameters():
        # Identify SSM transition parameter A, selective linear layer weights, dt projection, etc.
        is_ssm_param = any(keyword in name for keyword in ["conv1d", "A", "dt_proj", "x_proj"])
        
        if is_ssm_param:
            with torch.no_grad():
                if mode in ["hybrid", "log"]:
                    # Log quantize SSM parameters
                    param.copy_(log_quantize_dequantize(param))
                else:
                    # Pure linear int8
                    param.copy_(linear_quantize_dequantize(param))
                    
    return model_copy


class QuantizationAblationHarness:
    """
    Ablation harness to compare models quantized under the 4 configurations.
    """
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.modes = ["full", "int8", "hybrid", "log"]
        self.quantized_models = {}
        for m in self.modes:
            self.quantized_models[m] = quantize_model(base_model, mode=m)

    def evaluate_model(self, mode: str, dataloader, normalizer, device: str) -> float:
        """
        Computes the MSE reconstruction error of the quantized model on the test dataset.
        (Using pretraining dataset to check signal preservation).
        """
        model = self.quantized_models[mode]
        model.to(device)
        model.eval()
        
        total_error = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, _, _, _ in dataloader:
                x_batch = x_batch.to(device)
                
                # Apply same compaction/normalization
                from src.data.preprocessing import log_compact
                x_processed = log_compact(x_batch)
                if normalizer.is_fitted:
                    x_processed = normalizer.transform(x_processed)
                    
                # We measure how well the encoder embeddings match between quantized and full precision
                # Forward full precision
                _, z_full = self.base_model.encoder(x_processed)
                # Forward quantized
                _, z_quant = model.encoder(x_processed)
                
                # Mean squared difference in representation space
                mse = torch.mean((z_full - z_quant) ** 2)
                total_error += mse.item()
                num_batches += 1
                
        return total_error / max(1, num_batches)

    def get_ablation_results(self, dataloader, normalizer, device: str) -> dict[str, float]:
        results = {}
        for mode in self.modes:
            results[mode] = self.evaluate_model(mode, dataloader, normalizer, device)
        return results
