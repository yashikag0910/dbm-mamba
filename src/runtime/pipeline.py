import torch
import torch.nn as nn
import numpy as np
from src.device_id.fingerprinting import IdentityProfiler
from src.gating.gate import AdaptiveInferenceGate
from src.data.preprocessing import log_compact
from src.explain.shap_attribution import AnomalySHAPExplainer

class FastClassifier(nn.Module):
    """
    Lightweight, low-complexity classifier path for gating bypass.
    Directly maps averaged sequence features to class logits.
    """
    def __init__(self, num_features: int = 4, n_classes: int = 34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, num_features) -> mean pool to (B, num_features)
        x_mean = torch.mean(x, dim=1)
        return self.net(x_mean)


class DBMambaRuntimePipeline:
    """
    DBM-Mamba Real-time IoT Intrusion Detection pipeline.
    Coordinates device fingerprinting, gating, fast/slow classification paths,
    behavioral memory anomaly verification, and SHAP explanation generation.
    """
    def __init__(self, mamba_encoder: nn.Module, classifier_head: nn.Module,
                 fast_classifier: nn.Module, device_mem_banks: dict,
                 threshold_manager, normalizer, device: str = "cpu"):
        self.encoder = mamba_encoder
        self.classifier = classifier_head
        self.fast_classifier = fast_classifier
        self.mem_banks = device_mem_banks # Dict of device_class_id -> DeviceMemoryBank
        self.threshold_manager = threshold_manager
        self.normalizer = normalizer
        self.device = device
        
        # Modules
        self.profiler = IdentityProfiler()
        self.gate = AdaptiveInferenceGate()
        
        # Mapping from class ID to names
        self.class_names = {v: k for k, v in self.profiler.class_name_to_id.items()}
        
        # SHAP Explainer (lazy initialization per anomaly or globally)
        self.shap_explainer = None

    def initialize_shap(self, background_samples: torch.Tensor, device_class_id: int):
        """
        Pre-initializes the SHAP explainer using the specified device memory bank.
        """
        bank = self.mem_banks.get(device_class_id, None)
        if bank is None:
            # Fallback to generic
            bank = self.mem_banks.get(0)
            
        self.shap_explainer = AnomalySHAPExplainer(
            encoder=self.encoder,
            memory_bank=bank,
            normalizer=self.normalizer,
            background_samples=background_samples,
            device=self.device
        )

    def process_flow(self, seq_x: torch.Tensor, mac: str = None, 
                     tls_hello: str = None, dhcp_options: str = None) -> dict:
        """
        Processes a single flow sequence and outputs decision metadata.
        Args:
            seq_x: tensor of shape (1, L, 4) or (L, 4)
        """
        if len(seq_x.shape) == 2:
            seq_x = seq_x.unsqueeze(0)
            
        B, L, num_feats = seq_x.shape
        seq_x = seq_x.to(self.device)
        
        # 1. Device Identification
        device_class_id, is_ambiguous = self.profiler.profile_device(mac, tls_hello, dhcp_options)
        device_name = self.class_names.get(device_class_id, "Generic")
        
        # 2. Evaluate Adaptive Gating
        P_high, escalated = self.gate(seq_x)
        P_high_val = float(P_high[0].item())
        escalated_val = bool(escalated[0].item())
        
        # Decisions
        use_slow_path = escalated_val or (P_high_val >= 0.5)
        
        logits = None
        predicted_class_id = 0
        confidence = 0.0
        anomaly_score = 0.0
        is_anomaly = False
        path_taken = "Fast"
        
        # 3. Fast Path Evaluation
        if not use_slow_path:
            self.fast_classifier.eval()
            self.fast_classifier.to(self.device)
            with torch.no_grad():
                x_processed = log_compact(seq_x)
                if self.normalizer.is_fitted:
                    x_processed = self.normalizer.transform(x_processed)
                fast_logits = self.fast_classifier(x_processed)
                probs = torch.softmax(fast_logits, dim=-1)
                conf, pred_id = torch.max(probs, dim=-1)
                conf_val = float(conf[0].item())
                pred_id_val = int(pred_id[0].item())
                
                # Check confidence fallback: if confidence <= 0.7, fall back to slow path!
                if conf_val <= 0.7:
                    use_slow_path = True
                else:
                    logits = fast_logits
                    predicted_class_id = pred_id_val
                    confidence = conf_val
                    path_taken = "Fast"
                    
        # 4. Slow Path (Mamba) Evaluation
        if use_slow_path:
            path_taken = "Slow (Mamba)"
            self.encoder.eval()
            self.classifier.eval()
            self.encoder.to(self.device)
            self.classifier.to(self.device)
            
            with torch.no_grad():
                # Preprocess
                x_processed = log_compact(seq_x)
                if self.normalizer.is_fitted:
                    x_processed = self.normalizer.transform(x_processed)
                    
                # Forward selective Mamba
                _, z = self.encoder(x_processed)
                slow_logits = self.classifier(z)
                
                probs = torch.softmax(slow_logits, dim=-1)
                conf, pred_id = torch.max(probs, dim=-1)
                
                logits = slow_logits
                predicted_class_id = int(pred_id[0].item())
                confidence = float(conf[0].item())
                
                # Anomaly Score via Memory Bank
                bank = self.mem_banks.get(device_class_id, self.mem_banks[0]) # Fallback to generic if not found
                score_tensor = bank.compute_anomaly_score(z)
                anomaly_score = float(score_tensor[0].item())
                
                # Thresholding
                is_anomaly = self.threshold_manager.verify_anomaly(device_class_id, anomaly_score, is_ambiguous)
                
        # 5. SHAP attribution on anomaly detection
        shap_vals = None
        if is_anomaly and self.shap_explainer is not None:
            try:
                # Explain target sequence using pre-initialized explainer
                shap_vals = self.shap_explainer.explain_anomaly(seq_x[0])
            except Exception as e:
                print(f"[SHAP Explanation Error]: {e}")
                shap_vals = None
                
        return {
            "path": path_taken,
            "gate_probability": P_high_val,
            "escalated": escalated_val,
            "device_class": device_name,
            "is_ambiguous": is_ambiguous,
            "predicted_class": predicted_class_id,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "shap_attributions": shap_vals
        }
