import os
import yaml
import torch
import numpy as np
from src.data.loaders import generate_synthetic_data
from src.data.preprocessing import ZScoreNormalizer, log_compact
from src.models.mamba_encoder import MambaEncoder
from src.models.classifier_head import ClassifierHead
from src.runtime.pipeline import DBMambaRuntimePipeline, FastClassifier
from src.memory.online_refinement import OnlineMemoryRefiner

def main():
    print("=====================================================================")
    print("                  DBM-Mamba Pipeline Demo Run                        ")
    print("=====================================================================")
    
    # Load configuration
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dev_mode = config.get("dev_mode", True)
    seq_len = config["data"]["seq_len"]
    
    # Check device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    # Generate synthetic sequence samples for demo
    print("Generating demo sequence flows...")
    dataset = generate_synthetic_data(num_samples=100, seq_len=seq_len)
    
    # Load normalizer
    normalizer = ZScoreNormalizer(num_features=4)
    if os.path.exists("checkpoints/normalizer.pt"):
        normalizer.load_state_dict(torch.load("checkpoints/normalizer.pt", weights_only=False))
    else:
        # Fit on dataset
        normalizer.fit(log_compact(dataset.sequences))
        
    # Load models
    d_model = config["model"]["d_model"]
    d_state = config["model"]["d_state"]
    d_conv = config["model"]["d_conv"]
    expand = config["model"]["expand"]
    n_classes = config["model"]["n_classes"]
    
    encoder = MambaEncoder(num_features=4, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    classifier = ClassifierHead(d_model=d_model, n_classes=n_classes)
    fast_classifier = FastClassifier(num_features=4, n_classes=n_classes)
    
    # Load weights if available, else use random initializations
    if os.path.exists("checkpoints/finetune_encoder.pt"):
        encoder.load_state_dict(torch.load("checkpoints/finetune_encoder.pt", weights_only=False))
        print("Loaded fine-tuned Mamba encoder.")
    if os.path.exists("checkpoints/finetune_classifier.pt"):
        classifier.load_state_dict(torch.load("checkpoints/finetune_classifier.pt", weights_only=False))
        print("Loaded fine-tuned classifier head.")
    if os.path.exists("checkpoints/fast_classifier.pt"):
        fast_classifier.load_state_dict(torch.load("checkpoints/fast_classifier.pt", weights_only=False))
        print("Loaded fast classifier path.")
        
    # Load memory banks and threshold manager
    if os.path.exists("checkpoints/memory_banks.pt"):
        memory_banks = torch.load("checkpoints/memory_banks.pt", weights_only=False)
        threshold_manager = torch.load("checkpoints/threshold_manager.pt", weights_only=False)
        print("Loaded Memory Banks & Threshold Manager.")
    else:
        # Create mock memory banks and thresholds for demonstration
        print("[WARNING] Checkpoints not found. Creating mock memory banks for demonstration.")
        from src.memory.memory_bank import DeviceMemoryBank
        from src.memory.threshold import DeviceThresholdManager
        
        # Fit memory bank for class 0 (Generic) and class 3 (SmartPlug)
        encoder.to(device)
        encoder.eval()
        with torch.no_grad():
            x_proc = log_compact(dataset.sequences)
            if normalizer.is_fitted:
                x_proc = normalizer.transform(x_proc)
            _, z_all = encoder(x_proc.to(device))
            
        memory_banks = {}
        threshold_manager = DeviceThresholdManager()
        
        for dev_class in [0, 1, 2, 3, 4, 5]:
            bank = DeviceMemoryBank(device_class_id=dev_class)
            bank.fit(z_all)
            memory_banks[dev_class] = bank
            
            # Seed thresholds
            scores = bank.compute_anomaly_score(z_all).cpu().numpy()
            threshold_manager.get_tracker(dev_class).update(scores)
            threshold_manager.get_tracker(dev_class).recompute_threshold()
            
        # seed generic
        threshold_manager.generic_tracker.update(scores)
        threshold_manager.generic_tracker.recompute_threshold()
        
    # Instantiate Pipeline
    pipeline = DBMambaRuntimePipeline(
        mamba_encoder=encoder,
        classifier_head=classifier,
        fast_classifier=fast_classifier,
        device_mem_banks=memory_banks,
        threshold_manager=threshold_manager,
        normalizer=normalizer,
        device=device
    )
    
    # Select 20 benign sequences as background for SHAP
    bg_samples = dataset.sequences[:20]
    pipeline.initialize_shap(bg_samples, device_class_id=3)
    print("SHAP Explainer initialized.")
    
    # Process a few specific samples
    print("\n---------------------------------------------------------------------")
    print("Processing Sample Flows:")
    print("---------------------------------------------------------------------")
    
    # We choose 3 distinct samples to show gating and anomaly behavior
    # Sample A: Standard benign-looking flow
    x_benign = dataset.sequences[0].clone()
    x_benign[:, 0] = 60 # short packets (e.g. control)
    x_benign[:, 2] = 0.5 # normal IAT
    x_benign[:, 3] = 0.05 # normal flags
    
    # Sample B: DDoS Syn-Flood (causes TCP escalation and anomaly detection)
    x_ddos = dataset.sequences[1].clone()
    x_ddos[:, 0] = 1200 # large packets
    x_ddos[:, 2] = 0.001 # extremely small IAT (high rate)
    x_ddos[:, 3] = 0.9 # high TCP flags (e.g. SYN flood)
    
    # Sample C: Ambiguous flow with high entropy (causes Mamba slow-path gating)
    x_ambig = dataset.sequences[2].clone()
    # Alternate lengths to create high packet length entropy
    x_ambig[0::2, 0] = 60
    x_ambig[1::2, 0] = 1400
    x_ambig[:, 2] = 0.1
    x_ambig[:, 3] = 0.1
    
    test_cases = [
        ("Case 1: Standard benign-looking flow", x_benign, "00:1e:c0:b4:a1:02", None), # MAC indicates SmartPlug
        ("Case 2: DDoS Syn-Flood Attack", x_ddos, "00:1e:c0:b4:a1:02", None),
        ("Case 3: High-Entropy Flow (Ambiguous Device)", x_ambig, None, None) # No MAC (routes to Generic)
    ]
    
    for desc, seq, mac, tls in test_cases:
        print(f"\n>> Running {desc}...")
        decision = pipeline.process_flow(seq, mac=mac, tls_hello=tls)
        
        print(f"  Identified Device: {decision['device_class']} (Ambiguous: {decision['is_ambiguous']})")
        print(f"  Inference Pathway: {decision['path']} (TCP Escalation: {decision['escalated']})")
        print(f"  Gate High-Complexity Probability: {decision['gate_probability']:.4f}")
        print(f"  Predicted Intrusion Class ID: {decision['predicted_class']} (Confidence: {decision['confidence']:.4f})")
        print(f"  Memory Anomaly Score S(z): {decision['anomaly_score']:.4f} (Threshold: {pipeline.threshold_manager.get_threshold(3 if mac else 0, decision['is_ambiguous']):.4f})")
        print(f"  Intrusion Flagged: {decision['is_anomaly']}")
        
        # If anomaly, show SHAP
        if decision['is_anomaly'] and decision['shap_attributions'] is not None:
            print("  SHAP Feature Attributions (Mean over sequence steps):")
            feat_names = ["Length", "Direction", "IAT", "Flags"]
            attributions = np.mean(np.abs(decision['shap_attributions']), axis=0)
            for f_idx, f_name in enumerate(feat_names):
                print(f"    - {f_name}: {attributions[f_idx]:.6f}")
                
    # -------------------------------------------------------------
    # Online Centroid Refinement Demo
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------------------")
    print("Online Centroid Refinement (EMA + Triplet Loss updates) Demo:")
    print("---------------------------------------------------------------------")
    refiner = OnlineMemoryRefiner(beta=0.01, lr=0.05)
    
    # Take benign representation z from Case 1
    with torch.no_grad():
        x_proc = log_compact(x_benign.unsqueeze(0))
        if normalizer.is_fitted:
            x_proc = normalizer.transform(x_proc)
        _, z_benign = encoder(x_proc.to(device))
        
    # Select a valid device class present in memory banks, fallback to 0
    active_device_class = 3 if 3 in memory_banks else 0
    bank = memory_banks[active_device_class]
    old_centroid_val = bank.centroids[0].clone()
    
    loss_val, same_idx, other_idx = refiner.refine_step(
        z=z_benign[0],
        memory_bank=bank,
        other_banks=list(memory_banks.values())
    )
    
    new_centroid_val = bank.centroids[0]
    drift = torch.norm(new_centroid_val - old_centroid_val).item()
    print(f"  Fitted streaming benign sample to Cluster Centroid index: {same_idx}")
  # Check if other_idx is valid before printing
    other_msg = f"Mined Negative Centroid: {other_idx}" if other_idx != -1 else "No negative centroid (K=1)"
    print(f"  {other_msg}")
    print(f"  Triplet loss computed: {loss_val:.6f}")
    print(f"  Centroid drift ||c_new - c_old||_2 after EMA + Gradient Step: {drift:.6f}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
