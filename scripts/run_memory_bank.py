import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.loaders import load_iot_dataset
from src.data.preprocessing import ZScoreNormalizer, log_compact
from src.models.mamba_encoder import MambaEncoder
from src.memory.memory_bank import DeviceMemoryBank
from src.memory.threshold import DeviceThresholdManager

def main():
    print("=== DBM-Mamba Stage 3: Per-Device Behavioral Memory Bank Setup ===")
    
    # Load configuration
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dev_mode = config.get("dev_mode", True)
    seq_len = config["data"]["seq_len"]
    seed = config["data"]["seed"]
    
    torch.manual_seed(seed)
    
    # Check device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device} (Dev Mode: {dev_mode})")
    
    # Load dataset
    dataset_path = config["data"]["dataset_paths"]["ciciot2023"]
    dataset = load_iot_dataset(dataset_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type="ciciot2023")
    
    # Extract features, labels, device classes
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Load normalizer state
    normalizer = ZScoreNormalizer(num_features=4)
    if os.path.exists("checkpoints/normalizer.pt"):
        normalizer.load_state_dict(torch.load("checkpoints/normalizer.pt", weights_only=False))
        print("Loaded Z-Score Normalizer state.")
    else:
        print("[WARNING] Normalizer not found. Fitting default normalizer.")
        # fallback fit
        all_raw = []
        for x_b, _, _, _ in loader:
            all_raw.append(x_b)
            if len(all_raw) * 256 > 5000:
                break
        normalizer.fit(log_compact(torch.cat(all_raw, dim=0)))
        
    # Initialize fine-tuned encoder
    d_model = config["model"]["d_model"]
    d_state = config["model"]["d_state"]
    d_conv = config["model"]["d_conv"]
    expand = config["model"]["expand"]
    
    encoder = MambaEncoder(
        num_features=4,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand
    )
    
    # Load Stage 2 fine-tuned weights
    if os.path.exists("checkpoints/finetune_encoder.pt"):
        encoder.load_state_dict(torch.load("checkpoints/finetune_encoder.pt", weights_only=False))
        print("Loaded fine-tuned Mamba encoder weights.")
    else:
        print("[WARNING] Fine-tuned encoder weights not found. Initializing randomly.")
        
    encoder.to(device)
    encoder.eval()
    
    # We collect frozen representations z for all benign samples
    print("Collecting representations of benign traffic...")
    all_z = []
    all_device_classes = []
    
    with torch.no_grad():
        for x_batch, y_batch, dev_class_batch, _ in loader:
            # Filter benign traffic (label = 0)
            benign_mask = y_batch == 0
            if not torch.any(benign_mask):
                continue
                
            x_benign = x_batch[benign_mask].to(device)
            dev_class_benign = dev_class_batch[benign_mask]
            
            x_processed = log_compact(x_benign)
            if normalizer.is_fitted:
                x_processed = normalizer.transform(x_processed)
                
            _, z = encoder(x_processed)
            
            all_z.append(z.cpu())
            all_device_classes.append(dev_class_benign)
            
    if len(all_z) == 0:
        print("[WARNING] No benign samples found in dataset. Using a small synthetic batch for fitting memory banks.")
        # Create synthetic benign samples
        from src.data.loaders import generate_synthetic_data
        synth_db = generate_synthetic_data(num_samples=1000, seq_len=seq_len)
        synth_loader = DataLoader(synth_db, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x_batch, _, dev_class_batch, _ in synth_loader:
                x_processed = log_compact(x_batch.to(device))
                if normalizer.is_fitted:
                    x_processed = normalizer.transform(x_processed)
                _, z = encoder(x_processed)
                all_z.append(z.cpu())
                all_device_classes.append(dev_class_batch)
                
    z_benign_all = torch.cat(all_z, dim=0)
    dev_benign_all = torch.cat(all_device_classes, dim=0)
    
    print(f"Collected {len(z_benign_all)} benign sample embeddings.")
    
    # Set up per-device memory banks
    device_mem_banks = {}
    threshold_manager = DeviceThresholdManager(
        buffer_size=config["memory"]["buffer_size"],
        update_interval=config["memory"]["recompute_interval"],
        benign_percentile=config["memory"]["benign_percentile"],
        generic_percentile=config["memory"]["generic_percentile"]
    )
    
    # Distinct device classes (0 is Generic, 1..N are specific)
    # Ensure at least class 0 (Generic) is initialized
    unique_devices = torch.unique(dev_benign_all).tolist()
    if 0 not in unique_devices:
        unique_devices.append(0)
        
    for dev_class in unique_devices:
        print(f"Fitting Memory Bank for Device Class {dev_class}...")
        dev_mask = dev_benign_all == dev_class
        
        # If no samples, fit on the generic benign representation
        if not torch.any(dev_mask) or len(z_benign_all[dev_mask]) < 5:
            print(f"  No/few samples for Device Class {dev_class}. Initializing from generic benign embeddings.")
            z_dev = z_benign_all
        else:
            z_dev = z_benign_all[dev_mask]
            
        bank = DeviceMemoryBank(device_class_id=dev_class, k_candidates=config["memory"]["k_candidates"])
        bank.fit(z_dev.to(device))
        
        device_mem_banks[dev_class] = bank
        print(f"  Device Class {dev_class}: Selected optimal clusters K = {bank.optimal_k}")
        
        # Compute anomaly scores to seed the threshold manager circular buffer
        with torch.no_grad():
            scores = bank.compute_anomaly_score(z_dev.to(device)).cpu().numpy()
            
        # Add to threshold manager
        threshold_manager.get_tracker(dev_class).update(scores)
        # Recalculate initial threshold dynamically on the buffer contents
        threshold_manager.get_tracker(dev_class).recompute_threshold()
        print(f"  Device Class {dev_class}: Initialized threshold tau_d = {threshold_manager.get_threshold(dev_class):.4f}")
        
    # Initialize the fallback generic bank tracker
    with torch.no_grad():
        generic_bank = device_mem_banks[0]
        generic_scores = generic_bank.compute_anomaly_score(z_benign_all.to(device)).cpu().numpy()
        threshold_manager.generic_tracker.update(generic_scores)
        threshold_manager.generic_tracker.recompute_threshold()
        print(f"Generic IoT Fallback Bank: Initialized threshold tau_generic = {threshold_manager.get_threshold(0, is_ambiguous=True):.4f}")
        
    # Save the fitted memory banks and threshold manager
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(device_mem_banks, "checkpoints/memory_banks.pt")
    torch.save(threshold_manager, "checkpoints/threshold_manager.pt")
    print("Saved Device Memory Banks to checkpoints/memory_banks.pt")
    print("Saved Threshold Manager state to checkpoints/threshold_manager.pt")

if __name__ == "__main__":
    main()
