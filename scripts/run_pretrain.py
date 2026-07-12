import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.loaders import load_iot_dataset
from src.data.preprocessing import ZScoreNormalizer, log_compact
from src.models.mamba_encoder import MambaEncoder
from src.models.pretrain import PretrainModel, train_pretrain_epoch

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DBM-Mamba Stage 1 Pretraining")
    parser.add_argument("--dataset", type=str, default="ciciot2023", choices=["ciciot2023", "ton_iot", "iot_23"], help="Dataset to use")
    args = parser.parse_args()

    print(f"=== DBM-Mamba Stage 1: Masked Packet Pretraining ({args.dataset.upper()}) ===")
    
    # Load configuration
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dev_mode = config.get("dev_mode", True)
    seq_len = config["data"]["seq_len"]
    mask_ratio = config["data"]["mask_ratio"]
    seed = config["data"]["seed"]
    
    torch.manual_seed(seed)
    
    # Check device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device} (Dev Mode: {dev_mode})")
    
    # Determine batch size and epochs
    batch_size = config["data"]["batch_size_dev"] if dev_mode else config["data"]["batch_size_full"]
    epochs = config["finetune"]["epochs_dev"] if dev_mode else config["finetune"]["epochs_full"]
    
    # Load dataset
    dataset_path = config["data"]["dataset_paths"][args.dataset]
    dataset = load_iot_dataset(dataset_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type=args.dataset)
    
    # Split training / validation
    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Fit normalizer on raw inputs from training set
    print("Fitting Z-Score Normalizer on training data...")
    normalizer = ZScoreNormalizer(num_features=4)
    # Extract raw features from training loader
    all_raw = []
    for x_b, _, _, _ in train_loader:
        all_raw.append(x_b)
        if len(all_raw) * batch_size > 10000: # limit to first 10k samples for speed
            break
    all_raw_tensor = torch.cat(all_raw, dim=0)
    # Apply log compaction first before fitting z-score
    all_compacted = log_compact(all_raw_tensor)
    normalizer.fit(all_compacted)
    
    # Initialize Mamba Encoder and Pretrain Model
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
    
    pretrain_model = PretrainModel(encoder, num_features=4).to(device)
    optimizer = optim.AdamW(pretrain_model.parameters(), lr=config["finetune"]["lr"], weight_decay=1e-4)
    
    # Pretraining loop
    print(f"Starting pretraining for {epochs} epochs...")
    for epoch in range(epochs):
        loss = train_pretrain_epoch(
            model=pretrain_model,
            dataloader=train_loader,
            optimizer=optimizer,
            normalizer=normalizer,
            mask_ratio=mask_ratio,
            device=device
        )
        print(f"Epoch {epoch+1}/{epochs} | Reconstruction Loss (MSE): {loss:.6f}")
        
    # Create checkpoints folder
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save encoder and normalizer states
    suffix = f"_{args.dataset}" if args.dataset != "ciciot2023" else ""
    torch.save(encoder.state_dict(), f"checkpoints/pretrain_encoder{suffix}.pt")
    torch.save(normalizer.state_dict(), f"checkpoints/normalizer{suffix}.pt")
    print(f"Saved pretrained encoder to checkpoints/pretrain_encoder{suffix}.pt")
    print(f"Saved normalizer state to checkpoints/normalizer{suffix}.pt")

if __name__ == "__main__":
    main()
