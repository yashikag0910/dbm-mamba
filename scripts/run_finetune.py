import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.loaders import load_iot_dataset
from src.data.preprocessing import ZScoreNormalizer, log_compact
from src.models.mamba_encoder import MambaEncoder
from src.models.classifier_head import ClassifierHead
from src.models.focal_loss import FocalLoss
from src.models.parametric_umap import ParametricUMAPLoss
from src.models.finetune import JointFinetuningModel, get_lambda_umap, train_finetune_epoch
from src.runtime.pipeline import FastClassifier

def train_fast_classifier(model, dataloader, normalizer, lr, epochs, device):
    """
    Trains the lightweight fast classifier on raw/preprocessed sequence data.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Training Fast Classifier path...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for x_batch, y_batch, _, _ in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Preprocess
            x_processed = log_compact(x_batch)
            if normalizer.is_fitted:
                x_processed = normalizer.transform(x_processed)
                
            optimizer.zero_grad()
            logits = model(x_processed)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        print(f"  Fast Classifier Epoch {epoch+1}/{epochs} | Loss: {total_loss/max(1, num_batches):.6f}")


def main():
    print("=== DBM-Mamba Stage 2: Joint Classification & Parametric UMAP Finetuning ===")
    
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
    
    # Determine batch size and epochs
    batch_size = config["data"]["batch_size_dev"] if dev_mode else config["data"]["batch_size_full"]
    epochs = config["finetune"]["epochs_dev"] if dev_mode else config["finetune"]["epochs_full"]
    
    # Load dataset
    dataset_path = config["data"]["dataset_paths"]["ciciot2023"]
    dataset = load_iot_dataset(dataset_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type="ciciot2023")
    
    # Split training / validation
    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load normalizer state
    normalizer = ZScoreNormalizer(num_features=4)
    if os.path.exists("checkpoints/normalizer.pt"):
        normalizer.load_state_dict(torch.load("checkpoints/normalizer.pt", weights_only=False))
        print("Loaded Z-Score Normalizer state.")
    else:
        print("[WARNING] Normalizer checkpoint not found. Creating fit from training loader.")
        # fallback fit
        all_raw = []
        for x_b, _, _, _ in train_loader:
            all_raw.append(x_b)
            if len(all_raw) * batch_size > 5000:
                break
        normalizer.fit(log_compact(torch.cat(all_raw, dim=0)))
        
    # Initialize Models
    d_model = config["model"]["d_model"]
    d_state = config["model"]["d_state"]
    d_conv = config["model"]["d_conv"]
    expand = config["model"]["expand"]
    n_classes = config["model"]["n_classes"]
    
    encoder = MambaEncoder(
        num_features=4,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand
    )
    
    # Load Stage 1 weights
    if os.path.exists("checkpoints/pretrain_encoder.pt"):
        encoder.load_state_dict(torch.load("checkpoints/pretrain_encoder.pt", weights_only=False))
        print("Loaded pretrained Mamba encoder weights.")
    else:
        print("[WARNING] Pretrained encoder checkpoint not found. Initializing randomly.")
        
    classifier = ClassifierHead(d_model=d_model, n_classes=n_classes)
    joint_model = JointFinetuningModel(encoder, classifier).to(device)
    
    # Fast Classifier
    fast_classifier = FastClassifier(num_features=4, n_classes=n_classes).to(device)
    
    # Loss functions
    focal_alpha = config["finetune"]["focal_alpha"]
    focal_gamma = config["finetune"]["focal_gamma"]
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    umap_k = config["finetune"]["umap_k"]
    umap_loss_fn = ParametricUMAPLoss(k=umap_k)
    
    optimizer = optim.AdamW(joint_model.parameters(), lr=config["finetune"]["lr"], weight_decay=1e-4)
    
    lambda_init = config["finetune"]["lambda_umap_init"]
    lambda_final = config["finetune"]["lambda_umap_final"]
    
    print(f"Starting joint finetuning for {epochs} epochs...")
    for epoch in range(epochs):
        # Calculate annealed lambda
        lambda_umap = get_lambda_umap(epoch, epochs, lambda_init, lambda_final, schedule='linear')
        
        loss, focal_loss, umap_loss = train_finetune_epoch(
            model=joint_model,
            dataloader=train_loader,
            optimizer=optimizer,
            normalizer=normalizer,
            focal_loss_fn=focal_loss_fn,
            umap_loss_fn=umap_loss_fn,
            lambda_umap=lambda_umap,
            device=device
        )
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss:.5f} | Focal Loss: {focal_loss:.5f} | UMAP Loss: {umap_loss:.5f} | Lambda: {lambda_umap:.4f}")
        
    # Train Fast Path Classifier
    train_fast_classifier(fast_classifier, train_loader, normalizer, lr=0.002, epochs=max(1, epochs), device=device)
    
    # Save finetuning states
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/finetune_encoder.pt")
    torch.save(classifier.state_dict(), "checkpoints/finetune_classifier.pt")
    torch.save(fast_classifier.state_dict(), "checkpoints/fast_classifier.pt")
    print("Saved fine-tuned encoder to checkpoints/finetune_encoder.pt")
    print("Saved fine-tuned classifier head to checkpoints/finetune_classifier.pt")
    print("Saved fast path classifier to checkpoints/fast_classifier.pt")

if __name__ == "__main__":
    main()
