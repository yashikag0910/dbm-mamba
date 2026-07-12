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
    import argparse
    parser = argparse.ArgumentParser(description="DBM-Mamba Stage 3 Memory Bank Fitting")
    parser.add_argument("--dataset", type=str, default="ciciot2023",
                        choices=["ciciot2023", "ton_iot", "iot_23"],
                        help="Dataset to use")
    args = parser.parse_args()

    print(f"=== DBM-Mamba Stage 3: Per-Device Behavioral Memory Bank Setup ({args.dataset.upper()}) ===")

    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dev_mode = config.get("dev_mode", True)
    seq_len = config["data"]["seq_len"]
    seed = config["data"]["seed"]

    torch.manual_seed(seed)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device} (Dev Mode: {dev_mode})")

    dataset_path = config["data"]["dataset_paths"][args.dataset]
    dataset = load_iot_dataset(dataset_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type=args.dataset)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    suffix = f"_{args.dataset}" if args.dataset != "ciciot2023" else ""
    normalizer = ZScoreNormalizer(num_features=4)
    norm_path = f"checkpoints/normalizer{suffix}.pt"

    if os.path.exists(norm_path):
        normalizer.load_state_dict(torch.load(norm_path, weights_only=False))
        print(f"Loaded Z-Score Normalizer state from {norm_path}.")
    else:
        print("[WARNING] Normalizer not found. Fitting default normalizer.")
        all_raw = []
        for x_b, _, _, _ in loader:
            all_raw.append(x_b)
            if len(all_raw) * 256 > 5000:
                break
        normalizer.fit(log_compact(torch.cat(all_raw, dim=0)))

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

    ft_encoder_path = f"checkpoints/finetune_encoder{suffix}.pt"
    if os.path.exists(ft_encoder_path):
        encoder.load_state_dict(torch.load(ft_encoder_path, weights_only=False))
        print(f"Loaded fine-tuned Mamba encoder weights from {ft_encoder_path}.")
    elif os.path.exists("checkpoints/finetune_encoder.pt"):
        encoder.load_state_dict(torch.load("checkpoints/finetune_encoder.pt", weights_only=False))
        print("Loaded fine-tuned Mamba encoder weights from checkpoints/finetune_encoder.pt.")
    else:
        print("[WARNING] Fine-tuned encoder weights not found. Initializing randomly.")

    encoder.to(device)
    encoder.eval()

    print("Collecting representations of benign traffic...")
    all_z = []
    all_device_classes = []

    with torch.no_grad():
        for x_batch, y_batch, dev_class_batch, _ in loader:
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

    # Print per-device sample counts so you can spot underrepresented classes
    unique_devices_counts = {int(d): int((dev_benign_all == d).sum()) for d in torch.unique(dev_benign_all)}
    print(f"Benign samples per device class: {unique_devices_counts}")

    device_mem_banks = {}
    threshold_manager = DeviceThresholdManager(
        buffer_size=config["memory"]["buffer_size"],
        update_interval=config["memory"]["recompute_interval"],
        benign_percentile=config["memory"]["benign_percentile"],
        generic_percentile=config["memory"]["generic_percentile"]
    )

    unique_devices = torch.unique(dev_benign_all).tolist()
    if 0 not in unique_devices:
        unique_devices.append(0)

    min_samples_required = min(config["memory"]["k_candidates"]) + 1

    for dev_class in unique_devices:
        dev_class = int(dev_class)
        print(f"Fitting Memory Bank for Device Class {dev_class}...")

        dev_mask = dev_benign_all == dev_class
        n_samples = int(dev_mask.sum())

        if n_samples < min_samples_required:
            # Do NOT fall back to all-device embeddings -- that would give this
            # device class centroids spanning the entire benign distribution,
            # making its anomaly threshold far too broad and killing recall.
            # Instead, skip fitting and let process_flow route to the generic
            # bank (device_class_id=0) which is correctly fitted on all benign
            # data and is the designed fallback for ambiguous/sparse devices.
            print(f"  Device Class {dev_class}: only {n_samples} benign samples "
                  f"(need >= {min_samples_required}). Skipping -- will route to generic bank.")
            continue

        z_dev = z_benign_all[dev_mask]
        bank = DeviceMemoryBank(
            device_class_id=dev_class,
            k_candidates=config["memory"]["k_candidates"]
        )
        bank.fit(z_dev.to(device))
        device_mem_banks[dev_class] = bank
        print(f"  Device Class {dev_class}: fitted K={bank.optimal_k} centroids "
              f"on {n_samples} benign samples.")

        # Seed threshold manager with this device's benign anomaly scores
        # and force an immediate recompute so we don't use the hardcoded
        # initial_threshold=1.5 during evaluation.
        with torch.no_grad():
            scores = bank.compute_anomaly_score(z_dev.to(device)).cpu().numpy()
        threshold_manager.get_tracker(dev_class).update(scores)
        threshold_manager.get_tracker(dev_class).recompute_threshold()
        print(f"  Device Class {dev_class}: threshold tau_d = "
              f"{threshold_manager.get_threshold(dev_class):.4f}")

    # Generic bank (class 0) must always exist as the fallback.
    # If it wasn't fitted above (e.g. no device-0 benign samples),
    # fit it on all benign embeddings now.
    if 0 not in device_mem_banks:
        print("Fitting Generic IoT fallback bank (class 0) on all benign embeddings...")
        generic_bank = DeviceMemoryBank(
            device_class_id=0,
            k_candidates=config["memory"]["k_candidates"]
        )
        generic_bank.fit(z_benign_all.to(device))
        device_mem_banks[0] = generic_bank
        print(f"  Generic bank: fitted K={generic_bank.optimal_k} centroids "
              f"on {len(z_benign_all)} benign samples.")

    # Seed the generic tracker with all benign scores from the generic bank
    with torch.no_grad():
        generic_scores = device_mem_banks[0].compute_anomaly_score(
            z_benign_all.to(device)
        ).cpu().numpy()
    threshold_manager.generic_tracker.update(generic_scores)
    threshold_manager.generic_tracker.recompute_threshold()
    print(f"Generic IoT Fallback: threshold tau_generic = "
          f"{threshold_manager.get_threshold(0, is_ambiguous=True):.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(device_mem_banks, f"checkpoints/memory_banks{suffix}.pt")
    torch.save(threshold_manager, f"checkpoints/threshold_manager{suffix}.pt")
    print(f"\nSaved Device Memory Banks to checkpoints/memory_banks{suffix}.pt")
    print(f"Saved Threshold Manager to checkpoints/threshold_manager{suffix}.pt")
    print(f"Fitted banks: {sorted(device_mem_banks.keys())}")


if __name__ == "__main__":
    main()