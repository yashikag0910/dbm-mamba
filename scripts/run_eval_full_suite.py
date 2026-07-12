import os
import sys
import yaml
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc
from src.data.loaders import load_iot_dataset
from src.data.preprocessing import ZScoreNormalizer, log_compact
from src.models.mamba_encoder import MambaEncoder
from src.models.classifier_head import ClassifierHead
from src.runtime.pipeline import DBMambaRuntimePipeline, FastClassifier

def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Computes Precision, Recall, F1, PR-AUC, and False Positive Rate (FPR).
    y_scores must be the continuous anomaly score per sample (not binary prediction)
    for PR-AUC to be meaningful. Passing y_scores=None returns pr_auc=0.0.
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fpr = fp / (fp + tn + 1e-8)

    pr_auc = 0.0
    if y_scores is not None:
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec, prec)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "pr_auc": float(pr_auc)
    }


def run_fgsm_attack(encoder, classifier, x, y, eps=0.05, normalizer=None):
    """
    Applies FGSM perturbation on features to maximize classification loss.
    """
    encoder.eval()
    classifier.eval()

    x_adv = x.clone().detach().requires_grad_(True)

    x_processed = log_compact(x_adv)
    if normalizer is not None and normalizer.is_fitted:
        x_processed = normalizer.transform(x_processed)

    _, z = encoder(x_processed)
    logits = classifier(z)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)

    encoder.zero_grad()
    classifier.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv_perturbed = x + eps * torch.sign(x_adv.grad)

    return x_adv_perturbed.detach()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DBM-Mamba Evaluation Suite")
    parser.add_argument("--dataset", type=str, default="ciciot2023",
                        choices=["ciciot2023", "ton_iot", "iot_23"],
                        help="Primary dataset for base / zero-day / robustness evaluation. "
                             "Use iot_23 or ton_iot for device-aware evaluation (real per-flow "
                             "device identity); ciciot2023 reduces to a single global bank.")
    parser.add_argument("--cross_eval", action="store_true",
                        help="Run cross-dataset validation on ToN_IoT and IoT-23")
    args = parser.parse_args()

    print(f"=== DBM-Mamba Evaluation & Benchmarking Suite ({args.dataset.upper()}) ===")

    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dev_mode = config.get("dev_mode", True)
    seq_len = config["data"]["seq_len"]
    seed = config["data"]["seed"]

    # Per-dataset checkpoint suffix and class count (mirrors the training scripts).
    suffix = f"_{args.dataset}" if args.dataset != "ciciot2023" else ""
    n_classes = {"ciciot2023": 34, "ton_iot": 10, "iot_23": 12}[args.dataset]

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = "results/dev" if dev_mode else "results/final"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Operational Mode: {'DEVELOPMENT (dev_mode)' if dev_mode else 'FULL (paper-ready)'}")
    print(f"Results will be written to: {output_dir}")

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    enc_path = f"checkpoints/finetune_encoder{suffix}.pt"
    banks_path = f"checkpoints/memory_banks{suffix}.pt"
    if not (os.path.exists(enc_path) and os.path.exists(banks_path)):
        print(f"[ERROR] Missing checkpoints ({enc_path} / {banks_path}). "
              f"Run pretrain, finetune, and memory bank scripts for --dataset {args.dataset} first.")
        sys.exit(1)

    normalizer = ZScoreNormalizer(num_features=4)
    norm_path = f"checkpoints/normalizer{suffix}.pt"
    if os.path.exists(norm_path):
        normalizer.load_state_dict(torch.load(norm_path, weights_only=False))
    elif os.path.exists("checkpoints/normalizer.pt"):
        normalizer.load_state_dict(torch.load("checkpoints/normalizer.pt", weights_only=False))

    encoder = MambaEncoder(
        num_features=4,
        d_model=config["model"]["d_model"],
        d_state=config["model"]["d_state"],
        d_conv=config["model"]["d_conv"],
        expand=config["model"]["expand"]
    )
    encoder.load_state_dict(torch.load(enc_path, weights_only=False))
    encoder.to(device)

    classifier = ClassifierHead(d_model=config["model"]["d_model"], n_classes=n_classes)
    classifier.load_state_dict(torch.load(f"checkpoints/finetune_classifier{suffix}.pt", weights_only=False))
    classifier.to(device)

    fast_classifier = FastClassifier(num_features=4, n_classes=n_classes)
    fast_classifier.load_state_dict(torch.load(f"checkpoints/fast_classifier{suffix}.pt", weights_only=False))
    fast_classifier.to(device)

    memory_banks = torch.load(banks_path, weights_only=False)
    threshold_manager = torch.load(f"checkpoints/threshold_manager{suffix}.pt", weights_only=False)

    dataset_path = config["data"]["dataset_paths"][args.dataset]
    dataset = load_iot_dataset(dataset_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type=args.dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    pipeline = DBMambaRuntimePipeline(
        mamba_encoder=encoder,
        classifier_head=classifier,
        fast_classifier=fast_classifier,
        device_mem_banks=memory_banks,
        threshold_manager=threshold_manager,
        normalizer=normalizer,
        device=device
    )

    if args.cross_eval:
        print("\n=== Running Cross-Dataset Generalization & Robustness Evaluation ===")

        ton_path = config["data"]["dataset_paths"]["ton_iot"]
        ton_dataset = load_iot_dataset(ton_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type="ton_iot")
        ton_loader = DataLoader(ton_dataset, batch_size=64, shuffle=False)

        print("\nEvaluating on ToN_IoT...")
        ton_y_true, ton_y_pred, ton_scores = [], [], []
        for x_b, y_b, d_b, macs in ton_loader:
            for idx in range(len(x_b)):
                res = pipeline.process_flow(x_b[idx:idx+1], mac=macs[idx], device_class_id=int(d_b[idx]))
                ton_y_true.append(1 if y_b[idx].item() != 0 else 0)
                ton_y_pred.append(1 if res["is_anomaly"] else 0)
                ton_scores.append(res["anomaly_score"])

        ton_metrics = compute_metrics(ton_y_true, ton_y_pred, ton_scores)
        print(f"ToN_IoT | F1: {ton_metrics['f1']:.4f} | PR-AUC: {ton_metrics['pr_auc']:.4f} | FPR: {ton_metrics['fpr']:.4f}")

        iot23_path = config["data"]["dataset_paths"]["iot_23"]
        iot23_dataset = load_iot_dataset(iot23_path, seq_len=seq_len, dev_mode=dev_mode, dataset_type="iot_23")
        iot23_loader = DataLoader(iot23_dataset, batch_size=64, shuffle=False)

        print("\nEvaluating on IoT-23...")
        iot23_y_true, iot23_y_pred, iot23_scores = [], [], []
        for x_b, y_b, d_b, macs in iot23_loader:
            for idx in range(len(x_b)):
                res = pipeline.process_flow(x_b[idx:idx+1], mac=macs[idx], device_class_id=int(d_b[idx]))
                iot23_y_true.append(1 if y_b[idx].item() != 0 else 0)
                iot23_y_pred.append(1 if res["is_anomaly"] else 0)
                iot23_scores.append(res["anomaly_score"])

        iot23_metrics = compute_metrics(iot23_y_true, iot23_y_pred, iot23_scores)
        print(f"IoT-23 | F1: {iot23_metrics['f1']:.4f} | PR-AUC: {iot23_metrics['pr_auc']:.4f} | FPR: {iot23_metrics['fpr']:.4f}")

        cross_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dev_mode": dev_mode,
            "ton_iot_size": len(ton_dataset),
            "ton_iot_metrics": ton_metrics,
            "iot_23_size": len(iot23_dataset),
            "iot_23_metrics": iot23_metrics
        }
        report_path = os.path.join(output_dir, "cross_dataset_report.json")
        with open(report_path, "w") as f:
            json.dump(cross_report, f, indent=4)
        print(f"\nSaved cross-dataset report to: {report_path}")
        return

    # ------------------------------------------------------------------
    # 1. Base Pipeline Evaluation
    # ------------------------------------------------------------------
    print("\nRunning Base Pipeline Evaluation...")
    y_true_binary, y_pred_binary, y_scores = [], [], []

    for x_b, y_b, d_b, macs in loader:
        for idx in range(len(x_b)):
            res = pipeline.process_flow(x_b[idx:idx+1], mac=macs[idx], device_class_id=int(d_b[idx]))
            y_true_binary.append(1 if y_b[idx].item() != 0 else 0)
            y_pred_binary.append(1 if res["is_anomaly"] else 0)
            y_scores.append(res["anomaly_score"])

    base_metrics = compute_metrics(y_true_binary, y_pred_binary, y_scores)
    print(f"Base | F1: {base_metrics['f1']:.4f} | PR-AUC: {base_metrics['pr_auc']:.4f} | FPR: {base_metrics['fpr']:.4f}")

    # ------------------------------------------------------------------
    # 2. Zero-Day Leave-One-Attack-Class-Out Evaluation
    # ------------------------------------------------------------------
    print("\nRunning Zero-Day (Leave-One-Out) Validation...")
    labels_list = []
    for _, y_b, _, _ in loader:
        labels_list.extend(y_b.tolist())
    unique_labels = sorted(list(set(labels_list)))

    attack_classes = [c for c in unique_labels if c not in (0, 1)]
    classes_to_test = attack_classes[:2] if dev_mode else attack_classes[:5]

    zero_day_results = {}

    for left_out in classes_to_test:
        print(f"  Leaving out Attack Class {left_out}...")
        y_true_subset, y_pred_subset, scores_subset = [], [], []

        for x_b, y_b, d_b, macs in loader:
            eval_mask = (y_b == 0) | (y_b == left_out)
            if not torch.any(eval_mask):
                continue

            x_sub = x_b[eval_mask]
            y_sub = y_b[eval_mask]
            d_sub = d_b[eval_mask]
            macs_sub = [macs[i] for i, m in enumerate(eval_mask) if m]

            for idx in range(len(x_sub)):
                res = pipeline.process_flow(x_sub[idx:idx+1], mac=macs_sub[idx], device_class_id=int(d_sub[idx]))
                y_true_subset.append(1 if y_sub[idx].item() != 0 else 0)
                y_pred_subset.append(1 if res["is_anomaly"] else 0)
                scores_subset.append(res["anomaly_score"])

        subset_metrics = compute_metrics(y_true_subset, y_pred_subset, scores_subset)
        zero_day_results[f"left_out_class_{left_out}"] = subset_metrics
        print(f"    Class {left_out} | F1: {subset_metrics['f1']:.4f} | PR-AUC: {subset_metrics['pr_auc']:.4f}")

    # ------------------------------------------------------------------
    # 3. Robustness Tests
    # ------------------------------------------------------------------
    print("\nRunning Robustness (Distribution Shift) Tests...")

    # A. Timing Jitter
    jitter_y_true, jitter_y_pred, jitter_scores = [], [], []
    for x_b, y_b, d_b, macs in loader:
        x_jitter = x_b.clone()
        jitter = torch.randn(x_jitter.shape[0], x_jitter.shape[1]) * 0.005
        x_jitter[:, :, 2] += jitter
        x_jitter[:, :, 2] = torch.clamp(x_jitter[:, :, 2], min=0.0)

        for idx in range(len(x_jitter)):
            res = pipeline.process_flow(x_jitter[idx:idx+1], mac=macs[idx], device_class_id=int(d_b[idx]))
            jitter_y_true.append(1 if y_b[idx].item() != 0 else 0)
            jitter_y_pred.append(1 if res["is_anomaly"] else 0)
            jitter_scores.append(res["anomaly_score"])  # FIX: was missing

    jitter_metrics = compute_metrics(jitter_y_true, jitter_y_pred, jitter_scores)
    print(f"  Timing Jitter (5ms) | F1: {jitter_metrics['f1']:.4f} | PR-AUC: {jitter_metrics['pr_auc']:.4f}")

    # B. Feature Dropout
    dropout_y_true, dropout_y_pred, dropout_scores = [], [], []
    for x_b, y_b, d_b, macs in loader:
        x_dropout = x_b.clone()
        for i in range(len(x_dropout)):
            if np.random.rand() < 0.10:
                feat_to_drop = np.random.randint(0, 4)
                x_dropout[i, :, feat_to_drop] = 0.0

        for idx in range(len(x_dropout)):
            res = pipeline.process_flow(x_dropout[idx:idx+1], mac=macs[idx], device_class_id=int(d_b[idx]))
            dropout_y_true.append(1 if y_b[idx].item() != 0 else 0)
            dropout_y_pred.append(1 if res["is_anomaly"] else 0)
            dropout_scores.append(res["anomaly_score"])  # FIX: was missing

    dropout_metrics = compute_metrics(dropout_y_true, dropout_y_pred, dropout_scores)
    print(f"  Feature Dropout (10%) | F1: {dropout_metrics['f1']:.4f} | PR-AUC: {dropout_metrics['pr_auc']:.4f}")

    # C. Device-Bank Mismatch (proposal Sec 8.4): intentionally route 5% of
    # flows to the WRONG device bank, simulating device-identification errors.
    # Routing is now driven by device_class_id (real network identity), so the
    # mismatch is injected there rather than by corrupting the MAC.
    bank_ids = [k for k in memory_banks.keys() if k != 0]
    mismatch_y_true, mismatch_y_pred, mismatch_scores = [], [], []
    for x_b, y_b, d_b, macs in loader:
        for idx in range(len(x_b)):
            dev_id = int(d_b[idx])
            if bank_ids and np.random.rand() < 0.05:
                wrong = [b for b in bank_ids if b != dev_id]
                if wrong:
                    dev_id = int(np.random.choice(wrong))
            res = pipeline.process_flow(x_b[idx:idx+1], mac=macs[idx], device_class_id=dev_id)
            mismatch_y_true.append(1 if y_b[idx].item() != 0 else 0)
            mismatch_y_pred.append(1 if res["is_anomaly"] else 0)
            mismatch_scores.append(res["anomaly_score"])

    mismatch_metrics = compute_metrics(mismatch_y_true, mismatch_y_pred, mismatch_scores)
    print(f"  Bank Mismatch (5%) | F1: {mismatch_metrics['f1']:.4f} | PR-AUC: {mismatch_metrics['pr_auc']:.4f}")

    # ------------------------------------------------------------------
    # 4. Adversarial Attack (FGSM)
    # ------------------------------------------------------------------
    print("\nRunning Adversarial Robustness (FGSM) Tests...")
    adv_y_true, adv_y_pred, adv_scores = [], [], []

    for x_b, y_b, d_b, macs in loader:
        x_b_dev = x_b.to(device)
        y_b_dev = y_b.to(device)
        x_adv = run_fgsm_attack(encoder, classifier, x_b_dev, y_b_dev, eps=0.05, normalizer=normalizer)

        for idx in range(len(x_adv)):
            res = pipeline.process_flow(x_adv[idx:idx+1].cpu(), mac=macs[idx], device_class_id=int(d_b[idx]))
            adv_y_true.append(1 if y_b[idx].item() != 0 else 0)
            adv_y_pred.append(1 if res["is_anomaly"] else 0)
            adv_scores.append(res["anomaly_score"])  # FIX: was missing

    adv_metrics = compute_metrics(adv_y_true, adv_y_pred, adv_scores)
    print(f"  Adversarial (FGSM eps=0.05) | F1: {adv_metrics['f1']:.4f} | PR-AUC: {adv_metrics['pr_auc']:.4f}")

    # ------------------------------------------------------------------
    # 5. Latency Profiling
    # ------------------------------------------------------------------
    print("\nRunning Latency Benchmarking...")
    sample_x, _, sample_dev, sample_macs = dataset[0]
    sample_x = sample_x.unsqueeze(0)

    for _ in range(10):
        _ = pipeline.process_flow(sample_x, mac=sample_macs, device_class_id=int(sample_dev))

    iterations = 200
    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = fast_classifier(sample_x.to(device))
    fast_path_latency = (time.perf_counter() - start_time) / iterations * 1000.0

    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            x_p = log_compact(sample_x.to(device))
            if normalizer.is_fitted:
                x_p = normalizer.transform(x_p)
            _, z = encoder(x_p)
            _ = classifier(z)
    slow_path_latency = (time.perf_counter() - start_time) / iterations * 1000.0

    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = pipeline.gate(sample_x.to(device))
    gate_latency = (time.perf_counter() - start_time) / iterations * 1000.0

    print(f"  Fast Classifier Latency: {fast_path_latency:.4f} ms")
    print(f"  Slow Mamba Path Latency: {slow_path_latency:.4f} ms")
    print(f"  Gating Overhead Latency: {gate_latency:.4f} ms")

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    run_metadata = {
        "dev_mode": dev_mode,
        "seed": seed,
        "dataset_size": len(dataset),
        "sampling_method": "Random subsample" if dev_mode else "Full CICIoT2023 dataset",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_metrics": base_metrics,
        "zero_day_validation": zero_day_results,
        "robustness": {
            "timing_jitter_5ms": jitter_metrics,
            "feature_dropout_10pct": dropout_metrics,
            "bank_mismatch_5pct": mismatch_metrics
        },
        "adversarial_robustness_fgsm_0.05": adv_metrics,
        "latency_ms": {
            "fast_classifier": fast_path_latency,
            "slow_mamba_path": slow_path_latency,
            "gating_overhead": gate_latency
        }
    }

    metadata_path = os.path.join(output_dir, "evaluation_report.json")
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f"\nSaved complete evaluation report to: {metadata_path}")


if __name__ == "__main__":
    main()