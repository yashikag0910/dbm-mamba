# DBM-Mamba (Device-Aware Behavioral Memory for IoT Gateway Intrusion Detection)

DBM-Mamba is a three-stage IoT gateway intrusion detection architecture implementing selective Mamba sequence modeling, joint optimization with Parametric UMAP, per-device behavioral memory banks, and adaptive inference gating.

## Architectural Mapping to Proposal Sections

| Module File | Proposal Section & Details | Key Equations / Functionality |
| :--- | :--- | :--- |
| `src/data/preprocessing.py` | Sec 4 / 5.1: Packet Representation | Log compaction: $f(x) = \log(1+x)$ & Masking $M(X)$ |
| `src/data/loaders.py` | Sec 4: Token Sequences | Sequence formulation $p_t = [L_t, D_t, IAT_t, F_t]$ |
| `src/models/mamba_encoder.py` | Sec 4.1: Selective SSM Encoder | ZOH Discretization & Selective scan fallback |
| `src/models/pretrain.py` | Sec 4.2: Masked Pretraining | Stage 1: Masked packet attribute reconstruction |
| `src/models/parametric_umap.py` | Sec 5.2: Parametric UMAP | Fuzzy cross-entropy over mini-batch k-NN graph |
| `src/models/focal_loss.py` | Sec 5.1: Multi-class Focal Loss | Focal loss for imbalanced classes |
| `src/models/finetune.py` | Sec 5: Stage 2 Joint Optimization | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda \mathcal{L}_{\text{UMAP}}$ |
| `src/memory/memory_bank.py` | Sec 6.1: Behavioral Memory | Per-device K-Means & anomaly score $S(z) = \min \|z - c\|_2$ |
| `src/memory/threshold.py` | Sec 6.2: Dynamic Thresholding | Sliding percentile circular buffer ($N=10,000$, $P_{99}$) |
| `src/memory/online_refinement.py` | Sec 6.3: Online Adaptation | Streaming triplet loss refinement & EMA updates |
| `src/gating/gate.py` | Sec 7: Adaptive Inference Gating | Entropy & Var gating, self-distillation, TCP escalation |
| `src/device_id/fingerprinting.py` | Sec 7.1: Identity Profiler | MAC OUI, JA3/JA4, DHCP majority vote |
| `src/quantization/quantize.py` | Sec 8: Edge Quantization | Outlier-aware log-scale SSM + INT8 linear quantization |
| `src/explain/shap_attribution.py` | Explainability | SHAP feature attribution for flagged anomalies |
| `src/runtime/pipeline.py` | Sec 9: Gateway Runtime Pipeline | Unified real-time packet stream inference |

---

## Installation & Setup

1. **Clone & Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hardware Acceleration:**
   - On **macOS (Apple Silicon)**, the code runs on MPS (`mps`) or CPU. The built-in pure-PyTorch selective scan fallback will run out-of-the-box.
   - On **NVIDIA GPUs**, you can optionally install native Mamba CUDA kernels:
     ```bash
     pip install mamba-ssm causal-conv1d
     ```

---

## How to Run

### Development Mode (`dev_mode: true`)
By default, `configs/default.yaml` has `dev_mode: true`. This runs the scripts with  a generated synthetic dataset to quickly verify shapes, components, and pipelines in less than 2 minutes.

To run the pipeline in dev mode:
```bash
python scripts/run_demo_pipeline.py
```

### Full-Scale Mode (`dev_mode: false`)
1. Download the CICIoT2023, ToN_IoT, or IoT-23 datasets.
2. Edit `configs/default.yaml` to set `dev_mode: false` and point the `dataset_paths` to the folders containing the real CSV data.
3. Run the train and evaluation scripts:
   ```bash
   python scripts/run_pretrain.py
   python scripts/run_finetune.py
   python scripts/run_memory_bank.py
   python scripts/run_eval_full_suite.py
   ```

Note: `run_eval_full_suite.py` enforces that final paper-ready metrics are ONLY written to `results/final` when `dev_mode` is disabled. When `dev_mode` is enabled, all outputs are diverted to `results/dev`.



