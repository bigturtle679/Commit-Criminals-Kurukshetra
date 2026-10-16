# NeuroScope 3D — Model Card

## Model Overview

| Property | Value |
|---|---|
| **Name** | NeuroScope 8:2 Hybrid Ensemble |
| **Task** | 3D Brain Tumor Segmentation (BraTS-style) |
| **Output** | WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor) |
| **Label Convention** | BraTS 2021: 0=Background, 1=NCR, 2=ED, 4=ET |

## Architecture

The ensemble fuses two models at the probability level:

| Component | Architecture | Weight | Input Order | Source |
|---|---|---|---|---|
| **Model A** | SwinUNETR (Vision Transformer) | 80% | FLAIR, T1c, T1, T2 | MONAI / BraTS21 |
| **Model B** | nnU-Net ResEnc M (Residual Encoder UNet) | 20% | FLAIR, T1, T1c, T2 | nnU-Net v2 |

### Fusion Method
- Both models produce 3-channel sigmoid probabilities (TC, WT, ET)
- Weighted average: `prob = 0.8 × prob_A + 0.2 × prob_B` (or configured logit fusion)
- BraTS hierarchy enforcement: ET ≤ TC ≤ WT
- Threshold: 0.5
- Shannon Bernoulli entropy uncertainty map ($H(p) = -p\log_2 p - (1-p)\log_2(1-p)$) representing voxel classification boundary confidence
- Multi-model prediction disagreement ($|p_A - p_B|$), measuring genuine inter-model divergence between SwinUNETR and ResEnc M predictions

## Preprocessing

| Step | Details |
|---|---|
| Orientation | Canonical RAS via `nibabel.as_closest_canonical` |
| Resampling | 1mm isotropic via `resample_to_output` |
| Normalization | Per-channel z-score on nonzero voxels (`monai.NormalizeIntensity`) |
| Cropping | Foreground bounding box with 4-voxel padding |
| Inference | Sliding window (96³ patches, 50% overlap, Gaussian weighting) |

## Required Checkpoint Files

> **These files are NOT included in the Git repository. They must be obtained separately.**

| Model | Expected Path | Format |
|---|---|---|
| SwinUNETR | `models/swinunetr_brats/model_weights_BRATS-finetune.pt` | PyTorch state dict |
| ResEnc M | `models/resenc_brats/resenc_m_brats.pt` | PyTorch state dict |

Alternative paths (checked in order):
- SwinUNETR: `work/models/swinunetr_fold1.pt`
- ResEnc: `work/models/resenc_brats.pt`

If a checkpoint is not found at any of these locations, the application will raise a
`CheckpointNotFoundError` with the list of searched paths.

## Checkpoint Provenance & Verification

> **Integrity Hash Status: SYSTEM IMPLEMENTED · Provenance Status: UNVERIFIED**
>
> The application calculates SHA-256 hashes of the exact checkpoint files loaded for each inference run
> and records them in the output report (`report.json` and `evaluation_summary.json`).
>
> **Important Scientific Distinction**:
> SHA-256 hashes verify byte-level file integrity and run-to-run reproducibility. A cryptographic hash
> **does not** establish model authorship, training data provenance, licensing compliance, or clinical validity.
> The specific checkpoint files used in this project have not been independently audited or clinically certified.

### Independent Checkpoint Hash Verification:

To independently verify the SHA-256 hash of a local checkpoint file:

```bash
# Linux / macOS
sha256sum models/swinunetr_brats/model_weights_BRATS-finetune.pt
sha256sum models/resenc_brats/resenc_m_brats.pt

# Windows PowerShell
Get-FileHash models/swinunetr_brats/model_weights_BRATS-finetune.pt -Algorithm SHA256
Get-FileHash models/resenc_brats/resenc_m_brats.pt -Algorithm SHA256
```

### What must be verified before claiming benchmark results:

1. Training dataset origin and de-identification provenance (e.g. BraTS 2021, MSD, or other)
2. Training/validation split used during model training
3. Any fine-tuning performed after the original training
4. Commercial and research license terms of the checkpoint weights
5. Explicit confirmation that evaluation cases do not overlap with training data

## Intended Use

- **Primary use**: Research and educational demonstration of 3D multi-modal brain tumor segmentation
- **NOT intended for**: Primary clinical diagnosis, treatment planning, radiation therapy contouring, or medical decision-making
- **Users**: Researchers, machine learning engineers, educators, and hackathon evaluators

## Limitations

1. **Not a diagnostic tool** — Output represents statistical model estimates, not expert-verified clinical findings
2. **Dataset bias** — Models trained on BraTS data (adult gliomas from specific participating academic centers)
3. **Preprocessing dependency** — Predictions assume skull-stripped, co-registered MRI inputs on a 1mm isotropic grid
4. **No clinical validation** — Algorithmic Dice scores on benchmarks do not equate to clinical efficacy or patient safety
5. **Population generalization** — Performance may degrade on scans from unseen scanners, field strengths (e.g., 1.5T vs 3.0T), non-standard protocols, or pediatric populations

## Evaluation Protocol & Status

The evaluation suite (`scripts/evaluate_ensemble.py`) evaluates multi-modal cases against expert ground-truth masks:

```bash
# Run patient-level held-out evaluation
python scripts/evaluate_ensemble.py --manifest configs/evaluation_manifest_template.json --output-dir outputs/evaluation --device cuda
```

| Dimension | Verification Status | Current Evidence |
|---|---|---|
| **Pipeline Consolidation** | ✅ COMPLETE | Single authoritative `run_ensemble` path; `pipeline.py` delegates directly |
| **Config Validation** | ✅ COMPLETE | `config_loader.py` enforces strict types, non-finite rejection, and normalized weights |
| **Safe Checkpoint Loading** | ✅ COMPLETE | `weights_only=True` default enforced; no auto-fallback; strict state-dict validation |
| **24-Hour Retention** | ✅ COMPLETE | Persistent UTC timestamps; restart recovery; verified deletion; traversal protected |
| **Evaluation Harness** | ✅ COMPLETE | Reproducible script; patient separation; BraTS metrics (WT, TC, ET Dice/Sens) |
| **CPU Test Suite** | ✅ COMPLETE | Automated tests pass without checkpoints or GPU |
| **Patient-Level Evaluation** | ⚠️ **PENDING REAL DATA** | Requires ≥10 held-out, non-overlapping labelled patient MRI cases |
| **GPU Inference & VRAM** | ⚠️ **PENDING GPU** | Requires target GPU machine (≥6 GB VRAM) to verify runtime < 120s and memory |

## References

- Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images," BrainLes 2021
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods, 2021
- Myronenko, "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization," BrainLes 2018
- Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification," 2021

## Ethical Considerations

- All training and evaluation data must be de-identified public research datasets
- Uploaded scans are processed locally and not sent to external services
- Uploaded data should be deleted within 24 hours
- The application displays a research-only disclaimer throughout the interface
