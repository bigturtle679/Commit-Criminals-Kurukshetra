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
- Weighted average: `prob = 0.8 × prob_A + 0.2 × prob_B`
- BraTS hierarchy enforcement: ET ≤ TC ≤ WT
- Threshold: 0.5
- Shannon entropy uncertainty map from fused probabilities

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

## Checkpoint Provenance

> **Status: UNVERIFIED**
>
> The exact training data, training procedure, and licensing terms for the specific
> checkpoint files used in this project have not been independently verified. The
> checkpoint filenames and architecture configurations match publicly available
> BraTS challenge models, but their exact provenance must be confirmed by the team
> member who obtained them.

### What must be verified before claiming benchmark results:

1. Training dataset (BraTS 2021, MSD, or other)
2. Training/validation split used
3. Any fine-tuning performed after the original training
4. License terms of the checkpoint files
5. Whether evaluation cases overlap with training data

## Intended Use

- **Primary use**: Research and teaching demonstration of 3D brain tumor segmentation
- **NOT intended for**: Clinical diagnosis, treatment planning, or medical decision-making
- **Users**: Researchers, students, and hackathon evaluators

## Limitations

1. **Not a diagnostic tool** — Output represents model estimates, not expert-verified findings
2. **Dataset bias** — Models trained on BraTS data (adult gliomas from specific institutions)
3. **Preprocessing dependency** — Results assume skull-stripped, co-registered MRI inputs
4. **No clinical validation** — Segmentation accuracy ≠ clinical diagnostic accuracy
5. **Population generalization** — Performance may differ on scans from other scanners, protocols, or patient populations

## Evaluation

| Status | Description |
|---|---|
| ✅ Application code implemented | Backend, frontend, API, validation |
| ✅ CPU-compatible tests passed | Ensemble math, hierarchy, entropy, volume |
| ❓ Checkpoint loading verified | Requires GPU machine with checkpoints |
| ❓ Real MRI inference verified | Requires GPU machine with MRI data |
| ❓ Segmentation quality evaluated | Requires 10+ held-out labelled cases |
| ❓ GPU performance measured | Requires target hardware |

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
