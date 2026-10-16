# DI-02 pipeline decision — 16 October 2026

## Decision for the hackathon demo

Use **SwinUNETR** as the default segmentation engine, with **SegResNet** as a faster comparison option. The two checkpoints need different input channel orders. This is a small demonstration benchmark, not evidence of clinical performance or independent generalization.

```text
Four aligned, skull-stripped 3D MRI scans
    ↓ validate dimensions, affine and finite values
    ↓ canonical RAS orientation; 1 mm grid; per-channel nonzero normalization
    ├─ SwinUNETR: [FLAIR, T1c, T1, T2] → 96³ sliding windows
    └─ SegResNet: [T1c, T1, T2, FLAIR] → 96³ sliding windows
    ↓ sigmoid; threshold at 0.5; BraTS labels 0/1/2/4
    ↓ map labels to original MRI grid with nearest-neighbor resampling
    ↓ physical volumes, optional Dice/sensitivity, NiiVue slices and meshes
```

## Why the original Swin results were wrong

The initial shared pipeline used `[T1c, T1, T2, FLAIR]` for both checkpoints. The [official MONAI BraTS21 folds file](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/brats21_folds.json) lists SwinUNETR images as `[FLAIR, T1ce, T1, T2]`, while the [SegResNet bundle card](https://huggingface.co/monai-test/brats_mri_segmentation) specifies `[T1c, T1, T2, FLAIR]`. On GLI-00002 the old Swin pipeline produced zero Dice for all three regions; after correcting the order, the same installed model and case produced WT 0.8165, TC 0.9235, ET 0.8413. Old Swin runs are marked invalid in the app history and should not be cited.

## Measured comparison on the three downloaded cases

All cases are BraTS2023 GLI, with four co-registered MRI sequences and expert masks. Both checkpoints used the same 96³ sliding-window size, 0.5 overlap, mixed precision, and this laptop's RTX 4060. Each case was processed once per model after correcting the channel order. Dice measures voxel overlap with the expert mask.

| Case | Model | Whole tumor Dice | Tumor core Dice | Enhancing tumor Dice | End-to-end time |
|---|---|---:|---:|---:|---:|
| GLI-00000 | SegResNet | 0.9143 | 0.9524 | 0.8964 | 3.57 s |
| GLI-00000 | SwinUNETR | 0.9589 | 0.9822 | 0.9619 | 5.64 s |
| GLI-00002 | SegResNet | 0.4205 | 0.6129 | 0.4793 | 2.67 s |
| GLI-00002 | SwinUNETR | 0.8165 | 0.9235 | 0.8413 | 4.08 s |
| GLI-00003 | SegResNet | 0.8720 | 0.9689 | 0.8796 | 3.01 s |
| GLI-00003 | SwinUNETR | 0.9837 | 0.9788 | 0.9440 | 5.08 s |
| **Three-case mean** | **SegResNet** | **0.7356** | **0.8447** | **0.7518** | **3.08 s** |
| **Three-case mean** | **SwinUNETR** | **0.9197** | **0.9615** | **0.9157** | **4.93 s** |

The raw comparison and per-case sensitivity are in `work/benchmark/comparison_96.json`. Reproduce with `.venv/bin/python work/compare_models.py`. The live app was also rerun on GLI-00002 after the correction and returned the same Dice values.

The [published SwinUNETR implementation](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BRATS21/README.md) uses 128³ patches. Trying that size while the local app was using the 8 GB GPU caused CUDA out-of-memory. The 96³ setting is a laptop adaptation, so do not compare these numbers directly with published leaderboard scores.

## What is still needed to establish the best model

1. Build a patient-level evaluation set large enough to include diverse tumor sizes and scan sources. Confirm none of its patients appear in either checkpoint's training data; the current three cases have not been checked for overlap.
2. Keep modality naming, voxel spacing, orientation, normalization, label convention, inference patch, threshold and postprocessing recorded for every run. Each model retains its checkpoint-specific channel order.
3. Report per-case WT/TC/ET Dice, sensitivity, volume error, inference time and failures. Include aggregate spread, not just a mean. Inspect false negatives and alignment visually.
4. If the project later includes post-treatment scans, test those separately. The current models were trained for BraTS-style glioma segmentation and are not validated for arbitrary clinical MRIs.
5. Select the default based on the held-out evaluation and the 8 GB GPU limit. For this hackathon demonstration, SwinUNETR is the provisional default; SegResNet remains useful for speed and comparison.

The app's measured volumes are descriptive segmentation outputs. A single scan cannot establish histologic grade, progression, treatment response or a RANO 2.0 category. The earlier automatic RANO and phenotype labels were removed from new reports.
