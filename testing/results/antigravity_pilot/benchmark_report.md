# NeuroScope 3D — Quantitative Benchmark & Model Evaluation Report

**Lead Evaluator**: Antigravity  
**Branch**: `model-testing`  
**Commit**: `64c48df`  
**Execution Environment**: Local NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.8, PyTorch 2.6.0  
**Evaluated Cohort**: 20 unique labeled adult glioma patient cases from `reinheart/brats2022` (BraTS 2021)  
**Selection Seed**: `42` (Deterministic, frozen prior to inference)  
**Evaluation Date**: 2026-10-16 18:10:00 UTC (Local: 23:40:00 IST)  

---

> **⚠ Read first — corrected 2026-10-17 (see `outputs/testing/overlap_audit.md`).**
> All 20 cases are BraTS 2021 Task 1 *training* cases. The SwinUNETR checkpoint
> (`smilelab/BrainSegFounder`) was pretrained and fine-tuned on that same set.
> These Dice scores therefore measure performance on **training data**. They
> confirm the pipeline works end to end; they are **not** held-out or
> "state-of-the-art" accuracy. Runtime, VRAM and determinism results stand.

## 1. Executive Summary & Headline Findings

1. **20-Patient Pilot Benchmark Completed**:
   - The verified baseline model **SwinUNETR (Vision Transformer)** was benchmarked sequentially across all 20 frozen patients using the verified label convention ($ET=4, ED=2, NCR/NET=1, BG=0$).
   - **Mean Whole Tumor (WT) Dice**: **0.9050 (90.5%)** | Median: **0.9366 (93.7%)**
   - **Mean Tumor Core (TC) Dice**: **0.9235 (92.4%)** | Median: **0.9648 (96.5%)**
   - **Mean Enhancing Tumor (ET) Dice**: **0.8759 (87.6%)** | Median: **0.9056 (90.6%)**
   - **Mean WT Sensitivity**: **0.8910 (89.1%)**
   - **Mean WT Precision**: **0.9393 (93.9%)**
   - **Inference Speed**: Cold load: **10.69s**, Warm mean: **8.28s** per full 3D volume.
   - **Peak GPU VRAM**: **1.64 GB** (safely within the 8 GB laptop budget).

2. **Ensemble Status: BLOCKED Pursuant to Protocol Rule 8**:
   - Model B (`nnU-Net ResEnc`) checkpoint was missing from `models/resenc_brats/resenc_m_brats.pt`.
   - In accordance with Protocol Rule 8 (*"Never benchmark random weights as a trained model. If ResEnc is missing, mark the ensemble blocked and run the valid baseline only"*), the ensemble pipeline refused execution with:
     ```text
     FileNotFoundError: Trained checkpoint missing: ['models/resenc_brats/resenc_m_brats.pt']. Random inference is disabled.
     ```
   - **Did the ensemble help?** **No, the ensemble could not help because it cannot be evaluated without trained weights.** Fabricating numbers or evaluating random weights was strictly rejected.

3. **Bitwise Deterministic Reproducibility**:
   - Patient `BraTS2021_00003` was evaluated independently in the smoke run and repeated in the pilot run.
   - Comparison of the two 3D segmentation masks yielded:
     - **Exact bitwise identity**: `True`
     - **Mismatched voxels**: `0 / 8,928,000 (0.000000%)`

---

## 2. Comprehensive Per-Patient Benchmark Results (20 Cases)

| Patient Case ID | WT Dice | TC Dice | ET Dice | WT Sensitivity | WT Precision | End-to-End Latency | WT Volume (Pred / GT) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `BraTS2021_00003` | **0.9837** | **0.9788** | **0.9440** | 0.9822 | 0.9852 | 10.69s | 98.9 mL / 99.2 mL |
| `BraTS2021_00025` | 0.8262 | 0.8835 | 0.8380 | 0.7092 | 0.9894 | 10.05s | 21.9 mL / 30.6 mL |
| `BraTS2021_00658` | **0.9551** | **0.9717** | **0.9095** | 0.9467 | 0.9637 | 10.36s | 70.3 mL / 71.6 mL |
| `BraTS2021_00733` | **0.9682** | **0.9824** | **0.9448** | 0.9746 | 0.9620 | 8.36s | 82.8 mL / 81.8 mL |
| `BraTS2021_00529` | 0.8884 | 0.8225 | 0.7625 | 0.8277 | 0.9587 | 9.51s | 76.5 mL / 88.6 mL |
| `BraTS2021_00096` | **0.9638** | **0.9834** | **0.9644** | 0.9430 | 0.9854 | 9.60s | 114.2 mL / 119.3 mL |
| `BraTS2021_00774` | **0.9651** | **0.9696** | **0.9665** | 0.9383 | 0.9934 | 7.17s | 143.3 mL / 151.7 mL |
| `BraTS2021_00501` | 0.7309 | **0.9407** | **0.8894** | 0.5779 | 0.9942 | 7.62s | 48.0 mL / 82.5 mL |
| `BraTS2021_00684` | 0.8074 | 0.7674 | 0.6219 | 0.7410 | 0.8870 | 7.79s | 19.5 mL / 23.3 mL |
| `BraTS2021_00024` | **0.9671** | **0.9561** | **0.9467** | 0.9656 | 0.9685 | 7.60s | 68.3 mL / 68.5 mL |
| `BraTS2021_00731` | 0.6589 | 0.8566 | 0.8557 | 0.9981 | 0.4918 | 7.14s | 23.4 mL / 11.5 mL |
| `BraTS2021_00026` | 0.9180 | **0.9510** | **0.9355** | 0.8808 | 0.9585 | 9.13s | 72.8 mL / 79.2 mL |
| `BraTS2021_01157` | **0.9437** | **0.9632** | 0.8727 | 0.9419 | 0.9456 | 8.35s | 111.9 mL / 112.3 mL |
| `BraTS2021_00081` | **0.9760** | **0.9732** | **0.9708** | 0.9708 | 0.9813 | 9.18s | 114.7 mL / 116.0 mL |
| `BraTS2021_00054` | 0.9237 | 0.8704 | 0.7772 | 0.8735 | 0.9799 | 6.16s | 51.6 mL / 57.9 mL |
| `BraTS2021_00697` | **0.9613** | **0.9675** | **0.9360** | 0.9406 | 0.9830 | 8.24s | 133.7 mL / 139.7 mL |
| `BraTS2021_00714` | **0.9459** | **0.9664** | **0.9017** | 0.9001 | 0.9966 | 6.55s | 192.3 mL / 212.9 mL |
| `BraTS2021_01156` | 0.9295 | **0.9796** | 0.8639 | 0.9753 | 0.8879 | 5.22s | 113.4 mL / 103.2 mL |
| `BraTS2021_00008` | 0.9215 | 0.7048 | 0.6800 | 0.9630 | 0.8834 | 7.50s | 59.7 mL / 54.8 mL |
| `BraTS2021_00513` | 0.8659 | **0.9818** | **0.9375** | 0.7688 | 0.9911 | 11.70s | 13.9 mL / 17.9 mL |
| **MEAN** | **0.9050** | **0.9235** | **0.8759** | **0.8910** | **0.9393** | **8.40s** | — |
| **MEDIAN** | **0.9366** | **0.9648** | **0.9056** | **0.9412** | **0.9742** | **8.29s** | — |
| **MIN** | **0.6589** | **0.7048** | **0.6219** | **0.5779** | **0.4918** | **5.22s** | — |
| **MAX** | **0.9837** | **0.9834** | **0.9708** | **0.9981** | **0.9966** | **11.70s** | — |

---

## 3. Failure & Outlier Analysis

Numbers only; no scan was reviewed by a radiologist or visually inspected for
these notes. Causes are not established.

1. **`BraTS2021_00731` — WT over-segmentation**: lowest WT Dice (0.6589).
   Sensitivity 0.9981, precision 0.4918; predicted 23.4 mL vs 11.5 mL reference.
2. **`BraTS2021_00684` — lowest ET Dice (0.6219)**: reference ET is small
   (2.93 mL); predicted 1.66 mL. Sensitivity 0.4875, precision 0.8585 — the
   model under-segments enhancing tissue here. Small structures amplify Dice
   penalties.
3. **`BraTS2021_00501` — WT under-segmentation**: WT Dice 0.7309 with TC 0.9407
   and ET 0.8894. Precision 0.9942, sensitivity 0.5779; predicted 48.0 mL vs
   82.5 mL reference, i.e. the missed volume lies in the edema-only region.

**Why mean TC Dice (0.924) exceeds mean WT Dice (0.905):** region mapping was
audited (`src/metrics.py` WT={1,2,ET}, TC={1,ET}; `src/pipeline.py` channel
order TC/WT/ET) and is correct. The gap comes from edema-boundary errors in a
few cases (00731, 00501, 00025, 00513) that lower WT but not TC. Median WT
(0.937) vs TC (0.965) shows the same pattern.

---

## 4. UI & Localhost Verification (Protocol Rule 11)

- **Local Endpoint**: `http://127.0.0.1:8000` (FastAPI backend on port 8000).
- **Slice & Overlay Alignment**: *Not evidenced* — no screenshots or alignment measurements were saved. Treat as untested until screenshots exist under `outputs/testing/antigravity/screenshots/`.
- **3D Mesh Generation**: `marching_cubes` successfully extracts closed manifold meshes for `brain.obj`, `tumor.obj`, and `core.obj`.
- **Invalid-Input Guards Tested**:
  - Non-NIfTI screenshot upload (`scan.png`): Correctly rejected with HTTP 400 (`"Upload NIfTI MRI files (.nii or .nii.gz), not JPG/PNG screenshots."`).
  - Missing file / invalid job query: Correctly returns HTTP 404.
  - Dimension mismatch (4D series or 2D image): Correctly rejected with informative error.
- **CLI vs Web Parity**: Both paths call the same `src/pipeline.py` inference code (`roi_size=(96, 96, 96)`, `overlap=0.5`, `mode='gaussian'`). Shared code was verified; **verified on 2026-10-17**: `BraTS2021_00003` uploaded to `POST /api/analyze` matched the CLI pilot output with 0 / 8,928,000 mismatched voxels, identical affine, volumes and Dice (`cli_web_parity.json`).

---

## 5. Artifact Ledger

- **Manifest**: [outputs/testing/antigravity/manifest.json](file:///home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16/outputs/testing/antigravity/manifest.json)
- **Raw Pilot Results**: [outputs/testing/antigravity/pilot20_run/results.json](file:///home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16/outputs/testing/antigravity/pilot20_run/results.json)
- **Smoke Results**: [outputs/testing/antigravity/smoke_run/results.json](file:///home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16/outputs/testing/antigravity/smoke_run/results.json)
- **Metadata**: [outputs/testing/antigravity/pilot20_run/metadata.json](file:///home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16/outputs/testing/antigravity/pilot20_run/metadata.json)
