# Model testing analysis

Date: 2026-10-16 (Asia/Kolkata). Branch: model-testing.

## Executed checks

- 18 regression tests passed, including synthetic end-to-end CLI/web mask, entropy and metric parity.
- Six existing ensemble software checks passed.
- GPU numerical suite completed: entropy boundaries, sigmoid extremes, production empty-mask Dice, synthetic forward passes and FP16 passed. Swin weights were finite; trained ResEnc weight check skipped because weights are absent.
- Three existing real MRI cases ran on both SwinUNETR and SegResNet; all six runs completed. Ensemble correctly reported BLOCKED instead of running random weights.
- Source revision and checkpoint/input hashes are captured in the linked result artifacts.

- Two independent cold SwinUNETR runs on the first case produced identical voxel labels and matching affines.
- Antigravity's separate 20-patient manifest passed read-only file hash, label-value and geometry validation. No inference on that cohort was run in this audit.

## Real-case Dice

These are convenience smoke cases, with unknown training overlap, not an independent clinical validation cohort.

| Model | Patient | WT Dice | TC Dice | ET Dice | Seconds |
|---|---|---:|---:|---:|---:|
| swinunetr | BraTS-GLI-00000-000 | 0.9589 | 0.9822 | 0.9619 | 10.69 |
| swinunetr | BraTS-GLI-00002-000 | 0.8166 | 0.9235 | 0.8413 | 6.94 |
| swinunetr | BraTS-GLI-00003-000 | 0.9837 | 0.9788 | 0.9440 | 8.63 |
| segresnet | BraTS-GLI-00000-000 | 0.9144 | 0.9524 | 0.8964 | 5.30 |
| segresnet | BraTS-GLI-00002-000 | 0.4205 | 0.6128 | 0.4793 | 9.51 |
| segresnet | BraTS-GLI-00003-000 | 0.8719 | 0.9689 | 0.8796 | 13.19 |

Runtime includes loading on the first case, preprocessing and exports. Qwen CPU review was active, so these are observed runtimes under contention, not isolated performance measurements.

swinunetr mean Dice: WT 0.9197, TC 0.9615, ET 0.9157.

segresnet mean Dice: WT 0.7356, TC 0.8447, ET 0.7518.

## Fixed defects

Missing/incompatible checkpoints now fail; shared expert validation handles empty masks and rejects bad geometry/labels; production metrics agree between paths; low-precision entropy remains finite; web/CLI share the ensemble predictor and effective settings. See [code audit](results/code_audit/audit.md).

## Remaining blockers and scope

The compatible trained 3D ResEnc checkpoint is missing. The documented 100-epoch training experiment is a different 2D U-Net. Do not substitute its weights. ResEnc input semantics remain unverified against training artifacts. Full ensemble accuracy cannot be assessed.

Antigravity owns the separate 20-patient acquisition/benchmark and browser interaction. Its data files were left unchanged. This report does not claim that pilot is complete. HD95 is not implemented. Unknown spatial units retain the existing mm assumption; use documented mm inputs for benchmarks.

## Evidence

- [Regression log](results/code_audit/regressions.log)
- [Ensemble software-check log](results/code_audit/ensemble.log)
- [GPU stability log](results/code_audit/stability_gpu.log)
- [Case results](results/code_audit/real_smoke_results.json)
- [Run metadata](results/code_audit/real_smoke_metadata.json)
- [Input manifest](data/local_smoke_manifest.json)
- [Repeatability check](results/code_audit/repeatability.json)
- [20-patient manifest validation](results/code_audit/antigravity_manifest_validation.log)
- [Benchmark readiness](results/code_audit/benchmark_ready.md)

Raw predictions and detailed local artifacts: `outputs/testing/code_audit_real_smoke/`. Raw MRI and weights are not committed. Qwen review is still pending at this report update.


---

# Antigravity 20-Patient Pilot Benchmark & Data Acquisition Report

Date: 2026-10-16 23:35 (Asia/Kolkata). Branch: `model-testing`. Code commit: `5cbc5cc` (dirty diff hash `36bdccf4...`).
Executor: Antigravity.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), Driver 535.183.01, CUDA 12.4, PyTorch 2.6.0+cu124.

## 1. Dataset Acquisition & Validation Summary

- **Source**: `reinheart/brats2022` (Kaggle, 2.49 GB archive, BraTS 2021 3D NIfTI cohort).
- **Other candidate sources**:
  - `awsaf49/brats2020-training-data`: Audited and **REJECTED** (repackaged as 2D HDF5 slices, destroying 3D spatial affines; violates Protocol Rule 5).
  - `dschettler8845/brats-2021-task1`: Audited and **DEFERRED** (12.3 GB zip expanding to >25 GB uncompressed, exceeding partition free space threshold).
- **Inventory Audit**: 454 patient directories inspected.
  - Complete 5-modality labeled cases (`flair`, `t1`, `t1ce`, `t2`, `seg`): **185 cases**.
  - Incomplete/unlabeled cases excluded: **269 cases** (unlabeled validation cohort).
- **Geometry & Label Verification**:
  - All 185 cases strictly verify: 1 mm isotropic voxel spacing (1.0 x 1.0 x 1.0 mm), 240 x 240 x 155 matrix, LAS/RAS affine orientation (`[-1, -1, 1, 1]`).
  - Label dictionary: `0`: BG, `1`: NCR/NET, `2`: ED, `4`: ET.
- **Cohorts Selected** (Seed: `42`):
  - 3-Case Smoke Cohort: `BraTS2021_00003`, `BraTS2021_00025`, `BraTS2021_00658`.
  - 20-Case Pilot Cohort: `BraTS2021_00003`, `BraTS2021_00025`, `BraTS2021_00658`, `BraTS2021_00733`, `BraTS2021_00529`, `BraTS2021_00096`, `BraTS2021_00774`, `BraTS2021_00501`, `BraTS2021_00684`, `BraTS2021_00024`, `BraTS2021_00731`, `BraTS2021_00026`, `BraTS2021_01157`, `BraTS2021_00081`, `BraTS2021_00054`, `BraTS2021_00697`, `BraTS2021_00714`, `BraTS2021_01156`, `BraTS2021_00008`, `BraTS2021_00513`.
  - Manifest SHA-256: `8c6da29de2ac840fdfd27d03794e19e3cf093cd666addc9561a466a0eb139c69`.

## 2. Benchmark Results on Pilot Cohort (20 Patients)

### A. SwinUNETR Baseline (Trained BraTS21 Checkpoint)

> **⚠ Training-data overlap (added 2026-10-17):** every pilot case is a BraTS 2021
> training case, and this checkpoint (`smilelab/BrainSegFounder`) was trained on
> that set. These Dice values are **not held-out accuracy**. See
> `testing/results/overlap_audit.md`. Numbers below were recomputed from
> `outputs/testing/antigravity/pilot20_run/results.json`.
- **Checkpoint SHA-256**: `6fd4ec5dc690f365a99c10b92f0d5519ca8a6b68feb844ff3424e9f6a01f6a39`
- **Execution**: 20/20 patients completed successfully.
- **Aggregated Empirical Metrics**:
  - **Whole Tumor (WT) Dice**: Mean **0.9050 (90.5%)** | Median: **0.9366** | Min: 0.6589 | Max: 0.9837
  - **Tumor Core (TC) Dice**: Mean **0.9235 (92.4%)** | Median: **0.9648** | Min: 0.7048 | Max: 0.9834
  - **Enhancing Tumor (ET) Dice**: Mean **0.8759 (87.6%)** | Median: **0.9056** | Min: 0.6219 | Max: 0.9708
  - **WT Sensitivity**: Mean **0.8910 (89.1%)** | Median: **0.9412**
  - **WT Precision**: Mean **0.9393 (93.9%)** | Median: **0.9742**
- **Inference Latency**:
  - Cold start (Patient 1, includes weights load + PyTorch JIT): 10.69s.
  - Warm end-to-end mean (patients 2–20): **8.28s** (Min: 5.22s, Max: 11.70s). Whole 20-case run: 167.9s.
- **GPU Resource Utilization**:
  - Peak Allocated VRAM: **1.64 GB** on NVIDIA RTX 4060 Laptop (8GB capacity). Headroom: >6.3 GB.

### B. Custom ResEnc / 80:20 Ensemble Status
- **Status**: **BLOCKED** across all cases.
- **Reason**: Trained 3D checkpoint `models/resenc_brats/resenc_m_brats.pt` is absent.
- **Policy Compliance**: Randomly initialized weights were strictly rejected in accordance with testing standards. No fabricated accuracy scores are claimed.

### C. Bitwise Reproducibility Verification
- Patient `BraTS2021_00003` was run across independent benchmark invocations (smoke test vs. 20-patient pilot).
- Predicted segmentation arrays compared across all 240 x 240 x 155 = 8,928,000 voxels:
  - **Exact Bitwise Match**: **True** (0 mismatched voxels).
  - Deterministic sliding-window inference with fixed overlap is verified.

## 3. UI & API Integration Verification

- **API Endpoint Checks** (`http://127.0.0.1:8000`):
  - `GET /api/health` -> HTTP 200 `{"status": "ok"}`
  - `POST /api/predict` with non-MRI image (`.png`) -> HTTP 400 Bad Request (strictly rejected)
  - `GET /api/download/nonexistent` -> HTTP 404 Not Found
  - `GET /api/download/{job_id}/segmentation.nii.gz` -> HTTP 200 (verified)
  - `GET /api/download/{job_id}/report.json` -> HTTP 200 (verified)
  - 3D Mesh exports (`brain.obj`, `tumor.obj`, `core.obj`) -> HTTP 200 (verified)
- **Browser UI Testing**:
  - Web UI loaded cleanly with full dark theme, interactive modality selectors, and job status monitoring.

## 4. Tracked Evidence & Local Artifacts

- Tracked Manifests:
  - [testing/data/manifest.json](data/manifest.json) (20-case pilot manifest)
  - [testing/data/smoke_manifest.json](data/smoke_manifest.json) (3-case smoke manifest)
  - [testing/data/sources.json](data/sources.json) (audited dataset source registry)
- Tracked Results:
  - [testing/results/antigravity_pilot/pilot20_results.json](results/antigravity_pilot/pilot20_results.json) (per-patient pilot outputs)
  - [testing/results/antigravity_pilot/pilot20_metadata.json](results/antigravity_pilot/pilot20_metadata.json) (run metadata and commit hashes)
  - [testing/results/antigravity_pilot/per_patient_schema_results.json](results/antigravity_pilot/per_patient_schema_results.json) (validated against `result.schema.json`)
  - [testing/results/antigravity_pilot/smoke_ensemble_blocked.json](results/antigravity_pilot/smoke_ensemble_blocked.json) (blocked ensemble audit log)
- Local Artifacts (Detailed Logs & Reports):
  - Acquisition Report: `outputs/testing/antigravity/acquisition_report.md`
  - Benchmark Report: `outputs/testing/antigravity/benchmark_report.md`
  - Raw Output Volumes & Segmentations: `outputs/testing/antigravity/pilot20_run/`

## Corrections — 2026-10-17

| # | Change | Evidence |
|---|---|---|
| 1 | Pilot Dice reclassified as **training-data performance**, not held-out accuracy | `testing/results/overlap_audit.md` |
| 2 | Invented "clinical inspection" explanations removed from the failure analysis | `testing/results/antigravity_pilot/benchmark_report.md` §3 |
| 3 | Checkpoint traced to `smilelab/BrainSegFounder` (BraTS 2021 pretrain + fine-tune); all pilot, smoke and demo cases are `known_training_overlap` | `testing/results/overlap_audit.md` |
| 4 | Min/median/runtime figures above recomputed from raw `results.json` (several were wrong) | `testing/results/antigravity_pilot/pilot20_results.json` |
| 5 | TC > WT mean Dice explained: region mapping verified correct; edema-boundary errors in 4 cases | report §3 |
| 6 | CLI vs web parity measured: 0 / 8,928,000 voxels differ on `BraTS2021_00003` | `testing/results/antigravity_pilot/cli_web_parity.json` |
| 7 | UI: 8:2 Ensemble button now disabled when the ResEnc checkpoint is missing; latency labels use measured ~8 s | `src/web/app.js`, `src/web/index.html` |

### Still open

- **Held-out accuracy:** none of the downloaded data is unseen by this checkpoint. Needs BraTS-Africa or post-2021 GLI cases (separate access).
- **Ensemble:** blocked until a trained 3D ResEnc checkpoint and its training split are provided.
- **UI screenshots:** slice views, 3D view and invalid-upload error still need to be captured into `outputs/testing/antigravity/screenshots/`.
- **HD95:** not implemented; do not quote.
