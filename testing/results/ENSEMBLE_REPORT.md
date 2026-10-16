# 8:2 Ensemble Results — 2026-10-17

## Bottom line
The 80:20 SwinUNETR + ResEnc ensemble **runs end to end but does not improve accuracy**.
On both test sets it matches SwinUNETR alone to within ±0.001 mean Dice.

## Models
| Model | Checkpoint | Training | Overlap with test cases |
|---|---|---|---|
| SwinUNETR | `model_weights_BRATS-finetune.pt` (BrainSegFounder) sha256 `6fd4ec5d…` | BraTS 2021 (third party) | **Known overlap** — scores inflated |
| ResEnc-M | `resenc_m_brats.pt` sha256 `9e65a271…` | Trained here: 547 BraTS 2021 patients, **2,000 of 15,000 planned iterations (~13 min, early-stopped for time)** | **None** — test patients excluded (`heldout50/resenc_split.json`) |

## Test 1 — 50 random held-out patients (seed 42)
| Model | WT Dice | TC Dice | ET Dice |
|---|---|---|---|
| SwinUNETR | 0.949 | 0.959 | 0.936 |
| ResEnc (unseen data) | 0.876 | 0.874 | 0.859 |
| 80:20 ensemble | 0.949 | 0.959 | 0.936 |

Ensemble vs Swin per patient: WT better 26 / worse 21, TC 25 / 24, ET 23 / 25. Max per-patient change < 0.01.

## Test 2 — 20-patient pilot manifest (`benchmark_manifest.py`)
| Model | WT | TC | ET | Warm s/patient | Peak GPU |
|---|---|---|---|---|---|
| SwinUNETR | 0.905 | 0.924 | 0.876 | 6.8 | 1.65 GB |
| ResEnc | 0.817 | 0.768 | 0.759 | 4.0 | 1.08 GB |
| 80:20 ensemble | 0.905 | 0.923 | 0.875 | 9.2 | 1.68 GB |

Ensemble vs Swin: WT better 8 / worse 11, TC 10 / 10, ET 8 / 11 (mean change −0.0005).
SwinUNETR reproduced the earlier pilot run exactly (0.9050 / 0.9235 / 0.8759). 20/20 complete, no failures.

## Why the ensemble changes almost nothing
With weights 0.8/0.2 and threshold 0.5, ResEnc can only flip a voxel where Swin's
probability lies in [0.375, 0.625]. Elsewhere Swin decides alone.

## Honest claims for the demo
- ✅ Two trained 3D models, a working probability-fusion ensemble, ~9 s per patient on an RTX 4060.
- ✅ ResEnc trained in-house reaches ~0.87 Dice on 50 patients it never saw.
- ❌ Do not claim the ensemble beats SwinUNETR.
- ❌ Do not call SwinUNETR's Dice held-out accuracy (training overlap).

## Not done
Full 15k-iteration ResEnc training, tuning fusion weights on a separate validation set,
ensemble CLI-vs-web parity check, UI screenshots, HD95.
