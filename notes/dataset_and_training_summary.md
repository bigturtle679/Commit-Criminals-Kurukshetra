# NeuroScope 3D — Dataset & Local Training Documentation

This document archives the provenance, specification, preprocessing pipeline, and experimental results for the local brain tumor segmentation training runs prior to cleanup.

---

## 1. Dataset Origin & Provenance

- **Collection**: Medical Segmentation Decathlon (MSD) Task01 — Brain Tumour (BraTS)
- **Source Host**: AWS Open Data Registry (`msd-for-monai.s3.us-west-2.amazonaws.com/Task01_BrainTumour.tar`)
- **Total Archive Size**: 7.61 GB
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Downloaded Subset**: 50 complete, labelled retrospective adult glioma patient cases

### Modality Specifications
Each patient scan contained 4 co-registered 3D MRI volumes acquired during the same examination:
1. **FLAIR**: Fluid-attenuated inversion recovery (suppresses normal CSF; highlights peritumoral edema)
2. **T1w**: T1-weighted pre-contrast (clear grey/white matter anatomical structural boundaries)
3. **T1gd / T1ce**: T1-weighted post-gadolinium contrast (highlights active vascularized tumor core disruption)
4. **T2w**: T2-weighted (reveals fluid, hyperintensities, and structural tumor margins)

### Voxel Ground-Truth Label Definitions
The official MSD `dataset.json` defines four voxel classes:
- `0`: Background
- `1`: Edema (ED)
- `2`: Non-enhancing tumor / necrotic core (NCR/NET)
- `3`: Enhancing tumor (ET)

---

## 2. Preprocessing & Transformation Pipeline

### Data Acquisition via Selective Range Fetching (`scripts/fetch_msd.py`)
- Used HTTP Range byte requests to directly extract the first 50 image TAR blocks and the corresponding label TAR blocks from the archive tail without decompressing the 7.6 GB archive.
- SHA-256 integrity checksums were recorded per case in `work/data/msd/manifest.json`.

### Intensity Standardization
Non-zero voxel intensities were standardized per channel using z-score normalization:
$$\tilde{x}_c(v) = \frac{x_c(v) - \mu_{c, \text{nonzero}}}{\sigma_{c, \text{nonzero}}}$$
This prevents background air voxels ($0$) from skewing the statistical mean and ensures cross-patient contrast normalization.

### Spatial Resizing & Slice Extraction (`scripts/local_training.py prepare`)
- 3D volumetric NIfTI volumes were transposed into 2D axial slices.
- Interpolated to **$128 \times 128$** resolution:
  - Bilinear interpolation for the 4 MRI channels (stored as float16 `.npy`).
  - Nearest-neighbor interpolation for label masks (stored as uint8 `.npy`).
- Outputs stored in `work/prepared/msd/128/` totaling 1.3 GB across 50 patients.

### BraTS Nested Region Derivation
Converted multi-class labels into standard overlapping clinical evaluation regions:
- **Tumor Core (TC)**: Necrotic core + Enhancing tumor ($L \in \{2, 3\}$ in MSD schema)
- **Whole Tumor (WT)**: Core + Edema ($L \in \{1, 2, 3\}$)
- **Enhancing Tumor (ET)**: Contrast-enhancing active rim ($L == 3$)

---

## 3. Patient Cohort Split

Partitioned at the patient level using seed `42` to guarantee zero cross-slice data leakage:
- **Train Set**: 35 patients (used for gradient updates)
- **Validation Set**: 10 patients (evaluated at the close of every epoch for model checkpointing)
- **Test Set**: 5 held-out patients (`BRATS_277`, `BRATS_426`, `BRATS_306`, `BRATS_365`, `BRATS_150`) strictly untouched until final scoring

---

## 4. Model Architecture & Training Setup

- **Model**: MONAI 2D Residual U-Net (`UNet(spatial_dims=2, in_channels=4, out_channels=3)`)
- **Channel Progression**: `(16, 32, 64, 128)` with strides `(2, 2, 2)` and 1 residual unit
- **Total Parameters**: 206,720
- **Loss Function**: `DiceCELoss(sigmoid=True, squared_pred=True)`
- **Optimizer**: Adam ($\text{lr} = 1\times 10^{-4}$)
- **Hardware**: Apple Silicon GPU via Metal Performance Shaders (`mps`)
- **Throughput**: ~16.8 ms per training step

---

## 5. Training Progression & Experimental Results

### Progression Over 100 Epochs (20,000 Optimization Steps)
| Epoch | Training Loss | Validation Mean Dice | Status |
|:---:|:---:|:---:|:---|
| **1** | 1.0029 | 0.0164 | Random initialization |
| **10** | 0.9761 | 0.0946 | Foreground separation |
| **20** | 0.9381 | 0.1941 | End of initial run |
| **35** | 0.8920 | 0.4604 | Whole tumor structure formed |
| **50** | 0.8310 | 0.6621 | Major convergence breakthrough |
| **70** | 0.7850 | 0.6768 | Internal core delineation |
| **83** | 0.7410 | 0.7172 | Solid multi-region separation |
| **97 (Best)** | **0.7105** | **0.7402** | **Best model checkpoint (`best.pt`)** |
| **100** | 0.7180 | 0.7246 | Final checkpoint (`last.pt`) |

### Final Test Evaluation on 5 Unseen Held-Out Patients
Evaluated with the `best.pt` weights:

| Case ID | Tumor Core (TC) | Whole Tumor (WT) | Enhancing Tumor (ET) | Patient Mean Dice |
|---|:---:|:---:|:---:|:---:|
| `BRATS_277` | 0.694 | **0.843** | **0.799** | 0.779 |
| `BRATS_426` | **0.837** | **0.890** | **0.794** | **0.840** |
| `BRATS_306` | 0.589 | 0.639 | 0.617 | 0.615 |
| `BRATS_365` | 0.382 | **0.857** | 0.538 | 0.592 |
| `BRATS_150` | 0.484 | **0.831** | 0.357 | 0.557 |
| **Overall Mean** | **0.597** | **0.812** | **0.621** | **0.677** |

---

## 6. Output Artifacts & Data Cleanup

### Saved Model Deliverables
- Checkpoint: `outputs/unet2d_20261016_191019/best.pt` (Val Dice: 0.7402)
- Training history: `outputs/unet2d_20261016_191019/history.jsonl`
- Test metrics: `outputs/unet2d_20261016_191019/test.json`

### Data Cleanup Rationale
To preserve repository storage hygiene and prepare for the 8:2 ensemble pipeline:
- Removed raw NIfTI downloads (`work/data/msd/`)
- Removed cached byte ranges (`work/downloads/`)
- Removed resized axial slice arrays (`work/prepared/msd/`)
Reclaimed ~2.5 GB of disk space while keeping all code, configs, manifests, evaluation metrics, and model weights intact.
