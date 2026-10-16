# DI-02: NeuroScope 3D — Master Blueprint & Resource Hub

## 1. Executive Summary & Win Moats
- **Track**: Decentralized Intelligence (DI-02) — NeuroScope 3D
- **Target Win Probability**: 88%
- **Core Strategy**:
  1. Use pre-trained weights (`MONAI/brats_mri_segmentation` SegResNet) for zero-risk, high-accuracy inference.
  2. Implement sub-millimeter affine linear algebra coordinate transforms (RAS+ canonization + inverted affine projection).
  3. Integrate NiiVue (`@niivue/niivue`) in a modern React + Vite dashboard.
  4. Ship 5 judge-crushing extra features:
     - Automated RANO 2.0 Clinical Volumetric PDF Report
     - Test-Time Augmentation (TTA) Voxel Uncertainty Heatmap ($H(v)$ Shannon entropy)
     - Marching Cubes 3D Surface Mesh Extraction (.OBJ / .GLTF) for real-time 3D skull/tumor rendering
     - Longitudinal Progression Tracker (Delta-V % change across visits)
     - Mathematical Affine Resampling & Inversion Guardrail

---

## 2. Academic Research Papers to Cite
1. **Foundational SegResNet Paper (BraTS Winner)**
   - *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization* (Andriy Myronenko, NVIDIA, BrainLes / MICCAI 2018).
   - arXiv: [1810.11654](https://arxiv.org/abs/1810.11654)
   - Architecture: Encoder-Decoder 3D ResNet regularized with a variational autoencoder (VAE) reconstruction loss; GroupNorm for small batch sizes.

2. **Benchmark Challenge Paper**
   - *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification* (Ujjwal Baid et al., 2021).
   - arXiv: [2107.02314](https://arxiv.org/abs/2107.02314)
   - Standard clinical subregions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).

3. **SOTA Vision Transformer Paper**
   - *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images* (Ali Hatamizadeh et al., NVIDIA, MICCAI 2022).
   - arXiv: [2201.01266](https://arxiv.org/abs/2201.01266)
   - Shifted-window multi-head self-attention for 3D multi-modal volumetric MRI.

4. **Clinical Radiology Assessment Standard**
   - *Updated Response Assessment in Neuro-Oncology (RANO 2.0) criteria for adult diffuse glioma* (Patrick Y. Wen et al., *The Lancet Oncology*, 2023).
   - Mandates computer-assisted 3D volumetric assessments over legacy 2D cross-product measurements.

---

## 3. Open-Weight AI Models
- **Primary Model**: `MONAI/brats_mri_segmentation` (Hugging Face / MONAI Model Zoo)
  - Pretrained on BraTS multi-parametric MRI (T1, T1ce, T2, FLAIR) at 1mm isotropic resolution.
  - License: Apache 2.0.
  - Inputs: 4 modalities (`[B, 4, D, H, W]`).
  - Outputs: 3 subregions (`TC`, `WT`, `ET`).
- **Load via MONAI Bundle**:
  ```python
  from monai.bundle import load_bundle
  model = load_bundle("brats_mri_segmentation", repo="MONAI/brats_mri_segmentation", net_name="network_def")
  ```

---

## 4. Open-Source Datasets
1. **Kaggle**: `dschettler8845/brats-2021-task1` (1,251 cases with multi-modal NIfTI and ground truth masks).
   ```bash
   kaggle datasets download -d dschettler8845/brats-2021-task1 --unzip -p ./data/brats2021
   ```
2. **Hugging Face**: `Spirit-26/BraTS-2024-Complete` / `MedOtter/brats2023-gli-dataset`.
3. **Medical Segmentation Decathlon (MSD Task 1)**:
   ```python
   from monai.apps import DecathlonDataset
   ds = DecathlonDataset(root_dir="./data", task="Task01_BrainTumour", download=True)
   ```

---

## 5. Mathematical Formulas for Presentation & Scoring
1. **Physical Volume in Milliliters (mL)**:
   $$V_{\text{mL}} = \frac{N_{\text{voxels}} \times |\det(\mathbf{A}_{3\times3})|}{1000} = \frac{N_{\text{voxels}} \times (s_x \cdot s_y \cdot s_z)}{1000}$$
2. **Voxel-wise Predictive Entropy (Uncertainty)**:
   $$H(v) = -\bar{p}_v \log_2(\bar{p}_v + \epsilon) - (1 - \bar{p}_v) \log_2(1 - \bar{p}_v + \epsilon)$$
3. **Evaluation Metrics**:
   $$\text{Dice}(P, G) = \frac{2|P \cap G|}{|P| + |G|}, \quad \text{Sensitivity} = \frac{|P \cap G|}{|G|}$$
   $$\text{HD95}(P, G) = \max\left(P_{95}\min_{g \in \partial G} \|p - g\|_2, P_{95}\min_{p \in \partial P} \|g - p\|_2\right)$$
