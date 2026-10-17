# NeuroScope 3D — Brain Tumor Segmentation Research Platform

> **Research and teaching demonstration only. Not intended for clinical diagnosis.**

NeuroScope 3D is an AI-powered 3D brain tumor segmentation and visualization platform. It processes four MRI sequences (T1, T1c, T2, FLAIR), predicts tumor subregions using an 8:2 ensemble of SwinUNETR and nnU-Net ResEnc models, and provides interactive 3D visualization, volume measurements, and uncertainty estimation.

## Features

- **8:2 Hybrid Ensemble**: SwinUNETR (80%) + nnU-Net ResEnc M (20%) with probability-level fusion
- **BraTS-Standard Output**: WT, TC, ET segmentation with labels {0, 1, 2, 4}
- **3D Visualization**: NiiVue WebGL2 viewer with multiplanar slices and 3D mesh rendering
- **Volume Calculation**: Exact physical volumes using affine determinant (mL)
- **Uncertainty & Disagreement**: Shannon Bernoulli entropy boundary confidence and inter-model probability disagreement (|p_A - p_B|)
- **Expert Comparison**: Optional ground-truth mask upload with Dice and sensitivity metrics
- **Research Report**: JSON report with model provenance, preprocessing details, and SHA-256 input hashes

## Benchmark Performance & Validation (Multi-Year BraTS Cohorts)

Evaluated across multi-year BraTS datasets (2017, 2018, 2019, 2020/2021, 2023 GLI, and 2024 Multi-Institutional) following official [CBICA BraTS protocols](https://www.med.upenn.edu/cbica/brats/):

### Summary Across All Evaluated Cohorts
| Model Architecture | Overall Mean Dice | Overall Mean Sensitivity | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) |
|---|:---:|:---:|:---:|:---:|:---:|
| **80:20 Hybrid Ensemble** | **0.9125 (91.3%)** | **0.9057 (90.6%)** | **0.9228** (91.1% Sens) | **0.9282** (92.2% Sens) | **0.8866** (88.4% Sens) |
| **SwinUNETR (Model A)** | 0.9096 (91.0%) | 0.9007 (90.1%) | 0.9205 (90.6% Sens) | 0.9252 (91.8% Sens) | 0.8833 (87.9% Sens) |
| **ResEnc-M (Model B)** | 0.8350 (83.5%) | 0.8579 (85.8%) | 0.8525 (87.6% Sens) | 0.8390 (85.9% Sens) | 0.8134 (83.9% Sens) |

### Performance on Unseen / New Data (BraTS 2023-GLI, 2024 Multi-Center & Held-Out 50)
| Model Architecture | New Data Mean Dice | New Data Mean Sensitivity | WT (Dice / Sens) | TC (Dice / Sens) | ET (Dice / Sens) |
|---|:---:|:---:|:---:|:---:|:---:|
| **80:20 Hybrid Ensemble** | **0.9149 (91.5%)** | **0.9097 (91.0%)** | **0.9207** / **0.9048** | **0.9318** / **0.9292** | **0.8922** / **0.8951** |
| **SwinUNETR (Model A)** | 0.9108 (91.1%) | 0.9018 (90.2%) | 0.9169 / 0.8966 | 0.9282 / 0.9225 | 0.8874 / 0.8863 |
| **ResEnc-M (Model B)** | 0.8430 (84.3%) | 0.8725 (87.3%) | 0.8536 / 0.8809 | 0.8538 / 0.8799 | 0.8215 / 0.8568 |

> Comprehensive breakdowns by patient cohort, loss functions, and label schemas are archived in [training.md](training.md).

## Quick Start

### Prerequisites

- Python 3.10–3.12 (recommended)
- Model checkpoint files (see [MODEL_CARD.md](MODEL_CARD.md))
- GPU with ≥6 GB VRAM for inference (CPU mode available for development)

### Setup

```bash
# Clone the repository
git clone https://github.com/bigturtle679/Commit-Criminals-Kurukshetra.git
cd Commit-Criminals-Kurukshetra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install PyTorch (choose your platform)
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Place model checkpoints (see MODEL_CARD.md for paths)
# models/swinunetr_brats/model_weights_BRATS-finetune.pt
# models/resenc_brats/resenc_m_brats.pt
```

### Launch

```bash
# Linux/Mac
./start-neuroscope.sh

# Windows
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Then open http://127.0.0.1:8000
```

### Run Tests

```bash
# CPU-compatible unit tests
python -m pytest tests/ -v

# Metrics self-test
python src/metrics.py

# Integration test (requires running server)
python tests/check_local_app.py
```

## Project Structure

```
src/
  app.py              FastAPI application, endpoints, job lifecycle
  pipeline.py         Inference orchestration, spatial restoration, report
  metrics.py          Dice, sensitivity, volume calculation (single source)
  web/
    index.html        Frontend dashboard
    app.js            NiiVue viewer integration, API client
    style.css         UI stylesheet
    vendor/niivue.js  Bundled NiiVue WebGL2 engine

scripts/
  ensemble_inference.py   Core ensemble engine (8:2 fusion, hierarchy, entropy)
  evaluate_ensemble.py    Patient-level cohort evaluation harness
  models/
    model_loader.py       Checkpoint loading with validation
    resenc_unet.py        nnU-Net ResEnc architecture

configs/
  config_loader.py    Centralized configuration validation and schema checks
  ensemble.json       Ensemble weights, channel orders, thresholds
  evaluation_manifest_template.json  Held-out patient cohort manifest schema

models/               Model checkpoints (not in Git)
tests/                Comprehensive unit and integration test suite
notes/                Research documentation and design decisions
```

## Security & Privacy

- **Data Retention**: Uploaded patient MRI files and generated artifacts are automatically purged after 24 hours by a background cleanup worker (`RETENTION_HOURS=24`).
- **Strict Model Provenance**: Checkpoints require exact architecture validation and strict weight loading. Corrupt, partial, or mismatched checkpoints are rejected immediately without unsafe fallbacks. Checkpoint SHA-256 hashes are recorded in inference reports.
- **Path Traversal Protection**: File download endpoints strictly resolve paths within designated output directories and reject any path traversal attempts.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend dashboard |
| `/api/status` | GET | GPU info, available models, demo cases |
| `/api/demo` | POST | Run inference on a demo patient case |
| `/api/analyze` | POST | Upload and analyze custom MRI files |
| `/api/jobs/{id}` | GET | Job status and results |
| `/api/files/{id}/{name}` | GET | Download generated artifacts |

## Scoring Criteria (from Problem Statement)

| Category | Weight |
|---|---|
| Segmentation Quality | 25/100 |
| 3D + Linked Views | 15/100 |
| Volume Calculation | 10/100 |
| Explainability | 10/100 |
| Reliability | 10/100 |
| User Experience | 10/100 |
| Responsible Communication | 10/100 |
| Viva | 10/100 |

## Team

**Commit Criminals** — Kurukshetra Hackathon (DI-02: NeuroScope 3D)

## Dataset Credits & Citations

- **BraTS 2021**: U. Baid, et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification," arXiv:2107.02314, 2021. DOI: [10.7937/jc8x-9874](https://doi.org/10.7937/jc8x-9874). Accessible via The Cancer Imaging Archive (TCIA) under CC BY 4.0.
- **Medical Segmentation Decathlon (MSD)**: Brain Tumours (Task01_BrainTumour), 750 4D multi-modal volumes from BraTS 2016–2017. Accessible under CC BY-SA 4.0 (http://medicaldecathlon.com/).

## License

See individual component licenses in [MODEL_CARD.md](MODEL_CARD.md).
