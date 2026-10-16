# NeuroScope 3D — Brain Tumor Segmentation Research Platform

> **Research and teaching demonstration only. Not intended for clinical diagnosis.**

NeuroScope 3D is an AI-powered 3D brain tumor segmentation and visualization platform. It processes four MRI sequences (T1, T1c, T2, FLAIR), predicts tumor subregions using an 8:2 ensemble of SwinUNETR and nnU-Net ResEnc models, and provides interactive 3D visualization, volume measurements, and uncertainty estimation.

## Features

- **8:2 Hybrid Ensemble**: SwinUNETR (80%) + nnU-Net ResEnc M (20%) with probability-level fusion
- **BraTS-Standard Output**: WT, TC, ET segmentation with labels {0, 1, 2, 4}
- **3D Visualization**: NiiVue WebGL2 viewer with multiplanar slices and 3D mesh rendering
- **Volume Calculation**: Exact physical volumes using affine determinant (mL)
- **Uncertainty Maps**: Shannon entropy from ensemble disagreement
- **Expert Comparison**: Optional ground-truth mask upload with Dice and sensitivity metrics
- **Research Report**: JSON report with model provenance, preprocessing details, and SHA-256 input hashes

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
  models/
    model_loader.py       Checkpoint loading with validation
    resenc_unet.py        nnU-Net ResEnc architecture

configs/
  ensemble.json           Ensemble weights, channel orders, thresholds

models/                   Model checkpoints (not in Git)
tests/                    Unit and integration tests
notes/                    Research documentation and design decisions
```

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

## License

See individual component licenses in [MODEL_CARD.md](MODEL_CARD.md).
