# NeuroScope local application

Everything is in `/home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16`.

## Open the app

Visit http://127.0.0.1:8000 while the server is running.

If it is stopped, open a terminal in this folder and run:

```bash
./start-neuroscope.sh
```

Keep that terminal open. Ctrl+C stops the server. The server listens only on this laptop.

## First demonstration

Click **Analyze sample MRI**. This runs the real pretrained model on the downloaded research MRI case. Results include:

- Four-plane slice/3D viewer, FLAIR or T1c background, AI or expert mask, opacity control.
- Whole-tumor mesh and a brain surface derived from the skull-stripped FLAIR foreground.
- Whole tumor, tumor core, enhancing tumor and edema region volumes in mL.
- Dice and sensitivity against an optional expert reference.
- Downloads for the original-grid NIfTI mask, OBJ tumor surface and JSON report.
- Browser print/save-as-PDF for the displayed report.

No model training is needed. No LLM or remote inference API is involved. Once installed, the sample and UI work offline.

## Upload your own case

Provide four same-patient, aligned, skull-stripped 3D NIfTI volumes: T1c, T1, T2, FLAIR. JPEG/PNG screenshots are not supported by this model. Images are canonicalized to RAS and resampled to 1 mm for inference; predicted labels are mapped back to the original T1c grid with nearest-neighbor interpolation. The app does not register scans or skull-strip raw scans.

Expert mask is optional. Choose ET label 4 for traditional BraTS 2018–2021 masks or 3 for BraTS 2023 GLI. Output labels are always 0 background, 1 non-enhancing/necrotic core, 2 edema, 4 enhancing tumor. Composite regions are WT={1,2,4}, TC={1,4}, ET={4}. Regions overlap and should not be summed.

Uploads and results stay in `work/runs/<analysis-id>/`. Prior completed analyses are available from the dropdown even after server restart. Do not commit these data to Git.

## Sources and limitations

- Pretrained model: https://huggingface.co/monai-test/brats_mri_segmentation
- Demo case: BraTS-GLI-00000-000 from https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset
- Exact dataset revision recorded in `data/demo/source.json`.
- Viewer: https://niivue.com/docs/ ; installed version recorded in `work/viewer/package-lock.json`; runtime copied locally under `src/web/vendor/`.

Dataset mirror labels its license CC BY 4.0; retain attribution and consult original BraTS terms before redistribution. Dataset/weights are not committed. This is a research demonstration, not diagnosis. No claim of cancer detection in arbitrary images, clinical validation, or RANO response classification is made. Demo subject overlap with BraTS 2018 model training has not been excluded. Dice scores here describe this case only, not generalization accuracy. Smaller sliding windows differ from the model's original evaluation configuration.

## Checks

```bash
.venv/bin/python src/check_model.py
.venv/bin/python tests/check_local_app.py
```

The integration check requires the running server, performs an actual multipart sample upload, checks inference output geometry/labels, independently recomputes volume and Dice, and checks downloads and invalid-file rejection. `requirements-local.lock.txt` records the tested environment. CUDA PyTorch packages require the matching PyTorch wheel index when recreating the environment.
