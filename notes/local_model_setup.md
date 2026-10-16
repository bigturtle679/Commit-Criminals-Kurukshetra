# Local MONAI model setup

Shared workspace: `/home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16`.

Model: https://huggingface.co/monai-test/brats_mri_segmentation
Pinned model revision: `c41338c7694661bed59101e040b284a99716c3d8`.
Weights and original inference configuration are in `models/brats_mri_segmentation/`.
The model uses SegResNet. Model weights are approximately 18 MB; the GPU runtime is much larger.

## Verify offline execution

From the shared workspace:

```bash
.venv/bin/python src/check_model.py
```

This loads the real weights and runs a synthetic 96×96×96 patch on CUDA.
A passing result verifies installation, weight compatibility, and a GPU forward pass.
It does not validate segmentation accuracy or benchmark a full patient scan.

## Real MRI prerequisites

Obtain four co-registered, skull-stripped, BraTS-style MRI volumes for one patient, with 1 mm isotropic spacing as required by the model. Input channel order is T1c, T1, T2, FLAIR. The bundle normalizes nonzero intensity values separately per channel. Outputs are TC, WT, ET; apply sigmoid and threshold 0.5, following the provided configuration. Arbitrary web images or a single MRI screenshot cannot substitute for these inputs.

Use sliding-window inference with one patch at a time and mixed precision for the 8 GB laptop GPU. The original bundle uses a larger 240×240×160 window; start with smaller windows and validate memory and results. Preserve spatial coordinates and use matching grids for evaluation. The local app was subsequently tested end-to-end on a public BraTS MRI case. See APP_START_HERE.md for usage and limitations.

The Python environment, downloaded weights, and local data are Git-ignored. The pretrained bundle is a research example, not for diagnosis.
