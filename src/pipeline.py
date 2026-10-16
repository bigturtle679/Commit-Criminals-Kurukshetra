"""NeuroScope 3D inference pipeline — orchestration and output layer.

Coordinates MRI validation, inference dispatch, spatial restoration,
mesh generation, volume calculation, and report assembly.

ARCHITECTURE:
- Ensemble inference is delegated to scripts/ensemble_inference.py
  (single authoritative implementation — no duplicate inference here)
- Model loading is delegated to scripts/models/model_loader.py
- Configuration is loaded via configs/config_loader.py
- Metrics are computed via src/metrics.py

This module is responsible ONLY for:
- Input validation and case preparation
- Calling the authoritative inference API
- Mesh generation for 3D visualization
- Volume summary computation
- Expert-mask handling (label normalization for viewer)
- Report generation and provenance assembly
- Packaging the response expected by the web application
"""
import hashlib
import json
import logging
import subprocess
import time
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.measure import marching_cubes

from metrics import compute_dice, compute_sensitivity, calculate_volume_ml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {}
LOCK = threading.Lock()
torch.set_num_threads(4)

# Maximum upload size per file (1 GB)
MAX_FILE_BYTES = 1_073_741_824


def _get_device():
    """Select best available compute device."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _gpu_name():
    """Return GPU device name or 'CPU' without crashing on non-CUDA machines."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'Apple Silicon (MPS)'
    return 'CPU'


def _git_commit():
    """Return short Git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5, cwd=ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def _sha256(path):
    """Compute SHA-256 hash of a file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_name='ensemble'):
    """Load and cache the requested model(s).

    Raises FileNotFoundError (or CheckpointNotFoundError) if checkpoints are missing.
    This is intentional: no random-weight model is ever returned for inference.
    """
    global MODELS
    model_name = (model_name or 'ensemble').lower().strip()
    if model_name in MODELS:
        return MODELS[model_name]

    device = _get_device()

    if model_name in ('ensemble', '8_2', 'swin_resenc'):
        from scripts.models.model_loader import load_ensemble_models, get_device
        from configs.config_loader import load_ensemble_config
        vcfg = load_ensemble_config()
        ens = load_ensemble_models(device=get_device(device))
        title = f"8:2 Hybrid Ensemble ({ens['desc_a']} + {ens['desc_b']})"
        MODELS['ensemble'] = (ens, title, vcfg)
        return MODELS['ensemble']

    elif model_name in ('swinunetr', 'swin_unetr', 'swin'):
        from scripts.models.model_loader import load_swinunetr, get_device
        model, desc, path = load_swinunetr(device=get_device(device))
        title = f'MONAI {desc}'
        MODELS['swinunetr'] = (model, title, None)
        return MODELS['swinunetr']

    elif model_name in ('segresnet', 'resnet'):
        from scripts.models.model_loader import load_segresnet, get_device
        model, desc, path = load_segresnet(device=get_device(device))
        title = f'MONAI {desc}'
        MODELS['segresnet'] = (model, title, None)
        return MODELS['segresnet']

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'ensemble', 'swinunetr' or 'segresnet'.")


def read_image(path, is_expert_mask=False):
    """Validate and load a NIfTI MRI volume or expert mask.

    Checks: 3D shape, voxel count, spatial units, affine validity,
    finite values. For MRI volumes, also checks non-constant intensity.
    For expert masks, allows all-zero masks (valid in evaluation scenarios).

    Args:
        path: Path to NIfTI file
        is_expert_mask: If True, skip MRI-specific intensity validation
    """
    path = Path(path)

    # File size check
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'File exceeds 1 GB limit: {path.name} ({path.stat().st_size / 1e9:.1f} GB)')

    try:
        img = nib.load(str(path))
    except Exception as e:
        raise ValueError(f'Cannot read {path.name} as NIfTI: {e}') from e

    if len(img.shape) != 3:
        raise ValueError(
            f'{path.name} is {len(img.shape)}D (shape {img.shape}). '
            f'Each upload must be a single 3D MRI volume, not a 2D image or 4D series.'
        )
    if min(img.shape) < 8:
        raise ValueError(f'{path.name} has shape {img.shape} which is too small for brain MRI.')
    if np.prod(img.shape) > 35_000_000:
        raise ValueError(f'{path.name} exceeds 35 million voxels ({np.prod(img.shape):,}).')

    units = img.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError(f'{path.name}: spatial units must be millimeters, got {units!r}.')

    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError(f'{path.name}: invalid spatial affine in MRI header.')

    a = img.get_fdata(dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f'{path.name} contains NaN or Inf values.')

    if not is_expert_mask:
        # MRI-specific checks: reject all-zero and constant intensity
        if not np.any(a):
            raise ValueError(f'{path.name} is entirely zero/blank.')
        if np.std(a[a != 0]) < 1e-6:
            raise ValueError(f'{path.name} has constant intensity (no tissue contrast).')
    else:
        # Expert mask validation: check label values
        unique_labels = set(np.unique(a).astype(int))
        # Don't reject all-zero masks — valid evaluation scenario

    return img


def mesh(mask, affine, path):
    """Generate a triangulated OBJ mesh from a 3D binary mask."""
    if not mask.any():
        return False
    padded = np.pad(mask.astype(np.uint8), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, 0.5, step_size=2)
        verts = nib.affines.apply_affine(affine, verts - 1)
        with open(path, 'w') as f:
            for v in verts:
                f.write('v %.4f %.4f %.4f\n' % tuple(v))
            for face in faces + 1:
                f.write('f %d %d %d\n' % tuple(face))
        return True
    except Exception:
        return False


def region_masks(labels, enhancing=4):
    """Derive WT, TC, ET binary masks from BraTS-convention labels."""
    return {
        'WT': np.isin(labels, [1, 2, enhancing]),
        'TC': np.isin(labels, [1, enhancing]),
        'ET': labels == enhancing
    }


def compute_volume_summary(volumes):
    """Generate a descriptive volume summary from region volumes."""
    v_wt = volumes.get('WT', 0.0)
    v_tc = volumes.get('TC', 0.0)
    v_et = volumes.get('ET', 0.0)
    v_ed = volumes.get('ED', 0.0)

    if v_wt > 0:
        et_ratio = v_et / v_wt
        ed_ratio = v_ed / v_wt
    else:
        et_ratio = ed_ratio = 0.0

    return {
        "whole_tumor_ml": v_wt,
        "tumor_core_ml": v_tc,
        "enhancing_tumor_ml": v_et,
        "edema_region_ml": v_ed,
        "enhancing_fraction_pct": round(et_ratio * 100, 1),
        "edema_fraction_pct": round(ed_ratio * 100, 1),
        "summary": (
            f"Predicted whole-tumor region: {v_wt:.1f} mL; "
            f"enhancing region: {v_et:.1f} mL. "
            f"These are segmentation measurements, not a response assessment."
        )
    }


def run_case(paths, out, update=lambda message: None, expert=None,
             expert_et=4, case_name='Uploaded MRI', model_name='ensemble'):
    """Run the full inference pipeline on a set of MRI volumes.

    For ensemble mode (default), delegates to scripts/ensemble_inference.py
    which is the SINGLE authoritative inference implementation.

    Args:
        paths: dict with keys 't1c', 't1', 't2', 'flair' → file paths
        out: output directory for artifacts
        update: callback for progress messages
        expert: optional expert mask file path
        expert_et: enhancing tumor label in expert mask (3 or 4)
        case_name: display name for the case
        model_name: 'ensemble' (default), 'swinunetr', or 'segresnet'
    """
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Compute input hashes for provenance
    input_hashes = {}
    for k in ('t1c', 't1', 't2', 'flair'):
        try:
            input_hashes[k] = _sha256(paths[k])
        except Exception:
            input_hashes[k] = 'unavailable'

    update('Validating MRI dimensions and coordinate alignment')

    # Validate all four MRI modalities
    imgs = [read_image(paths[k]) for k in ('t1c', 't1', 't2', 'flair')]
    original = imgs[0]  # T1c is the output reference grid
    for img in imgs[1:]:
        if img.shape != original.shape:
            raise ValueError(
                'MRI volume shapes do not match. All four sequences must be '
                'co-registered from the same patient examination.'
            )
        if not np.allclose(img.affine, original.affine, atol=1e-3):
            raise ValueError(
                'MRI affine transforms do not match. All four sequences must '
                'share the same spatial coordinate system.'
            )

    # Expert mask validation (shape AND affine, with mask-specific rules)
    expert_img = read_image(expert, is_expert_mask=True) if expert else None
    if expert_img is not None:
        if expert_img.shape != original.shape:
            raise ValueError('Expert mask shape does not match the MRI volumes.')
        if not np.allclose(expert_img.affine, original.affine, atol=1e-3):
            raise ValueError(
                'Expert mask affine does not match the MRI affine. '
                'Both must be in the same spatial coordinate system.'
            )

    device = _get_device()
    peak = 0.0

    with LOCK:
        bundle_tuple = load_model(model_name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if model_name in ('ensemble', '8_2', 'swin_resenc'):
            # ============================================================
            # ENSEMBLE MODE: Delegate to the authoritative implementation
            # in scripts/ensemble_inference.py. No duplicate inference here.
            # ============================================================
            bundle, model_title, vcfg = bundle_tuple

            from scripts.ensemble_inference import run_ensemble
            ens_result = run_ensemble(
                mri_paths=paths,
                out_dir=str(out),
                validated_config=vcfg,
                expert_path=expert,
                expert_et=expert_et,
                device=device,
                update=update,
                models_bundle=bundle,
            )

            # Extract results from authoritative inference
            seg_data = ens_result['_seg_data']
            regions = ens_result['_regions']
            ref_img = ens_result['_ref_img']
            volumes = ens_result['volumes_ml']
            metrics = ens_result.get('metrics')
            uncertainty_info = {
                'available': (out / 'uncertainty.nii.gz').exists(),
                'mean_in_tumor': ens_result.get('mean_uncertainty_in_tumor'),
                'metric': ens_result.get('uncertainty_metric'),
                'interpretation': ens_result.get('uncertainty_interpretation'),
            }
            model_disagreement = ens_result.get('model_disagreement')
            effective_config = ens_result.get('effective_config')

            input_channel_order = {
                'model_a': vcfg['model_a']['channel_order'],
                'model_b': vcfg['model_b']['channel_order'],
            }
            roi_size = vcfg['model_a']['roi_size']
            checkpoint_hashes = ens_result.get('checkpoint_hashes', {})

        else:
            # Single-model mode (swinunetr or segresnet) — inline inference
            # These are secondary modes; ensemble is the production default.
            bundle, model_title, _ = bundle_tuple
            model = bundle

            from monai.inferers import sliding_window_inference
            from monai.transforms import NormalizeIntensity
            from nibabel.processing import resample_to_output, resample_from_to
            from configs.config_loader import load_ensemble_config

            # Try to read ROI from config; fall back for single-model mode
            try:
                vcfg = load_ensemble_config()
                roi_size = vcfg['model_a']['roi_size']
            except Exception:
                roi_size = (96, 96, 96)

            update(f'Standardizing to 1 mm isotropic RAS grid')
            canonical = nib.as_closest_canonical(original)
            if not np.allclose(nib.affines.voxel_sizes(canonical.affine), [1, 1, 1], atol=1e-3):
                canonical = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1)
            target = (canonical.shape, canonical.affine)
            if np.prod(canonical.shape) > 35_000_000:
                raise ValueError('Resampled MRI is too large. Use a prepared BraTS-style case.')

            channels = []
            for img in imgs:
                can = nib.as_closest_canonical(img)
                if can.shape != target[0] or not np.allclose(can.affine, target[1], atol=1e-3):
                    can = resample_from_to(can, target, order=1)
                channels.append(can.get_fdata(dtype=np.float32))

            raw_arr = np.stack(channels)
            foreground = np.any(raw_arr != 0, axis=0)
            positions = np.where(foreground)
            if len(positions[0]) == 0:
                raise ValueError("No non-zero foreground voxels found across MRI sequences")
            slices = tuple(
                slice(max(0, int(p.min()) - 4), min(raw_arr.shape[i + 1], int(p.max()) + 5))
                for i, p in enumerate(positions)
            )
            norm_arr = NormalizeIntensity(nonzero=True, channel_wise=True)(raw_arr).as_tensor()

            model_display = "SwinUNETR Transformer" if "Swin" in model_title else "SegResNet CNN"
            update(f'Segmenting with {model_display} ({roi_size[0]}³ sliding patches)')
            # SwinUNETR: [FLAIR, T1c, T1, T2]; SegResNet: [T1c, T1, T2, FLAIR]
            mod_indices = [3, 0, 1, 2] if model_name == 'swinunetr' else [0, 1, 2, 3]
            x = norm_arr[mod_indices][(slice(None),) + slices].unsqueeze(0).contiguous().to(device)

            with torch.inference_mode():
                logits = sliding_window_inference(
                    x, roi_size, 1, model,
                    overlap=0.5, mode='gaussian', sw_device=device, device='cpu'
                )
            pred = (torch.sigmoid(logits[0].float()) >= 0.5).numpy()

            if model_name == 'swinunetr':
                input_channel_order = ['FLAIR', 'T1c', 'T1', 'T2']
            else:
                input_channel_order = ['T1c', 'T1', 'T2', 'FLAIR']

            # Convert to BraTS labels
            labels = np.where(pred[2], 4, np.where(pred[0], 1, np.where(pred[1], 2, 0))).astype(np.uint8)
            full = np.zeros(canonical.shape, np.uint8)
            full[slices] = labels

            update('Projecting predictions back to native patient space')
            seg = resample_from_to(
                nib.Nifti1Image(full, canonical.affine),
                (original.shape, original.affine), order=0
            )
            seg_data = np.asarray(seg.dataobj, dtype=np.uint8)
            header = original.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(seg_data, original.affine, header), out / 'segmentation.nii.gz')

            ref_img = original
            regions = region_masks(seg_data)
            volumes = {k: calculate_volume_ml(m, original.affine) for k, m in regions.items()}
            volumes['ED'] = calculate_volume_ml(seg_data == 2, original.affine)
            checkpoint_hashes = {}

            # Expert mask evaluation for single-model mode
            metrics = None
            if expert_img is not None:
                gt = expert_img.get_fdata()
                allowed = {0, 1, 2, expert_et}
                if not set(np.unique(gt).astype(int)).issubset(allowed):
                    raise ValueError(
                        f'Expert mask labels do not match convention. '
                        f'Expected subset of {allowed}, got {set(np.unique(gt).astype(int))}'
                    )
                gt_regions = region_masks(gt, expert_et)
                metrics = {}
                for k, p in regions.items():
                    g = gt_regions[k]
                    metrics[k] = {
                        'dice': compute_dice(p, g),
                        'sensitivity': compute_sensitivity(p, g),
                    }

            uncertainty_info = {
                'available': False,
                'mean_in_tumor': None,
                'metric': None,
                'interpretation': 'Uncertainty map generation is supported only in ensemble mode.',
            }
            model_disagreement = None
            effective_config = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    # Save reference MRI files for viewer
    nib.save(imgs[3], out / 'flair.nii.gz')
    nib.save(original, out / 't1c.nii.gz')
    nib.save(imgs[1], out / 't1.nii.gz')
    nib.save(imgs[2], out / 't2.nii.gz')

    # Ensure segmentation is saved (ensemble mode saves it internally)
    if not (out / 'segmentation.nii.gz').exists():
        header = ref_img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(seg_data, ref_img.affine, header), out / 'segmentation.nii.gz')

    # Compute edema volume if not already present
    if 'ED' not in volumes:
        volumes['ED'] = calculate_volume_ml(seg_data == 2, ref_img.affine)

    # Expert mask: normalize and save for viewer
    if expert_img is not None and metrics is not None:
        gt = expert_img.get_fdata()
        normalized = np.where(gt == expert_et, 4, gt).astype(np.uint8)
        header = ref_img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(normalized, ref_img.affine, header), out / 'expert.nii.gz')

    update('Synthesizing 3D polygonal meshes and measurement report')
    meshes = []
    # 1. Brain shell from FLAIR foreground
    if mesh(imgs[3].get_fdata() != 0, original.affine, out / 'brain.obj'):
        meshes.append('brain')
    # 2. Whole Tumor surface
    if mesh(regions['WT'], original.affine, out / 'tumor.obj'):
        meshes.append('tumor')
    # 3. Tumor Core surface
    if mesh(regions['TC'], original.affine, out / 'core.obj'):
        meshes.append('core')
    # 4. Enhancing Tumor surface
    if mesh(regions['ET'], original.affine, out / 'et.obj'):
        meshes.append('et')

    volume_summary = compute_volume_summary(volumes)

    result = {
        'case_name': case_name,
        'model_name': model_name,
        'model': model_title,
        'input_channel_order': input_channel_order,
        'sliding_window_roi': list(roi_size),
        'volumes_ml': volumes,
        'volume_summary': volume_summary,
        'metrics': metrics,
        'uncertainty': uncertainty_info,
        'model_disagreement': model_disagreement,
        'effective_config': effective_config,
        'runtime_seconds': round(time.perf_counter() - started, 2),
        'peak_gpu_mib': peak,
        'gpu': _gpu_name(),
        'device': device,
        'shape': list(original.shape),
        'affine': original.affine.tolist(),
        'available_modalities': ['t1', 't1c', 't2', 'flair'],
        'input_hashes': input_hashes,
        'spacing_mm': nib.affines.voxel_sizes(original.affine).round(4).tolist(),
        'meshes': meshes,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'input_hashes': input_hashes,
        'checkpoint_hashes': checkpoint_hashes,
        'code_commit': _git_commit(),
        'preprocessing': {
            'canonical': 'RAS',
            'resampling': '1mm isotropic',
            'normalization': 'z-score (nonzero, channel-wise)',
            'foreground_crop': True,
        },
        'disclaimer': (
            'Research demonstration only. Not intended for primary clinical '
            'diagnosis or treatment decisions. Predicted regions are model '
            'estimates, not expert-verified findings.'
        ),
        'note': (
            'Voxel volumes computed via exact 3x3 affine determinant. '
            'Segmentation quality depends on model training data and '
            'input preprocessing. See MODEL_CARD.md for details.'
        ),
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    return result
