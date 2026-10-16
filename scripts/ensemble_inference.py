"""8:2 Weighted Multi-Model Ensemble Inference Engine.

Fuses:
- Model A (80%): SwinUNETR BraTS21 Vision Transformer
- Model B (20%): nnU-Net ResEnc Residual Encoder UNet (Preset M)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_to_output, resample_from_to
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.models.model_loader import load_ensemble_models, get_device
from configs.config_loader import load_ensemble_config

# Canonical modality ordering used during preprocessing.
# channels[0]=t1c, channels[1]=t1, channels[2]=t2, channels[3]=flair
MODALITY_INDEX = {'t1c': 0, 't1': 1, 't2': 2, 'flair': 3}


def _build_channel_indices(channel_order: list[str]) -> list[int]:
    """Convert a config channel_order list to numpy/tensor index list.

    Example: ['flair', 't1c', 't1', 't2'] → [3, 0, 1, 2]
    """
    return [MODALITY_INDEX[m] for m in channel_order]


_HASH_CACHE = {}


def _sha256_file(path) -> str:
    """Compute SHA-256 hash of a file for provenance with mtime/size caching."""
    if path is None:
        return 'missing'
    p = Path(path)
    if not p.exists():
        return 'not_found'
    stat = p.stat()
    cache_key = (str(p.resolve()), stat.st_mtime, stat.st_size)
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]

    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    digest = h.hexdigest()
    _HASH_CACHE[cache_key] = digest
    return digest


def validate_mri(path):
    img = nib.load(str(path))
    if len(img.shape) != 3 or min(img.shape) < 8:
        raise ValueError(f"Each input must be a 3D MRI volume, got shape {img.shape} from {path}")
    units = img.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError(f"Spatial units must be millimeters, got {units}")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid spatial affine matrix in MRI header")
    data = img.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError("MRI contains non-finite numbers or is completely empty")
    return img


def enforce_hierarchy(prob_tc, prob_wt, prob_et):
    """Enforce BraTS anatomical containment: ET <= TC <= WT."""
    p_tc_corr = torch.maximum(prob_tc, prob_et)
    p_wt_corr = torch.maximum(prob_wt, p_tc_corr)
    return p_tc_corr, p_wt_corr, prob_et


def compute_voxel_entropy(probs, eps=1e-7):
    """Compute Shannon entropy H(v) = -p*log2(p) - (1-p)*log2(1-p)."""
    if not torch.isfinite(probs).all():
        raise ValueError('Entropy input contains nonfinite probabilities')
    # Promote to FP32: an eps of 1e-7 rounds to 0 in FP16/BF16 and log2(0) = -inf.
    eps = max(float(eps), torch.finfo(torch.float32).eps)
    if not 0 < eps < 0.5:
        raise ValueError('Entropy epsilon must be between zero and 0.5')
    p = torch.clamp(probs.float(), eps, 1.0 - eps)
    entropy = - (p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    return entropy


def run_ensemble(mri_paths, out_dir, config_path=None, validated_config=None,
                 expert_path=None, expert_et=4, device='auto',
                 update=lambda msg: None, models_bundle=None):
    """Run the authoritative 8:2 ensemble inference.

    This is the SINGLE implementation of ensemble inference used by both
    the CLI and the web application. src/pipeline.py delegates here.

    Args:
        mri_paths: dict with keys 't1c', 't1', 't2', 'flair' → file paths
        out_dir: output directory for NIfTI and report artifacts
        config_path: path to ensemble.json (used if validated_config is None)
        validated_config: pre-validated config dict from config_loader
        expert_path: optional expert reference mask path
        expert_et: ET label value in expert mask (3 or 4)
        device: 'auto', 'cuda', 'mps', or 'cpu'
        update: progress callback function
        models_bundle: pre-loaded models dict (avoids reloading per call)

    Returns:
        dict with inference results, volumes, metrics, and provenance

    Raises:
        ValueError: invalid inputs
        CheckpointNotFoundError: missing model checkpoints
        CheckpointLoadError: incompatible checkpoints
    """
    started = time.perf_counter()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    update("Loading ensemble models & validating inputs")

    # Use validated config from config_loader (authoritative source)
    if validated_config is not None:
        vcfg = validated_config
    else:
        vcfg = load_ensemble_config(config_path)

    dev = get_device(device)

    # Load models if not pre-loaded
    if models_bundle is None:
        models_bundle = load_ensemble_models(config_path=config_path, device=dev)

    w_a = vcfg['weight_a']
    w_b = vcfg['weight_b']

    # Load and check 4 modalities: t1c, t1, t2, flair
    mod_order = ['t1c', 't1', 't2', 'flair']
    imgs = [validate_mri(mri_paths[k]) for k in mod_order]
    ref = imgs[0]
    for m, img in zip(mod_order[1:], imgs[1:]):
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"Modality {m} geometry does not match reference T1c geometry")

    update("Standardizing to 1 mm isotropic canonical RAS space")
    canonical = nib.as_closest_canonical(ref)
    if not np.allclose(nib.affines.voxel_sizes(canonical.affine), [1, 1, 1], atol=1e-3):
        canonical = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1)
    target_geom = (canonical.shape, canonical.affine)

    channels = []
    for img in imgs:
        can = nib.as_closest_canonical(img)
        if can.shape != target_geom[0] or not np.allclose(can.affine, target_geom[1], atol=1e-3):
            can = resample_from_to(can, target_geom, order=1)
        channels.append(can.get_fdata(dtype=np.float32))

    # Stack channels in canonical order: [t1c, t1, t2, flair]
    arr_raw = np.stack(channels)
    foreground = np.any(arr_raw != 0, axis=0)
    positions = np.where(foreground)
    if len(positions[0]) == 0:
        raise ValueError("No non-zero foreground voxels found across MRI sequences")
    slices = tuple(slice(max(0, int(p.min()) - 4), min(arr_raw.shape[i + 1], int(p.max()) + 5)) for i, p in enumerate(positions))

    # Normalized tensors
    norm_tf = NormalizeIntensity(nonzero=True, channel_wise=True)
    normed = norm_tf(arr_raw).as_tensor()

    # Build model-specific channel reordering from config (not hardcoded)
    idx_a = _build_channel_indices(vcfg['model_a']['channel_order'])
    idx_b = _build_channel_indices(vcfg['model_b']['channel_order'])
    x_a = normed[idx_a][(slice(None),) + slices].unsqueeze(0).contiguous()
    x_b = normed[idx_b][(slice(None),) + slices].unsqueeze(0).contiguous()

    # Read inference parameters from validated config
    roi_a = vcfg['model_a']['roi_size']
    roi_b = vcfg['model_b']['roi_size']
    overlap_a = vcfg['model_a']['sliding_window_overlap']
    overlap_b = vcfg['model_b']['sliding_window_overlap']
    mode_a = vcfg['model_a'].get('mode', 'gaussian')
    mode_b = vcfg['model_b'].get('mode', 'gaussian')
    threshold = vcfg['postprocessing']['threshold']
    fusion_mode = vcfg.get('fusion_mode', 'probability')
    temp_cfg = vcfg.get('temperature', {})
    temp_a = float(temp_cfg.get('model_a', 1.0))
    temp_b = float(temp_cfg.get('model_b', 1.0))
    min_et_voxels = vcfg['postprocessing'].get('min_et_volume_voxels', 0)
    export_uncertainty = vcfg['postprocessing'].get('export_uncertainty_map', True)

    update(f"Executing Model A ({w_a * 100:.0f}%): {models_bundle['desc_a']}")
    with torch.inference_mode():
        logits_a = sliding_window_inference(
            x_a.to(dev), roi_a, 1, models_bundle['model_a'],
            overlap=overlap_a, mode=mode_a, sw_device=dev, device='cpu'
        )

    # Validate model A output
    if not torch.isfinite(logits_a).all():
        raise RuntimeError("Model A produced non-finite logits. Checkpoint or input may be corrupt.")

    update(f"Executing Model B ({w_b * 100:.0f}%): {models_bundle['desc_b']}")
    with torch.inference_mode():
        logits_b = sliding_window_inference(
            x_b.to(dev), roi_b, 1, models_bundle['model_b'],
            overlap=overlap_b, mode=mode_b, sw_device=dev, device='cpu'
        )

    # Validate model B output
    if not torch.isfinite(logits_b).all():
        raise RuntimeError("Model B produced non-finite logits. Checkpoint or input may be corrupt.")

    # Verify output shapes match
    if logits_a.shape != logits_b.shape:
        raise RuntimeError(
            f"Model output shape mismatch: A={list(logits_a.shape)}, B={list(logits_b.shape)}. "
            f"Cannot fuse incompatible tensors."
        )

    # Apply temperature scaling and fuse according to configured fusion_mode
    la = logits_a[0].float() / max(temp_a, 1e-4)
    lb = logits_b[0].float() / max(temp_b, 1e-4)

    if fusion_mode == 'logit':
        update(f"Fusing logits with {w_a:.1f} : {w_b:.1f} ratio")
        logits_ens = w_a * la + w_b * lb
        prob_ens = torch.sigmoid(logits_ens)
        prob_a = torch.sigmoid(la)
        prob_b = torch.sigmoid(lb)
    else:  # 'probability' (default)
        update(f"Fusing probabilities with {w_a:.1f} : {w_b:.1f} ratio")
        prob_a = torch.sigmoid(la)
        prob_b = torch.sigmoid(lb)
        if not (torch.isfinite(prob_a).all() and torch.isfinite(prob_b).all()):
            raise RuntimeError("Non-finite probabilities computed after sigmoid.")
        prob_ens = w_a * prob_a + w_b * prob_b

    if not torch.isfinite(prob_ens).all():
        raise RuntimeError("Ensemble produced non-finite probability values.")

    # Multi-model prediction disagreement (|p_a - p_b|)
    disagreement_crop = torch.abs(prob_a - prob_b)

    # Post-processing: Channel 0: TC, Channel 1: WT, Channel 2: ET
    p_tc, p_wt, p_et = prob_ens[0], prob_ens[1], prob_ens[2]
    if vcfg['postprocessing']['enforce_hierarchy']:
        p_tc, p_wt, p_et = enforce_hierarchy(p_tc, p_wt, p_et)

    pred_tc = (p_tc >= threshold).numpy()
    pred_wt = (p_wt >= threshold).numpy()
    pred_et = (p_et >= threshold).numpy()

    # Apply minimum enhancing tumor volume postprocessing
    if min_et_voxels > 0 and int(np.count_nonzero(pred_et)) < min_et_voxels:
        pred_et = np.zeros_like(pred_et)

    # Discrete BraTS labels: 4: ET, 1: NET/Core, 2: Edema, 0: BG
    labels_crop = np.where(pred_et, 4, np.where(pred_tc, 1, np.where(pred_wt, 2, 0))).astype(np.uint8)
    full_labels = np.zeros(canonical.shape, dtype=np.uint8)
    full_labels[slices] = labels_crop

    # Compute Shannon entropy uncertainty map if enabled
    if export_uncertainty:
        entropy_crop = compute_voxel_entropy(torch.stack([p_tc, p_wt, p_et])).mean(dim=0).numpy()
        full_entropy = np.zeros(canonical.shape, dtype=np.float32)
        full_entropy[slices] = entropy_crop
    else:
        full_entropy = None

    update("Projecting segmentation & uncertainty to native patient space")
    seg_canonical_img = nib.Nifti1Image(full_labels, canonical.affine)
    seg_native = resample_from_to(seg_canonical_img, (ref.shape, ref.affine), order=0)
    seg_data = np.asarray(seg_native.dataobj, dtype=np.uint8)

    if export_uncertainty and full_entropy is not None:
        ent_canonical_img = nib.Nifti1Image(full_entropy, canonical.affine)
        ent_native = resample_from_to(ent_canonical_img, (ref.shape, ref.affine), order=1)
        ent_data = np.asarray(ent_native.dataobj, dtype=np.float32)
    else:
        ent_data = None

    # Save NIfTI outputs
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(seg_data, ref.affine, header), out_dir / 'segmentation.nii.gz')

    if export_uncertainty and ent_data is not None:
        ent_header = ref.header.copy()
        ent_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(ent_data, ref.affine, ent_header), out_dir / 'uncertainty.nii.gz')

    # Compute physical volumes using metrics.py (single source of truth)
    sys.path.insert(0, str(ROOT / 'src'))
    from metrics import calculate_volume_ml, compute_dice, compute_sensitivity

    regions = {
        'WT': np.isin(seg_data, [1, 2, 4]),
        'TC': np.isin(seg_data, [1, 4]),
        'ET': seg_data == 4,
        'ED': seg_data == 2
    }
    volumes_ml = {k: calculate_volume_ml(m, ref.affine) for k, m in regions.items()}

    # Compute validation metrics if ground truth expert mask provided
    metrics = None
    if expert_path is not None and Path(expert_path).exists():
        gt_img = nib.load(str(expert_path))
        # Validate expert mask geometry against reference
        if gt_img.shape != ref.shape:
            raise ValueError('Expert mask shape does not match the MRI volumes.')
        if not np.allclose(gt_img.affine, ref.affine, atol=1e-3):
            raise ValueError('Expert mask affine does not match the MRI affine.')
        gt = gt_img.get_fdata()
        allowed = {0, 1, 2, expert_et}
        gt_unique = set(np.unique(gt).astype(int))
        if not gt_unique.issubset(allowed):
            raise ValueError(
                f'Expert mask labels do not match convention. Expected subset of {allowed}, got {gt_unique}'
            )
        gt_regions = {
            'WT': np.isin(gt, [1, 2, expert_et]),
            'TC': np.isin(gt, [1, expert_et]),
            'ET': gt == expert_et
        }
        metrics = {}
        for k in ['WT', 'TC', 'ET']:
            metrics[k] = {
                'dice': compute_dice(regions[k], gt_regions[k]),
                'sensitivity': compute_sensitivity(regions[k], gt_regions[k]),
            }

    # Compute checkpoint hashes for provenance
    ckpt_hashes = {}
    for key, path_key in [('model_a', 'path_a'), ('model_b', 'path_b')]:
        ckpt_path = models_bundle.get(path_key)
        if ckpt_path and Path(ckpt_path).exists():
            try:
                ckpt_hashes[key] = _sha256_file(ckpt_path)
            except Exception:
                ckpt_hashes[key] = 'unavailable'
        else:
            ckpt_hashes[key] = 'unavailable'

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        'ensemble': vcfg['ensemble_name'],
        'weights': {'model_a': w_a, 'model_b': w_b},
        'models': {
            'model_a': models_bundle['desc_a'],
            'model_b': models_bundle['desc_b']
        },
        'checkpoint_hashes': ckpt_hashes,
        'device': str(dev),
        'runtime_seconds': elapsed,
        'volumes_ml': volumes_ml,
        'metrics': metrics,
        'mean_uncertainty_in_tumor': (
            round(float(ent_data[regions['WT']].mean()), 4)
            if (export_uncertainty and ent_data is not None and regions['WT'].any())
            else None
        ),
        'uncertainty_metric': 'voxel_bernoulli_entropy' if export_uncertainty else None,
        'uncertainty_interpretation': (
            'Mean Shannon Bernoulli entropy across tumor regions (WT), representing '
            'voxel-level prediction confidence. It is not multi-model disagreement or clinical confidence.'
            if export_uncertainty else 'Uncertainty map export disabled.'
        ),
        'model_disagreement': {
            'whole_crop_mean': round(float(disagreement_crop.mean().item()), 4),
            'tumor_region_mean': round(float(disagreement_crop[:, pred_wt].mean().item()), 4) if pred_wt.any() else 0.0,
            'metric': 'mean_absolute_probability_difference',
            'interpretation': (
                'Voxel-wise absolute difference between Model A (SwinUNETR) and Model B (ResEnc M) '
                'predicted probabilities (|p_A - p_B|), measuring genuine inter-model divergence.'
            )
        },
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'effective_config': {
            'threshold': threshold,
            'fusion_mode': fusion_mode,
            'temperature': {'model_a': temp_a, 'model_b': temp_b},
            'roi_a': list(roi_a),
            'roi_b': list(roi_b),
            'overlap_a': overlap_a,
            'overlap_b': overlap_b,
            'mode_a': mode_a,
            'mode_b': mode_b,
            'channel_order_a': vcfg['model_a']['channel_order'],
            'channel_order_b': vcfg['model_b']['channel_order'],
            'enforce_hierarchy': vcfg['postprocessing']['enforce_hierarchy'],
            'min_et_volume_voxels': min_et_voxels,
            'export_uncertainty_map': export_uncertainty,
        },
        'provenance_note': (
            'Checkpoint SHA-256 hashes certify file integrity and reproducibility. '
            'They do not constitute clinical authorization, peer review, or diagnostic validation.'
        ),
        'disclaimer': (
            'Research demonstration only. Not intended for clinical diagnosis. '
            'Predicted regions are model estimates, not expert-verified findings.'
        ),
        'artifacts': {
            'segmentation': str(out_dir / 'segmentation.nii.gz'),
            'uncertainty': str(out_dir / 'uncertainty.nii.gz') if export_uncertainty else None
        },
        # Pass through raw data for pipeline integration
        '_seg_data': seg_data,
        '_ent_data': ent_data,
        '_ref_img': ref,
        '_regions': regions,
    }

    report_path = out_dir / 'ensemble_report.json'
    # Write report without internal fields
    public_report = {k: v for k, v in report.items() if not k.startswith('_')}
    report_path.write_text(json.dumps(public_report, indent=2))
    update(f"Completed in {elapsed}s. Results written to {out_dir}")
    return report


def main():
    p = argparse.ArgumentParser(description="NeuroScope 8:2 Ensemble Inference")
    p.add_argument("--t1c", required=True, help="Path to T1ce/T1Gd contrast MRI")
    p.add_argument("--t1", required=True, help="Path to T1 pre-contrast MRI")
    p.add_argument("--t2", required=True, help="Path to T2 MRI")
    p.add_argument("--flair", required=True, help="Path to FLAIR MRI")
    p.add_argument("--output", default="outputs/ensemble_inference", help="Output directory")
    p.add_argument("--config", default="configs/ensemble.json", help="Path to ensemble config")
    p.add_argument("--expert", default=None, help="Optional expert reference mask")
    p.add_argument("--expert-et", type=int, default=4, help="ET label in expert mask (3 or 4)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    paths = {'t1c': args.t1c, 't1': args.t1, 't2': args.t2, 'flair': args.flair}
    rep = run_ensemble(
        mri_paths=paths,
        out_dir=args.output,
        config_path=args.config,
        expert_path=args.expert,
        expert_et=args.expert_et,
        device=args.device,
        update=print
    )
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
