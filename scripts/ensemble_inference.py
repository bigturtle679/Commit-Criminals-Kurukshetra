"""8:2 Weighted Multi-Model Ensemble Inference Engine.

Fuses:
- Model A (80%): SwinUNETR BraTS21 Vision Transformer
- Model B (20%): nnU-Net ResEnc Residual Encoder UNet (Preset M)
"""
import argparse
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
from src.validation import read_expert
from src.metrics import evaluate_case


def validate_mri(path):
    img = nib.load(str(path))
    if len(img.shape) != 3 or min(img.shape) < 8 or np.prod(img.shape) > 35_000_000:
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
    eps = max(float(eps), torch.finfo(torch.float32).eps)
    if not 0 < eps < 0.5:
        raise ValueError('Entropy epsilon must be between zero and 0.5')
    p = torch.clamp(probs.float(), eps, 1.0 - eps)
    entropy = - (p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    return entropy


def predict_ensemble(normed_crop, bundle, device):
    """Shared CLI/web inference, using each model's declared settings."""
    cfg = bundle['config']
    probabilities = []
    raw_order = ['t1c', 't1', 't2', 'flair']
    for key in ('model_a', 'model_b'):
        spec = cfg[key]
        order = spec['channel_order']
        if sorted(order) != sorted(raw_order):
            raise ValueError('Invalid model channel order')
        x = normed_crop[[raw_order.index(k) for k in order]].unsqueeze(0).contiguous()
        with torch.inference_mode():
            logits = sliding_window_inference(x.to(device), tuple(spec['roi_size']), 1,
                bundle[key], overlap=spec.get('sliding_window_overlap', 0.5),
                mode=spec.get('mode', 'gaussian'), sw_device=device, device='cpu')
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite model output')
        temperature = float(cfg.get('temperature', {}).get(key, 1.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        probabilities.append(torch.sigmoid(logits[0].float() / temperature))
    if cfg.get('fusion_mode', 'probability') != 'probability':
        raise ValueError('Only probability fusion is supported')
    if cfg.get('postprocessing', {}).get('min_et_volume_voxels', 0):
        raise ValueError('min_et_volume_voxels is unsupported; set it to 0')
    prob = bundle['weight_a'] * probabilities[0] + bundle['weight_b'] * probabilities[1]
    post = cfg.get('postprocessing', {})
    if post.get('enforce_hierarchy', True):
        prob = torch.stack(enforce_hierarchy(*prob))
    threshold = float(post.get('threshold', 0.5))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must lie strictly between zero and one')
    return (prob >= threshold).numpy(), compute_voxel_entropy(prob).mean(dim=0).numpy()


def run_ensemble(mri_paths, out_dir, config_path=None, expert_path=None, expert_et=4, device='auto', update=lambda msg: None):
    started = time.perf_counter()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    update("Loading ensemble models & validating inputs")

    dev = get_device(device)
    models_bundle = load_ensemble_models(config_path=config_path, device=dev)
    cfg = models_bundle['config']
    w_a = models_bundle['weight_a']
    w_b = models_bundle['weight_b']

    # Load and check 4 modalities: t1c, t1, t2, flair
    mod_order = ['t1c', 't1', 't2', 'flair']
    imgs = [validate_mri(mri_paths[k]) for k in mod_order]
    ref = imgs[0]
    for m, img in zip(mod_order[1:], imgs[1:]):
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"Modality {m} geometry does not match reference T1c geometry")

    expert_img = read_expert(expert_path, ref, expert_et) if expert_path is not None else None

    update("Standardizing to 1 mm isotropic canonical RAS space")
    canonical = nib.as_closest_canonical(ref)
    if not np.allclose(nib.affines.voxel_sizes(canonical.affine), [1, 1, 1], atol=1e-3):
        canonical = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1)
    if np.prod(canonical.shape) > 35_000_000:
        raise ValueError('Resampled MRI exceeds 35 million voxels')
    target_geom = (canonical.shape, canonical.affine)

    channels = []
    for img in imgs:
        can = nib.as_closest_canonical(img)
        if can.shape != target_geom[0] or not np.allclose(can.affine, target_geom[1], atol=1e-3):
            can = resample_from_to(can, target_geom, order=1)
        channels.append(can.get_fdata(dtype=np.float32))

    # Raw channel indices: 0: t1c, 1: t1, 2: t2, 3: flair
    # SwinUNETR expects [flair, t1c, t1, t2] -> indices [3, 0, 1, 2]
    # nnUNet ResEnc expects [flair, t1, t1c, t2] -> indices [3, 1, 0, 2]
    arr_raw = np.stack(channels)
    foreground = np.any(arr_raw != 0, axis=0)
    positions = np.where(foreground)
    if len(positions[0]) == 0:
        raise ValueError("No non-zero foreground voxels found across MRI sequences")
    slices = tuple(slice(max(0, int(p.min()) - 4), min(arr_raw.shape[i + 1], int(p.max()) + 5)) for i, p in enumerate(positions))

    # Normalized tensors
    norm_tf = NormalizeIntensity(nonzero=True, channel_wise=True)
    normed = norm_tf(arr_raw).as_tensor()

    pred, entropy_crop = predict_ensemble(normed[(slice(None),) + slices], models_bundle, dev)
    pred_tc, pred_wt, pred_et = pred

    # Discrete BraTS labels: 4: ET, 1: NET/Core, 2: Edema, 0: BG
    labels_crop = np.where(pred_et, 4, np.where(pred_tc, 1, np.where(pred_wt, 2, 0))).astype(np.uint8)
    full_labels = np.zeros(canonical.shape, dtype=np.uint8)
    full_labels[slices] = labels_crop

    # Compute Shannon entropy uncertainty map
    full_entropy = np.zeros(canonical.shape, dtype=np.float32)
    full_entropy[slices] = entropy_crop

    update("Projecting segmentation & uncertainty to native patient space")
    seg_canonical_img = nib.Nifti1Image(full_labels, canonical.affine)
    seg_native = resample_from_to(seg_canonical_img, (ref.shape, ref.affine), order=0)
    seg_data = np.asarray(seg_native.dataobj, dtype=np.uint8)

    ent_canonical_img = nib.Nifti1Image(full_entropy, canonical.affine)
    ent_native = resample_from_to(ent_canonical_img, (ref.shape, ref.affine), order=1)
    ent_data = np.asarray(ent_native.dataobj, dtype=np.float32)

    # Save NIfTI outputs
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(seg_data, ref.affine, header), out_dir / 'segmentation.nii.gz')

    ent_header = ref.header.copy()
    ent_header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(ent_data, ref.affine, ent_header), out_dir / 'uncertainty.nii.gz')

    # Compute physical volumes in mL
    voxel_ml = abs(float(np.linalg.det(ref.affine[:3, :3]))) / 1000.0
    regions = {
        'WT': np.isin(seg_data, [1, 2, 4]),
        'TC': np.isin(seg_data, [1, 4]),
        'ET': seg_data == 4,
        'ED': seg_data == 2
    }
    volumes_ml = {k: round(float(m.sum()) * voxel_ml, 3) for k, m in regions.items()}

    # Compute validation metrics if ground truth expert mask provided
    metrics = evaluate_case(seg_data, expert_img.get_fdata(), ref.affine, expert_et) if expert_img is not None else None

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        'ensemble': cfg.get('ensemble_name', 'neuroscope_8_2'),
        'weights': {'model_a': w_a, 'model_b': w_b},
        'models': {
            'model_a': models_bundle['desc_a'],
            'model_b': models_bundle['desc_b']
        },
        'device': str(dev),
        'checkpoint_sha256': models_bundle['checkpoint_sha256'],
        'settings': cfg,
        'runtime_seconds': elapsed,
        'volumes_ml': volumes_ml,
        'metrics': metrics,
        'mean_uncertainty_in_tumor': round(float(ent_data[regions['WT']].mean()), 4) if regions['WT'].any() else 0.0,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        'artifacts': {
            'segmentation': str(out_dir / 'segmentation.nii.gz'),
            'uncertainty': str(out_dir / 'uncertainty.nii.gz')
        }
    }

    report_path = out_dir / 'ensemble_report.json'
    report_path.write_text(json.dumps(report, indent=2))
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
