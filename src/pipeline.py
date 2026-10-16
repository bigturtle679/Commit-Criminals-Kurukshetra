"""Local BraTS-style research inference with SwinUNETR and SegResNet support."""
import json, time, threading
from src.validation import read_expert
from src.metrics import evaluate_case
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_to_output, resample_from_to
from monai.networks.nets import SegResNet, SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
MODELS = {}
LOCK = threading.Lock()
torch.set_num_threads(4)


def load_model(model_name='swinunetr'):
    global MODELS
    model_name = (model_name or 'ensemble').lower().strip()
    if model_name in MODELS:
        return MODELS[model_name]

    device = 'cuda' if torch.cuda.is_available() else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'

    if model_name in ('ensemble', '8_2', 'swin_resenc'):
        from scripts.models.model_loader import load_ensemble_models, get_device
        ens = load_ensemble_models(device=get_device(device))
        title = f"8:2 Hybrid Ensemble ({ens['desc_a']} + {ens['desc_b']})"
        MODELS['ensemble'] = (ens, title)
        return MODELS['ensemble']

    elif model_name in ('swinunetr', 'swin_unetr', 'swin'):
        weights_path = ROOT / 'models/swinunetr_brats/model_weights_BRATS-finetune.pt'
        if not weights_path.exists():
            weights_path = ROOT / 'work/models/swinunetr_fold1.pt'
        if not weights_path.exists():
            raise FileNotFoundError(f'SwinUNETR weights not found in models/ or work/models/')

        model = SwinUNETR(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=False,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24]
        )
        state = torch.load(weights_path, map_location='cpu', weights_only=False)
        sd = state.get('state_dict', state.get('model', state))
        model.load_state_dict(sd, strict=True)
        model = model.eval().to(device)
        title = 'MONAI SwinUNETR (Vision Transformer) · BraTS21 checkpoint'
        MODELS['swinunetr'] = (model, title)
        return MODELS['swinunetr']

    elif model_name == 'resenc':
        from scripts.models.model_loader import load_resenc
        model, title, _ = load_resenc(device=device)
        MODELS['resenc'] = (model, title)
        return MODELS['resenc']

    elif model_name in ('segresnet', 'resnet'):
        config_path = ROOT / 'models/brats_mri_segmentation/configs/inference.json'
        weights_path = ROOT / 'models/brats_mri_segmentation/models/model.pt'
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError('SegResNet bundle not found in models/brats_mri_segmentation')

        config = json.loads(config_path.read_text())['network_def']
        config.pop('_target_', None)
        model = SegResNet(**config)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state.get('model', state), strict=True)
        model = model.eval().to(device)
        title = 'MONAI SegResNet (CNN Autoencoder) · BraTS Winner'
        MODELS['segresnet'] = (model, title)
        return MODELS['segresnet']

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'ensemble', 'swinunetr' or 'segresnet'.")


def read_image(path):
    img = nib.load(str(path))
    if len(img.shape) != 3 or min(img.shape) < 8 or np.prod(img.shape) > 35_000_000:
        raise ValueError('Each upload must be a 3D MRI volume (not a screenshot or 4D series), under 35 million voxels.')
    units = img.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError('MRI spatial units must be millimeters.')
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3,:3])) < 1e-8:
        raise ValueError('Invalid spatial affine in MRI header.')
    a = img.get_fdata(dtype=np.float32)
    if not np.isfinite(a).all() or not np.any(a):
        raise ValueError('MRI contains invalid numbers or is entirely blank.')
    return img


def mesh(mask, affine, path):
    if not mask.any():
        return False
    # Padding closes surfaces touching the image boundary; apply full affine, not just spacing.
    padded = np.pad(mask.astype(np.uint8), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, 0.5, step_size=2)
        verts = nib.affines.apply_affine(affine, verts - 1)
        with open(path, 'w') as f:
            for v in verts: f.write('v %.4f %.4f %.4f\n' % tuple(v))
            for face in faces + 1: f.write('f %d %d %d\n' % tuple(face))
        return True
    except Exception:
        return False


def region_masks(labels, enhancing=4):
    return {
        'WT': np.isin(labels, [1, 2, enhancing]),
        'TC': np.isin(labels, [1, enhancing]),
        'ET': labels == enhancing
    }


def compute_volume_summary(volumes):
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
        "summary": f"Predicted whole-tumor region: {v_wt:.1f} mL; enhancing region: {v_et:.1f} mL. These are segmentation measurements, not a response assessment."
    }


def run_case(paths, out, update=lambda message: None, expert=None, expert_et=4, case_name='Uploaded MRI', model_name='swinunetr'):
    model_name = {'swin': 'swinunetr', 'swin_unetr': 'swinunetr', 'resnet': 'segresnet', '8_2': 'ensemble', 'swin_resenc': 'ensemble'}.get(model_name, model_name)
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    
    update('Validating MRI dimensions and coordinate alignment')
    # Keep T1c as the output reference grid. The two checkpoints were trained
    # with different channel orders, so arrange channels only after validation.
    imgs = [read_image(paths[k]) for k in ('t1c', 't1', 't2', 'flair')]
    original = imgs[0]
    for img in imgs[1:]:
        if img.shape != original.shape or not np.allclose(img.affine, original.affine, atol=1e-3):
            raise ValueError('The four MRI volumes are not aligned. Use co-registered scans from the same patient. This app does not perform registration or skull stripping.')
            
    expert_img = read_expert(expert, original, expert_et) if expert else None
    if expert_img is not None and (expert_img.shape != original.shape or not np.allclose(expert_img.affine, original.affine, atol=1e-3)):
        raise ValueError('Expert mask must share the original MRI grid and affine.')

    update('Standardizing to 1 mm isotropic RAS grid')
    canonical = nib.as_closest_canonical(original)
    if not np.allclose(nib.affines.voxel_sizes(canonical.affine), [1, 1, 1], atol=1e-3):
        canonical = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1)
    target = (canonical.shape, canonical.affine)
    if np.prod(canonical.shape) > 35_000_000:
        raise ValueError('Resampled MRI is too large for this laptop. Use a prepared BraTS-style case.')

    channels = []
    for img in imgs:
        can = nib.as_closest_canonical(img)
        if can.shape != target[0] or not np.allclose(can.affine, target[1], atol=1e-3):
            can = resample_from_to(can, target, order=1)
        channels.append(can.get_fdata(dtype=np.float32))

    raw_arr = np.stack(channels)
    foreground = np.any(raw_arr != 0, axis=0)
    positions = np.where(foreground)
    slices = tuple(slice(max(0, int(p.min()) - 4), min(raw_arr.shape[i + 1], int(p.max()) + 5)) for i, p in enumerate(positions))
    norm_arr = NormalizeIntensity(nonzero=True, channel_wise=True)(raw_arr).as_tensor()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
    roi_size = (96, 96, 96)
    peak = 0.0

    with LOCK:
        bundle, model_title = load_model(model_name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if model_name in ('ensemble', '8_2', 'swin_resenc'):
            update(f'Segmenting with 8:2 Hybrid Ensemble ({bundle["desc_a"]} + {bundle["desc_b"]})')
            from scripts.ensemble_inference import predict_ensemble
            pred, ent_crop = predict_ensemble(norm_arr[(slice(None),) + slices], bundle, device)
            full_ent = np.zeros(canonical.shape, dtype=np.float32)
            full_ent[slices] = ent_crop
            ent_native = resample_from_to(nib.Nifti1Image(full_ent, canonical.affine), (original.shape, original.affine), order=1)
            ent_header = original.header.copy()
            ent_header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(np.asarray(ent_native.dataobj, dtype=np.float32), original.affine, ent_header), out / 'uncertainty.nii.gz')

        else:
            model = bundle
            model_display = model_title
            update(f'Segmenting with {model_display} ({roi_size[0]}³ sliding patches)')
            mod_indices = {'swinunetr': [3, 0, 1, 2], 'resenc': [3, 1, 0, 2]}.get(model_name, [0, 1, 2, 3])
            x = norm_arr[mod_indices][(slice(None),) + slices].unsqueeze(0).contiguous().to(device)
            
            with torch.inference_mode():
                logits = sliding_window_inference(x, roi_size, 1, model, overlap=0.5, mode='gaussian', sw_device=device, device='cpu')
            pred = (torch.sigmoid(logits[0].float()) >= 0.5).numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    # BraTS channels: 0: TC, 1: WT, 2: ET
    labels = np.where(pred[2], 4, np.where(pred[0], 1, np.where(pred[1], 2, 0))).astype(np.uint8)
    full = np.zeros(canonical.shape, np.uint8)
    full[slices] = labels

    update('Projecting predictions back to native patient space')
    seg = resample_from_to(nib.Nifti1Image(full, canonical.affine), (original.shape, original.affine), order=0)
    seg_data = np.asarray(seg.dataobj, dtype=np.uint8)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)

    nib.save(nib.Nifti1Image(seg_data, original.affine, header), out / 'segmentation.nii.gz')
    nib.save(imgs[3], out / 'flair.nii.gz')
    nib.save(original, out / 't1c.nii.gz')

    regions = region_masks(seg_data)
    voxel_ml = abs(float(np.linalg.det(original.affine[:3, :3]))) / 1000
    volumes = {k: round(float(m.sum()) * voxel_ml, 3) for k, m in regions.items()}
    volumes['ED'] = round(float(np.count_nonzero(seg_data == 2)) * voxel_ml, 3)

    metrics = None
    if expert_img is not None:
        gt = expert_img.get_fdata()
        if not np.isin(gt, [0, 1, 2, expert_et]).all():
            raise ValueError('Expert mask labels do not match the selected convention (0,1,2,3 or 0,1,2,4).')
        metrics = evaluate_case(seg_data, gt, original.affine, expert_et)
        normalized = np.where(gt == expert_et, 4, gt).astype(np.uint8)
        nib.save(nib.Nifti1Image(normalized, original.affine, header), out / 'expert.nii.gz')

    update('Synthesizing 3D polygonal meshes and measurement report')
    meshes = []
    # 1. Brain shell
    if mesh(imgs[3].get_fdata() != 0, original.affine, out / 'brain.obj'):
        meshes.append('brain')
    # 2. Whole tumor surface
    if mesh(regions['WT'], original.affine, out / 'tumor.obj'):
        meshes.append('tumor')
    # 3. Tumor Core surface
    if mesh(regions['TC'], original.affine, out / 'core.obj'):
        meshes.append('core')

    volume_summary = compute_volume_summary(volumes)

    result = {
        'case_name': case_name,
        'model_name': model_name,
        'model': model_title,
        'input_channel_order': {k: bundle['config'][k]['channel_order'] for k in ('model_a', 'model_b')} if model_name == 'ensemble' else ({'swinunetr': ['FLAIR', 'T1c', 'T1', 'T2'], 'resenc': ['FLAIR', 'T1', 'T1c', 'T2']}.get(model_name, ['T1c', 'T1', 'T2', 'FLAIR'])),
        'ensemble_settings': bundle['config'] if model_name == 'ensemble' else None,
        'sliding_window_roi': list(roi_size),
        'volumes_ml': volumes,
        'volume_summary': volume_summary,
        'metrics': metrics,
        'runtime_seconds': round(time.perf_counter() - started, 2),
        'peak_gpu_mib': peak,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
        'shape': list(original.shape),
        'spacing_mm': nib.affines.voxel_sizes(original.affine).round(4).tolist(),
        'meshes': meshes,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        'note': 'Research demonstration, not intended for primary clinical diagnosis. Voxel metrics computed via exact spatial affine determinant.'
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    return result
