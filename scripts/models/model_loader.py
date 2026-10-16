"""Unified model loader for NeuroScope ensemble and baseline inference.

Supports:
- Model A: SwinUNETR (Transformer) / SegResNet (CNN Autoencoder)
- Model B: ResEncUNet (nnU-Net Residual Encoder preset M/L/XL)
- Automatic device routing: CUDA -> MPS (Apple Silicon) -> CPU
"""
from pathlib import Path
import json
import torch
from monai.networks.nets import SwinUNETR, SegResNet
from scripts.models.resenc_unet import ResEncUNet

ROOT = Path(__file__).resolve().parents[2]


def get_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_swinunetr(weights_path=None, fallback_path=None, device=None):
    device = device or get_device()
    candidates = [
        Path(weights_path) if weights_path else None,
        ROOT / 'models/swinunetr_brats/model_weights_BRATS-finetune.pt',
        Path(fallback_path) if fallback_path else None,
        ROOT / 'work/models/swinunetr_fold1.pt'
    ]
    resolved_path = None
    for p in candidates:
        if p and p.exists():
            resolved_path = p
            break

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24]
    )

    if resolved_path:
        state = torch.load(resolved_path, map_location='cpu', weights_only=False)
        sd = state.get('state_dict', state.get('model', state))
        model.load_state_dict(sd, strict=False)
        desc = f'SwinUNETR BraTS21 ({resolved_path.name})'
    else:
        desc = 'SwinUNETR BraTS21 (uninitialized weights)'

    model.to(device)
    model.eval()
    return model, desc, resolved_path


def load_resenc(preset='M', weights_path=None, device=None):
    device = device or get_device()
    candidates = [
        Path(weights_path) if weights_path else None,
        ROOT / 'models/resenc_brats/resenc_m_brats.pt',
        ROOT / 'work/models/resenc_brats.pt'
    ]
    resolved_path = None
    for p in candidates:
        if p and p.exists():
            resolved_path = p
            break

    model = ResEncUNet(in_channels=4, out_channels=3, preset=preset)
    if resolved_path:
        state = torch.load(resolved_path, map_location='cpu', weights_only=False)
        sd = state.get('state_dict', state.get('model', state))
        model.load_state_dict(sd, strict=False)
        desc = f'nnU-Net ResEnc {preset} ({resolved_path.name})'
    else:
        desc = f'nnU-Net ResEnc {preset} (initialized)'

    model.to(device)
    model.eval()
    return model, desc, resolved_path


def load_ensemble_models(config_path=None, device=None):
    device = device or get_device()
    if config_path is None:
        config_path = ROOT / 'configs/ensemble.json'
    cfg = json.loads(Path(config_path).read_text())

    model_a_cfg = cfg.get('model_a', {})
    model_b_cfg = cfg.get('model_b', {})

    model_a, desc_a, path_a = load_swinunetr(
        weights_path=model_a_cfg.get('weights_path'),
        fallback_path=model_a_cfg.get('fallback_weights_path'),
        device=device
    )

    model_b, desc_b, path_b = load_resenc(
        preset=model_b_cfg.get('preset', 'M'),
        weights_path=model_b_cfg.get('weights_path'),
        device=device
    )

    weights = cfg.get('weights', {'model_a': 0.8, 'model_b': 0.2})
    return {
        'model_a': model_a,
        'model_b': model_b,
        'desc_a': desc_a,
        'desc_b': desc_b,
        'path_a': path_a,
        'path_b': path_b,
        'weight_a': float(weights['model_a']),
        'weight_b': float(weights['model_b']),
        'config': cfg,
        'device': device
    }
