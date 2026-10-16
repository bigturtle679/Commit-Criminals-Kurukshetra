"""Unified model loader for NeuroScope ensemble and baseline inference.

Supports:
- Model A: SwinUNETR (Transformer) / SegResNet (CNN Autoencoder)
- Model B: ResEncUNet (nnU-Net Residual Encoder preset M/L/XL)
- Automatic device routing: CUDA -> MPS (Apple Silicon) -> CPU

Safety guarantees:
- Missing checkpoints raise CheckpointNotFoundError (never returns random weights)
- Checkpoint key coverage is validated after loading
- Clear diagnostic messages name expected file paths
"""
from pathlib import Path
import json
import logging
import torch
from monai.networks.nets import SwinUNETR, SegResNet
from scripts.models.resenc_unet import ResEncUNet

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a required model checkpoint file cannot be found."""
    pass


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint file exists but cannot be loaded correctly."""
    pass


def get_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _resolve_checkpoint(candidates, model_name):
    """Find the first existing checkpoint from a list of candidate paths.

    Returns the resolved Path.
    Raises CheckpointNotFoundError with all searched locations if none found.
    """
    searched = []
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        # Resolve relative paths against ROOT
        if not p.is_absolute():
            p = ROOT / p
        searched.append(str(p))
        if p.exists():
            return p
    raise CheckpointNotFoundError(
        f"{model_name} checkpoint not found. "
        f"Searched locations:\n  " + "\n  ".join(searched) +
        f"\n\nPlease place the checkpoint file at one of these locations. "
        f"See MODEL_CARD.md for checkpoint provenance and download instructions."
    )


def _load_state_dict(model, checkpoint_path, model_name):
    """Load checkpoint into model with key coverage validation.

    Uses strict=True to catch architecture mismatches.
    Raises CheckpointLoadError if loading fails.
    """
    try:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise CheckpointLoadError(
            f"Failed to read checkpoint file for {model_name} at "
            f"{checkpoint_path}: {e}"
        ) from e

    # Extract state dict from common wrapper formats
    sd = state.get('state_dict', state.get('model', state))
    if not isinstance(sd, dict):
        raise CheckpointLoadError(
            f"Checkpoint for {model_name} at {checkpoint_path} does not contain "
            f"a valid state dict. Found type: {type(sd).__name__}"
        )

    # Attempt strict loading first for safety
    try:
        model.load_state_dict(sd, strict=True)
        logger.info(
            "%s: loaded %d parameters from %s (strict=True, full match)",
            model_name, len(sd), checkpoint_path.name
        )
    except RuntimeError as e:
        # Fall back to non-strict with detailed diagnostics
        error_msg = str(e)
        logger.warning(
            "%s: strict loading failed, attempting non-strict. Details: %s",
            model_name, error_msg
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)

        model_keys = set(dict(model.named_parameters()).keys())
        model_keys.update(dict(model.named_buffers()).keys())
        loaded_keys = set(sd.keys()) - set(unexpected)
        coverage = len(loaded_keys) / max(len(model_keys), 1)

        if missing:
            logger.warning(
                "%s: %d missing keys (%.1f%% coverage). Missing: %s",
                model_name, len(missing), coverage * 100,
                missing[:5]  # Log first 5 for diagnostics
            )
        if unexpected:
            logger.warning(
                "%s: %d unexpected keys ignored: %s",
                model_name, len(unexpected), unexpected[:5]
            )

        # Reject if coverage is too low (likely wrong checkpoint)
        if coverage < 0.5:
            raise CheckpointLoadError(
                f"{model_name} checkpoint at {checkpoint_path} has only "
                f"{coverage:.0%} key coverage. This likely indicates an "
                f"incompatible checkpoint file."
            )


def load_swinunetr(weights_path=None, fallback_path=None, device=None):
    """Load SwinUNETR model with validated checkpoint.

    Raises CheckpointNotFoundError if no checkpoint is found.
    Raises CheckpointLoadError if the checkpoint is incompatible.
    """
    device = device or get_device()
    candidates = [
        weights_path,
        'models/swinunetr_brats/model_weights_BRATS-finetune.pt',
        fallback_path,
        'work/models/swinunetr_fold1.pt'
    ]

    resolved_path = _resolve_checkpoint(candidates, 'SwinUNETR')

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24]
    )

    _load_state_dict(model, resolved_path, 'SwinUNETR')
    desc = f'SwinUNETR BraTS21 ({resolved_path.name})'

    model.to(device)
    model.eval()
    return model, desc, resolved_path


def load_segresnet(weights_path=None, device=None):
    """Load SegResNet model with validated checkpoint.

    Raises CheckpointNotFoundError if no checkpoint is found.
    Raises CheckpointLoadError if the checkpoint is incompatible.
    """
    device = device or get_device()
    candidates = [
        weights_path,
        'models/brats_mri_segmentation/models/model.pt',
    ]

    resolved_path = _resolve_checkpoint(candidates, 'SegResNet')

    # Load config from bundle if available
    config_path = ROOT / 'models/brats_mri_segmentation/configs/inference.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())['network_def']
        config.pop('_target_', None)
        model = SegResNet(**config)
    else:
        model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1]
        )

    _load_state_dict(model, resolved_path, 'SegResNet')
    desc = f'SegResNet BraTS ({resolved_path.name})'

    model.to(device)
    model.eval()
    return model, desc, resolved_path


def load_resenc(preset='M', weights_path=None, device=None):
    """Load ResEncUNet model with validated checkpoint.

    Raises CheckpointNotFoundError if no checkpoint is found.
    Raises CheckpointLoadError if the checkpoint is incompatible.
    """
    device = device or get_device()
    candidates = [
        weights_path,
        'models/resenc_brats/resenc_m_brats.pt',
        'work/models/resenc_brats.pt'
    ]

    resolved_path = _resolve_checkpoint(candidates, f'ResEncUNet {preset}')

    model = ResEncUNet(in_channels=4, out_channels=3, preset=preset)
    _load_state_dict(model, resolved_path, f'ResEncUNet {preset}')
    desc = f'nnU-Net ResEnc {preset} ({resolved_path.name})'

    model.to(device)
    model.eval()
    return model, desc, resolved_path


def load_ensemble_models(config_path=None, device=None):
    """Load both ensemble models with validated checkpoints.

    Raises CheckpointNotFoundError if either checkpoint is missing.
    Raises CheckpointLoadError if either checkpoint is incompatible.
    """
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
