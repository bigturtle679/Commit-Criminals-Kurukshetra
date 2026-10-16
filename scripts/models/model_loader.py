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

# Register safe globals for PyTorch 2.6+ compatibility with checkpoints containing NumPy scalar types
try:
    import numpy as np
    if hasattr(torch.serialization, 'add_safe_globals'):
        _safe = []
        if hasattr(np, '_core') and hasattr(np._core, 'multiarray'):
            _safe.append((np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'))
        elif hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
            _safe.append((np.core.multiarray.scalar, 'numpy.core.multiarray.scalar'))
        if hasattr(np, 'dtype'):
            _safe.append((np.dtype, 'numpy.dtype'))
        if hasattr(np, 'dtypes') and hasattr(np.dtypes, 'Float32DType'):
            _safe.append(np.dtypes.Float32DType)
        if _safe:
            torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

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
    Raises CheckpointNotFoundError with searched locations if none found.
    """
    searched = []
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        # Resolve relative paths against ROOT
        if not p.is_absolute():
            p = ROOT / p
        try:
            rel = p.relative_to(ROOT)
            searched.append(str(rel))
        except ValueError:
            searched.append(p.name)
        if p.exists():
            return p
    raise CheckpointNotFoundError(
        f"{model_name} checkpoint not found. "
        f"Searched locations:\n  " + "\n  ".join(searched) +
        f"\n\nPlease place the checkpoint file at one of these locations. "
        f"See MODEL_CARD.md for checkpoint provenance and download instructions."
    )


def _load_state_dict(model, checkpoint_path, model_name, allow_unsafe_legacy_weights: bool = False):
    """Load checkpoint into model with strict validation.

    Safety:
    1. Loads using weights_only=True by default for secure deserialization.
       Never automatically falls back to unsafe weights_only=False.
       An explicit, opt-in allow_unsafe_legacy_weights=True is required for legacy files.
    2. Try strict=True with the raw state dict.
    3. If that fails, try known prefix normalization (e.g. 'module.' from
       DataParallel) and retry strict=True.
    4. If both fail, reject the checkpoint entirely.

    Never returns partially loaded models.
    Raises CheckpointLoadError if loading fails for any reason.
    """
    try:
        if not allow_unsafe_legacy_weights:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        else:
            try:
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception as e:
                logger.warning(
                    "%s: Explicit opt-in allow_unsafe_legacy_weights=True used after weights_only=True failed: %s",
                    model_name, e
                )
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise CheckpointLoadError(
            f"Failed to read checkpoint file for {model_name} (weights_only=True): {e}"
        ) from e

    # Extract state dict from common wrapper formats
    sd = state.get('state_dict', state.get('model', state)) if isinstance(state, dict) else state
    if not isinstance(sd, dict):
        raise CheckpointLoadError(
            f"Checkpoint for {model_name} does not contain "
            f"a valid state dict. Found type: {type(sd).__name__}"
        )

    # Validate tensor shapes against model parameters
    model_sd = model.state_dict()
    _validate_tensor_shapes(sd, model_sd, model_name, checkpoint_path)

    # Attempt 1: strict loading with raw keys
    try:
        model.load_state_dict(sd, strict=True)
        logger.info(
            "%s: loaded %d parameters from %s (strict match)",
            model_name, len(sd), checkpoint_path.name
        )
        return
    except RuntimeError as e:
        first_error = str(e)

    # Attempt 2: strip known wrapper prefixes and retry strict
    stripped_sd = _strip_known_prefixes(sd)
    if stripped_sd is not sd:
        try:
            _validate_tensor_shapes(stripped_sd, model_sd, model_name, checkpoint_path)
            model.load_state_dict(stripped_sd, strict=True)
            logger.info(
                "%s: loaded %d parameters from %s (after prefix normalization, strict match)",
                model_name, len(stripped_sd), checkpoint_path.name
            )
            return
        except RuntimeError:
            pass

    # Both attempts failed — produce a clear diagnostic
    model_keys = set(model_sd.keys())
    ckpt_keys = set(sd.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    diagnostic_parts = [
        f"{model_name} checkpoint is incompatible with the model architecture."
    ]
    if missing:
        diagnostic_parts.append(
            f"  Missing keys ({len(missing)}): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    if unexpected:
        diagnostic_parts.append(
            f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
            + (" ..." if len(unexpected) > 5 else "")
        )
    diagnostic_parts.append(f"  Original error: {first_error[:200]}")

    raise CheckpointLoadError("\n".join(diagnostic_parts))


def _strip_known_prefixes(sd: dict) -> dict:
    """Strip known wrapper prefixes (e.g. 'module.' from DataParallel).

    Returns a new dict if any prefix was stripped, otherwise returns
    the original dict unchanged.
    """
    KNOWN_PREFIXES = ('module.',)
    for prefix in KNOWN_PREFIXES:
        if all(k.startswith(prefix) for k in sd.keys()):
            logger.info("Stripping '%s' prefix from checkpoint keys.", prefix)
            return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _validate_tensor_shapes(sd: dict, model_sd: dict, model_name: str, checkpoint_path: Path):
    """Validate that checkpoint tensor shapes match model expectations.

    Only checks keys that exist in both dicts (missing/unexpected keys
    are caught by strict=True loading).
    """
    shape_mismatches = []
    common_keys = set(sd.keys()) & set(model_sd.keys())
    for key in sorted(common_keys):
        ckpt_val = sd[key]
        model_val = model_sd[key]
        if hasattr(ckpt_val, 'shape') and hasattr(model_val, 'shape'):
            if ckpt_val.shape != model_val.shape:
                shape_mismatches.append(
                    f"  {key}: checkpoint={list(ckpt_val.shape)}, "
                    f"model={list(model_val.shape)}"
                )
    if shape_mismatches:
        raise CheckpointLoadError(
            f"{model_name} checkpoint has tensor shape mismatches:\n"
            + "\n".join(shape_mismatches[:10])
            + (f"\n  ... and {len(shape_mismatches) - 10} more"
               if len(shape_mismatches) > 10 else "")
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


def load_ensemble_models(config_path=None, validated_config=None, device=None):
    """Load both ensemble models with validated checkpoints.

    Uses configs/config_loader.py to guarantee authoritative validation.
    Raises CheckpointNotFoundError if either checkpoint is missing.
    Raises CheckpointLoadError if either checkpoint is incompatible.
    """
    device = device or get_device()
    if validated_config is not None:
        cfg = validated_config
    else:
        from configs.config_loader import load_ensemble_config
        cfg = load_ensemble_config(config_path)

    model_a_cfg = cfg['model_a']
    model_b_cfg = cfg['model_b']

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

    return {
        'model_a': model_a,
        'model_b': model_b,
        'desc_a': desc_a,
        'desc_b': desc_b,
        'path_a': path_a,
        'path_b': path_b,
        'weight_a': cfg['weight_a'],
        'weight_b': cfg['weight_b'],
        'config': cfg,
        'device': device
    }
