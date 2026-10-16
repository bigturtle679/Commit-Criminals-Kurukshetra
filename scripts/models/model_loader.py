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
    Raises CheckpointNotFoundError with searched locations if none found.
    """
    explicit = candidates[0] if candidates else None
    if explicit is not None:
        ep = Path(explicit)
        ep = ep if ep.is_absolute() else ROOT / ep
        if not ep.exists():
            # An explicitly requested checkpoint must never be silently swapped for a default.
            raise CheckpointNotFoundError(f"{model_name} checkpoint not found at requested path: {ep}")
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


def resolve_weights(explicit, defaults):
    """Return the requested checkpoint, or the first existing default; never random weights."""
    return _resolve_checkpoint([explicit] + list(defaults) if explicit else list(defaults), 'Model')


def checkpoint_hash(path):
    """SHA-256 of a checkpoint file, recorded in benchmark metadata."""
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _numpy_safe_globals():
    """NumPy scalar types found in common research checkpoints (e.g. a stored best_acc).

    Allowlisting these keeps weights_only=True (no arbitrary code execution) while
    accepting checkpoints such as the BrainSegFounder SwinUNETR file.
    """
    import numpy as np
    core = getattr(np, '_core', None) or np.core
    scalar = core.multiarray.scalar
    # Checkpoints saved under NumPy 1.x reference the pre-2.0 module path.
    return [scalar, (scalar, 'numpy.core.multiarray.scalar'), np.dtype,
            type(np.dtype('float64')), type(np.dtype('float32'))]


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
            with torch.serialization.safe_globals(_numpy_safe_globals()):
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        else:
            try:
                with torch.serialization.safe_globals(_numpy_safe_globals()):
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


SWINUNETR_HF_URL = "https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-finetune.pt"
SWINUNETR_DEFAULT_PATH = ROOT / "models/swinunetr_brats/model_weights_BRATS-finetune.pt"


def ensure_swinunetr_checkpoint(target_path=None, progress_callback=None) -> Path:
    """Ensure SwinUNETR checkpoint exists locally; downloads from Hugging Face if missing."""
    target = Path(target_path) if target_path else SWINUNETR_DEFAULT_PATH
    if target.is_file() and target.stat().st_size > 100_000_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix('.pt.download')
    logger.info("SwinUNETR checkpoint missing locally. Auto-downloading from Hugging Face (%s)...", SWINUNETR_HF_URL)
    if progress_callback:
        progress_callback("Connecting to Hugging Face to download SwinUNETR weights (256 MB)...")

    import urllib.request
    req = urllib.request.Request(
        SWINUNETR_HF_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeuroScope/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp, open(temp_target, "wb") as f:
            total = int(resp.headers.get("Content-Length", 256365127))
            downloaded = 0
            last_pct = -1
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_callback:
                    pct = int(downloaded / total * 100)
                    if pct != last_pct and pct % 5 == 0:
                        last_pct = pct
                        mb_cur = downloaded // (1024 * 1024)
                        mb_tot = total // (1024 * 1024)
                        progress_callback(f"Downloading SwinUNETR weights: {pct}% ({mb_cur}/{mb_tot} MB)...")

        temp_target.rename(target)
        logger.info("Successfully downloaded SwinUNETR checkpoint to %s", target)
        return target
    except Exception as e:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
        logger.error("Failed to auto-download SwinUNETR checkpoint: %s", e)
        err_msg = (
            f'SwinUNETR checkpoint not found and auto-download failed: {e}\n\n'
            f'Please run python scripts/download_weights.py or download_weights.bat on Windows\n'
            f'or download manually from {SWINUNETR_HF_URL}\n'
            f'and place it at {target}'
        )
        raise CheckpointNotFoundError(err_msg) from e


def load_swinunetr(weights_path=None, fallback_path=None, device=None, progress_callback=None):
    """Load SwinUNETR model with validated checkpoint.

    Automatically retrieves official weights from Hugging Face if default weights are missing.
    Raises CheckpointNotFoundError if no checkpoint is found and auto-download fails.
    Raises CheckpointLoadError if the checkpoint is incompatible.
    """
    device = device or get_device()
    candidates = [
        weights_path,
        'models/swinunetr_brats/model_weights_BRATS-finetune.pt',
        fallback_path,
        'work/models/swinunetr_fold1.pt'
    ]

    try:
        resolved_path = _resolve_checkpoint(candidates, 'SwinUNETR')
    except CheckpointNotFoundError:
        # If user did not provide a custom explicit path, auto-download default weights
        norm_explicit = str(weights_path).replace(chr(92), '/') if weights_path else ''
        if weights_path is None or norm_explicit.endswith('model_weights_BRATS-finetune.pt'):
            resolved_path = ensure_swinunetr_checkpoint(progress_callback=progress_callback)
        else:
            raise

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


def load_ensemble_models(config_path=None, validated_config=None, device=None, progress_callback=None):
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
        device=device,
        progress_callback=progress_callback
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
