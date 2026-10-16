"""Centralized ensemble configuration loader and validator.

Makes configs/ensemble.json the single authoritative source for:
- Model identifiers and checkpoint paths
- Ensemble weights and fusion mode
- Per-model channel ordering
- ROI dimensions, sliding-window overlap, probability threshold
- Postprocessing settings

All production inference paths MUST use load_ensemble_config() to obtain
parameters. No hardcoded ROI, overlap, threshold, or channel-order values
should exist in the inference path.
"""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / 'configs' / 'ensemble.json'

# Valid modality identifiers recognized by the pipeline
VALID_MODALITIES = {'flair', 't1', 't1c', 't2'}


class ConfigValidationError(ValueError):
    """Raised when ensemble configuration fails validation."""
    pass


def _check_field(cfg: dict, field: str, expected_type: type, context: str = ''):
    """Check that a required field exists and has the expected type."""
    prefix = f"{context}." if context else ""
    if field not in cfg:
        raise ConfigValidationError(
            f"Missing required field '{prefix}{field}' in ensemble configuration."
        )
    value = cfg[field]
    if not isinstance(value, expected_type):
        raise ConfigValidationError(
            f"Field '{prefix}{field}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _validate_model_config(model_cfg: dict, model_key: str) -> dict:
    """Validate a single model's configuration block."""
    ctx = model_key

    name = _check_field(model_cfg, 'name', str, ctx)
    arch = _check_field(model_cfg, 'architecture', str, ctx)
    weights_path = _check_field(model_cfg, 'weights_path', str, ctx)
    in_ch = _check_field(model_cfg, 'in_channels', int, ctx)
    out_ch = _check_field(model_cfg, 'out_channels', int, ctx)

    if in_ch < 1 or in_ch > 16:
        raise ConfigValidationError(
            f"{ctx}.in_channels must be between 1 and 16, got {in_ch}"
        )
    if out_ch < 1 or out_ch > 16:
        raise ConfigValidationError(
            f"{ctx}.out_channels must be between 1 and 16, got {out_ch}"
        )

    # ROI size validation
    roi = _check_field(model_cfg, 'roi_size', list, ctx)
    if len(roi) != 3:
        raise ConfigValidationError(
            f"{ctx}.roi_size must have exactly 3 elements, got {len(roi)}"
        )
    for i, dim in enumerate(roi):
        if not isinstance(dim, int) or dim < 8 or dim > 512:
            raise ConfigValidationError(
                f"{ctx}.roi_size[{i}] must be an integer between 8 and 512, got {dim}"
            )

    # Channel order validation
    channel_order = _check_field(model_cfg, 'channel_order', list, ctx)
    if len(channel_order) != in_ch:
        raise ConfigValidationError(
            f"{ctx}.channel_order must have {in_ch} entries (matching in_channels), "
            f"got {len(channel_order)}"
        )
    for ch in channel_order:
        if ch not in VALID_MODALITIES:
            raise ConfigValidationError(
                f"{ctx}.channel_order contains invalid modality '{ch}'. "
                f"Valid modalities: {sorted(VALID_MODALITIES)}"
            )
    if len(set(channel_order)) != len(channel_order):
        raise ConfigValidationError(
            f"{ctx}.channel_order has duplicate entries: {channel_order}"
        )

    # Sliding window overlap
    overlap = _check_field(model_cfg, 'sliding_window_overlap', (int, float), ctx)
    if not (0.0 <= overlap < 1.0):
        raise ConfigValidationError(
            f"{ctx}.sliding_window_overlap must be in [0.0, 1.0), got {overlap}"
        )

    return {
        'name': name,
        'architecture': arch,
        'weights_path': weights_path,
        'fallback_weights_path': model_cfg.get('fallback_weights_path'),
        'in_channels': in_ch,
        'out_channels': out_ch,
        'roi_size': tuple(roi),
        'channel_order': channel_order,
        'sliding_window_overlap': float(overlap),
        'mode': model_cfg.get('mode', 'gaussian'),
        'feature_size': model_cfg.get('feature_size'),
        'preset': model_cfg.get('preset'),
        'channels': model_cfg.get('channels'),
        'ratio': model_cfg.get('ratio'),
    }


def _validate_weights(cfg: dict) -> tuple[float, float]:
    """Validate and normalize ensemble weights."""
    weights = _check_field(cfg, 'weights', dict)
    w_a = weights.get('model_a')
    w_b = weights.get('model_b')

    if w_a is None or w_b is None:
        raise ConfigValidationError(
            "weights must contain 'model_a' and 'model_b' entries."
        )

    w_a = float(w_a)
    w_b = float(w_b)

    import math
    if not (math.isfinite(w_a) and math.isfinite(w_b)):
        raise ConfigValidationError(
            f"Ensemble weights must be finite numbers, got model_a={w_a}, model_b={w_b}"
        )
    if w_a < 0 or w_b < 0:
        raise ConfigValidationError(
            f"Ensemble weights must be nonnegative, got model_a={w_a}, model_b={w_b}"
        )

    total = w_a + w_b
    if total <= 0:
        raise ConfigValidationError("Ensemble weights must sum to a positive value.")

    # Normalize to sum to 1.0
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        logger.warning(
            "Ensemble weights (%.4f, %.4f) do not sum to 1.0. "
            "Normalizing to (%.4f, %.4f).",
            w_a, w_b, w_a / total, w_b / total
        )
        w_a /= total
        w_b /= total

    return w_a, w_b


def _validate_postprocessing(cfg: dict) -> dict:
    """Validate postprocessing configuration."""
    pp = cfg.get('postprocessing', {})
    if not isinstance(pp, dict):
        raise ConfigValidationError("postprocessing must be a dict.")

    threshold = pp.get('threshold', 0.5)
    if not isinstance(threshold, (int, float)):
        raise ConfigValidationError(
            f"postprocessing.threshold must be numeric, got {type(threshold).__name__}"
        )
    threshold = float(threshold)
    if not (0.0 < threshold < 1.0):
        raise ConfigValidationError(
            f"postprocessing.threshold must be in (0.0, 1.0), got {threshold}"
        )

    enforce_hierarchy = pp.get('enforce_hierarchy', True)
    if not isinstance(enforce_hierarchy, bool):
        raise ConfigValidationError("postprocessing.enforce_hierarchy must be boolean.")

    min_et = pp.get('min_et_volume_voxels', 0)
    if not isinstance(min_et, int) or min_et < 0:
        raise ConfigValidationError(
            f"postprocessing.min_et_volume_voxels must be a non-negative integer, got {min_et}"
        )

    export_uncertainty = pp.get('export_uncertainty_map', True)
    if not isinstance(export_uncertainty, bool):
        raise ConfigValidationError("postprocessing.export_uncertainty_map must be boolean.")

    return {
        'threshold': threshold,
        'enforce_hierarchy': enforce_hierarchy,
        'min_et_volume_voxels': min_et,
        'export_uncertainty_map': export_uncertainty,
    }


def load_ensemble_config(config_path: str | Path | None = None) -> dict:
    """Load and fully validate the ensemble configuration.

    Returns a validated configuration dict with normalized values.
    Raises ConfigValidationError if any field is invalid or missing.
    Raises FileNotFoundError if the config file does not exist.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Ensemble configuration not found at {config_path}. "
            f"Expected configs/ensemble.json in the project root."
        )

    try:
        raw = json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise ConfigValidationError(
            f"Ensemble configuration is not valid JSON: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise ConfigValidationError("Ensemble configuration must be a JSON object.")

    # Validate top-level fields
    ensemble_name = raw.get('ensemble_name', 'unnamed_ensemble')
    fusion_mode = raw.get('fusion_mode', 'probability')
    if fusion_mode not in ('probability', 'logit'):
        raise ConfigValidationError(
            f"fusion_mode must be 'probability' or 'logit', got '{fusion_mode}'"
        )

    # Validate model configs
    if 'model_a' not in raw:
        raise ConfigValidationError("Missing required 'model_a' configuration block.")
    if 'model_b' not in raw:
        raise ConfigValidationError("Missing required 'model_b' configuration block.")

    model_a = _validate_model_config(raw['model_a'], 'model_a')
    model_b = _validate_model_config(raw['model_b'], 'model_b')

    # Validate weights
    w_a, w_b = _validate_weights(raw)

    # Validate postprocessing
    postprocessing = _validate_postprocessing(raw)

    validated = {
        'ensemble_name': ensemble_name,
        'fusion_mode': fusion_mode,
        'weight_a': w_a,
        'weight_b': w_b,
        'model_a': model_a,
        'model_b': model_b,
        'postprocessing': postprocessing,
        # Preserve raw config for report provenance
        '_raw': raw,
        '_config_path': str(config_path),
    }

    logger.info(
        "Loaded ensemble config '%s': %s (%.0f%%) + %s (%.0f%%), "
        "threshold=%.2f, roi=%s",
        ensemble_name,
        model_a['name'], w_a * 100,
        model_b['name'], w_b * 100,
        postprocessing['threshold'],
        model_a['roi_size'],
    )

    return validated
