"""Tests for the centralized configuration loader (configs/config_loader.py).

Validates:
- Happy path: valid ensemble.json loads successfully
- Missing fields: each required field triggers ConfigValidationError
- Invalid types: wrong types for weights, roi_size, etc.
- Weight normalization: non-unity weights are normalized
- Channel order validation: duplicate, unknown, and mismatched entries
- Postprocessing bounds: threshold out of (0, 1) range
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.config_loader import (
    load_ensemble_config,
    ConfigValidationError,
)


def _write_config(tmp_path, cfg_dict):
    """Write a config dict to a temporary JSON file and return its path."""
    p = tmp_path / 'test_ensemble.json'
    p.write_text(json.dumps(cfg_dict))
    return p


def _valid_config():
    """Return a minimal valid configuration dict."""
    return {
        "ensemble_name": "test_ensemble",
        "fusion_mode": "probability",
        "weights": {"model_a": 0.8, "model_b": 0.2},
        "model_a": {
            "name": "TestSwin",
            "architecture": "SwinUNETR",
            "weights_path": "models/test_a.pt",
            "in_channels": 4,
            "out_channels": 3,
            "roi_size": [96, 96, 96],
            "channel_order": ["flair", "t1c", "t1", "t2"],
            "sliding_window_overlap": 0.5,
            "mode": "gaussian",
            "feature_size": 48,
        },
        "model_b": {
            "name": "TestResEnc",
            "architecture": "ResEncUNet",
            "weights_path": "models/test_b.pt",
            "in_channels": 4,
            "out_channels": 3,
            "roi_size": [96, 96, 96],
            "channel_order": ["flair", "t1", "t1c", "t2"],
            "sliding_window_overlap": 0.5,
            "mode": "gaussian",
            "preset": "M",
        },
        "postprocessing": {
            "threshold": 0.5,
            "enforce_hierarchy": True,
            "min_et_volume_voxels": 16,
            "export_uncertainty_map": True,
        },
    }


class TestValidConfig:
    def test_loads_valid_config(self, tmp_path):
        cfg = _valid_config()
        path = _write_config(tmp_path, cfg)
        result = load_ensemble_config(path)

        assert result['ensemble_name'] == 'test_ensemble'
        assert result['weight_a'] == 0.8
        assert result['weight_b'] == 0.2
        assert result['model_a']['roi_size'] == (96, 96, 96)
        assert result['model_a']['channel_order'] == ['flair', 't1c', 't1', 't2']
        assert result['postprocessing']['threshold'] == 0.5

    def test_default_config_path_exists(self):
        """Verify configs/ensemble.json exists and is loadable."""
        default_path = ROOT / 'configs' / 'ensemble.json'
        assert default_path.exists(), "configs/ensemble.json must exist"
        result = load_ensemble_config(default_path)
        assert result['weight_a'] == 0.8
        assert result['weight_b'] == 0.2


class TestMissingFields:
    def test_missing_model_a(self, tmp_path):
        cfg = _valid_config()
        del cfg['model_a']
        with pytest.raises(ConfigValidationError, match='model_a'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_missing_model_b(self, tmp_path):
        cfg = _valid_config()
        del cfg['model_b']
        with pytest.raises(ConfigValidationError, match='model_b'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_missing_weights(self, tmp_path):
        cfg = _valid_config()
        del cfg['weights']
        with pytest.raises(ConfigValidationError, match='weights'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_missing_roi_size(self, tmp_path):
        cfg = _valid_config()
        del cfg['model_a']['roi_size']
        with pytest.raises(ConfigValidationError, match='roi_size'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_missing_channel_order(self, tmp_path):
        cfg = _valid_config()
        del cfg['model_b']['channel_order']
        with pytest.raises(ConfigValidationError, match='channel_order'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_missing_weights_path(self, tmp_path):
        cfg = _valid_config()
        del cfg['model_a']['weights_path']
        with pytest.raises(ConfigValidationError, match='weights_path'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestInvalidTypes:
    def test_weights_not_a_dict(self, tmp_path):
        cfg = _valid_config()
        cfg['weights'] = [0.8, 0.2]
        with pytest.raises(ConfigValidationError, match='dict'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_roi_size_wrong_length(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['roi_size'] = [96, 96]
        with pytest.raises(ConfigValidationError, match='3 elements'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_roi_size_too_small(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['roi_size'] = [4, 4, 4]
        with pytest.raises(ConfigValidationError, match='between 8 and 512'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_in_channels_zero(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['in_channels'] = 0
        with pytest.raises(ConfigValidationError, match='between 1 and 16'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestWeightNormalization:
    def test_weights_sum_to_one(self, tmp_path):
        cfg = _valid_config()
        cfg['weights'] = {'model_a': 0.8, 'model_b': 0.2}
        result = load_ensemble_config(_write_config(tmp_path, cfg))
        assert abs(result['weight_a'] + result['weight_b'] - 1.0) < 1e-6

    def test_weights_normalized_if_not_unity(self, tmp_path):
        cfg = _valid_config()
        cfg['weights'] = {'model_a': 4.0, 'model_b': 1.0}
        result = load_ensemble_config(_write_config(tmp_path, cfg))
        assert abs(result['weight_a'] - 0.8) < 1e-6
        assert abs(result['weight_b'] - 0.2) < 1e-6

    def test_negative_weights_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['weights'] = {'model_a': -0.5, 'model_b': 0.5}
        with pytest.raises(ConfigValidationError, match='nonnegative'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestChannelOrder:
    def test_duplicate_channels_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['channel_order'] = ['flair', 'flair', 't1', 't2']
        with pytest.raises(ConfigValidationError, match='duplicate'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_unknown_modality_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['channel_order'] = ['flair', 't1c', 't1', 'dwi']
        with pytest.raises(ConfigValidationError, match='invalid modality'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_channel_count_mismatch(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['channel_order'] = ['flair', 't1c', 't1']  # 3 != in_channels=4
        with pytest.raises(ConfigValidationError, match='4 entries'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestPostprocessing:
    def test_threshold_too_high(self, tmp_path):
        cfg = _valid_config()
        cfg['postprocessing']['threshold'] = 1.5
        with pytest.raises(ConfigValidationError, match='\\(0.0, 1.0\\)'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_threshold_zero(self, tmp_path):
        cfg = _valid_config()
        cfg['postprocessing']['threshold'] = 0.0
        with pytest.raises(ConfigValidationError, match='\\(0.0, 1.0\\)'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_invalid_fusion_mode(self, tmp_path):
        cfg = _valid_config()
        cfg['fusion_mode'] = 'argmax'
        with pytest.raises(ConfigValidationError, match='probability'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestBooleanRejection:
    def test_in_channels_boolean_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['in_channels'] = True
        with pytest.raises(ConfigValidationError, match='got bool'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_roi_dim_boolean_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['roi_size'] = [96, True, 96]
        with pytest.raises(ConfigValidationError, match='must be an integer'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_weights_boolean_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['weights']['model_a'] = True
        with pytest.raises(ConfigValidationError, match='boolean'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_threshold_boolean_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['postprocessing']['threshold'] = True
        with pytest.raises(ConfigValidationError, match='numeric'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_min_et_boolean_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['postprocessing']['min_et_volume_voxels'] = True
        with pytest.raises(ConfigValidationError, match='non-negative integer'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestNonFiniteRejection:
    def test_nan_weights_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['weights']['model_a'] = float('nan')
        with pytest.raises(ConfigValidationError, match='finite'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_inf_weights_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['weights']['model_b'] = float('inf')
        with pytest.raises(ConfigValidationError, match='finite'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_nan_overlap_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['sliding_window_overlap'] = float('nan')
        with pytest.raises(ConfigValidationError, match='finite'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_nan_threshold_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['postprocessing']['threshold'] = float('nan')
        with pytest.raises(ConfigValidationError, match='finite'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestTemperatureValidation:
    def test_valid_temperature_loaded(self, tmp_path):
        cfg = _valid_config()
        cfg['temperature'] = {'model_a': 1.2, 'model_b': 0.8}
        v = load_ensemble_config(_write_config(tmp_path, cfg))
        assert v['temperature']['model_a'] == 1.2
        assert v['temperature']['model_b'] == 0.8

    def test_invalid_temperature_type(self, tmp_path):
        cfg = _valid_config()
        cfg['temperature'] = 'hot'
        with pytest.raises(ConfigValidationError, match='temperature must be a dict'):
            load_ensemble_config(_write_config(tmp_path, cfg))

    def test_non_positive_temperature_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['temperature'] = {'model_a': 0.0, 'model_b': 1.0}
        with pytest.raises(ConfigValidationError, match='positive finite number'):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestModeValidation:
    def test_valid_mode_constant(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['mode'] = 'constant'
        v = load_ensemble_config(_write_config(tmp_path, cfg))
        assert v['model_a']['mode'] == 'constant'

    def test_invalid_mode_rejected(self, tmp_path):
        cfg = _valid_config()
        cfg['model_a']['mode'] = 'linear'
        with pytest.raises(ConfigValidationError, match="must be 'gaussian' or 'constant'"):
            load_ensemble_config(_write_config(tmp_path, cfg))


class TestFileErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ensemble_config(tmp_path / 'nonexistent.json')

    def test_invalid_json(self, tmp_path):
        p = tmp_path / 'bad.json'
        p.write_text('{not valid json')
        with pytest.raises(ConfigValidationError, match='valid JSON'):
            load_ensemble_config(p)

    def test_non_object_json(self, tmp_path):
        p = tmp_path / 'array.json'
        p.write_text('[1, 2, 3]')
        with pytest.raises(ConfigValidationError, match='JSON object'):
            load_ensemble_config(p)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

