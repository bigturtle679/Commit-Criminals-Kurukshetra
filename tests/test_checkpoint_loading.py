"""Tests for strict checkpoint loading (scripts/models/model_loader.py).

Validates:
- Missing checkpoint raises CheckpointNotFoundError
- Corrupt checkpoint raises CheckpointLoadError
- Shape-mismatched checkpoint raises CheckpointLoadError
- Partial/incomplete checkpoint raises CheckpointLoadError (no 50% fallback)
- module. prefix stripping works correctly
- Valid checkpoint loads successfully with strict match

All tests are CPU-safe and create synthetic checkpoints — no real model weights needed.
"""
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.models.model_loader import (
    _resolve_checkpoint,
    _load_state_dict,
    _strip_known_prefixes,
    _validate_tensor_shapes,
    CheckpointNotFoundError,
    CheckpointLoadError,
)


class SimpleModel(nn.Module):
    """Minimal model for checkpoint loading tests."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(3, affine=True)

    def forward(self, x):
        return self.norm(self.conv(x))


class TestCheckpointResolution:
    def test_missing_all_candidates_raises(self, tmp_path):
        with pytest.raises(CheckpointNotFoundError, match='not found'):
            _resolve_checkpoint(
                [str(tmp_path / 'a.pt'), str(tmp_path / 'b.pt')],
                'TestModel'
            )

    def test_finds_first_existing(self, tmp_path):
        existing = tmp_path / 'found.pt'
        existing.write_bytes(b'dummy')
        result = _resolve_checkpoint([None, str(tmp_path / 'missing.pt'), str(existing)], 'TestModel')
        assert result == existing

    def test_missing_explicit_checkpoint_is_not_swapped(self, tmp_path):
        existing = tmp_path / 'found.pt'
        existing.write_bytes(b'dummy')
        with pytest.raises(CheckpointNotFoundError):
            _resolve_checkpoint([str(tmp_path / 'missing.pt'), str(existing)], 'TestModel')

    def test_none_candidates_skipped(self, tmp_path):
        existing = tmp_path / 'found.pt'
        existing.write_bytes(b'dummy')
        result = _resolve_checkpoint([None, str(existing)], 'TestModel')
        assert result == existing


class UnsafeDummyObject:
    """Non-tensor object used to test weights_only=True rejection."""
    def __init__(self):
        self.secret = 42


class TestStrictLoading:
    def test_valid_checkpoint_loads(self, tmp_path):
        """A checkpoint that exactly matches the model should load."""
        model = SimpleModel()
        ckpt_path = tmp_path / 'valid.pt'
        torch.save(model.state_dict(), ckpt_path)

        fresh_model = SimpleModel()
        _load_state_dict(fresh_model, ckpt_path, 'TestModel')
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), fresh_model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameter {n1} doesn't match"

    def test_corrupt_file_raises(self, tmp_path):
        """A corrupt file should raise CheckpointLoadError."""
        ckpt_path = tmp_path / 'corrupt.pt'
        ckpt_path.write_bytes(b'not a valid pytorch file')

        model = SimpleModel()
        with pytest.raises(CheckpointLoadError, match='Failed to read'):
            _load_state_dict(model, ckpt_path, 'TestModel')

    def test_wrong_keys_rejected(self, tmp_path):
        """A checkpoint with completely different keys should be rejected."""
        wrong_sd = {'layer1.weight': torch.zeros(10), 'layer1.bias': torch.zeros(10)}
        ckpt_path = tmp_path / 'wrong_keys.pt'
        torch.save(wrong_sd, ckpt_path)

        model = SimpleModel()
        with pytest.raises(CheckpointLoadError, match='incompatible'):
            _load_state_dict(model, ckpt_path, 'TestModel')

    def test_shape_mismatch_rejected(self, tmp_path):
        """A checkpoint with matching keys but wrong shapes should be rejected."""
        model = SimpleModel()
        sd = model.state_dict()
        # Corrupt shape of conv weight
        sd['conv.weight'] = torch.zeros(8, 8, 5, 5, 5)  # Wrong shape
        ckpt_path = tmp_path / 'wrong_shape.pt'
        torch.save(sd, ckpt_path)

        fresh_model = SimpleModel()
        with pytest.raises(CheckpointLoadError, match='shape mismatch'):
            _load_state_dict(fresh_model, ckpt_path, 'TestModel')

    def test_partial_checkpoint_rejected(self, tmp_path):
        """A checkpoint with only some keys should be rejected (no partial loading)."""
        model = SimpleModel()
        sd = model.state_dict()
        # Keep only one key
        partial_sd = {k: v for i, (k, v) in enumerate(sd.items()) if i == 0}
        ckpt_path = tmp_path / 'partial.pt'
        torch.save(partial_sd, ckpt_path)

        fresh_model = SimpleModel()
        with pytest.raises(CheckpointLoadError, match='incompatible'):
            _load_state_dict(fresh_model, ckpt_path, 'TestModel')

    def test_non_dict_state_raises(self, tmp_path):
        """A checkpoint containing a non-dict value should raise."""
        ckpt_path = tmp_path / 'not_dict.pt'
        torch.save(torch.zeros(10), ckpt_path)

        model = SimpleModel()
        with pytest.raises(CheckpointLoadError, match='valid state dict'):
            _load_state_dict(model, ckpt_path, 'TestModel')

    def test_weights_only_safe_loading_enforced(self, tmp_path):
        """A checkpoint containing arbitrary non-tensor classes must be rejected by default (weights_only=True)."""
        ckpt_path = tmp_path / 'arbitrary_payload.pt'
        torch.save({'state_dict': SimpleModel().state_dict(), 'custom': UnsafeDummyObject()}, ckpt_path)

        model = SimpleModel()
        # Default must reject unsafe objects
        with pytest.raises(CheckpointLoadError, match='weights_only=True'):
            _load_state_dict(model, ckpt_path, 'TestModel', allow_unsafe_legacy_weights=False)

        # Opt-in allows loading if explicitly permitted
        fresh_model = SimpleModel()
        _load_state_dict(fresh_model, ckpt_path, 'TestModel', allow_unsafe_legacy_weights=True)


class TestPrefixStripping:
    def test_module_prefix_stripped(self):
        sd = {'module.conv.weight': torch.zeros(3), 'module.conv.bias': torch.zeros(3)}
        result = _strip_known_prefixes(sd)
        assert 'conv.weight' in result
        assert 'conv.bias' in result
        assert result is not sd  # New dict returned

    def test_no_prefix_returns_original(self):
        sd = {'conv.weight': torch.zeros(3), 'conv.bias': torch.zeros(3)}
        result = _strip_known_prefixes(sd)
        assert result is sd  # Same dict returned

    def test_module_prefix_checkpoint_loads(self, tmp_path):
        """A checkpoint with module. prefix should load after stripping."""
        model = SimpleModel()
        sd = {f'module.{k}': v for k, v in model.state_dict().items()}
        ckpt_path = tmp_path / 'module_prefix.pt'
        torch.save(sd, ckpt_path)

        fresh_model = SimpleModel()
        _load_state_dict(fresh_model, ckpt_path, 'TestModel')
        # Should succeed without error


class TestTensorShapeValidation:
    def test_matching_shapes_pass(self):
        sd = {'a': torch.zeros(3, 4), 'b': torch.zeros(5)}
        model_sd = {'a': torch.zeros(3, 4), 'b': torch.zeros(5)}
        # Should not raise
        _validate_tensor_shapes(sd, model_sd, 'Test', Path('test.pt'))

    def test_mismatched_shapes_raise(self):
        sd = {'a': torch.zeros(3, 4)}
        model_sd = {'a': torch.zeros(5, 6)}
        with pytest.raises(CheckpointLoadError, match='shape mismatch'):
            _validate_tensor_shapes(sd, model_sd, 'Test', Path('test.pt'))

    def test_extra_keys_ignored(self):
        """Keys only in checkpoint (not in model) should not cause shape errors."""
        sd = {'a': torch.zeros(3, 4), 'extra': torch.zeros(99)}
        model_sd = {'a': torch.zeros(3, 4)}
        # Should not raise — extra key checking is done by strict=True
        _validate_tensor_shapes(sd, model_sd, 'Test', Path('test.pt'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
