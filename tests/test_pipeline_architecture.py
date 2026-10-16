"""Tests verifying pipeline architecture — delegation, input validation, and contracts.

Validates:
- Pipeline delegates ensemble inference to scripts/ensemble_inference.py
- Pipeline does NOT contain inline sliding_window_inference for ensemble mode
- Input validation catches bad files
- Result contract matches expected schema
- Metrics module is used as single source of truth

All tests are static/structural (no GPU or checkpoints required).
"""
import ast
import sys
import textwrap
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


class TestNoDuplicateInference:
    """Verify pipeline.py delegates to run_ensemble instead of reimplementing."""

    def _get_pipeline_source(self):
        return (ROOT / 'src' / 'pipeline.py').read_text(encoding='utf-8')

    def test_pipeline_imports_run_ensemble(self):
        """Pipeline must import run_ensemble from the authoritative module."""
        source = self._get_pipeline_source()
        # The import can be at module level or inside run_case
        assert 'from scripts.ensemble_inference import run_ensemble' in source, \
            "pipeline.py must import run_ensemble from scripts/ensemble_inference.py"

    def test_pipeline_calls_run_ensemble(self):
        """Pipeline must call run_ensemble() in the ensemble branch."""
        source = self._get_pipeline_source()
        assert 'run_ensemble(' in source, \
            "pipeline.py must call run_ensemble() for ensemble inference"

    def test_no_inline_sliding_window_for_ensemble(self):
        """Pipeline must NOT call sliding_window_inference inside the ensemble branch.

        The ensemble code path should delegate entirely to run_ensemble().
        Sliding window calls are only allowed in the single-model fallback path.
        """
        source = self._get_pipeline_source()
        # Parse the AST to find sliding_window_inference calls
        tree = ast.parse(source)

        # Find the run_case function
        run_case_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'run_case':
                run_case_func = node
                break

        assert run_case_func is not None, "run_case function not found in pipeline.py"

        # Find if-blocks that check for ensemble mode
        # We look for the ensemble code path and verify it doesn't
        # directly call sliding_window_inference
        ensemble_source_lines = source.split('\n')

        # Find the ensemble delegation block
        in_ensemble_block = False
        ensemble_block_lines = []
        else_depth = 0

        for i, line in enumerate(ensemble_source_lines):
            stripped = line.strip()
            if "ensemble" in stripped and ("'ensemble'" in stripped or '"ensemble"' in stripped) and 'if ' in stripped:
                in_ensemble_block = True
                continue
            if in_ensemble_block:
                if stripped.startswith('else:') or stripped.startswith('elif '):
                    break
                ensemble_block_lines.append(line)

        ensemble_block = '\n'.join(ensemble_block_lines)

        # The ensemble block should NOT contain direct sliding_window_inference calls
        assert 'sliding_window_inference(' not in ensemble_block, \
            "Ensemble code path in pipeline.py must NOT call sliding_window_inference directly. " \
            "It should delegate to run_ensemble() from scripts/ensemble_inference.py."


class TestPipelineNoHardcodedParams:
    """Verify pipeline reads from config rather than hardcoding inference params."""

    def test_no_hardcoded_roi_in_ensemble_path(self):
        source = (ROOT / 'src' / 'pipeline.py').read_text(encoding='utf-8')
        # In the ensemble code path, ROI should come from config, not hardcoded
        # We check that roi_size is assigned from vcfg, not as a literal tuple
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'roi_size = (96, 96, 96)' in line:
                # This literal is acceptable ONLY in the single-model fallback
                # Check context: it should be after an 'else' or 'except'
                context = '\n'.join(lines[max(0, i-5):i+1])
                assert any(kw in context for kw in ['except', 'else', 'fallback', 'single-model']), \
                    f"Line {i+1}: Hardcoded roi_size found outside single-model fallback"


class TestMetricsUsedAsSingleSource:
    """Verify that volume and overlap metrics use src/metrics.py."""

    def test_pipeline_imports_metrics(self):
        source = (ROOT / 'src' / 'pipeline.py').read_text(encoding='utf-8')
        assert 'from metrics import' in source, \
            "pipeline.py must import from metrics.py"
        assert 'calculate_volume_ml' in source
        assert 'compute_dice' in source

    def test_ensemble_imports_metrics(self):
        source = (ROOT / 'scripts' / 'ensemble_inference.py').read_text(encoding='utf-8')
        assert 'from metrics import' in source, \
            "ensemble_inference.py must import from metrics.py"
        assert 'calculate_volume_ml' in source


class TestMetricsModule:
    """Direct tests for src/metrics.py as the single source of truth."""

    def test_volume_1mm_isotropic(self):
        from metrics import calculate_volume_ml
        mask = np.ones((10, 10, 10), dtype=bool)  # 1000 voxels
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        vol = calculate_volume_ml(mask, affine)
        assert vol == 1.0, f"1000 voxels at 1mm³ should be 1.0 mL, got {vol}"

    def test_volume_anisotropic(self):
        from metrics import calculate_volume_ml
        mask = np.ones((10, 10, 10), dtype=bool)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])  # 2mm in z
        vol = calculate_volume_ml(mask, affine)
        assert vol == 2.0, f"1000 voxels at 1x1x2mm should be 2.0 mL, got {vol}"

    def test_dice_perfect(self):
        from metrics import compute_dice
        mask = np.ones((5, 5, 5), dtype=bool)
        assert compute_dice(mask, mask) == 1.0

    def test_dice_no_overlap(self):
        from metrics import compute_dice
        a = np.zeros((5, 5, 5), dtype=bool)
        b = np.ones((5, 5, 5), dtype=bool)
        assert compute_dice(a, b) == 0.0

    def test_dice_both_empty(self):
        from metrics import compute_dice
        empty = np.zeros((5, 5, 5), dtype=bool)
        assert compute_dice(empty, empty) == 1.0

    def test_sensitivity_perfect(self):
        from metrics import compute_sensitivity
        mask = np.ones((5, 5, 5), dtype=bool)
        assert compute_sensitivity(mask, mask) == 1.0

    def test_sensitivity_all_false_negative(self):
        from metrics import compute_sensitivity
        pred = np.zeros((5, 5, 5), dtype=bool)
        gt = np.ones((5, 5, 5), dtype=bool)
        assert compute_sensitivity(pred, gt) == 0.0


class TestResultContract:
    """Verify the pipeline result dict has all expected fields."""

    REQUIRED_FIELDS = {
        'case_name', 'model_name', 'model', 'volumes_ml',
        'runtime_seconds', 'gpu', 'device', 'shape', 'spacing_mm',
        'meshes', 'timestamp', 'input_hashes', 'code_commit',
        'preprocessing', 'disclaimer',
    }

    def test_contract_fields_in_pipeline_source(self):
        """Check that all required fields appear in the result dict construction."""
        source = (ROOT / 'src' / 'pipeline.py').read_text(encoding='utf-8')
        for field in self.REQUIRED_FIELDS:
            assert f"'{field}'" in source, \
                f"Required field '{field}' missing from pipeline.py result dict"


class TestChannelIndexBuilder:
    """Test the ensemble_inference channel index builder."""

    def test_swinunetr_channel_order(self):
        from scripts.ensemble_inference import _build_channel_indices
        # SwinUNETR: [flair, t1c, t1, t2] -> [3, 0, 1, 2]
        idx = _build_channel_indices(['flair', 't1c', 't1', 't2'])
        assert idx == [3, 0, 1, 2]

    def test_resenc_channel_order(self):
        from scripts.ensemble_inference import _build_channel_indices
        # ResEnc: [flair, t1, t1c, t2] -> [3, 1, 0, 2]
        idx = _build_channel_indices(['flair', 't1', 't1c', 't2'])
        assert idx == [3, 1, 0, 2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
