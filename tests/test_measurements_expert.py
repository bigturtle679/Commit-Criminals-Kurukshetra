"""Expert comparison measurement tests.

Verifies:
- No expert → all expert_* fields are None
- Empty expert + empty pred → both-empty Dice = 1.0, sensitivity = None
- Cube expert vs cube pred → exact volume arithmetic using affine determinant
"""
import io
import tempfile
import shutil
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from metrics import (
    measurement_summary,
    compute_dice,
    compute_sensitivity,
    calculate_volume_ml,
    evaluate_case,
)


# ---------------------------------------------------------------------------
# Both-empty Dice = 1.0
# ---------------------------------------------------------------------------

def test_both_empty_dice_is_one():
    pred = np.zeros((10, 10, 10), dtype=bool)
    gt = np.zeros((10, 10, 10), dtype=bool)
    assert compute_dice(pred, gt) == 1.0


def test_one_empty_dice_is_zero():
    pred = np.ones((10, 10, 10), dtype=bool)
    gt = np.zeros((10, 10, 10), dtype=bool)
    assert compute_dice(pred, gt) == 0.0


# ---------------------------------------------------------------------------
# sensitivity is None when expert region is empty
# ---------------------------------------------------------------------------

def test_sensitivity_none_for_empty_expert():
    pred = np.ones((10, 10, 10), dtype=bool)
    gt = np.zeros((10, 10, 10), dtype=bool)
    result = compute_sensitivity(pred, gt)
    assert result is None


# ---------------------------------------------------------------------------
# Volume arithmetic: N voxels × |det(A)| / 1000 = mL
# ---------------------------------------------------------------------------

def test_volume_arithmetic_isotropic_1mm():
    """1000 voxels at 1×1×1 mm = 1.0 mL."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[:10, :10, :10] = True  # 1000 voxels
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    vol = calculate_volume_ml(mask, affine)
    assert abs(vol - 1.0) < 1e-6, f'Expected 1.0 mL, got {vol}'


def test_volume_arithmetic_anisotropic():
    """1000 voxels at 1×1×2 mm = 2.0 mL."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[:10, :10, :10] = True
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    vol = calculate_volume_ml(mask, affine)
    assert abs(vol - 2.0) < 1e-6, f'Expected 2.0 mL, got {vol}'


def test_volume_arithmetic_rotated_affine():
    """Volume is invariant to rotation: |det| accounts for the Jacobian."""
    mask = np.ones((10, 10, 10), dtype=bool)  # 1000 voxels
    # 45-degree rotation in x-y plane, 1mm voxel size
    theta = np.pi / 4
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ], dtype=float)
    vol = calculate_volume_ml(mask, rot)
    assert abs(vol - 1.0) < 1e-5, f'Rotation should not change volume: got {vol}'


# ---------------------------------------------------------------------------
# measurement_summary: no expert → all expert_* fields are None
# ---------------------------------------------------------------------------

def test_measurement_summary_no_expert():
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[3:7, 3:7, 3:7] = 4  # 64 ET voxels
    pred[2:8, 2:8, 2:8][pred[2:8, 2:8, 2:8] == 0] = 1  # necrotic fill
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    result = measurement_summary(pred, affine, expert_labels=None)
    for region, row in result.items():
        assert row['expert_voxels'] is None, f'{region}.expert_voxels should be None'
        assert row['expert_ml'] is None, f'{region}.expert_ml should be None'
        assert row['dice'] is None, f'{region}.dice should be None'
        assert row['sensitivity'] is None, f'{region}.sensitivity should be None'


def test_measurement_summary_empty_expert_empty_pred():
    """Both empty → Dice = 1.0 for each region, sensitivity = None."""
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    expert = np.zeros((10, 10, 10), dtype=np.float64)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    result = measurement_summary(pred, affine, expert_labels=expert, enhancing_label=4)
    for region, row in result.items():
        assert row['dice'] == 1.0, f'{region}: expected Dice 1.0 when both empty, got {row["dice"]}'
        assert row['sensitivity'] is None, f'{region}: expected sensitivity None when expert empty'


def test_measurement_summary_cube_expert():
    """Expert cube: 2×2×2 voxels at 1mm = 8 voxels = 8.0 mm³ = 0.008 mL.
    
    If the cube is all 'enhancing tumor' (label 4), then:
    - WT expert = TC expert = ET expert = 8 voxels = 0.008 mL
    """
    shape = (10, 10, 10)
    pred = np.zeros(shape, dtype=np.uint8)
    pred[4:6, 4:6, 4:6] = 4  # 8 ET voxels

    expert = np.zeros(shape, dtype=np.float64)
    expert[4:6, 4:6, 4:6] = 4  # 8 ET voxels (label convention 4 = ET)

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    result = measurement_summary(pred, affine, expert_labels=expert, enhancing_label=4)

    # ET: 8 voxels at 1mm³ = 0.008 mL
    assert abs(result['ET']['predicted_ml'] - 0.008) < 1e-6
    assert abs(result['ET']['expert_ml'] - 0.008) < 1e-6
    # Perfect overlap → Dice = 1.0
    assert result['ET']['dice'] == 1.0
    # Sensitivity = 1.0 (all expert voxels are predicted)
    assert result['ET']['sensitivity'] == 1.0


def test_evaluate_case_matches_measurement_summary():
    """evaluate_case and measurement_summary must agree on Dice values."""
    from metrics import evaluate_case

    shape = (12, 12, 12)
    pred = np.zeros(shape, dtype=np.uint8)
    pred[3:9, 3:9, 3:9] = 4   # ET cube

    gt = np.zeros(shape, dtype=np.uint8)
    gt[4:8, 4:8, 4:8] = 4     # smaller ET cube (partial overlap)

    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    scores = evaluate_case(pred, gt, affine, enhancing_label=4)
    summary = measurement_summary(pred, affine, expert_labels=gt.astype(float), enhancing_label=4)

    for region in ('WT', 'TC', 'ET'):
        assert abs(scores[region]['dice'] - summary[region]['dice']) < 1e-8, \
            f'{region} Dice mismatch: evaluate_case={scores[region]["dice"]}, measurement_summary={summary[region]["dice"]}'
