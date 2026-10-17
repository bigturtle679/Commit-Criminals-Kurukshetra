"""Comprehensive evaluation metrics for 3D multi-modal brain tumor segmentation.

Region-based research metrics (not an official challenge evaluator):
- Subregions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- Overlap metrics: Dice Similarity Coefficient (DSC), Sensitivity (Recall)
- Volumetric metrics: Absolute Volume Error (|Delta-V| mL), Relative Volume Error (%)
- Coordinate verification: Affine determinant physical volume integration
"""
import numpy as np


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    """Compute Dice Similarity Coefficient (DSC) between two binary boolean masks.
    
    If both masks are empty, return 1.0 (perfect agreement on negative case).
    If only one mask is empty, return 0.0.
    """
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    
    p_sum = int(np.count_nonzero(p))
    g_sum = int(np.count_nonzero(g))
    
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
        
    intersection = int(np.count_nonzero(p & g))
    return round(2.0 * intersection / (p_sum + g_sum), 4)


def compute_sensitivity(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    """Compute Sensitivity (Recall / True Positive Rate)."""
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    
    g_sum = int(np.count_nonzero(g))
    if g_sum == 0:
        return None
        
    intersection = int(np.count_nonzero(p & g))
    return round(float(intersection) / float(g_sum), 4)


def compute_specificity(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    """Compute Specificity (True Negative Rate)."""
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    
    tn = int(np.count_nonzero((~p) & (~g)))
    fp = int(np.count_nonzero(p & (~g)))
    
    denom = tn + fp
    return round(float(tn) / float(denom), 4) if denom > 0 else 1.0


def calculate_volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    """Compute exact physical volume in milliliters (mL = cm^3).
    
    Uses the absolute determinant of the 3x3 spatial submatrix from the
    NIfTI homogeneous affine transformation matrix:
    V_mL = (N_voxels * |det(A_3x3)|) / 1000
    """
    n_voxels = int(np.count_nonzero(mask))
    if n_voxels == 0:
        return 0.0
    voxel_volume_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return round((n_voxels * voxel_volume_mm3) / 1000.0, 3)


def evaluate_case(pred_labels: np.ndarray, gt_labels: np.ndarray, affine: np.ndarray, enhancing_label: int = 3) -> dict:
    """Evaluate predicted labels against expert reference ground truth for all 3 BraTS subregions."""
    if pred_labels.shape != gt_labels.shape:
        raise ValueError('Prediction and reference shapes must match')
    if enhancing_label not in (3, 4) or not np.isin(gt_labels, [0, 1, 2, enhancing_label]).all():
        raise ValueError('Invalid expert label convention')
    if not np.isin(pred_labels, [0, 1, 2, 4]).all():
        raise ValueError('Invalid predicted labels')
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError('Invalid affine')
    # Label definitions
    # Pred: 1 (Necrotic core), 2 (Edema), 4 (Enhancing)
    pred_regions = {
        'WT': np.isin(pred_labels, [1, 2, 4]),
        'TC': np.isin(pred_labels, [1, 4]),
        'ET': pred_labels == 4
    }
    
    # GT: BraTS 2023 uses label 3 for ET; BraTS 2018-2021 uses label 4 for ET
    gt_regions = {
        'WT': np.isin(gt_labels, [1, 2, enhancing_label]),
        'TC': np.isin(gt_labels, [1, enhancing_label]),
        'ET': gt_labels == enhancing_label
    }
    
    scores = {}
    for region in ['WT', 'TC', 'ET']:
        p = pred_regions[region]
        g = gt_regions[region]
        
        pred_vol = calculate_volume_ml(p, affine)
        gt_vol = calculate_volume_ml(g, affine)
        abs_err = round(abs(pred_vol - gt_vol), 3)
        rel_err = round((abs_err / gt_vol * 100.0), 1) if gt_vol > 0 else None
        
        scores[region] = {
            'tp': int(np.count_nonzero(p & g)),
            'fp': int(np.count_nonzero(p & ~g)),
            'fn': int(np.count_nonzero(~p & g)),
            'precision': round(float(np.count_nonzero(p & g)) / int(p.sum()), 4) if p.any() else None,
            'dice': compute_dice(p, g),
            'sensitivity': compute_sensitivity(p, g),
            'pred_volume_ml': pred_vol,
            'gt_volume_ml': gt_vol,
            'abs_volume_error_ml': abs_err,
            'rel_volume_error_pct': rel_err
        }
        
    return scores

def measurement_summary(pred_labels, affine, expert_labels=None, enhancing_label=4):
    """Summary of tumor subregions (WT, TC, ET) predicted and expert volume/metrics."""
    pred_regions = {
        'WT': np.isin(pred_labels, [1, 2, 4]),
        'TC': np.isin(pred_labels, [1, 4]),
        'ET': pred_labels == 4,
    }
    gt_regions = None
    if expert_labels is not None:
        gt_regions = {
            'WT': np.isin(expert_labels, [1, 2, enhancing_label]),
            'TC': np.isin(expert_labels, [1, enhancing_label]),
            'ET': expert_labels == enhancing_label,
        }

    summary = {}
    for region in ['WT', 'TC', 'ET']:
        p = pred_regions[region]
        p_voxels = int(p.sum())
        p_ml = calculate_volume_ml(p, affine)

        if gt_regions is not None:
            g = gt_regions[region]
            g_voxels = int(g.sum())
            g_ml = calculate_volume_ml(g, affine)
            d = compute_dice(p, g)
            s = compute_sensitivity(p, g)
        else:
            g_voxels = None
            g_ml = None
            d = None
            s = None

        summary[region] = {
            'predicted_voxels': p_voxels,
            'predicted_ml': p_ml,
            'expert_voxels': g_voxels,
            'expert_ml': g_ml,
            'dice': d,
            'sensitivity': s,
        }
    return summary


if __name__ == '__main__':
    # Self-test synthetic check: 1,000 voxels at 1x1x2 mm spacing must produce 2.0 mL
    synthetic_mask = np.ones((10, 10, 10), dtype=bool) # 1,000 voxels
    synthetic_affine = np.diag([1.0, 1.0, 2.0, 1.0])
    vol = calculate_volume_ml(synthetic_mask, synthetic_affine)
    assert vol == 2.0, f"Synthetic test failed: expected 2.0 mL, got {vol}"
    
    dice = compute_dice(synthetic_mask, synthetic_mask)
    assert dice == 1.0, f"Expected DSC 1.0, got {dice}"
    print("Metrics module self-test PASSED!")
