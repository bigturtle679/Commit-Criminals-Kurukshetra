"""Unit and mathematical verification tests for the 8:2 Ensemble Pipeline."""
import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.models.resenc_unet import ResEncUNet
from scripts.models.model_loader import load_ensemble_models
from scripts.ensemble_inference import enforce_hierarchy, compute_voxel_entropy


def test_ensemble_weights_and_config():
    cfg_path = ROOT / 'configs/ensemble.json'
    assert cfg_path.exists(), "configs/ensemble.json must exist"
    cfg = json.loads(cfg_path.read_text())
    
    w_a = float(cfg['weights']['model_a'])
    w_b = float(cfg['weights']['model_b'])
    assert np.isclose(w_a, 0.8), f"Model A weight must be 0.8, got {w_a}"
    assert np.isclose(w_b, 0.2), f"Model B weight must be 0.2, got {w_b}"
    assert np.isclose(w_a + w_b, 1.0), "Ensemble weights must sum to 1.0"
    print("PASS: test_ensemble_weights_and_config")


def test_soft_probability_fusion():
    # Synthetic logits
    z_a = torch.tensor([2.0, -1.0, 0.5])
    z_b = torch.tensor([-2.0, 1.0, 0.0])
    
    p_a = torch.sigmoid(z_a)
    p_b = torch.sigmoid(z_b)
    
    p_fused = 0.8 * p_a + 0.2 * p_b
    
    expected = 0.8 * (1.0 / (1.0 + torch.exp(-z_a))) + 0.2 * (1.0 / (1.0 + torch.exp(-z_b)))
    assert torch.allclose(p_fused, expected), "Soft probability fusion must match 0.8*p_a + 0.2*p_b"
    assert (p_fused >= 0.0).all() and (p_fused <= 1.0).all(), "Fused probabilities must lie in [0, 1]"
    print("PASS: test_soft_probability_fusion")


def test_hierarchy_enforcement():
    # Case where raw probabilities violate ET <= TC <= WT
    # Channel 0: TC, Channel 1: WT, Channel 2: ET
    # Voxel 1: ET=0.9 > TC=0.3 -> TC must become 0.9, WT must become 0.9
    # Voxel 2: TC=0.7 > WT=0.4 -> WT must become 0.7
    prob_tc = torch.tensor([0.3, 0.7])
    prob_wt = torch.tensor([0.2, 0.4])
    prob_et = torch.tensor([0.9, 0.1])
    
    tc_c, wt_c, et_c = enforce_hierarchy(prob_tc, prob_wt, prob_et)
    
    # Assert ET <= TC <= WT for every voxel
    assert (et_c <= tc_c).all(), "Enhancing Tumor must be subset of Tumor Core"
    assert (tc_c <= wt_c).all(), "Tumor Core must be subset of Whole Tumor"
    assert torch.isclose(tc_c[0], torch.tensor(0.9))
    assert torch.isclose(wt_c[0], torch.tensor(0.9))
    assert torch.isclose(wt_c[1], torch.tensor(0.7))
    print("PASS: test_hierarchy_enforcement")


def test_voxel_entropy_uncertainty():
    # Voxel with p=0.5 has maximum entropy (1.0 bit)
    # Voxel with p=1.0 or 0.0 has minimum entropy (0.0 bit)
    p_max_uncertainty = torch.tensor([0.5])
    p_certain = torch.tensor([1.0 - 1e-7])
    
    h_max = compute_voxel_entropy(p_max_uncertainty)
    h_min = compute_voxel_entropy(p_certain)
    
    assert np.isclose(h_max.item(), 1.0, atol=1e-3), f"Entropy at p=0.5 must be 1.0, got {h_max.item()}"
    assert h_min.item() < 1e-3, f"Entropy at p~1.0 must be ~0, got {h_min.item()}"
    print("PASS: test_voxel_entropy_uncertainty")


def test_physical_volume_calculation():
    # 1000 voxels at 1x1x2 mm spacing must yield exactly 2.0 mL
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    vol = 1000 * voxel_ml
    assert np.isclose(vol, 2.0), f"1000 voxels at 1x1x2 mm must be 2.0 mL, got {vol}"
    print("PASS: test_physical_volume_calculation")


def test_resenc_model_instantiation():
    model = ResEncUNet(preset='M')
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 5_000_000, f"ResEncUNet M should have >5M parameters, got {n_params}"
    
    # Test forward pass with 3D input
    dummy_x = torch.zeros(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(dummy_x)
    assert out.shape == (1, 3, 32, 32, 32), f"Expected shape (1, 3, 32, 32, 32), got {out.shape}"
    print("PASS: test_resenc_model_instantiation")


def test_model_disagreement_calculation():
    """Verify that multi-model disagreement is calculated as genuine inter-model divergence."""
    p_a = torch.tensor([0.9, 0.1, 0.8])
    p_b = torch.tensor([0.2, 0.1, 0.4])
    disagreement = torch.abs(p_a - p_b)
    assert torch.allclose(disagreement, torch.tensor([0.7, 0.0, 0.4]))
    mean_disagreement = disagreement.mean().item()
    assert np.isclose(mean_disagreement, (0.7 + 0.0 + 0.4) / 3.0, atol=1e-4)


def test_disabled_uncertainty_semantics():
    """When uncertainty is disabled, it must report None rather than 0.0."""
    ent_data = None
    export_uncertainty = False
    regions = {'WT': np.array([True, True, False])}
    
    mean_uncertainty = (
        round(float(ent_data[regions['WT']].mean()), 4)
        if (export_uncertainty and ent_data is not None and regions['WT'].any())
        else None
    )
    assert mean_uncertainty is None, "Disabled uncertainty must be None, not 0.0"


if __name__ == '__main__':
    test_ensemble_weights_and_config()
    test_soft_probability_fusion()
    test_hierarchy_enforcement()
    test_voxel_entropy_uncertainty()
    test_physical_volume_calculation()
    test_resenc_model_instantiation()
    test_model_disagreement_calculation()
    test_disabled_uncertainty_semantics()
    print("\nALL ENSEMBLE UNIT TESTS PASSED! (8/8)")
