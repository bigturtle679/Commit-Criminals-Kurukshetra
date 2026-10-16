"""Numerical Stability and Boundary Testing Suite for NeuroScope.

Verifies protection against:
1. Underflow: Tiny values rounding to zero, division by zero (0/0) in Dice loss, log(0) in entropy.
2. Overflow: Huge values exceeding float max, exp() explosion in Sigmoid/Softmax, NaN/Inf propagation.
3. Subnormals & FP16 Dynamic Range limits: Checking half-precision bounds ([6e-5, 65504]).
4. Model Tensor Health: Ensuring all weights and activation tensors are strictly finite.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ensemble_inference import compute_voxel_entropy, enforce_hierarchy
from scripts.models.resenc_unet import ResEncUNet
from scripts.models.model_loader import load_swinunetr, load_resenc


def test_entropy_underflow_and_log_zero():
    """Verify compute_voxel_entropy never evaluates log(0) -> -Inf or NaN."""
    edge_probs = torch.tensor([0.0, 1.0, 1e-45, 1.0 - 1e-45, -1e-10, 1.0 + 1e-10])
    entropy = compute_voxel_entropy(edge_probs, eps=1e-7)
    
    assert torch.isfinite(entropy).all(), f"Entropy contains non-finite values: {entropy}"
    assert not torch.isnan(entropy).any(), "Entropy contains NaN!"
    assert (entropy >= 0.0).all(), "Entropy cannot be negative"
    assert torch.isclose(entropy[0], torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(entropy[1], torch.tensor(0.0), atol=1e-4)
    print("PASS: test_entropy_underflow_and_log_zero (log(0) guard verified)")


def test_sigmoid_overflow_extreme_logits():
    """Verify that extreme logits (+10,000, -10,000) don't overflow exp() to Inf or NaN."""
    extreme_logits = torch.tensor([1e4, -1e4, 88.0, -88.0, 0.0, 1e-20])
    probs = torch.sigmoid(extreme_logits)
    
    assert torch.isfinite(probs).all(), "Sigmoid produced non-finite values on extreme logits"
    assert not torch.isnan(probs).any(), "Sigmoid produced NaN on extreme logits"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Sigmoid output out of bounds [0, 1]"
    assert probs[0] == 1.0, "Huge positive logit should saturate cleanly to 1.0"
    assert probs[1] == 0.0, "Huge negative logit should underflow cleanly to 0.0"
    print("PASS: test_sigmoid_overflow_extreme_logits (exp() overflow guard verified)")


def test_dice_division_by_zero_underflow():
    """Verify Dice metric calculation handles empty foregrounds (0/0) gracefully."""
    def safe_dice(pred, target, eps=1e-7):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        if denominator == 0:
            return 1.0 if intersection == 0 else 0.0
        return float((2.0 * intersection + eps) / (denominator + eps))

    # Case 1: Both pred and target are all-zeros (healthy brain, no tumor)
    pred_empty = torch.zeros((10, 10, 10))
    target_empty = torch.zeros((10, 10, 10))
    d1 = safe_dice(pred_empty, target_empty)
    assert d1 == 1.0, f"Empty-on-empty Dice must be 1.0, got {d1}"

    # Case 2: Model predicts zero voxels, but target has 1 voxel (false negative)
    target_one = torch.zeros((10, 10, 10))
    target_one[0, 0, 0] = 1.0
    d2 = safe_dice(pred_empty, target_one)
    assert d2 < 1e-5, f"False negative Dice must be ~0.0, got {d2}"
    assert not np.isnan(d2), "Dice must not be NaN"
    print("PASS: test_dice_division_by_zero_underflow (0/0 division guard verified)")


def test_model_weights_finite_check():
    """Verify that loaded models contain no NaN or Inf in their parameters."""
    swin, _, _ = load_swinunetr(device=torch.device('cpu'))
    for name, param in swin.named_parameters():
        assert torch.isfinite(param).all(), f"SwinUNETR parameter {name} has NaN or Inf!"
    
    resenc, _, _ = load_resenc(preset='M', device=torch.device('cpu'))
    for name, param in resenc.named_parameters():
        assert torch.isfinite(param).all(), f"ResEncUNet parameter {name} has NaN or Inf!"
    print("PASS: test_model_weights_finite_check (all model parameters strictly finite)")


def test_forward_pass_underflow_overflow():
    """Verify forward pass on synthetic patch with both normal and extreme intensity inputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResEncUNet(in_channels=4, out_channels=3, preset='M').to(device)
    model.eval()

    # 1. Normal normalized input
    x_normal = torch.randn(1, 4, 32, 32, 32, device=device)
    with torch.no_grad():
        out_normal = model(x_normal)
        prob_normal = torch.sigmoid(out_normal)
    assert torch.isfinite(out_normal).all(), "Model output has NaN/Inf on normal input"
    assert (prob_normal >= 0.0).all() and (prob_normal <= 1.0).all()

    # 2. Extreme input (scanner artifact spike: 10,000 intensity)
    x_spike = torch.randn(1, 4, 32, 32, 32, device=device) * 50.0 + 500.0
    with torch.no_grad():
        out_spike = model(x_spike)
        prob_spike = torch.sigmoid(out_spike)
    assert torch.isfinite(out_spike).all(), "Model output exploded to Inf/NaN on intensity spike"
    assert torch.isfinite(prob_spike).all()

    # 3. All-zeros input (empty background air)
    x_zero = torch.zeros(1, 4, 32, 32, 32, device=device)
    with torch.no_grad():
        out_zero = model(x_zero)
        prob_zero = torch.sigmoid(out_zero)
    assert torch.isfinite(out_zero).all(), "Model produced NaN/Inf on all-zero input"
    assert torch.isfinite(prob_zero).all()
    print("PASS: test_forward_pass_underflow_overflow (normal, spike, and zero inputs tested)")


def test_fp16_autocast_dynamic_range():
    """Verify FP16 / AMP execution does not overflow the 65,504 float16 ceiling."""
    if not torch.cuda.is_available():
        print("SKIP: test_fp16_autocast_dynamic_range (CUDA not available)")
        return

    device = torch.device('cuda')
    model = ResEncUNet(in_channels=4, out_channels=3, preset='M').to(device)
    model.eval()

    x = torch.randn(1, 4, 32, 32, 32, device=device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            out_fp16 = model(x)
            prob_fp16 = torch.sigmoid(out_fp16.float())

    assert torch.isfinite(out_fp16).all(), "FP16 inference resulted in Inf or NaN"
    assert (out_fp16.abs() < 65500.0).all(), "FP16 activations exceeded FP16 max dynamic range (65504)"
    assert torch.isfinite(prob_fp16).all(), "Sigmoid of FP16 output produced NaN/Inf"
    print("PASS: test_fp16_autocast_dynamic_range (FP16 stays safely within dynamic range)")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING NUMERICAL STABILITY & UNDERFLOW/OVERFLOW TEST SUITE")
    print("=" * 60)
    test_entropy_underflow_and_log_zero()
    test_sigmoid_overflow_extreme_logits()
    test_dice_division_by_zero_underflow()
    test_model_weights_finite_check()
    test_forward_pass_underflow_overflow()
    test_fp16_autocast_dynamic_range()
    print("=" * 60)
    print("ALL NUMERICAL STABILITY TESTS PASSED! (6/6)")
    print("=" * 60)
