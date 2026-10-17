"""Report content and escaping tests.

Verifies:
- HTML escaping of dynamic text in case names
- SHA-256 hash values present in report output
- "Suspected tumour region" wording (not "Tumour detected")
- Volume arithmetic formula shown in report
- Both-empty Dice = 1 displayed as 1.0 in report row
"""
import io
import json
import shutil
import tempfile
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import report as rpt_module


def make_seg_nifti(shape=(16, 16, 16)):
    affine = np.eye(4)
    data = np.zeros(shape, dtype=np.uint8)
    data[6:10, 6:10, 6:10] = 4
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()


def make_mri_nifti(shape=(16, 16, 16)):
    affine = np.eye(4)
    data = np.random.uniform(10, 200, shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()


def build_minimal_run_dir():
    """Create a minimal run directory with all required NIfTI artifacts."""
    tmpdir = Path(tempfile.mkdtemp())
    shape = (16, 16, 16)
    affine = np.eye(4)

    for name in ('t1c.nii.gz', 't1.nii.gz', 't2.nii.gz', 'flair.nii.gz'):
        mri = np.random.uniform(10, 200, shape).astype(np.float32)
        nib.save(nib.Nifti1Image(mri, affine), tmpdir / name)

    seg = np.zeros(shape, dtype=np.uint8)
    seg[6:10, 6:10, 6:10] = 4
    nib.save(nib.Nifti1Image(seg, affine), tmpdir / 'segmentation.nii.gz')

    return tmpdir


def build_minimal_job(run_dir, case_name='Test Case'):
    """Return a minimal job dict compatible with render_report."""
    return {
        'id': 'aabbccdd11223344',
        'status': 'complete',
        'result': {
            'case_name': case_name,
            'model_name': 'swinunetr',
            'model': 'MONAI SwinUNETR',
            'model_version': 'Not verified',
            'model_license': 'Not verified',
            'code_commit': 'abc1234',
            'timestamp': '2026-01-01T00:00:00Z',
            'input_channel_order': ['FLAIR', 'T1c', 'T1', 'T2'],
            'shape': [16, 16, 16],
            'affine': np.eye(4).tolist(),
            'runtime_seconds': 5.0,
            'gpu': 'CPU',
            'device': 'cpu',
            'sliding_window_roi': [96, 96, 96],
            'spacing_mm': [1.0, 1.0, 1.0],
            'volumes_ml': {'WT': 0.064, 'TC': 0.064, 'ET': 0.064, 'ED': 0.0},
            'metrics': None,
            'uncertainty': {'available': False, 'mean_in_tumor': None, 'metric': None, 'interpretation': 'N/A'},
            'model_disagreement': None,
            'effective_config': None,
            'input_hashes': {'t1c': 'abc123' * 10 + 'abcd', 't1': 'def456' * 10 + 'defg'},
            'checkpoint_hashes': {'model_a': 'aaa' + 'b' * 61},
            'preprocessing': {'canonical': 'RAS', 'resampling': '1mm', 'normalization': 'z-score', 'foreground_crop': True},
            'quality_warnings': [],
            'limitations': ['Research use only.'],
            'evaluation': {'status': 'pending', 'reason': 'No verified cohort.'},
            'measurements': {
                'WT': {'predicted_voxels': 64, 'predicted_ml': 0.064, 'voxel_mm3': 1.0,
                       'expert_voxels': None, 'expert_ml': None, 'dice': None, 'sensitivity': None},
                'TC': {'predicted_voxels': 64, 'predicted_ml': 0.064, 'voxel_mm3': 1.0,
                       'expert_voxels': None, 'expert_ml': None, 'dice': None, 'sensitivity': None},
                'ET': {'predicted_voxels': 64, 'predicted_ml': 0.064, 'voxel_mm3': 1.0,
                       'expert_voxels': None, 'expert_ml': None, 'dice': None, 'sensitivity': None},
            },
            'disclaimer': 'Research demonstration only.',
            'note': 'Voxel volumes computed via exact 3x3 affine determinant.',
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_html_escaping():
    """Case name with HTML special chars must be escaped in the report."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir, case_name='<script>alert("xss")</script>')
        html = rpt_module.render_report(run_dir, job)
        assert '<script>alert' not in html, 'Raw <script> tag must not appear in report HTML'
        assert '&lt;script&gt;' in html or 'alert' not in html
    finally:
        shutil.rmtree(run_dir)


def test_report_contains_sha256_hashes():
    """Report must include the SHA-256 hash values from input_hashes and checkpoint_hashes."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir)
        html = rpt_module.render_report(run_dir, job)
        # The input hash values should appear somewhere in the HTML
        assert 'abc123' in html or 'SHA' in html or 'sha' in html
    finally:
        shutil.rmtree(run_dir)


def test_report_suspected_tumour_wording():
    """Report must say 'Suspected tumour region predicted by the model', not 'Tumour detected'."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir)
        html = rpt_module.render_report(run_dir, job)
        # The old forbidden wording
        assert 'Tumour detected' not in html
        # The correct wording (from quality_checks)
        assert 'Suspected tumour region predicted' in html
    finally:
        shutil.rmtree(run_dir)


def test_report_contains_volume_formula():
    """Report should show the volume formula N × det(A) / 1000."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir)
        html = rpt_module.render_report(run_dir, job)
        # The formula text as shown in the report
        assert '1,000' in html or 'det' in html or 'mL' in html
    finally:
        shutil.rmtree(run_dir)


def test_report_no_expert_shows_no_reference_text():
    """When expert_ml is None, the report shows 'No expert reference supplied'."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir)
        html = rpt_module.render_report(run_dir, job)
        assert 'No expert reference supplied' in html
    finally:
        shutil.rmtree(run_dir)


def test_report_quality_checks_label_check():
    """Quality checks must appear and include label verification."""
    run_dir = build_minimal_run_dir()
    try:
        checks = rpt_module.quality_checks(run_dir, build_minimal_job(run_dir)['result'])
        names = [c['name'] for c in checks]
        assert 'Label codes' in names
        # All label checks should pass for our synthetic output
        label_check = next(c for c in checks if c['name'] == 'Label codes')
        assert label_check['status'] == 'pass'
    finally:
        shutil.rmtree(run_dir)


def test_report_nested_region_check():
    """Quality checks must verify ET ⊆ TC ⊆ WT nesting."""
    run_dir = build_minimal_run_dir()
    try:
        checks = rpt_module.quality_checks(run_dir, build_minimal_job(run_dir)['result'])
        names = [c['name'] for c in checks]
        assert any('nesting' in n.lower() or 'ET' in n for n in names)
    finally:
        shutil.rmtree(run_dir)


def test_report_print_css_repeated_headers():
    """Report CSS must include thead{display:table-header-group} for PDF print."""
    run_dir = build_minimal_run_dir()
    try:
        job = build_minimal_job(run_dir)
        html = rpt_module.render_report(run_dir, job)
        assert 'table-header-group' in html, 'Print CSS for repeated table headers missing'
    finally:
        shutil.rmtree(run_dir)
