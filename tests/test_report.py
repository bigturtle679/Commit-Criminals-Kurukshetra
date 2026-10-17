"""Quality checks re-verify the saved mask; the report escapes text and states the disclaimer."""
import nibabel as nib
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import report  # noqa: E402

RESULT = {'volumes_ml': {'WT': 0.08, 'TC': 0.016, 'ET': 0.008, 'ED': 0.064}, 'uncertainty': {'available': False},
          'case_name': '<script>alert(1)</script>', 'model': 'test', 'metrics': {'WT': {'dice': 0.9, 'sensitivity': 0.8}}}


@pytest.fixture()
def run_dir(tmp_path):
    brain = np.zeros((20, 24, 16), np.float32)
    brain[3:17, 4:20, 2:14] = 100.0
    seg = np.zeros(brain.shape, np.uint8)
    seg[5:9, 6:10, 4:8] = 2                       # 64 voxels edema
    seg[6:8, 7:9, 5:7] = 4                        # 8 voxels enhancing
    seg[6:8, 7:9, 4] = 1                          # 4 voxels... overwritten region below keeps counts simple
    for m in ('t1', 't1c', 't2', 'flair'):
        nib.save(nib.Nifti1Image(brain, np.eye(4)), tmp_path / f'{m}.nii.gz')
    nib.save(nib.Nifti1Image(seg, np.eye(4)), tmp_path / 'segmentation.nii.gz')
    return tmp_path


def by_name(checks):
    return {c['name']: c['status'] for c in checks}


def test_quality_checks_pass_on_valid_mask(run_dir):
    seg = np.asarray(nib.load(str(run_dir / 'segmentation.nii.gz')).dataobj)
    v = {k: float(n) / 1000 for k, n in (('WT', np.isin(seg, [1, 2, 4]).sum()), ('TC', np.isin(seg, [1, 4]).sum()), ('ET', (seg == 4).sum()))}
    v['ED'] = v['WT'] - v['TC']
    checks = by_name(report.quality_checks(run_dir, {'volumes_ml': v, 'uncertainty': {}}))
    assert checks['Label codes'] == 'pass'
    assert checks['Mask aligned with MRI'] == 'pass'
    assert checks['Region nesting ET ⊆ TC ⊆ WT'] == 'pass'
    assert checks['Volume consistency WT = TC + ED'] == 'pass'
    assert checks['Uncertainty map produced'] == 'n/a'


def test_quality_checks_fail_on_bad_labels_and_inconsistent_volumes(run_dir):
    seg = np.asarray(nib.load(str(run_dir / 'segmentation.nii.gz')).dataobj).copy()
    seg[0, 0, 0] = 3
    nib.save(nib.Nifti1Image(seg, np.eye(4)), run_dir / 'segmentation.nii.gz')
    checks = by_name(report.quality_checks(run_dir, {'volumes_ml': {'WT': 1, 'TC': 0.2, 'ED': 0.5}}))
    assert checks['Label codes'] == 'fail'
    assert checks['Volume consistency WT = TC + ED'] == 'fail'


def test_report_escapes_text_and_contains_disclaimer(run_dir):
    page = report.render_report(run_dir, {'id': 'abc', 'result': RESULT})
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page
    assert 'does not present model output as a diagnosis' in page
    assert page.count('data:image/png;base64,') == 3
