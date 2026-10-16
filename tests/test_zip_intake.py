"""Zip upload: scan-type detection from file names and archive safety."""
import io
import zipfile

import pytest

from src.zip_intake import classify, detect


@pytest.mark.parametrize('name, role', [
    ('BraTS2021_00144/BraTS2021_00144_t1.nii.gz', 't1'),
    ('BraTS2021_00144_t1ce.nii.gz', 't1c'),
    ('BraTS2021_00144_t2.nii.gz', 't2'),
    ('BraTS2021_00144_flair.nii.gz', 'flair'),
    ('BraTS2021_00144_seg.nii.gz', 'expert'),
    ('BraTS-GLI-00000-000-t1n.nii.gz', 't1'),
    ('BraTS-GLI-00000-000-t1c.nii.gz', 't1c'),
    ('BraTS-GLI-00000-000-t2w.nii.gz', 't2'),
    ('BraTS-GLI-00000-000-t2f.nii.gz', 'flair'),
    ('patient_T1_CE.nii', 't1c'),
    ('patient_T1Gd.nii.gz', 't1c'),
    ('patient_ground_truth.nii.gz', 'expert'),
    ('00000017_brain_flair.nii', 'flair'),
    ('notes.txt', None),
    ('scan.png', None),
])
def test_classify(name, role):
    assert classify(name) == role


def _zip(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        for n in names:
            z.writestr(n, b'x')
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_detect_full_case_ignores_junk():
    names = ['case/p_t1.nii.gz', 'case/p_t1ce.nii.gz', 'case/p_t2.nii.gz', 'case/p_flair.nii.gz',
             'case/p_seg.nii.gz', '__MACOSX/case/._p_t1.nii.gz', 'case/HOW_TO_UPLOAD.txt']
    found = detect(_zip(names))
    assert {r: m.filename for r, m in found.items()} == {
        't1': 'case/p_t1.nii.gz', 't1c': 'case/p_t1ce.nii.gz', 't2': 'case/p_t2.nii.gz',
        'flair': 'case/p_flair.nii.gz', 'expert': 'case/p_seg.nii.gz'}


def test_missing_modality_is_reported():
    with pytest.raises(ValueError, match='FLAIR'):
        detect(_zip(['p_t1.nii.gz', 'p_t1ce.nii.gz', 'p_t2.nii.gz']))


def test_two_patients_rejected():
    with pytest.raises(ValueError, match='More than one'):
        detect(_zip(['a_t1.nii.gz', 'a_t1ce.nii.gz', 'a_t2.nii.gz', 'a_flair.nii.gz', 'b_t1.nii.gz']))
