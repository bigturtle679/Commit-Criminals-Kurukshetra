"""Failure-pack tests: all rejection scenarios must return 400 with a readable message.

All tests use in-memory NIfTI fixtures so no disk I/O above the fixture is needed.
"""
import io
import zipfile
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from app import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Synthetic NIfTI helpers
# ---------------------------------------------------------------------------

def make_nifti_bytes(shape=(16, 16, 16)):
    affine = np.eye(4)
    data = np.random.uniform(10.0, 200.0, shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()


def make_zip_with_files(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Non-NIfTI (PNG) via /api/analyze → 400
# ---------------------------------------------------------------------------

def test_png_upload_rejected():
    png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    files = {
        't1c': ('brain.png', png, 'image/png'),
        't1':  ('brain_t1.png', png, 'image/png'),
        't2':  ('brain_t2.png', png, 'image/png'),
        'flair': ('brain_flair.png', png, 'image/png'),
    }
    resp = client.post('/api/analyze', files=files, data={'model': 'swinunetr'})
    assert resp.status_code == 400
    detail = resp.json().get('detail', '').lower()
    assert 'nifti' in detail or 'nii' in detail or 'screenshot' in detail


# ---------------------------------------------------------------------------
# 2. File size limit: test validation module directly (avoids allocating 1 GB)
# ---------------------------------------------------------------------------

def test_file_size_limit_validation():
    from validation import read_image
    import tempfile, unittest.mock as mock

    nii_bytes = make_nifti_bytes()
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        f.write(nii_bytes)
        tmp_path = Path(f.name)

    try:
        class FakeStat:
            st_size = 1_000_000_001
            st_ctime = 0
            st_mtime = 0

        with mock.patch.object(Path, 'stat', return_value=FakeStat()):
            with pytest.raises((ValueError, OSError), match='1 GB|limit'):
                read_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. ZIP with > 200 entries → 400
# ---------------------------------------------------------------------------

def test_zip_too_many_entries():
    nii_bytes = make_nifti_bytes()
    files = {f'extra_{i:04d}_t1.nii.gz': nii_bytes for i in range(201)}
    zip_bytes = make_zip_with_files(files)
    resp = client.post(
        '/api/analyze-zip',
        files={'archive': ('patient.zip', zip_bytes, 'application/zip')},
        data={'model': 'swinunetr'}
    )
    assert resp.status_code == 400
    assert '200' in resp.json().get('detail', '') or 'entries' in resp.json().get('detail', '').lower()


# ---------------------------------------------------------------------------
# 4. ZIP with duplicate modality → 400
# ---------------------------------------------------------------------------

def test_zip_duplicate_modality():
    nii_bytes = make_nifti_bytes()
    zip_bytes = make_zip_with_files({
        'scan_t1c.nii.gz': nii_bytes,
        'BraTS_t1ce.nii.gz': nii_bytes,  # second match for t1c
        'scan_t1.nii.gz': nii_bytes,
        'scan_t2.nii.gz': nii_bytes,
        'scan_flair.nii.gz': nii_bytes,
    })
    resp = client.post(
        '/api/analyze-zip',
        files={'archive': ('patient.zip', zip_bytes, 'application/zip')},
        data={'model': 'swinunetr'}
    )
    assert resp.status_code == 400
    body = resp.json()['detail'].lower()
    assert 't1c' in body or 'duplicate' in body or 'more than one' in body


# ---------------------------------------------------------------------------
# 5. ZIP with missing modality → 400
# ---------------------------------------------------------------------------

def test_zip_missing_modality():
    nii_bytes = make_nifti_bytes()
    # FLAIR is missing
    zip_bytes = make_zip_with_files({
        'scan_t1c.nii.gz': nii_bytes,
        'scan_t1.nii.gz': nii_bytes,
        'scan_t2.nii.gz': nii_bytes,
    })
    resp = client.post(
        '/api/analyze-zip',
        files={'archive': ('patient.zip', zip_bytes, 'application/zip')},
        data={'model': 'swinunetr'}
    )
    assert resp.status_code == 400
    assert 'FLAIR' in resp.json().get('detail', '')


# ---------------------------------------------------------------------------
# 6. Expert mask with both labels 3 and 4 → ValueError from detect_et_label
# ---------------------------------------------------------------------------

def test_expert_mask_ambiguous_labels():
    from app import detect_et_label
    import tempfile

    affine = np.eye(4)
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[0, 0, 0] = 3
    data[1, 1, 1] = 4
    img = nib.Nifti1Image(data, affine)
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        nib.save(img, f.name)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match='3 and 4'):
            detect_et_label(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. Expert mask with mismatched geometry → InputValidationError
# ---------------------------------------------------------------------------

def test_expert_mask_geometry_mismatch():
    from validation import validate_study, InputValidationError
    import tempfile, shutil

    nii = make_nifti_bytes(shape=(16, 16, 16))
    expert_nii = make_nifti_bytes(shape=(8, 8, 8))  # different shape

    tmpdir = Path(tempfile.mkdtemp())
    try:
        for name, data in [('t1c.nii.gz', nii), ('t1.nii.gz', nii),
                           ('t2.nii.gz', nii), ('flair.nii.gz', nii),
                           ('expert.nii.gz', expert_nii)]:
            (tmpdir / name).write_bytes(data)
        paths = {k: tmpdir / f'{k}.nii.gz' for k in ('t1c', 't1', 't2', 'flair')}
        with pytest.raises((InputValidationError, ValueError)):
            validate_study(paths, tmpdir / 'expert.nii.gz', enhancing=4)
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# 8. Malicious ZIP path traversal → 400
# ---------------------------------------------------------------------------

def test_zip_path_traversal():
    nii_bytes = make_nifti_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        info = zipfile.ZipInfo('../../etc/passwd')
        zf.writestr(info, b'root:x:0:0:root:/root:/bin/bash')
        zf.writestr('scan_t1c.nii.gz', nii_bytes)
        zf.writestr('scan_t1.nii.gz', nii_bytes)
        zf.writestr('scan_t2.nii.gz', nii_bytes)
        zf.writestr('scan_flair.nii.gz', nii_bytes)
    resp = client.post(
        '/api/analyze-zip',
        files={'archive': ('patient.zip', buf.getvalue(), 'application/zip')},
        data={'model': 'swinunetr'}
    )
    assert resp.status_code == 400
    body = resp.json().get('detail', '').lower()
    assert 'unsafe' in body or 'path' in body or 'relative' in body


# ---------------------------------------------------------------------------
# 9. Corrupt ZIP archive → 400
# ---------------------------------------------------------------------------

def test_zip_corrupt_archive():
    corrupt = b'PK\x03\x04' + b'\x00' * 20 + b'this is not a valid zip'
    resp = client.post(
        '/api/analyze-zip',
        files={'archive': ('bad.zip', corrupt, 'application/zip')},
        data={'model': 'swinunetr'}
    )
    assert resp.status_code == 400
    assert 'zip' in resp.json().get('detail', '').lower()
