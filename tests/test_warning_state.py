"""Warning state machine: confirm/cancel/double-confirm cycle.

Uses synthetic NIfTI that triggers foreground heuristic warnings.
Does NOT run real inference (pool.submit is mocked after queuing).
"""
import io
import shutil
import tempfile
import threading
from pathlib import Path
from unittest import mock
import sys

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def make_noisy_nifti():
    """Synthetic noise — triggers suspicious_foreground warning (occupancy > 60%)."""
    affine = np.eye(4)
    data = np.random.uniform(1.0, 100.0, (16, 16, 16)).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()


def make_nifti_bytes():
    """Sparse valid brain-like NIfTI (foreground occupancy < 1% if mostly zeros)."""
    affine = np.eye(4)
    data = np.zeros((32, 32, 32), dtype=np.float32)
    # Fill only a small central sphere
    cx, cy, cz = 16, 16, 16
    for x in range(12, 20):
        for y in range(12, 20):
            for z in range(12, 20):
                if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < 25:
                    data[x, y, z] = float((x+y+z) % 50 + 10)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_noisy_input_triggers_warning():
    """Noise-filled NIfTI causes validate_study to produce quality_warnings."""
    from validation import validate_study
    import tempfile, shutil

    nii = make_noisy_nifti()
    tmpdir = Path(tempfile.mkdtemp())
    try:
        for name in ('t1c.nii.gz', 't1.nii.gz', 't2.nii.gz', 'flair.nii.gz'):
            (tmpdir / name).write_bytes(nii)
        paths = {k: tmpdir / f'{k}.nii.gz' for k in ('t1c', 't1', 't2', 'flair')}
        _, warnings = validate_study(paths)
        assert len(warnings) > 0, 'Expected quality warnings for noise-filled NIfTI'
        codes = [w['code'] for w in warnings]
        assert 'suspicious_foreground' in codes
    finally:
        shutil.rmtree(tmpdir)


def test_cancel_job_removes_directory():
    """cancel_job must delete the run directory and return status='cancelled'."""
    from app import JOBS, RUNS, GUARD

    uid = 'aaaabbbbccccddddeeeeffff00001111'
    job_dir = RUNS / uid
    job_dir.mkdir(parents=True, exist_ok=True)
    job_record = {
        'id': uid,
        'status': 'awaiting_confirmation',
        'message': 'Review input quality warnings before inference',
        'pending_input': {'paths': {}, 'expert': None, 'expert_et': 4, 'name': 'test', 'model_name': 'swinunetr'},
    }
    with GUARD:
        JOBS[uid] = job_record

    try:
        from fastapi.testclient import TestClient
        from app import app
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(f'/api/jobs/{uid}/cancel')
        assert resp.status_code == 200
        assert resp.json()['status'] == 'cancelled'
        assert not job_dir.exists(), 'Run directory must be deleted on cancel'
    finally:
        with GUARD:
            JOBS.pop(uid, None)
        shutil.rmtree(job_dir, ignore_errors=True)


def test_double_confirm_returns_409():
    """Confirming a non-awaiting_confirmation job must return 409."""
    from app import JOBS, RUNS, GUARD
    from fastapi.testclient import TestClient
    from app import app

    uid = 'bbbbbbbbbbbbbbbb1111111111111111'
    job_dir = RUNS / uid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'job.json').write_text('{}')

    # Set the job to 'running' (already confirmed)
    job_record = {'id': uid, 'status': 'running', 'message': 'Running'}
    with GUARD:
        JOBS[uid] = job_record

    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.post(f'/api/jobs/{uid}/confirm')
        assert resp.status_code == 409
    finally:
        with GUARD:
            JOBS.pop(uid, None)
        shutil.rmtree(job_dir, ignore_errors=True)


def test_cancel_non_awaiting_returns_409():
    """Cancelling a job that is not awaiting_confirmation must return 409."""
    from app import JOBS, RUNS, GUARD
    from fastapi.testclient import TestClient
    from app import app

    uid = 'cccccccccccccccc2222222222222222'
    job_dir = RUNS / uid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'job.json').write_text('{}')

    job_record = {'id': uid, 'status': 'complete', 'message': 'Analysis complete', 'result': {}}
    with GUARD:
        JOBS[uid] = job_record

    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.post(f'/api/jobs/{uid}/cancel')
        assert resp.status_code == 409
    finally:
        with GUARD:
            JOBS.pop(uid, None)
        shutil.rmtree(job_dir, ignore_errors=True)
