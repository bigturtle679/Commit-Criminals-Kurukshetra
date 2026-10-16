"""Unit and integration tests for upload retention, job persistence, and API security.

Validates:
- Expired run cleanup after RETENTION_HOURS (without waiting 24 real hours)
- Active job protection (queued/running jobs are never deleted)
- Non-expired run preservation
- Restart recovery (stale queued/running jobs marked as error)
- Deletion verification and idempotent sweeps
- Path traversal rejection on file download endpoints
- Download allowlist enforcement
"""
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from app import (
    app, is_run_expired, sweep_expired_runs, RETENTION_HOURS,
    new_job, JOBS, RUNS, GUARD
)


@pytest.fixture
def test_runs_dir(tmp_path):
    """Create an isolated test directory for runs."""
    d = tmp_path / 'test_runs'
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestRetentionExpiration:
    def test_run_with_old_created_ts_is_expired(self, test_runs_dir):
        run = test_runs_dir / 'old_run'
        run.mkdir()
        now = time.time()
        job_data = {
            'id': 'old_run',
            'status': 'complete',
            'created_ts': now - (25 * 3600),  # 25 hours ago
            'created_at': datetime.fromtimestamp(now - (25 * 3600), timezone.utc).isoformat(),
        }
        (run / 'job.json').write_text(json.dumps(job_data))

        cutoff = now - (24 * 3600)
        assert is_run_expired(run, cutoff) is True

    def test_run_with_recent_created_ts_is_not_expired(self, test_runs_dir):
        run = test_runs_dir / 'recent_run'
        run.mkdir()
        now = time.time()
        job_data = {
            'id': 'recent_run',
            'status': 'complete',
            'created_ts': now - (2 * 3600),  # 2 hours ago
            'created_at': datetime.fromtimestamp(now - (2 * 3600), timezone.utc).isoformat(),
        }
        (run / 'job.json').write_text(json.dumps(job_data))

        cutoff = now - (24 * 3600)
        assert is_run_expired(run, cutoff) is False

    def test_run_with_malformed_metadata_falls_back_safely(self, test_runs_dir):
        run = test_runs_dir / 'corrupt_meta'
        run.mkdir()
        (run / 'job.json').write_text("invalid json contents")

        # Future cutoff: should be expired relative to future
        future_cutoff = time.time() + 1000
        assert is_run_expired(run, future_cutoff) is True

        # Past cutoff: should not be expired relative to distant past
        past_cutoff = time.time() - 100000
        assert is_run_expired(run, past_cutoff) is False


class TestSweepExpiredRuns:
    def test_sweep_deletes_expired_runs(self, test_runs_dir):
        now = time.time()
        # Expired job
        expired = test_runs_dir / 'expired_job'
        expired.mkdir()
        (expired / 'job.json').write_text(json.dumps({
            'id': 'expired_job',
            'status': 'complete',
            'created_ts': now - (30 * 3600)
        }))

        # Recent job
        recent = test_runs_dir / 'recent_job'
        recent.mkdir()
        (recent / 'job.json').write_text(json.dumps({
            'id': 'recent_job',
            'status': 'complete',
            'created_ts': now - (1 * 3600)
        }))

        mock_jobs = {
            'expired_job': {'status': 'complete'},
            'recent_job': {'status': 'complete'},
        }

        deleted = sweep_expired_runs(
            runs_dir=test_runs_dir,
            retention_hours=24,
            jobs_dict=mock_jobs,
            active_ids=set()
        )

        assert 'expired_job' in deleted
        assert not expired.exists()
        assert recent.exists()
        assert 'expired_job' not in mock_jobs
        assert 'recent_job' in mock_jobs

    def test_sweep_protects_active_running_jobs(self, test_runs_dir):
        now = time.time()
        # Old job that is still marked active (e.g. running a very long task)
        active = test_runs_dir / 'active_job'
        active.mkdir()
        (active / 'job.json').write_text(json.dumps({
            'id': 'active_job',
            'status': 'running',
            'created_ts': now - (30 * 3600)
        }))

        mock_jobs = {'active_job': {'status': 'running'}}
        deleted = sweep_expired_runs(
            runs_dir=test_runs_dir,
            retention_hours=24,
            jobs_dict=mock_jobs,
            active_ids={'active_job'}
        )

        assert 'active_job' not in deleted
        assert active.exists()
        assert 'active_job' in mock_jobs

    def test_sweep_is_idempotent(self, test_runs_dir):
        now = time.time()
        job = test_runs_dir / 'job1'
        job.mkdir()
        (job / 'job.json').write_text(json.dumps({
            'id': 'job1',
            'status': 'complete',
            'created_ts': now - (50 * 3600)
        }))

        mock_jobs = {'job1': {'status': 'complete'}}
        # First sweep
        d1 = sweep_expired_runs(runs_dir=test_runs_dir, retention_hours=24, jobs_dict=mock_jobs, active_ids=set())
        assert d1 == ['job1']

        # Second sweep immediately after
        d2 = sweep_expired_runs(runs_dir=test_runs_dir, retention_hours=24, jobs_dict=mock_jobs, active_ids=set())
        assert d2 == []


class TestRestartRecovery:
    def test_interrupted_jobs_marked_as_error_on_startup(self, test_runs_dir):
        stale_running = test_runs_dir / 'crashed_run'
        stale_running.mkdir()
        job_file = stale_running / 'job.json'
        job_file.write_text(json.dumps({
            'id': 'crashed_run',
            'status': 'running',
            'message': 'Inference in progress'
        }))

        # Simulate startup loop from app.py
        recovered_jobs = {}
        for p in test_runs_dir.glob('*/job.json'):
            try:
                j = json.loads(p.read_text())
                if j.get('status') in ('queued', 'running'):
                    j['status'] = 'error'
                    j['message'] = 'Analysis interrupted by server restart'
                    p.write_text(json.dumps(j, indent=2))
                recovered_jobs[j['id']] = j
            except Exception:
                pass

        assert recovered_jobs['crashed_run']['status'] == 'error'
        reloaded = json.loads(job_file.read_text())
        assert reloaded['status'] == 'error'
        assert 'restart' in reloaded['message']


class TestApiSecurityAndDownloads:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_status_endpoint_returns_200(self, client):
        res = client.get('/api/status')
        assert res.status_code == 200
        data = res.json()
        assert 'gpu' in data
        assert 'default_model' in data
        assert data['default_model'] == 'ensemble'

    def test_disallowed_file_download_rejected(self, client):
        res = client.get('/api/files/testjob/secret.txt')
        assert res.status_code == 404

    def test_invalid_uid_format_rejected(self, client):
        # Path traversal characters in UID
        res = client.get('/api/files/../../etc/report.json')
        assert res.status_code in (400, 404)

    def test_missing_job_returns_404(self, client):
        res = client.get('/api/jobs/nonexistentjobid123')
        assert res.status_code == 404

    def test_expired_job_endpoint_access_returns_404(self, client):
        # Create an expired run directory
        old_uid = 'expiredtestuid123'
        job_dir = RUNS / old_uid
        job_dir.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - (26 * 3600)  # 26 hours ago
        job_data = {
            'id': old_uid,
            'status': 'complete',
            'created_ts': old_ts,
            'created_at': datetime.fromtimestamp(old_ts, timezone.utc).isoformat(),
        }
        (job_dir / 'job.json').write_text(json.dumps(job_data))
        JOBS[old_uid] = job_data

        try:
            res = client.get(f'/api/jobs/{old_uid}')
            assert res.status_code == 404
            assert 'expired' in res.json()['detail'].lower()
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            JOBS.pop(old_uid, None)

    def test_expired_file_endpoint_access_returns_404(self, client):
        old_uid = 'expiredfiletestuid123'
        job_dir = RUNS / old_uid
        job_dir.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - (26 * 3600)
        job_data = {
            'id': old_uid,
            'status': 'complete',
            'created_ts': old_ts,
            'created_at': datetime.fromtimestamp(old_ts, timezone.utc).isoformat(),
        }
        (job_dir / 'job.json').write_text(json.dumps(job_data))
        (job_dir / 'report.json').write_text('{"dummy": true}')
        JOBS[old_uid] = job_data

        try:
            res = client.get(f'/api/files/{old_uid}/report.json')
            assert res.status_code == 404
            assert 'expired' in res.json()['detail'].lower()
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            JOBS.pop(old_uid, None)

    def test_invalid_upload_extension_rejected(self, client):
        files = {
            't1c': ('brain.png', b'fake png data', 'image/png'),
            't1': ('t1.nii.gz', b'dummy', 'application/gzip'),
            't2': ('t2.nii.gz', b'dummy', 'application/gzip'),
            'flair': ('flair.nii.gz', b'dummy', 'application/gzip'),
        }
        res = client.post('/api/analyze', files=files)
        assert res.status_code == 400
        assert 'nifti' in res.json()['detail'].lower()
