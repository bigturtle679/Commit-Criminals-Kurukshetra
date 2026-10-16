from pathlib import Path
import json, uuid, threading, shutil, time, traceback, logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT, run_case, _gpu_name

logger = logging.getLogger(__name__)

app = FastAPI(title='NeuroScope Local · 3D Brain Tumor Segmentation')
RUNS = ROOT / 'work/runs'
RUNS.mkdir(parents=True, exist_ok=True)
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)
GUARD = threading.Lock()
app.mount('/static', StaticFiles(directory=ROOT / 'src/web'), name='static')

# Upload retention: delete run directories older than this many hours (default: 24h)
RETENTION_HOURS = 24


def is_run_expired(run_dir: Path, cutoff_ts: float) -> bool:
    """Determine if a run directory is expired based on persistent metadata or timestamps.

    Checks created_ts (Unix timestamp) or created_at (ISO 8601 UTC string) in job.json.
    Falls back to directory filesystem timestamps if metadata is unparseable.
    """
    job_file = run_dir / 'job.json'
    if job_file.exists():
        try:
            data = json.loads(job_file.read_text(encoding='utf-8'))
            if 'created_ts' in data and isinstance(data['created_ts'], (int, float)):
                return float(data['created_ts']) < cutoff_ts
            if 'created_at' in data and isinstance(data['created_at'], str):
                dt = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
                return dt.timestamp() < cutoff_ts
        except Exception:
            pass
    try:
        stat = run_dir.stat()
        ts = getattr(stat, 'st_ctime', stat.st_mtime)
        return ts < cutoff_ts
    except Exception:
        return False


def sweep_expired_runs(runs_dir: Path = RUNS, retention_hours: float = RETENTION_HOURS,
                       jobs_dict: dict | None = None, active_ids: set | None = None) -> list[str]:
    """Perform a sweep of expired run directories older than retention_hours.

    Returns a list of successfully deleted job IDs.
    Thread-safe, verifies deletion, and safeguards active jobs.
    """
    if jobs_dict is None:
        jobs_dict = JOBS
    cutoff = time.time() - (retention_hours * 3600.0)

    if active_ids is None:
        with GUARD:
            active_ids = {
                uid for uid, job in jobs_dict.items()
                if job.get('status') in ('queued', 'running')
            }

    deleted = []
    if not runs_dir.exists():
        return deleted

    for run_dir in list(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        uid = run_dir.name
        if uid in active_ids:
            continue
        try:
            if is_run_expired(run_dir, cutoff):
                shutil.rmtree(run_dir, ignore_errors=False)
                if not run_dir.exists():
                    with GUARD:
                        jobs_dict.pop(uid, None)
                    deleted.append(uid)
                    logger.info("Cleaned up expired run: %s", uid)
                else:
                    logger.error("Failed to delete expired run directory: %s", run_dir)
        except Exception as e:
            logger.warning("Error during cleanup of %s: %s", uid, e)

    return deleted


def _cleanup_worker():
    """Background worker thread that periodically triggers cleanup sweeps."""
    while True:
        try:
            sweep_expired_runs()
        except Exception as e:
            logger.warning("Background cleanup sweep error: %s", e)
        time.sleep(3600)


# Start cleanup thread as daemon
_cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True, name='run-cleanup')
_cleanup_thread.start()

PATIENT_CASES = {
    '00000': {
        'id': '00000',
        'code': 'BraTS-GLI-00000-000',
        'name': 'BraTS GLI-00000',
        'folder': ROOT / 'data/demo',
        'et_label': 3,
        'description': 'Four MRI sequences with an expert reference mask.'
    },
    '00002': {
        'id': '00002',
        'code': 'BraTS-GLI-00002-000',
        'name': 'BraTS GLI-00002',
        'folder': ROOT / 'data/BraTS-GLI-00002-000',
        'et_label': 3,
        'description': 'Four MRI sequences with an expert reference mask.'
    },
    '00003': {
        'id': '00003',
        'code': 'BraTS-GLI-00003-000',
        'name': 'BraTS GLI-00003',
        'folder': ROOT / 'data/BraTS-GLI-00003-000',
        'et_label': 3,
        'description': 'Four MRI sequences with an expert reference mask.'
    }
}


@app.get('/')
def index():
    return FileResponse(ROOT / 'src/web/index.html')


@app.get('/api/status')
def status():
    import torch
    dev_name = _gpu_name()
    return {
        'gpu': dev_name,
        'cuda_available': torch.cuda.is_available(),
        'demo_ready': len(list((ROOT / 'data/demo').glob('*.nii.gz'))) >= 5 or len(list((ROOT / 'data/synthetic_demo').glob('*.nii.gz'))) >= 4,
        'models_available': [
            {'id': 'ensemble', 'name': '8:2 Ensemble (SwinUNETR + nnU-Net ResEnc)', 'badge': 'High-Accuracy 8:2 Fusion', 'speed': '~5s'},
            {'id': 'swinunetr', 'name': 'SwinUNETR (Vision Transformer)', 'badge': 'Tested on three cases', 'speed': '~5s'},
            {'id': 'segresnet', 'name': 'SegResNet (3D CNN Autoencoder)', 'badge': 'BraTS baseline', 'speed': '~3s'}
        ],
        'default_model': 'ensemble',
        'patient_cases': [
            {
                'id': k,
                'name': v['name'],
                'description': v['description'],
                'available': all((v['folder'] / f"{v['code']}-{s}.nii.gz").exists() for s in ['t1c', 't1n', 't2w', 't2f'])
            }
            for k, v in PATIENT_CASES.items()
        ]
    }


@app.get('/api/jobs')
def jobs():
    return list(JOBS.values())


def execute(job_id, paths, expert, expert_et, name, model_name='ensemble'):
    job = JOBS[job_id]
    def update(message):
        job.update(status='running', message=message)
    try:
        result = run_case(paths, RUNS / job_id, update, expert, expert_et, name, model_name=model_name)
        job.update(status='complete', message='Analysis complete', result=result)
    except Exception as e:
        traceback.print_exc()
        job.update(status='error', message=str(e))
    (RUNS / job_id / 'job.json').write_text(json.dumps(job, indent=2))


def new_job():
    with GUARD:
        if any(j['status'] in ('queued', 'running') for j in JOBS.values()):
            raise HTTPException(409, 'An analysis is already running. Please wait for it to finish.')
        uid = uuid.uuid4().hex
        job_dir = RUNS / uid
        job_dir.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        job_record = {
            'id': uid,
            'status': 'queued',
            'message': f'Queued for {_gpu_name()} inference',
            'created_at': now_utc.isoformat(),
            'created_ts': now_utc.timestamp(),
        }
        (job_dir / 'job.json').write_text(json.dumps(job_record, indent=2))
        JOBS[uid] = job_record
        return uid


@app.post('/api/demo')
def demo(model: str = Query('ensemble'), case: str = Query('00000')):
    case_info = PATIENT_CASES.get(case, PATIENT_CASES['00000'])
    folder = case_info['folder']
    code = case_info['code']

    paths = {
        't1c': folder / f'{code}-t1c.nii.gz',
        't1': folder / f'{code}-t1n.nii.gz',
        't2': folder / f'{code}-t2w.nii.gz',
        'flair': folder / f'{code}-t2f.nii.gz'
    }
    expert = folder / f'{code}-seg.nii.gz'
    if not all(p.exists() for p in paths.values()):
        raise HTTPException(404, f"MRI scans missing for case {case_info['name']}.")

    uid = new_job()
    model_name = model if model in ('ensemble', 'swinunetr', 'segresnet') else 'ensemble'
    POOL.submit(
        execute,
        uid,
        paths,
        expert if expert.exists() else None,
        case_info['et_label'],
        f"{case_info['name']} ({model_name.upper()})",
        model_name=model_name
    )
    return {'id': uid}


@app.post('/api/analyze')
async def analyze(
    t1c: UploadFile = File(...),
    t1: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    expert: UploadFile | None = File(None),
    expert_et: int = Form(4),
    model: str = Form('ensemble')
):
    if expert_et not in (3, 4):
        raise HTTPException(400, 'Expert ET label must be 3 or 4.')
    files = {'t1c': t1c, 't1': t1, 't2': t2, 'flair': flair}
    if expert and expert.filename:
        files['expert'] = expert
    for f in files.values():
        if not (f.filename or '').lower().endswith(('.nii', '.nii.gz')):
            raise HTTPException(400, 'Upload NIfTI MRI files (.nii or .nii.gz), not JPG/PNG screenshots.')
    uid = new_job()
    paths = {}
    try:
        for key, f in files.items():
            suffix = '.nii.gz' if f.filename.lower().endswith('.nii.gz') else '.nii'
            path = RUNS / uid / (key + suffix)
            size = 0
            with path.open('wb') as dest:
                while chunk := await f.read(1024 * 1024):
                    size += len(chunk)
                    if size > 150 * 1024 * 1024:
                        raise ValueError('Each uploaded file must be under 150 MB.')
                    dest.write(chunk)
            paths[key] = path
    except Exception as e:
        JOBS[uid].update(status='error', message=str(e))
        (RUNS / uid / 'job.json').write_text(json.dumps(JOBS[uid], indent=2))
        raise HTTPException(400, str(e))
    model_name = model if model in ('ensemble', 'swinunetr', 'segresnet') else 'ensemble'
    POOL.submit(execute, uid, paths, paths.get('expert'), expert_et, f'Patient MRI ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


@app.get('/api/jobs/{uid}')
def job(uid: str):
    if not uid.isalnum() or len(uid) > 64 or uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    return JOBS[uid]


@app.get('/api/files/{uid}/{name}')
def file(uid: str, name: str):
    allowed = {
        'flair.nii.gz', 't1c.nii.gz', 'segmentation.nii.gz', 'expert.nii.gz',
        'uncertainty.nii.gz', 'brain.obj', 'tumor.obj', 'core.obj', 'et.obj', 'report.json'
    }
    if not uid.isalnum() or len(uid) > 64 or uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    if name not in allowed:
        raise HTTPException(404, 'Artifact not found.')

    runs_root = RUNS.resolve()
    job_dir = (RUNS / uid).resolve()
    if not job_dir.is_relative_to(runs_root):
        raise HTTPException(403, 'Path traversal forbidden.')

    path = (job_dir / name).resolve()
    if not path.is_relative_to(job_dir) or not path.exists():
        raise HTTPException(404, 'Artifact not found.')
    return FileResponse(path, filename=name)


# Startup recovery: reload persisted jobs and mark unfinished jobs as interrupted
for p in RUNS.glob('*/job.json'):
    try:
        j = json.loads(p.read_text())
        if j.get('status') in ('queued', 'running'):
            j['status'] = 'error'
            j['message'] = 'Analysis interrupted by server restart'
            p.write_text(json.dumps(j, indent=2))
        JOBS[j['id']] = j
    except (ValueError, KeyError, json.JSONDecodeError):
        pass

# Initial cleanup sweep at server startup
try:
    sweep_expired_runs()
except Exception as _e:
    logger.warning("Startup cleanup sweep failed: %s", _e)
