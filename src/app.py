from pathlib import Path
import json, uuid, threading, shutil, time, traceback, logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT, run_case, _gpu_name
from review_store import load_review, save_review, ReviewConflict
import zip_intake
import slices
import report
from validation import validate_upload_filenames, InputValidationError

logger = logging.getLogger(__name__)

app = FastAPI(title='NeuroScope Local · 3D Brain Tumor Segmentation')
RUNS = ROOT / 'work/runs'
RUNS.mkdir(parents=True, exist_ok=True)
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)
GUARD = threading.Lock()
app.mount('/static', StaticFiles(directory=ROOT / 'src/web'), name='static')
app.mount('/landing-static', StaticFiles(directory=ROOT / 'frontend'), name='landing-static')
# Atharva's landing + upload pages (frontend-1 branch); upload.js calls the real API above.
app.mount('/site', StaticFiles(directory=ROOT / 'neuroscope', html=True), name='site')

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
        'et_label': 4,
        'description': 'Four MRI sequences with an expert reference mask.'
    },
    '00002': {
        'id': '00002',
        'code': 'BraTS-GLI-00002-000',
        'name': 'BraTS GLI-00002',
        'folder': ROOT / 'data/BraTS-GLI-00002-000',
        'et_label': 4,
        'description': 'Four MRI sequences with an expert reference mask.'
    },
    '00003': {
        'id': '00003',
        'code': 'BraTS-GLI-00003-000',
        'name': 'BraTS GLI-00003',
        'folder': ROOT / 'data/BraTS-GLI-00003-000',
        'et_label': 4,
        'description': 'Four MRI sequences with an expert reference mask.'
    }
}


@app.get('/')
def index(request: Request):
    if 'analysis' in request.query_params:
        return RedirectResponse('/workspace?' + request.url.query, status_code=307)
    return RedirectResponse('/site/', status_code=307)


@app.get('/upload')
def upload_page():
    return RedirectResponse('/site/upload.html', status_code=307)


@app.get('/workspace')
def workspace():
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
            {'id': 'ensemble', 'name': '8:2 Ensemble (SwinUNETR + nnU-Net ResEnc)', 'badge': 'Weighted probability fusion',
             'available': (ROOT / 'models/resenc_brats/resenc_m_brats.pt').is_file() and (ROOT / 'models/swinunetr_brats/model_weights_BRATS-finetune.pt').is_file()},
            {'id': 'swinunetr', 'name': 'SwinUNETR (Vision Transformer)', 'badge': 'Transformer baseline',
             'available': (ROOT / 'models/swinunetr_brats/model_weights_BRATS-finetune.pt').is_file()},
            {'id': 'segresnet', 'name': 'SegResNet (3D CNN Autoencoder)', 'badge': 'Convolutional baseline',
             'available': (ROOT / 'models/brats_mri_segmentation/models/model.pt').is_file()}
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


@app.post('/api/download-weights')
def download_weights_endpoint():
    from scripts.models.model_loader import ensure_swinunetr_checkpoint
    try:
        path = ensure_swinunetr_checkpoint()
        return {'status': 'ok', 'message': f'SwinUNETR checkpoint verified at {path.name}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/jobs')
def jobs():
    return list(JOBS.values())


def execute(job_id, paths, expert, expert_et, name, model_name='ensemble'):
    job = JOBS[job_id]
    def update(message):
        job.update(status='running', message=message)
    try:
        result = run_case(paths, RUNS / job_id, update, expert, expert_et, name, model_name=model_name)
        if expert:
            # Pre-compute (and cache) the AI-vs-expert comparison here, so opening the result
            # never runs the heavy HD95 calculation while the browser is loading the MRI.
            update('Comparing prediction with the expert mask')
            try:
                report.expert_comparison(RUNS / job_id)
            except Exception:
                logger.warning('Expert comparison pre-compute failed for %s', job_id, exc_info=True)
        job.update(status='complete', message='Analysis complete', result=result)
    except Exception as e:
        traceback.print_exc()
        msg = str(e).replace(str(ROOT), '.')
        job.update(status='error', message=msg)
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


def validate_model(model):
    if model not in ('ensemble', 'swinunetr', 'segresnet'):
        raise HTTPException(400, 'Unknown model. Choose ensemble, swinunetr or segresnet.')
    return model


# Held-out BraTS 2021 patients (never used to train ResEnc; see testing/results/heldout50).
SAMPLE_DIR = ROOT / 'data/external/brats2021_task1'
SAMPLE_IDS = ['BraTS2021_00144', 'BraTS2021_00723', 'BraTS2021_00120', 'BraTS2021_00685', 'BraTS2021_00737',
              'BraTS2021_01050', 'BraTS2021_00305', 'BraTS2021_00418', 'BraTS2021_00156', 'BraTS2021_01014']
SAMPLE_FILES = {'t1': 't1', 't1c': 't1ce', 't2': 't2', 'flair': 'flair', 'expert': 'seg'}
SAMPLE_README = """NeuroScope 3D - held-out test case {pid} (BraTS 2021 Task 1)

Upload in the NeuroScope "Upload MRI" form:
  T1c slot    -> {pid}_t1ce.nii.gz
  T1 slot     -> {pid}_t1.nii.gz
  T2 slot     -> {pid}_t2.nii.gz
  FLAIR slot  -> {pid}_flair.nii.gz
  Expert mask -> {pid}_seg.nii.gz   (set enhancing-tumour label to 4)

The expert mask is only used to score the prediction (Dice), never as model input.
This patient was excluded from ResEnc training. Research demonstration only - not for diagnosis.
"""


def sample_paths(pid):
    if pid not in SAMPLE_IDS:
        raise HTTPException(404, 'Unknown sample case.')
    paths = {k: SAMPLE_DIR / pid / f'{pid}_{suffix}.nii.gz' for k, suffix in SAMPLE_FILES.items()}
    if not all(p.is_file() for p in paths.values()):
        raise HTTPException(404, f'Sample case {pid} is not downloaded on this machine.')
    return paths


@app.get('/api/samples')
def samples():
    return [{'id': pid, 'available': all((SAMPLE_DIR / pid / f'{pid}_{s}.nii.gz').is_file() for s in SAMPLE_FILES.values())}
            for pid in SAMPLE_IDS]


@app.get('/api/samples/{pid}/download')
def sample_download(pid: str):
    import io, zipfile
    from fastapi.responses import Response
    paths = sample_paths(pid)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as archive:
        for path in paths.values():
            archive.write(path, f'NeuroScope_TestCase_{pid}/{path.name}')
        archive.writestr(f'NeuroScope_TestCase_{pid}/HOW_TO_UPLOAD.txt', SAMPLE_README.format(pid=pid))
    name = f'NeuroScope_TestCase_{pid}.zip'
    return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})


@app.post('/api/samples/{pid}/run')
def sample_run(pid: str, model: str = Query('ensemble')):
    model_name = validate_model(model)
    paths = sample_paths(pid)
    expert = paths.pop('expert')
    uid = new_job()
    POOL.submit(execute, uid, paths, expert, 4, f'{pid} · unseen test case ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


@app.post('/api/demo')
def demo(model: str = Query('ensemble'), case: str = Query('00000')):
    model_name = validate_model(model)
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

    expert_file = expert if expert.exists() else None
    demo_et = case_info.get('et_label', 4)
    if expert_file:
        try:
            demo_et = detect_et_label(expert_file)
        except Exception:
            demo_et = case_info.get('et_label', 4)

    uid = new_job()
    POOL.submit(
        execute,
        uid,
        paths,
        expert_file,
        demo_et,
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
    model_name = validate_model(model)
    if expert_et not in (3, 4):
        raise HTTPException(400, 'Expert ET label must be 3 or 4.')
    files = {'t1c': t1c, 't1': t1, 't2': t2, 'flair': flair}
    if expert and expert.filename:
        files['expert'] = expert
    try:
        validate_upload_filenames(files)
    except (ValueError, InputValidationError) as e:
        raise HTTPException(400, str(e))
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
                    if size > 1024 * 1024 * 1024:
                        raise ValueError('Each uploaded file must be under 1 GB.')
                    dest.write(chunk)
            paths[key] = path
    except Exception as e:
        with GUARD:
            JOBS.pop(uid, None)
        shutil.rmtree(RUNS / uid, ignore_errors=True)
        msg = str(e).replace(str(ROOT), '.')
        raise HTTPException(400, msg)
    POOL.submit(execute, uid, paths, paths.get('expert'), expert_et, f'Patient MRI ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


def detect_et_label(mask_path):
    """BraTS 2018-2021 masks use 4 for enhancing tumour, BraTS 2023+ use 3."""
    import nibabel as nib, numpy as np
    labels = set(np.unique(np.asarray(nib.load(str(mask_path)).dataobj)).astype(int).tolist())
    if 4 in labels and 3 in labels:
        raise ValueError(f'Expert mask contains both labels 3 and 4 ({sorted(labels)}); cannot tell the BraTS version.')
    return 3 if 3 in labels else 4


@app.post('/api/analyze-zip')
async def analyze_zip(archive: UploadFile = File(...), model: str = Form('ensemble'),
                      expert: UploadFile | None = File(None)):
    """Zip of the four MRI scans (T1, T1c, T2, FLAIR); an expert mask is a separate, optional file.

    Any seg/mask file inside the zip is ignored so the zip path only ever carries scans.
    """
    import zipfile
    model_name = validate_model(model)
    if not (archive.filename or '').lower().endswith('.zip'):
        raise HTTPException(400, 'Upload a .zip containing the four NIfTI MRI scans.')
    uid = new_job()
    job_dir = RUNS / uid
    try:
        zip_path = job_dir / 'upload.zip'
        size = 0
        with zip_path.open('wb') as dest:
            while chunk := await archive.read(1024 * 1024):
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise ValueError('Zip must be under 1 GB.')
                dest.write(chunk)
        try:
            bundle = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            raise ValueError('That file is not a valid zip archive.')
        with bundle:
            members = zip_intake.detect(bundle)
            paths, detected = {}, {}
            members.pop('expert', None)        # scans only from the zip
            for role, member in members.items():
                suffix = '.nii.gz' if member.filename.lower().endswith('.nii.gz') else '.nii'
                target = job_dir / (role + suffix)  # fixed name: never the archive's own path
                with bundle.open(member) as src, target.open('wb') as dest:
                    shutil.copyfileobj(src, dest, 1024 * 1024)
                paths[role] = target
                detected[role] = Path(member.filename).name
        zip_path.unlink(missing_ok=True)
        expert_path = None
        if expert is not None and expert.filename:
            if not expert.filename.lower().endswith(('.nii', '.nii.gz')):
                raise ValueError('The expert mask must be a NIfTI file (.nii or .nii.gz).')
            expert_path = job_dir / ('expert_upload' + ('.nii.gz' if expert.filename.lower().endswith('.nii.gz') else '.nii'))
            size = 0
            with expert_path.open('wb') as dest:
                while chunk := await expert.read(1024 * 1024):
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise ValueError('Expert mask must be under 1 GB.')
                    dest.write(chunk)
            detected['expert'] = Path(expert.filename).name
        expert = expert_path
        expert_et = detect_et_label(expert) if expert else 4
    except Exception as e:
        with GUARD:
            JOBS.pop(uid, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(e).replace(str(ROOT), '.'))
    stem = Path(archive.filename).stem.replace('NeuroScope_TestCase_', '')[:40]
    POOL.submit(execute, uid, paths, expert, expert_et, f'{stem} · zip upload ({model_name.upper()})', model_name=model_name)
    return {'id': uid, 'detected': detected, 'expert_et': expert_et if expert else None}


@app.get('/api/jobs/{uid}')
def job(uid: str):
    if not uid.isalnum() or len(uid) > 64 or uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    job_dir = RUNS / uid
    cutoff = time.time() - (RETENTION_HOURS * 3600.0)
    if is_run_expired(job_dir, cutoff):
        sweep_expired_runs()
        raise HTTPException(404, 'Analysis has expired under the 24-hour retention policy.')
    return JOBS[uid]


def review_metadata(uid):
    if uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    if JOBS[uid]['status'] != 'complete':
        raise HTTPException(409, 'Wait for the analysis to complete before reviewing it.')
    folder = RUNS / uid
    result = JOBS[uid]['result']
    affine, shape = result.get('affine'), result.get('shape')
    # Older runs do not have geometry in their report; read the header only.
    if affine is None and (folder / 't1c.nii.gz').exists():
        import nibabel as nib
        reference = nib.load(str(folder / 't1c.nii.gz'))
        affine, shape = reference.affine.tolist(), list(reference.shape)
    return dict(affine=affine, shape=shape,
                available_modalities=[m for m in ('t1', 't1c', 't2', 'flair') if (folder / f'{m}.nii.gz').exists()],
                input_hashes=result.get('input_hashes'), has_expert=(folder / 'expert.nii.gz').exists())


@app.get('/api/jobs/{uid}/review')
def get_review(uid: str):
    metadata = review_metadata(uid)
    return {**load_review(RUNS / uid), 'metadata': metadata}


@app.put('/api/jobs/{uid}/review')
def put_review(uid: str, document: dict = Body(...)):
    metadata = review_metadata(uid)
    try:
        return {**save_review(RUNS / uid, document, metadata), 'metadata': metadata}
    except ReviewConflict as error:
        raise HTTPException(409, str(error))
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(400, str(error))


def completed_run_dir(uid: str) -> Path:
    """Run folder for a finished, unexpired analysis (same guards as file downloads)."""
    if not uid.isalnum() or len(uid) > 64 or uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    if JOBS[uid].get('status') != 'complete':
        raise HTTPException(409, 'Analysis is not complete yet.')
    job_dir = (RUNS / uid).resolve()
    if not job_dir.is_relative_to(RUNS.resolve()):
        raise HTTPException(403, 'Path traversal forbidden.')
    if is_run_expired(job_dir, time.time() - RETENTION_HOURS * 3600.0):
        sweep_expired_runs()
        raise HTTPException(404, 'Analysis has expired under the 24-hour retention policy.')
    return job_dir


@app.get('/api/jobs/{uid}/geometry')
def job_geometry(uid: str):
    try:
        return slices.geometry(completed_run_dir(uid))
    except FileNotFoundError:
        raise HTTPException(404, 'Scan files for this analysis are no longer available.')


@app.get('/api/jobs/{uid}/quality')
def job_quality(uid: str):
    run_dir = completed_run_dir(uid)
    try:
        return {'checks': report.quality_checks(run_dir, JOBS[uid].get('result') or {})}
    except FileNotFoundError:
        raise HTTPException(404, 'Segmentation files for this analysis are no longer available.')


@app.get('/api/jobs/{uid}/expert-comparison')
def job_expert_comparison(uid: str):
    run_dir = completed_run_dir(uid)
    try:
        result = report.expert_comparison(run_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))
    if result is None:
        raise HTTPException(404, 'No expert mask was supplied for this analysis.')
    return result


@app.get('/api/jobs/{uid}/report.html')
def job_report(uid: str):
    from fastapi.responses import HTMLResponse
    run_dir = completed_run_dir(uid)
    try:
        page = report.render_report(run_dir, JOBS[uid])
    except FileNotFoundError:
        raise HTTPException(404, 'Files for this analysis are no longer available.')
    return HTMLResponse(page, headers={'Content-Security-Policy': "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"})


@app.get('/api/jobs/{uid}/slice.png')
def job_slice(uid: str, modality: str = Query('flair'), axis: str = Query('axial'), index: int = Query(..., ge=0, le=4096),
              overlay: str = Query('none'), size: int = Query(128, ge=32, le=512)):
    from fastapi.responses import Response
    run_dir = completed_run_dir(uid)
    try:
        png = slices.render(run_dir, modality, axis, index, overlay, size)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return Response(png, media_type='image/png', headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/api/files/{uid}/{name}')
def file(uid: str, name: str):
    allowed = {
        'flair.nii.gz', 't1c.nii.gz', 't1.nii.gz', 't2.nii.gz', 'segmentation.nii.gz', 'expert.nii.gz',
        'uncertainty.nii.gz', 'brain.obj', 'tumor.obj', 'core.obj', 'et.obj', 'report.json'
    }
    if not uid.isalnum() or len(uid) > 64 or uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    job_dir = (RUNS / uid).resolve()
    cutoff = time.time() - (RETENTION_HOURS * 3600.0)
    if is_run_expired(job_dir, cutoff):
        sweep_expired_runs()
        raise HTTPException(404, 'Analysis has expired under the 24-hour retention policy.')
    if name not in allowed:
        raise HTTPException(404, 'Artifact not found.')

    runs_root = RUNS.resolve()
    if not job_dir.is_relative_to(runs_root):
        raise HTTPException(403, 'Path traversal forbidden.')

    path = (job_dir / name).resolve()
    if not path.is_relative_to(job_dir) or not path.exists():
        raise HTTPException(404, 'Artifact not found.')
    case = str((JOBS.get(uid, {}).get('result') or {}).get('case_name', 'case'))
    model = str((JOBS.get(uid, {}).get('result') or {}).get('model_name', 'model'))
    slug = ''.join(c if c.isalnum() else '_' for c in case.split('·')[0].strip())[:40].strip('_') or 'case'
    return FileResponse(path, filename=f'NeuroScope_{slug}_{model}_{name}')


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
