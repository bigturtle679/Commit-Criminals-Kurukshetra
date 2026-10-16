from pathlib import Path
import json, uuid, threading, shutil, traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pipeline import ROOT, run_case

app = FastAPI(title='NeuroScope Local · 3D Brain Tumor Segmentation')
RUNS = ROOT / 'work/runs'
RUNS.mkdir(parents=True, exist_ok=True)
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)
GUARD = threading.Lock()
app.mount('/static', StaticFiles(directory=ROOT / 'src/web'), name='static')


@app.get('/')
def index():
    return FileResponse(ROOT / 'src/web/index.html')


@app.get('/api/status')
def status():
    import torch
    return {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable',
        'cuda_available': torch.cuda.is_available(),
        'demo_ready': len(list((ROOT / 'data/demo').glob('*.nii.gz'))) >= 5,
        'models_available': [
            {'id': 'swinunetr', 'name': 'SwinUNETR (Vision Transformer)', 'badge': 'SOTA Foundation Model', 'speed': '3.1s'},
            {'id': 'segresnet', 'name': 'SegResNet (3D CNN Autoencoder)', 'badge': 'BraTS Winner', 'speed': '1.2s'}
        ],
        'default_model': 'swinunetr'
    }


@app.get('/api/jobs')
def jobs():
    return list(JOBS.values())


def execute(job_id, paths, expert, expert_et, name, model_name='swinunetr'):
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
        (RUNS / uid).mkdir()
        JOBS[uid] = {'id': uid, 'status': 'queued', 'message': 'Queued for local RTX 4060 GPU'}
        return uid


@app.post('/api/demo')
def demo(model: str = Query('swinunetr')):
    folder = ROOT / 'data/demo'
    case = 'BraTS-GLI-00000-000'
    paths = {k: folder / f'{case}-{s}.nii.gz' for k, s in [('t1c', 't1c'), ('t1', 't1n'), ('t2', 't2w'), ('flair', 't2f')]}
    expert = folder / f'{case}-seg.nii.gz'
    if not all(p.exists() for p in [*paths.values(), expert]):
        raise HTTPException(404, 'Demo MRI volumes are missing in data/demo.')
    uid = new_job()
    model_name = model if model in ('swinunetr', 'segresnet') else 'swinunetr'
    POOL.submit(execute, uid, paths, expert, 3, f'BraTS Demo · GLI-00000 ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


@app.post('/api/analyze')
async def analyze(
    t1c: UploadFile = File(...),
    t1: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    expert: UploadFile | None = File(None),
    expert_et: int = Form(4),
    model: str = Form('swinunetr')
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
        raise HTTPException(400, str(e))
    model_name = model if model in ('swinunetr', 'segresnet') else 'swinunetr'
    POOL.submit(execute, uid, paths, paths.get('expert'), expert_et, f'Patient MRI ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


@app.get('/api/jobs/{uid}')
def job(uid: str):
    if uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
    return JOBS[uid]


@app.get('/api/files/{uid}/{name}')
def file(uid: str, name: str):
    allowed = {'flair.nii.gz', 't1c.nii.gz', 'segmentation.nii.gz', 'expert.nii.gz', 'brain.obj', 'tumor.obj', 'core.obj', 'report.json'}
    if uid not in JOBS or name not in allowed:
        raise HTTPException(404)
    path = RUNS / uid / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, filename=name)


for p in RUNS.glob('*/job.json'):
    try:
        j = json.loads(p.read_text())
        if j['status'] in ('complete', 'error'):
            JOBS[j['id']] = j
    except (ValueError, KeyError):
        pass
