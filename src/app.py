from pathlib import Path
import json, uuid, threading, shutil, traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pipeline import ROOT, run_case
from review_store import load_review, save_review, ReviewConflict

app = FastAPI(title='NeuroScope Local · 3D Brain Tumor Segmentation')
RUNS = ROOT / 'work/runs'
RUNS.mkdir(parents=True, exist_ok=True)
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)
GUARD = threading.Lock()
app.mount('/static', StaticFiles(directory=ROOT / 'src/web'), name='static')

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
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Apple Silicon (MPS)' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'CPU'
    return {
        'gpu': dev_name,
        'cuda_available': torch.cuda.is_available(),
        'demo_ready': len(list((ROOT / 'data/demo').glob('*.nii.gz'))) >= 5 or len(list((ROOT / 'data/synthetic_demo').glob('*.nii.gz'))) >= 4,
        'models_available': [
            {'id': 'ensemble', 'name': '8:2 Ensemble (SwinUNETR + nnU-Net ResEnc)', 'badge': 'Weighted probability fusion'},
            {'id': 'swinunetr', 'name': 'SwinUNETR (Vision Transformer)', 'badge': 'Transformer baseline'},
            {'id': 'segresnet', 'name': 'SegResNet (3D CNN Autoencoder)', 'badge': 'Convolutional baseline'}
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


def validate_model(model):
    if model not in ('ensemble', 'swinunetr', 'segresnet'):
        raise HTTPException(400, 'Unknown model. Choose ensemble, swinunetr or segresnet.')
    return model


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

    uid = new_job()
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
    model_name = validate_model(model)
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
    POOL.submit(execute, uid, paths, paths.get('expert'), expert_et, f'Patient MRI ({model_name.upper()})', model_name=model_name)
    return {'id': uid}


@app.get('/api/jobs/{uid}')
def job(uid: str):
    if uid not in JOBS:
        raise HTTPException(404, 'Analysis not found.')
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


@app.get('/api/files/{uid}/{name}')
def file(uid: str, name: str):
    allowed = {'flair.nii.gz', 't1c.nii.gz', 't1.nii.gz', 't2.nii.gz', 'segmentation.nii.gz', 'expert.nii.gz', 'brain.obj', 'tumor.obj', 'core.obj', 'report.json'}
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
