"""Score SwinUNETR, ResEnc and the 80:20 ensemble on held-out BraTS 2021 patients.

Each model runs once per case; the ensemble fuses those same probabilities with the
weights, hierarchy and threshold from configs/ensemble.json (as predict_ensemble does).
Single models use the pipeline's label conversion without hierarchy enforcement.

  .venv/bin/python scripts/eval_ensemble_heldout.py --split work/resenc_training/split.json \
      --resenc work/resenc_training/latest.pt --out testing/results/heldout50
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ensemble_inference import enforce_hierarchy  # noqa: E402
from scripts.models.model_loader import load_resenc, load_swinunetr  # noqa: E402
from src.metrics import evaluate_case  # noqa: E402

RAW = ['t1ce', 't1', 't2', 'flair']  # inference raw order: t1c, t1, t2, flair
RAW_KEYS = ['t1c', 't1', 't2', 'flair']


def to_labels(tc, wt, et):
    return np.where(et, 4, np.where(tc, 1, np.where(wt, 2, 0))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=Path, default=ROOT / 'work/resenc_training/split.json')
    ap.add_argument('--data', type=Path, default=ROOT / 'data/external/brats2021_task1')
    ap.add_argument('--resenc', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ROOT / 'configs/ensemble.json').read_text())
    wa, wb = cfg['weights']['model_a'], cfg['weights']['model_b']
    thr = cfg['postprocessing']['threshold']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    swin = load_swinunetr(device=dev)[0]
    resenc, _, resenc_path = load_resenc(weights_path=str(args.resenc), device=dev)
    ck = torch.load(args.resenc, map_location='cpu', weights_only=False)
    cases = json.loads(args.split.read_text())['test']
    if args.limit:
        cases = cases[:args.limit]
    norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    results = []
    for n, pid in enumerate(cases, 1):
        t0 = time.time()
        d = args.data / pid
        imgs = [nib.as_closest_canonical(nib.load(d / f'{pid}_{m}.nii.gz')) for m in RAW]
        gt_img = nib.as_closest_canonical(nib.load(d / f'{pid}_seg.nii.gz'))
        raw = np.stack([i.get_fdata(dtype=np.float32) for i in imgs])
        pos = np.where(np.any(raw != 0, axis=0))
        sl = tuple(slice(max(0, int(p.min()) - 4), min(raw.shape[i + 1], int(p.max()) + 5)) for i, p in enumerate(pos))
        normed = norm(raw).as_tensor()[(slice(None),) + sl]
        probs = {}
        for key, model in (('model_a', swin), ('model_b', resenc)):
            spec = cfg[key]
            x = normed[[RAW_KEYS.index(k) for k in spec['channel_order']]].unsqueeze(0).contiguous()
            with torch.inference_mode():
                logits = sliding_window_inference(x.to(dev), tuple(spec['roi_size']), 1, model,
                                                  overlap=spec['sliding_window_overlap'], mode=spec['mode'],
                                                  sw_device=dev, device='cpu')
            probs[key] = torch.sigmoid(logits[0].float())
        fused = torch.stack(enforce_hierarchy(*(wa * probs['model_a'] + wb * probs['model_b'])))
        preds = {'swinunetr': (probs['model_a'] >= 0.5).numpy(), 'resenc': (probs['model_b'] >= 0.5).numpy(),
                 'ensemble_80_20': (fused >= thr).numpy()}
        gt = np.asarray(gt_img.dataobj).astype(np.uint8)
        row = {'patient_id': pid, 'seconds': None}
        for name, p in preds.items():
            full = np.zeros(gt.shape, np.uint8)
            full[sl] = to_labels(*p)
            m = evaluate_case(full, gt, gt_img.affine, enhancing_label=4)
            row[name] = {r: {k: m[r][k] for k in ('dice', 'sensitivity', 'precision', 'pred_volume_ml', 'gt_volume_ml')} for r in ('WT', 'TC', 'ET')}
        row['seconds'] = round(time.time() - t0, 2)
        results.append(row)
        print(f"[{n}/{len(cases)}] {pid} " + ' | '.join(
            f"{k}: WT {row[k]['WT']['dice']:.3f} TC {row[k]['TC']['dice']:.3f} ET {row[k]['ET']['dice']:.3f}"
            for k in preds) + f" ({row['seconds']}s)", flush=True)

    models = ['swinunetr', 'resenc', 'ensemble_80_20']
    summary = {m: {r: {'mean_dice': round(float(np.mean([x[m][r]['dice'] for x in results])), 4),
                       'median_dice': round(float(np.median([x[m][r]['dice'] for x in results])), 4)}
                   for r in ('WT', 'TC', 'ET')} for m in models}
    paired = {r: {'ensemble_better': sum(x['ensemble_80_20'][r]['dice'] > x['swinunetr'][r]['dice'] for x in results),
                  'ensemble_worse': sum(x['ensemble_80_20'][r]['dice'] < x['swinunetr'][r]['dice'] for x in results),
                  'mean_delta': round(float(np.mean([x['ensemble_80_20'][r]['dice'] - x['swinunetr'][r]['dice'] for x in results])), 4)}
              for r in ('WT', 'TC', 'ET')}
    meta = {'cases': len(results), 'seed': 42, 'resenc_checkpoint': str(args.resenc), 'resenc_iteration': ck.get('iteration'),
            'resenc_note': 'early-stopped checkpoint' if ck.get('iteration', 0) < 15000 else 'full schedule',
            'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu',
            'training_overlap': {'swinunetr': 'known_training_overlap (BrainSegFounder trained on BraTS 2021)',
                                 'resenc': 'excluded_from_known_training (held out by split.json)'}}
    (args.out / 'results.json').write_text(json.dumps({'meta': meta, 'summary': summary, 'paired_vs_swin': paired, 'cases': results}, indent=1))
    print(json.dumps({'summary': summary, 'paired_vs_swin': paired}, indent=1))


if __name__ == '__main__':
    main()
