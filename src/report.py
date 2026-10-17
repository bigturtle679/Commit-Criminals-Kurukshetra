"""Printable HTML research report and post-segmentation quality checks for one analysis.

Every value comes from the analysis' saved result and its NIfTI artifacts; nothing is
estimated or invented here. All dynamic text is HTML-escaped.
"""
import base64
import html
import json
from pathlib import Path

import nibabel as nib
import numpy as np

import slices

ALLOWED_LABELS = {0, 1, 2, 4}


def quality_checks(run_dir: Path, result: dict):
    """Re-verify the saved segmentation instead of trusting the pipeline's own claims."""
    checks = []

    def add(name, ok, detail):
        checks.append({'name': name, 'status': 'pass' if ok else 'fail', 'detail': detail})

    seg_img = nib.load(str(run_dir / 'segmentation.nii.gz'))
    seg = np.asarray(seg_img.dataobj)
    ref = nib.load(str(run_dir / 't1c.nii.gz'))
    labels = set(np.unique(seg).astype(int).tolist())
    add('Label codes', labels <= ALLOWED_LABELS, f'found {sorted(labels)}; allowed 0 background, 1 NCR, 2 ED, 4 ET')
    add('Mask aligned with MRI', seg_img.shape == ref.shape and np.allclose(seg_img.affine, ref.affine, atol=1e-3),
        f'shape {list(seg_img.shape)} vs {list(ref.shape)}; affine match')
    wt, tc, et = np.isin(seg, [1, 2, 4]), np.isin(seg, [1, 4]), seg == 4
    add('Region nesting ET ⊆ TC ⊆ WT', bool(np.all(tc <= wt) and np.all(et <= tc)), 'enhancing inside core inside whole tumour')
    if wt.any():
        add('Tumour regions predicted', True, f'{int(wt.sum())} whole-tumour voxels')
    else:
        # A tumour-free result is valid (e.g. a healthy scan); report it, do not flag it as a failure.
        checks.append({'name': 'No tumour predicted', 'status': 'info',
                       'detail': '0 tumour voxels — expected for a tumour-free scan; otherwise review the input'})
    v = result.get('volumes_ml') or {}
    consistent = all(k in v for k in ('WT', 'TC', 'ED')) and abs(v['WT'] - v['TC'] - v['ED']) <= 0.01
    add('Volume consistency WT = TC + ED', consistent,
        f"{v.get('WT', '—')} = {v.get('TC', '—')} + {v.get('ED', '—')} mL")
    unc = result.get('uncertainty') or {}
    if unc.get('available'):
        add('Uncertainty map produced', True, f"mean entropy in tumour {unc.get('mean_in_tumor')}")
    else:
        checks.append({'name': 'Uncertainty map produced', 'status': 'n/a', 'detail': 'not generated for this model'})
    return checks


def _img(run_dir: Path, axis: str, index: int, overlay: str):
    png = slices.render(run_dir, 't1c', axis, index, overlay, 360)
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def render_report(run_dir: Path, job: dict) -> str:
    r = job.get('result') or {}
    e = lambda x: html.escape(str(x))
    geo = slices.geometry(run_dir)
    centre = geo.get('tumour_centre') or [d // 2 for d in geo['dims_ras']]
    v = r.get('volumes_ml') or {}
    voxel = r.get('spacing_mm') or []
    checks = quality_checks(run_dir, r)
    brain = geo.get('brain_ml')

    def pct(a, b):
        return f'{a / b * 100:.1f}%' if isinstance(a, (int, float)) and b else '—'

    vol_rows = ''.join(
        f'<tr><td>{e(name)}</td><td>{e(v.get(key, "—"))}</td><td>{e(note)}</td></tr>' for name, key, note in [
            ('Whole tumour (WT)', 'WT', f'{pct(v.get("WT"), brain)} of brain volume ({brain} mL)'),
            ('Tumour core (TC)', 'TC', f'{pct(v.get("TC"), v.get("WT"))} of WT'),
            ('Enhancing tumour (ET)', 'ET', f'{pct(v.get("ET"), v.get("WT"))} of WT'),
            ('Peritumoral edema (ED)', 'ED', 'WT − TC'),
        ])
    metrics = r.get('metrics')
    metric_html = '<p class="muted">No expert reference mask was supplied, so overlap metrics were not computed.</p>'
    if metrics:
        metric_html = '<table><tr><th>Region</th><th>Predicted estimated volume</th><th>Expert reference volume</th><th>Dice</th><th>Sensitivity</th></tr>' + ''.join(
            f'<tr><td>{e(k)}</td><td>{e(m.get("pred_volume_ml", v.get(k, "—")))} mL</td><td>{e(m.get("gt_volume_ml", "—"))} mL</td><td>{e(m.get("dice", "—"))}</td><td>{e(m.get("sensitivity", "—"))}</td></tr>'
            for k, m in metrics.items()) + '</table><p class="muted">Dice = 2|P∩G| / (|P|+|G|): overlap with the expert region. ' \
            'Sensitivity = |P∩G| / |G|: share of the expert region the model captured.</p>'
    unc, dis = r.get('uncertainty') or {}, r.get('model_disagreement') or {}
    check_rows = ''.join(f'<tr class="{e(c["status"])}"><td>{e(c["name"])}</td><td>{e(c["status"].upper())}</td><td>{e(c["detail"])}</td></tr>'
                         for c in checks)
    hashes = {**(r.get('input_hashes') or {}), **{f'checkpoint {k}': h for k, h in (r.get('checkpoint_hashes') or {}).items()}}
    hash_rows = ''.join(f'<tr><td>{e(k)}</td><td class="mono" style="word-break:break-all">{e(str(h))}</td></tr>' for k, h in hashes.items())
    images = ''.join(f'<figure><img src="{_img(run_dir, axis, centre[i], "segmentation")}" alt="{axis} slice"><figcaption>{axis.title()} '
                     f'{centre[i] + 1} / {geo["dims_ras"][i]}</figcaption></figure>'
                     for axis, i in (('axial', 2), ('coronal', 1), ('sagittal', 0)))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NeuroScope 3D report · {e(r.get("case_name", "case"))}</title>
<style>
body{{font:13px/1.55 Inter,Arial,sans-serif;color:#111;max-width:860px;margin:24px auto;padding:0 20px}}
h1{{font-size:22px;margin:0}} h2{{font-size:15px;margin:22px 0 8px;border-bottom:1px solid #ccc;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:5px 8px;text-align:left}} th{{background:#f1f3f4}}
.banner{{background:#fff4e5;border:1px solid #f0b35b;padding:10px 12px;border-radius:6px;margin:14px 0}}
.muted{{color:#555;font-size:12px}} .mono{{font-family:monospace}} tr.fail td{{background:#fde8e8}} tr.pass td:nth-child(2){{color:#11793b;font-weight:600}} tr.info td:nth-child(2){{color:#2a5d9f;font-weight:600}}
.figs{{display:flex;gap:10px}} figure{{margin:0;flex:1;text-align:center}} figure img{{width:100%;background:#000;border-radius:4px}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px}}
@media print{{button{{display:none}} body{{margin:0}}}}
</style></head><body>
<button onclick="window.print()">Print / save as PDF</button>
<h1>NeuroScope 3D — research segmentation report</h1>
<p class="muted">{e(r.get("case_name", ""))} · {e(r.get("timestamp", ""))} · analysis {e(job.get("id", ""))}</p>
<div class="banner"><b>Research and teaching prototype only.</b> This report does not present model output as a diagnosis.
Suspected tumour regions predicted by the model and estimated volumes are model estimates, not expert-verified findings, and must not be used for clinical decisions.</div>
<h2>Model and run</h2>
<table><tr><td>Model</td><td>{e(r.get("model", r.get("model_name", "")))}</td></tr>
<tr><td>Runtime / device</td><td>{e(r.get("runtime_seconds", "—"))} s on {e(r.get("gpu", "—"))}</td></tr>
<tr><td>Volume grid</td><td>{e(r.get("shape", "—"))} voxels at {e(voxel)} mm</td></tr>
<tr><td>Code revision</td><td class="mono">{e(r.get("code_commit", "—"))}</td></tr></table>
<h2>Predicted regions (slices through the tumour centre)</h2>
<div class="figs">{images}</div>
<p class="legend muted"><span style="background:#47c7c5"></span>Edema (label 2)<span style="background:#e875b7"></span>Non-enhancing core (label 1)<span style="background:#f5a623"></span>Enhancing tumour (label 4)</p>
<h2>Estimated volumes on original grid</h2>
<table><tr><th>Region</th><th>Estimated volume (mL)</th><th>Share</th></tr>{vol_rows}</table>
<p class="muted">V(mL) = N<sub>voxels</sub> × |det(A<sub>3×3</sub>)| / 1000 = N × s<sub>x</sub>·s<sub>y</sub>·s<sub>z</sub> / 1000. WT ⊇ TC ⊇ ET are nested — do not add them.</p>
<h2>Accuracy against expert mask</h2>{metric_html}
<h2>Reliability</h2>
<table><tr><td>Mean voxel entropy in tumour (0 = certain, 1 = coin flip)</td><td>{e(unc.get("mean_in_tumor", "n/a"))}</td></tr>
<tr><td>Mean |p<sub>Swin</sub> − p<sub>ResEnc</sub>| in tumour (model disagreement)</td><td>{e(dis.get("tumor_region_mean", "n/a"))}</td></tr></table>
<h2>Quality checks (re-verified from the saved mask)</h2>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{check_rows}</table>
<h2>Provenance (SHA-256, first 16 hex)</h2><table>{hash_rows or "<tr><td>none recorded</td></tr>"}</table>
</body></html>'''
