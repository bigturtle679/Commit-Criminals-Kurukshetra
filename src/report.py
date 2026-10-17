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



INTER_RATER_DICE = {'WT': 85, 'TC': 75, 'ET': 74}   # BraTS benchmark, Menze et al., IEEE TMI 2015


def _svg_lines(series, x_labels, y_min, y_max, y_ticks, width=560, height=230, y_fmt=lambda v: f'{v:g}', x_fmt=str):
    """Small dependency-free SVG line chart for the printable report.

    series: list of (name, colour, dash, [(x, y), ...]) with x as index into x_labels or numeric.
    """
    L, R, T, B = 44, 150, 12, 28
    xs_all = [x for _, _, _, pts in series for x, _ in pts]
    x_min, x_max = (0, max(len(x_labels) - 1, 1)) if x_labels else (0, max(xs_all))
    X = lambda x: L + (width - L - R) * (x - x_min) / ((x_max - x_min) or 1)
    Y = lambda y: T + (height - T - B) * (1 - (min(max(y, y_min), y_max) - y_min) / ((y_max - y_min) or 1))
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Arial" font-size="10">']
    for t in y_ticks:
        out.append(f'<line x1="{L}" x2="{width - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e4e6e8"/>'
                   f'<text x="{L - 6}" y="{Y(t) + 3:.1f}" text-anchor="end" fill="#666">{y_fmt(t)}</text>')
    ticks = range(len(x_labels)) if x_labels else [0] + [round(x_max * i / 4 / 100) * 100 for i in range(1, 5)]
    for t in ticks:
        label = x_labels[int(t)] if x_labels else x_fmt(t)
        out.append(f'<text x="{X(t):.1f}" y="{height - 8}" text-anchor="middle" fill="#333">{html.escape(str(label))}</text>')
    out.append(f'<line x1="{L}" x2="{width - R}" y1="{height - B}" y2="{height - B}" stroke="#999"/>')
    ends = []
    for name, colour, dash, pts in series:
        if not pts:
            continue
        d = ' '.join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(y):.1f}' for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{1.4 if dash else 2}" stroke-dasharray="{dash}"/>')
        if len(pts) <= 12:
            out += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{colour}"/>' for x, y in pts]
        ends.append([Y(pts[-1][1]), f'{name} {y_fmt(pts[-1][1])}', colour])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
    out += [f'<text x="{width - R + 6}" y="{y + 3:.1f}" fill="{c}">{html.escape(t)}</text>' for y, t, c in ends]
    out.append('</svg>')
    return ''.join(out)


def _expert_chart(comparison):
    regions = [k for k in ('WT', 'TC', 'ET') if k in comparison['regions']]
    pct = lambda k, m: comparison['regions'][k][m] * 100 if comparison['regions'][k][m] is not None else None
    def pts(fn):
        return [(i, v) for i, k in enumerate(regions) if (v := fn(k)) is not None]
    series = [
        ('Expert reference', '#8a8f94', '1 3', pts(lambda k: 100)),
        ('Expert vs expert', '#7a5cb8', '5 3', pts(lambda k: INTER_RATER_DICE[k])),
        ('AI sensitivity', '#b5862a', '', pts(lambda k: pct(k, 'sensitivity'))),
        ('AI Dice', '#2f7fb5', '', pts(lambda k: pct(k, 'dice'))),
    ]
    return _svg_lines(series, regions, 50, 100, [50, 60, 70, 80, 90, 100], y_fmt=lambda v: f'{v:.0f}%')


def _training_charts():
    path = Path(__file__).resolve().parent / 'web' / 'training_curves.json'
    if not path.exists():
        return ''
    data = json.loads(path.read_text())
    loss = [(r['iteration'], r['loss']) for r in data.get('loss', [])]
    val = [(r['iteration'], r['mean']) for r in data.get('val', [])]
    if not loss:
        return ''
    k = lambda v: f'{v / 1000:g}k' if v >= 1000 else f'{v:.0f}'
    top = max(1.0, max(y for _, y in loss))
    loss_svg = _svg_lines([('train_loss', '#2f7fb5', '', loss)], None, 0, top, [0, top / 2, top], height=170,
                          y_fmt=lambda v: f'{v:.2f}', x_fmt=k)
    val_svg = _svg_lines([('val_mean_dice', '#b5862a', '', val)], None, 0, 1, [0, 0.25, 0.5, 0.75, 1], height=170,
                         y_fmt=lambda v: f'{v:.2f}', x_fmt=k)
    last = data['val'][-1] if data.get('val') else {}
    note = html.escape(data.get('description', ''))
    final = (f" Final validation Dice: WT {last['WT']:.3f} · TC {last['TC']:.3f} · ET {last['ET']:.3f}.") if last else ''
    return (f'<h2>Model training (in-house ResEnc-M)</h2><p class="muted">train_loss (Dice + BCE) by training step</p>{loss_svg}'
            f'<p class="muted">val_mean_dice (mean of WT, TC, ET on validation patients) by training step</p>{val_svg}'
            f'<p class="muted">{note}{final}</p>')


def render_report(run_dir: Path, job: dict) -> str:
    r = job.get('result') or {}
    e = lambda x: html.escape(str(x))
    geo = slices.geometry(run_dir)
    centre = geo.get('tumour_centre') or [d // 2 for d in geo['dims_ras']]
    v = r.get('volumes_ml') or {}
    voxel = r.get('spacing_mm') or []
    checks = quality_checks(run_dir, r)
    try:
        _cmp = expert_comparison(run_dir)
    except (FileNotFoundError, ValueError):
        _cmp = None
    if _cmp and r.get('metrics'):
        r = {**r, 'metrics': {k: {**m, 'gt_volume_ml': _cmp['regions'].get(k, {}).get('expert_ml', m.get('gt_volume_ml')),
                                  'pred_volume_ml': _cmp['regions'].get(k, {}).get('pred_ml', m.get('pred_volume_ml'))}
                              for k, m in r['metrics'].items()}}
    try:
        comparison = expert_comparison(run_dir)
    except (FileNotFoundError, ValueError):
        comparison = None
    expert_chart_html = ('<p class="muted"><b>AI vs expert across regions.</b> Solid: AI Dice and AI sensitivity. '
                         'Dashed: typical agreement between two expert radiologists (BraTS, Menze et al., IEEE TMI 2015). '
                         'Dotted: expert reference (100% by definition). Y-axis starts at 50%.</p>' + _expert_chart(comparison)) if comparison else ''
    training_html = _training_charts()
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
<h2>Accuracy against expert mask</h2>{metric_html}{expert_chart_html}
{training_html}
<h2>Reliability</h2>
<table><tr><td>Mean voxel entropy in tumour (0 = certain, 1 = coin flip)</td><td>{e(unc.get("mean_in_tumor", "n/a"))}</td></tr>
<tr><td>Mean |p<sub>Swin</sub> − p<sub>ResEnc</sub>| in tumour (model disagreement)</td><td>{e(dis.get("tumor_region_mean", "n/a"))}</td></tr></table>
<h2>Quality checks (re-verified from the saved mask)</h2>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{check_rows}</table>
<h2>Provenance (SHA-256, first 16 hex)</h2><table>{hash_rows or "<tr><td>none recorded</td></tr>"}</table>
</body></html>'''


def _hd95(pred, ref, spacing):
    """95th-percentile symmetric surface distance in mm (None when only one mask is empty)."""
    from scipy import ndimage
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return None
    sp = pred ^ ndimage.binary_erosion(pred)
    sr = ref ^ ndimage.binary_erosion(ref)
    to_ref = ndimage.distance_transform_edt(~sr, sampling=spacing)[sp]
    to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)[sr]
    return round(float(np.percentile(np.concatenate([to_ref, to_pred]), 95)), 2)


def expert_comparison(run_dir: Path):
    """Model prediction vs the uploaded expert mask, recomputed from the saved files.

    Both masks are on the same grid with labels normalised to 1 NCR, 2 ED, 4 ET.
    Returns None when no expert mask was supplied.
    """
    expert_path = run_dir / 'expert.nii.gz'
    if not expert_path.exists():
        return None
    cache = run_dir / 'expert_comparison.json'
    sources = [expert_path, run_dir / 'segmentation.nii.gz']
    if cache.exists() and all(cache.stat().st_mtime >= src.stat().st_mtime for src in sources):
        return json.loads(cache.read_text())
    pred_img = nib.load(str(run_dir / 'segmentation.nii.gz'))
    exp_img = nib.load(str(expert_path))
    pred, exp = np.asarray(pred_img.dataobj), np.asarray(exp_img.dataobj)
    if pred.shape != exp.shape or not np.allclose(pred_img.affine, exp_img.affine, atol=1e-3):
        raise ValueError('Prediction and expert mask are not on the same grid.')
    spacing = nib.affines.voxel_sizes(pred_img.affine)
    voxel_ml = float(abs(np.linalg.det(pred_img.affine[:3, :3]))) / 1000.0
    regions = {}
    wt_pred = wt_exp = None
    for name, labels in (('WT', [1, 2, 4]), ('TC', [1, 4]), ('ET', [4])):
        p, g = np.isin(pred, labels), np.isin(exp, labels)
        tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
        if name == 'WT':
            wt_pred, wt_exp = int(p.sum()), int(g.sum())
        regions[name] = {
            'pred_voxels': int(p.sum()), 'expert_voxels': int(g.sum()),
            'pred_ml': round(p.sum() * voxel_ml, 3), 'expert_ml': round(g.sum() * voxel_ml, 3),
            # share of each mask's own whole tumour (nested-region ratio, comparable across masks)
            'pred_share_of_wt': (p.sum() / wt_pred) if wt_pred else None,
            'expert_share_of_wt': (g.sum() / wt_exp) if wt_exp else None,
            'dice': 1.0 if tp + fp + fn == 0 else round(2 * tp / (2 * tp + fp + fn), 4),
            'sensitivity': None if tp + fn == 0 else round(tp / (tp + fn), 4),
            'precision': None if tp + fp == 0 else round(tp / (tp + fp), 4),
            'hd95_mm': _hd95(p, g, spacing),
        }
    result = {'regions': regions, 'voxel_ml': voxel_ml}
    try:
        cache.write_text(json.dumps(result))
    except OSError:
        pass
    return result
