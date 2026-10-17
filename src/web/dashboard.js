// Dashboard layer on top of the existing NiiVue viewer: workflow state, display modes,
// slice navigation, filmstrip, modality thumbnails, split view, region callout and the
// measurement rings. All images and numbers come from the active analysis (backend
// endpoints /api/jobs/<id>/geometry and /api/jobs/<id>/slice.png) — nothing is hardcoded.
export const REGION_COLORS = Object.freeze({ edema:'#47c7c5', necrotic:'#e875b7', enhancing:'#f5a623' });

// Backend segmentation labels (pipeline output is normalised to 1/2/4).
export const LABELS = Object.freeze({
  1: { name: 'Non-enhancing core (NC)', color: REGION_COLORS.necrotic, volume: v => Math.max(0, v.TC - v.ET) },
  2: { name: 'Peritumoral edema (ED)', color: REGION_COLORS.edema, volume: v => v.ED },
  4: { name: 'Enhancing tumour (ET)', color: REGION_COLORS.enhancing, volume: v => v.ET },
});

const AXIS_NAMES = ['sagittal', 'coronal', 'axial']; // RAS axis index -> plane name
// Validated (dark surface, CVD-safe) two-series palette for the AI-vs-expert chart.
export const COMPARE_COLORS = Object.freeze({ dice: '#3b93c9', sensitivity: '#b5862a', interrater: '#8e6fc4', reference: '#a7b0b6' });
// Typical agreement between two expert raters (Dice), BraTS benchmark: Menze et al., IEEE TMI 2015.
export const INTER_RATER_DICE = Object.freeze({ WT: 85, TC: 75, ET: 74 });

// Values for the grouped chart: per region, Dice and sensitivity against the expert mask (in %).
export function compareSeries(regions) {
  const pct = v => (v == null ? null : v * 100);
  return ['WT', 'TC', 'ET'].filter(k => regions?.[k]).map(k => ({ region: k, dice: pct(regions[k].dice), sensitivity: pct(regions[k].sensitivity) }));
}

export function allowedOverlay(selected, access, uncertaintyAvailable) {
  if (selected === 'expert') return access.expert ? 'expert' : 'none';
  if (selected === 'uncertainty') return access.prediction && uncertaintyAvailable ? 'uncertainty' : 'none';
  return selected === 'segmentation' && access.prediction ? 'segmentation' : 'none';
}

export function composition(volumes) {
  const { WT, TC, ET, ED } = volumes || {};
  if (![WT,TC,ET,ED].every(Number.isFinite) || WT <= 0 || ET < 0 || TC < ET || WT < TC ||
      ED < 0 || Math.abs(ED - (WT-TC)) > 0.02) return [];
  return [
    { label:'Edema', value:ED, color:REGION_COLORS.edema },
    { label:'Non-enhancing core', value:TC-ET, color:REGION_COLORS.necrotic },
    { label:'Enhancing tumour', value:ET, color:REGION_COLORS.enhancing }
  ].map(part => ({...part, share:part.value/WT}));
}

export function slicePosition(fraction, count) {
  if (!Number.isInteger(count) || count < 1) return null;
  const index = Math.max(0, Math.min(count-1, Math.floor(fraction*count)));
  return { index, count, fraction:(index+0.5)/count };
}

// Map a backend job status message to the workflow stage it belongs to (0 upload, 1 validate, 2 segment).
export function stageFromMessage(message) {
  const m = String(message || '').toLowerCase();
  if (/queued|waiting/.test(m)) return 1;            // upload received, validation next
  if (/validat|loading ensemble|alignment|checking/.test(m)) return 1;
  if (/standardiz|segment|executing|fusing|projecting|synthesiz|mesh|inference|model/.test(m)) return 2;
  return 0; // queued / uploading / anything before validation
}

// Ring percentages with explicit denominators: WT / brain, TC / WT, ET / WT.
export function measureShares(volumes, brainMl) {
  const { WT, TC, ET } = volumes || {};
  const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && b > 0 ? Math.min(1, Math.max(0, a / b)) : null);
  return { WT: pct(WT, brainMl), TC: pct(TC, WT), ET: pct(ET, WT) };
}

// Evenly spaced filmstrip slices that always include the active slice.
export function filmstripIndices(count, active, n = 15) {
  if (!Number.isInteger(count) || count < 1) return [];
  n = Math.min(n, count);
  const step = count / n;
  const picks = Array.from({ length: n }, (_, i) => Math.min(count - 1, Math.floor((i + 0.5) * step)));
  const nearest = picks.reduce((best, v, i) => (Math.abs(v - active) < Math.abs(picks[best] - active) ? i : best), 0);
  picks[nearest] = active;
  return picks;
}

export function bindDashboard({ nv, getState, getJob, getView, changeOverlay, changeBackground, setStatus }) {
  const $ = id => document.getElementById(id);
  const stageButtons = [...document.querySelectorAll('[data-stage]')];
  const modeButtons = [...document.querySelectorAll('[data-display]')];
  const modalityButtons = [...document.querySelectorAll('[data-modality]')];
  let mode = 'overlay';
  let activeStage = 0;
  let geometry = null;          // { dims_ras, brain_ml, voxel_ml, tumour_centre }
  let geometryJob = null;
  let stripKey = '';
  let splitKey = '';
  const grid = document.querySelector('.workspace-grid');

  const job = () => getJob();
  const axisIndex = () => Number($('slice-axis').value);
  const dims = () => nv.volumes[0]?.dimsRAS?.slice(1, 4) || geometry?.dims_ras;
  const sliceUrl = (axis, index, overlay, size, modality = $('select-background').value) =>
    `/api/jobs/${encodeURIComponent(job().id)}/slice.png?modality=${modality}&axis=${axis}&index=${index}&overlay=${overlay}&size=${size}`;

  function currentIndices() {
    const d = dims();
    if (!d || !nv.volumes.length) return null;
    return [0, 1, 2].map(a => slicePosition(nv.scene.crosshairPos[a], d[a]));
  }

  function moveCrosshair(axis, index) {
    const d = dims();
    if (!d) return;
    nv.scene.crosshairPos[axis] = (index + 0.5) / d[axis];
    nv.drawScene();
    nv.onLocationChange?.({ mm: Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) });
  }

  // ---------- region under the crosshair (real label value from the loaded segmentation) ----------
  function regionAtCrosshair() {
    const seg = nv.volumes[1];
    if (!seg || mode === 'uncertainty' || mode === 'original') return null;
    try {
      const vox = seg.mm2vox(Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3));
      const value = Math.round(seg.getValue(vox[0], vox[1], vox[2]));
      return LABELS[value] ? value : null;
    } catch { return null; }
  }

  function syncCallout(indices) {
    const box = $('region-callout');
    const label = getState().ready ? regionAtCrosshair() : null;
    const volumes = job()?.result?.volumes_ml;
    if (!label || !volumes || mode === 'expert' || mode === 'split') { box.hidden = true; return; }
    const info = LABELS[label];
    $('region-dot').style.background = info.color;
    $('region-name').textContent = info.name;
    const axial = indices?.[2];
    $('region-detail').textContent = `Volume ${info.volume(volumes).toFixed(2)} mL · Axial ${axial ? axial.index + 1 : '—'} / ${axial ? axial.count : '—'}`;
    box.hidden = false;
  }

  // ---------- slice tile labels for the 3D + slices layout ----------
  function syncTileLabels(indices) {
    const show = getState().ready && getView() === 3 && mode !== 'split';
    $('slice-tile-labels').hidden = !show;
    if (!show || !indices) return;
    for (const [axis, id] of [[2, 'tile-axial'], [1, 'tile-coronal'], [0, 'tile-sagittal']]) {
      const p = indices[axis];
      $(id).textContent = p ? `${p.index + 1} / ${p.count}` : '— / —';
    }
  }

  // ---------- filmstrip of real slices ----------
  function renderFilmstrip(indices) {
    const track = $('filmstrip-track');
    const ready = getState().ready && getState().prediction;
    $('filmstrip-prev').disabled = $('filmstrip-next').disabled = !ready;
    if (!ready || !indices) { track.replaceChildren(); stripKey = ''; return; }
    const axis = axisIndex();
    const active = indices[axis];
    const key = `${job().id}|${axis}|${$('select-background').value}`;
    if (key !== stripKey) {
      stripKey = key;
      track.replaceChildren();
      for (const index of filmstripIndices(active.count, active.index)) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'film-thumb';
        item.dataset.index = index;
        item.setAttribute('role', 'option');
        item.setAttribute('aria-label', `${AXIS_NAMES[axis]} slice ${index + 1} of ${active.count}`);
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.alt = '';
        img.src = sliceUrl(AXIS_NAMES[axis], index, 'segmentation', 96);
        item.append(img);
        item.onclick = () => moveCrosshair(axis, index);
        track.append(item);
      }
    }
    // Highlight the thumbnail nearest to the active slice.
    let best = null;
    for (const item of track.children) {
      const d = Math.abs(Number(item.dataset.index) - active.index);
      if (!best || d < best.d) best = { item, d };
    }
    for (const item of track.children) {
      const on = item === best?.item;
      item.classList.toggle('active', on);
      item.setAttribute('aria-selected', String(on));
    }
    if (best) track.scrollLeft = best.item.offsetLeft - track.clientWidth / 2 + best.item.clientWidth / 2;
  }

  // ---------- split view: original vs overlay of the same real slice ----------
  function syncSplit(indices) {
    const show = mode === 'split' && getState().ready && indices;
    $('split-view').hidden = !show;
    if (!show) return;
    const axis = axisIndex();
    const key = `${job().id}|${axis}|${indices[axis].index}|${$('select-background').value}`;
    if (key === splitKey) return;
    splitKey = key;
    $('split-original').src = sliceUrl(AXIS_NAMES[axis], indices[axis].index, 'none', 512);
    $('split-overlay').src = sliceUrl(AXIS_NAMES[axis], indices[axis].index, 'segmentation', 512);
  }

  function syncSlice() {
    const state = getState();
    const indices = state.ready ? currentIndices() : null;
    const position = indices?.[axisIndex()] || null;
    $('slice-position').disabled = !position;
    $('centre-crosshair').disabled = !position;
    $('slice-position-label').textContent = position ? `${position.index + 1} / ${position.count}` : '— / —';
    $('slice-position').max = position ? position.count - 1 : 0;
    $('slice-position').value = position?.index || 0;
    syncTileLabels(indices);
    syncCallout(indices);
    renderFilmstrip(indices);
    syncSplit(indices);
  }

  // ---------- modality thumbnails ----------
  function renderModalities() {
    const state = getState();
    const current = $('select-background').value;
    const available = job()?.result?.available_modalities || [];
    const centre = geometry?.tumour_centre;
    const d = geometry?.dims_ras;
    for (const button of modalityButtons) {
      const modality = button.dataset.modality;
      const ok = state.ready && available.includes(modality);
      button.disabled = !ok;
      button.setAttribute('aria-checked', String(modality === current));
      const holder = button.querySelector('.thumb-img');
      if (ok && d) {
        const index = centre ? centre[2] : Math.floor(d[2] / 2);
        const src = sliceUrl('axial', index, 'none', 160, modality);
        let img = holder.querySelector('img');
        if (!img) { img = document.createElement('img'); img.alt = ''; holder.append(img); }
        if (img.getAttribute('src') !== src) img.src = src;
      } else {
        holder.replaceChildren();
      }
    }
  }

  // ---------- measurement rings (denominators stated in the UI) ----------
  function renderMeasures() {
    const volumes = job()?.result?.volumes_ml;
    if (!volumes) return;
    const shares = measureShares(volumes, geometry?.brain_ml);
    $('no-tumour-note').hidden = !(Number.isFinite(volumes.WT) && volumes.WT === 0);
    for (const key of ['WT', 'TC', 'ET']) {
      const share = shares[key];
      $(`ring-${key}`).style.setProperty('--p', share === null ? 0 : share * 100);
      $(`pct-${key}`).textContent = share === null ? 'n/a' : `${(share * 100).toFixed(share < 0.1 ? 1 : 0)}%`;
      $(`bar-${key}`).style.width = `${share === null ? 0 : share * 100}%`;
    }
  }

  async function loadGeometry() {
    const current = job();
    if (!current || geometryJob === current.id) return;
    geometryJob = current.id;
    geometry = null;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(current.id)}/geometry`);
      geometry = response.ok ? await response.json() : null;
    } catch { geometry = null; }
    if (job()?.id !== current.id) return;
    renderMeasures();
    loadExpertComparison(current.id);
    loadQuality(current.id);
    const reportLink = $('link-open-report');
    if (reportLink) reportLink.href = `/api/jobs/${encodeURIComponent(current.id)}/report.html`;
    renderModalities();
    centreOnTumour();
  }

  // ---------- AI vs expert grouped bar chart ----------
  let comparison = null;
  let compareMeasure = 'ml';
  const NS = 'http://www.w3.org/2000/svg';
  const fmt = (v, measure) => (v == null ? 'n/a' : measure === 'share' ? `${v.toFixed(1)}%` : `${v.toFixed(v < 10 ? 2 : 1)}`);

  async function loadExpertComparison(id) {
    comparison = null;
    $('expert-chart').replaceChildren();
    $('expert-metrics').replaceChildren();
    if (!getState().expert) return;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/expert-comparison`);
      comparison = response.ok ? await response.json() : null;
    } catch { comparison = null; }
    if (job()?.id !== id) return;
    renderComparison();
  }

  function renderComparison() {
    const chart = $('expert-chart');
    chart.replaceChildren();
    $('expert-metrics').replaceChildren();
    $('expert-table').replaceChildren();
    if (!comparison) {
      chart.textContent = 'Expert comparison unavailable for this analysis.';
      return;
    }
    const data = compareSeries(comparison.regions);
    // Line chart, x = region (WT -> TC -> ET), y = percent. Four lines: AI Dice, AI sensitivity,
    // expert-vs-expert agreement and the expert reference (100% by definition).
    const W = 340, H = 190, L = 34, R = 96, T = 12, B = 24, yMin = 50, yMax = 100;
    const xs = data.map((_, i) => L + (W - L - R) * (data.length === 1 ? 0.5 : i / (data.length - 1)));
    const y = v => T + (H - T - B) * (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin));
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    const el = (tag, attrs, parent = svg) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); parent.append(n); return n; };
    for (const v of [50, 60, 70, 80, 90, 100]) {
      el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'cmp-grid' });
      el('text', { x: L - 5, y: y(v) + 3, class: 'cmp-axis', 'text-anchor': 'end' }).textContent = `${v}%`;
    }
    data.forEach((d, i) => { el('text', { x: xs[i], y: H - 7, class: 'cmp-cat', 'text-anchor': 'middle' }).textContent = d.region; });
    const lines = [
      { key: 'reference', name: 'Expert reference', values: data.map(() => 100), dash: '1 3', tip: () => 'Expert mask = reference (100% by definition)' },
      { key: 'interrater', name: 'Expert vs expert', values: data.map(d => INTER_RATER_DICE[d.region]), dash: '5 3',
        tip: (d, v) => `${d.region}: two expert raters typically agree at ~${v}% Dice (BraTS, Menze et al. 2015)` },
      { key: 'sensitivity', name: 'AI sensitivity', values: data.map(d => d.sensitivity), dash: '',
        tip: (d, v) => `${d.region} AI sensitivity ${v.toFixed(1)}% — share of the expert's tumour the AI found` },
      { key: 'dice', name: 'AI Dice', values: data.map(d => d.dice), dash: '',
        tip: (d, v) => `${d.region} AI Dice ${v.toFixed(1)}% — overlap with the expert outline` },
    ];
    const ends = [];
    for (const line of lines) {
      const pts = line.values.map((v, i) => (v == null ? null : [xs[i], y(v), v, data[i]])).filter(Boolean);
      if (!pts.length) continue;
      el('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(''), fill: 'none',
        stroke: COMPARE_COLORS[line.key], 'stroke-width': line.dash ? 1.4 : 2, 'stroke-dasharray': line.dash, 'stroke-linejoin': 'round' });
      for (const p of pts) {
        const g = el('g', { class: 'hbar-row cmp-point', tabindex: 0, 'data-tip': line.tip(p[3], p[2]) });
        g.setAttribute('aria-label', g.getAttribute('data-tip'));
        el('circle', { cx: p[0], cy: p[1], r: 6, fill: 'transparent' }, g);                       // hit target
        el('circle', { cx: p[0], cy: p[1], r: line.dash ? 2.5 : 3.5, fill: COMPARE_COLORS[line.key], stroke: '#111419', 'stroke-width': 1.5 }, g);
      }
      const last = pts.at(-1);
      ends.push({ y: last[1], text: `${line.name} ${Math.round(last[2])}%`, color: COMPARE_COLORS[line.key] });
    }
    // direct labels at line ends, nudged apart so they never overlap
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 10) ends[i].y = ends[i - 1].y + 10;
    for (const e of ends) el('text', { x: W - R + 6, y: e.y + 3, class: 'cmp-end', fill: '#d6dde1' }).textContent = e.text;
    chart.append(svg);

    // agreement strip
    for (const k of ['WT', 'TC', 'ET']) {
      const r = comparison.regions[k];
      if (!r) continue;
      const cell = document.createElement('div');
      cell.className = 'cmp-metric';
      const pct = v => (v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`);
      cell.innerHTML = '';
      const title = document.createElement('b'); title.textContent = k;
      const dice = document.createElement('span'); dice.className = 'cmp-dice'; dice.textContent = `Dice ${pct(r.dice)}`;
      const rest = document.createElement('small');
      rest.textContent = `Sens ${pct(r.sensitivity)} · Prec ${pct(r.precision)} · HD95 ${r.hd95_mm == null ? 'n/a' : r.hd95_mm + ' mm'}`;
      cell.append(title, dice, rest);
      $('expert-metrics').append(cell);
    }
    // table view (accessible equivalent of the chart)
    const table = $('expert-table');
    const head = table.createTHead().insertRow();
    for (const h of ['Region', 'AI mL', 'Expert mL', 'AI voxels', 'Expert voxels', 'Dice', 'Sens.', 'Prec.', 'HD95 mm']) {
      const th = document.createElement('th'); th.textContent = h; head.append(th);
    }
    const body = table.createTBody();
    for (const k of ['WT', 'TC', 'ET']) {
      const r = comparison.regions[k]; if (!r) continue;
      const row = body.insertRow();
      for (const v of [k, r.pred_ml, r.expert_ml, r.pred_voxels.toLocaleString(), r.expert_voxels.toLocaleString(), r.dice, r.sensitivity ?? 'n/a', r.precision ?? 'n/a', r.hd95_mm ?? 'n/a'])
        row.insertCell().textContent = v;
    }
  }

  for (const button of document.querySelectorAll('[data-measure]')) button.addEventListener('click', () => {
    compareMeasure = button.dataset.measure;
    for (const b of document.querySelectorAll('[data-measure]')) b.setAttribute('aria-pressed', String(b === button));
    renderComparison();
  });

  async function loadQuality(id) {
    const list = $('qc-list');
    list.replaceChildren(Object.assign(document.createElement('li'), { className: 'qc-pending', textContent: 'Checking…' }));
    let checks = null;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/quality`);
      checks = response.ok ? (await response.json()).checks : null;
    } catch { checks = null; }
    if (job()?.id !== id) return;
    list.replaceChildren();
    if (!checks) {
      list.append(Object.assign(document.createElement('li'), { className: 'qc-pending', textContent: 'Quality checks unavailable for this analysis.' }));
      return;
    }
    for (const check of checks) {
      const item = document.createElement('li');
      item.className = `qc-item qc-${['pass', 'fail', 'info'].includes(check.status) ? check.status : 'na'}`;
      const mark = document.createElement('span');
      mark.className = 'qc-mark';
      mark.textContent = check.status === 'pass' ? '✓' : check.status === 'fail' ? '✗' : check.status === 'info' ? 'i' : '–';
      mark.setAttribute('aria-label', check.status);
      const text = document.createElement('span');
      text.textContent = check.name;
      text.title = check.detail;
      item.append(mark, text);
      list.append(item);
    }
  }

  function centreOnTumour() {
    const centre = geometry?.tumour_centre;
    const d = dims();
    if (!centre || !d || !nv.volumes.length) return false;
    nv.scene.crosshairPos = [0, 1, 2].map(a => (centre[a] + 0.5) / d[a]);
    nv.drawScene();
    nv.onLocationChange?.({ mm: Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) });
    return true;
  }

  // ---------- workflow + display modes ----------
  function sync() {
    const state = getState();
    // Real progress: while a job runs the stage follows the backend status message;
    // after a result is ready, Upload/Validate/Segment are complete and the user moves freely.
    const running = state.busy ? stageFromMessage(state.message) : null;
    const resultDone = state.ready || state.loadingResult;   // segmentation finished (result may still be drawing)
    const current = running ?? (state.failedStage ?? (resultDone ? Math.max(activeStage, 3) : 0));
    for (const button of stageButtons) {
      const stage = Number(button.dataset.stage);
      button.disabled = state.busy || (stage >= 3 && !state.ready);
      button.classList.toggle('active', stage === current);
      button.classList.toggle('complete', running !== null ? stage < running : resultDone && stage < 3);
      button.classList.toggle('failed', !state.busy && state.failedStage === stage);
      if (button.classList.contains('active')) button.setAttribute('aria-current','step');
      else button.removeAttribute('aria-current');
    }
    for (const button of modeButtons) {
      const m = button.dataset.display;
      button.disabled = !state.ready || (m !== 'original' && m !== 'expert' && !state.prediction) ||
        (m === 'uncertainty' && !state.uncertainty) || (m === 'expert' && !state.expert);
      button.setAttribute('aria-pressed', String(m === mode));
    }
    renderModalities();
    syncSlice();
  }

  function applyDisplay() {
    if (!getState().ready || !nv.volumes.length) return;
    nv.setOpacity(0, mode === 'segmentation' ? 0 : 1);
  }

  async function selectMode(nextMode) {
    mode = nextMode;
    sync();
    const overlay = ({ original:'none', overlay:'segmentation', segmentation:'segmentation', uncertainty:'uncertainty',
      expert:'expert', split:'segmentation' })[mode];
    await changeOverlay(overlay);
    applyDisplay();
    sync();
  }

  for (const button of modeButtons) button.addEventListener('click', () => {
    selectMode(button.dataset.display).catch(error => setStatus(`Could not change view: ${error.message}`, true));
  });
  for (const button of modalityButtons) button.addEventListener('click', () => {
    changeBackground(button.dataset.modality).then(sync).catch(error => setStatus(`Could not change MRI sequence: ${error.message}`, true));
  });
  for (const button of stageButtons) button.addEventListener('click', () => {
    activeStage = Number(button.dataset.stage);
    const targets = ['.upload-card','.status-card','.model-card','.viewer-card','.measurements-heading','.export-card'];
    const side = activeStage < 3 ? 'inputs' : activeStage > 3 ? 'measurements' : null;
    if (side) { grid.classList.remove(`${side}-collapsed`); $(`toggle-${side}`).setAttribute('aria-expanded','true'); }
    document.querySelector(targets[activeStage])?.scrollIntoView({ block:'nearest', behavior:'auto' });
    sync();
  });
  for (const side of ['inputs','measurements']) $(`toggle-${side}`).onclick = () => {
    const collapsed = grid.classList.toggle(`${side}-collapsed`);
    $(`toggle-${side}`).setAttribute('aria-expanded', String(!collapsed));
  };
  // Clicking a measurement row moves the crosshair to the tumour centre.
  for (const row of document.querySelectorAll('.measure-row')) {
    const go = () => { if (getState().ready) centreOnTumour(); };
    row.addEventListener('click', go);
    row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  }
  $('slice-axis').onchange = () => { stripKey = ''; syncSlice(); };
  $('slice-position').oninput = () => {
    if (!getState().ready) return;
    moveCrosshair(axisIndex(), Number($('slice-position').value));
  };
  const step = delta => {
    const p = currentIndices()?.[axisIndex()];
    if (p) moveCrosshair(axisIndex(), Math.max(0, Math.min(p.count - 1, p.index + delta)));
  };
  $('filmstrip-prev').onclick = () => step(-1);
  $('filmstrip-next').onclick = () => step(1);
  $('centre-crosshair').onclick = () => {
    if (!centreOnTumour()) {
      nv.scene.crosshairPos = [0.5, 0.5, 0.5];
      nv.drawScene();
      nv.onLocationChange?.({ mm: Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) });
    }
  };
  $('viewer-fullscreen').onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.querySelector('.viewer-card').requestFullscreen();
    } catch (error) { setStatus(`Fullscreen unavailable: ${error.message}`, true); }
  };
  sync();
  return {
    sync, syncSlice, applyDisplay,
    isSegmentation: () => mode === 'segmentation',
    loaded() {
      activeStage = 3;
      applyDisplay();
      sync();
      loadGeometry();
    },
    reset() { geometry = null; geometryJob = null; stripKey = ''; splitKey = ''; sync(); },
    overlayChanged(value) {
      mode = value === 'none' ? 'original' : value === 'uncertainty' ? 'uncertainty' : value === 'expert' ? 'expert' : 'overlay';
      sync();
    }
  };
}
