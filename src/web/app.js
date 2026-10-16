import { Niivue, SHOW_RENDER, MULTIPLANAR_TYPE } from '/static/vendor/niivue.js';
import { ReviewController, worldToVoxel } from '/static/review.js';

const $ = id => document.getElementById(id);

let currentJob = null;
let currentView = 3; // 3: Multiplanar Grid, 0: Axial, 1: Coronal, 2: Sagittal, 4: 3D Render
let isBusy = false;
let selectedModel = 'ensemble';
let selectedCase = '00002';
let viewerReady = true;
let review = null;
let openingGeneration = 0;
let requestedReviewJob = null;
let viewerGeneration = 0;
let viewerQueue = Promise.resolve();
let loadedJobId = null;
let pendingBlindJob = (() => { try { return sessionStorage.getItem('neuroscope-blind-job'); } catch { return null; } })();
let opening = false;
let gateSignature = '';
const denyAccess = {prediction:false, expert:false, report:false};
// Overlay the user last picked; loading gates temporarily force 'none', this restores it.
let preferredOverlay = 'segmentation';
// Validated (dark surface, CVD-safe) steps of the viewer legend hues; order Edema, Necrotic, Enhancing.
const REGION_COLORS = { edema: '#13a090', necrotic: '#c9651a', enhancing: '#c03a8c' };

// Horizontal bar chart: label · bar from a shared zero baseline · direct value label; hover/focus tooltip.
function renderHBars(container, rows, { max = 1, ref = null } = {}) {
  container.replaceChildren();
  for (const row of rows) {
    const item = document.createElement('div');
    item.className = 'hbar-row';
    item.setAttribute('role', 'listitem');
    item.tabIndex = 0;
    item.dataset.tip = row.tip;
    item.setAttribute('aria-label', row.tip);
    const label = document.createElement('span');
    label.className = 'hbar-label';
    label.textContent = row.label;
    const track = document.createElement('span');
    track.className = 'hbar-track';
    const bar = document.createElement('span');
    bar.className = 'hbar-fill';
    bar.style.width = `${Math.max(0, Math.min(1, row.value / max)) * 100}%`;
    bar.style.background = row.color;
    track.append(bar);
    if (ref !== null) {
      const line = document.createElement('span');
      line.className = 'hbar-ref';
      line.style.left = `${(ref / max) * 100}%`;
      track.append(line);
    }
    const value = document.createElement('span');
    value.className = 'hbar-value';
    value.textContent = row.text;
    item.append(label, track, value);
    container.append(item);
  }
}

const chartTip = document.createElement('div');
chartTip.className = 'chart-tooltip';
chartTip.hidden = true;
document.body.append(chartTip);
function showTip(row) {
  chartTip.textContent = row.dataset.tip;
  chartTip.hidden = false;
  const r = row.getBoundingClientRect();
  chartTip.style.left = `${Math.min(window.innerWidth - chartTip.offsetWidth - 8, Math.max(8, r.left))}px`;
  chartTip.style.top = `${Math.max(8, r.top - chartTip.offsetHeight - 6)}px`;
}
for (const type of ['mouseover', 'focusin']) document.addEventListener(type, e => { const row = e.target.closest?.('.hbar-row'); if (row) showTip(row); });
for (const type of ['mouseout', 'focusout']) document.addEventListener(type, e => { if (e.target.closest?.('.hbar-row')) chartTip.hidden = true; });

function restoreOverlay() {
  if (review?.attempt) return;
  const overlay = $('select-overlay');
  const wanted = [...overlay.options].find(o => o.value === preferredOverlay && !o.disabled);
  overlay.value = wanted ? preferredOverlay : 'segmentation';
}

// Initialize NiiVue WebGL2 Engine
const nv = new Niivue({
  backColor: [0.027, 0.039, 0.063, 1], // Deep obsidian match
  show3Dcrosshair: true,
  isColorbar: false,
  multiplanarShowRender: SHOW_RENDER.ALWAYS,
  multiplanarLayout: MULTIPLANAR_TYPE.GRID
});

try {
  if (!$('gl-canvas').getContext('webgl2', {alpha:true, antialias:navigator.hardwareConcurrency > 6})) {
    throw new Error('WebGL2 unavailable');
  }
  await nv.attachTo('gl-canvas');
  nv.setSliceType(3);
  // The viewer box is sized by the flex layout; re-measure the WebGL buffer whenever it changes.
  // Watch the canvas itself too: it is hidden while results load, and un-hiding it does not resize the parent.
  const viewerResize = new ResizeObserver(() => requestAnimationFrame(() => { try { nv.resizeListener(); } catch {} }));
  viewerResize.observe($('gl-canvas').parentElement);
  viewerResize.observe($('gl-canvas'));
} catch (e) {
  viewerReady = false;
  setStatus('3D WebGL2 viewer unavailable in this environment. Inference and downloads remain functional.', true);
}

// Coordinate & HUD Listener
if (viewerReady) {
  nv.onLocationChange = data => {
    if (!data) return;
    const { mm } = data;
    if (mm && mm.length >= 3) {
      let voxel = '';
      try { voxel = ` | voxel: ${worldToVoxel(Array.from(mm).slice(0, 3), review.document.metadata.affine, review.document.metadata.shape).map(x => x.toFixed(1)).join(', ')}`; } catch {}
      $('hud-crosshair').textContent = `X: ${mm[0].toFixed(1)} mm | Y: ${mm[1].toFixed(1)} mm | Z: ${mm[2].toFixed(1)} mm${voxel}`;
    }
  };
}

function setStatus(text, isError = false) {
  $('status-text').textContent = text;
  $('status-text').classList.toggle('error', isError);
  $('status-icon').hidden = !isBusy;
}

function setBusy(value) {
  isBusy = value;
  document.querySelectorAll('.patient-case-btn, .model-btn').forEach(b => b.disabled = value);
  $('btn-analyze').disabled = value;
  $('history-select').disabled = value;
  $('progress-container').hidden = !value;
  $('status-icon').hidden = !value;
}

async function fetchApi(endpoint, options = {}) {
  const res = await fetch(endpoint, options);
  const data = await res.json();
  if (!res.ok) {
    const msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    throw new Error(msg);
  }
  return data;
}

// Descriptive download names, e.g. NeuroScope_BraTS2021_00144_ensemble_segmentation.nii.gz
function downloadName(filename) {
  const res = currentJob?.result || {};
  const slug = String(res.case_name || 'case').split('·')[0].trim().replace(/[^A-Za-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 40) || 'case';
  return `NeuroScope_${slug}_${res.model_name || 'model'}_${filename}`;
}

function setDownload(id, filename) {
  const link = $(id);
  link.href = getFileUrl(filename);
  link.download = downloadName(filename);
}

function getFileUrl(filename) {
  return `/api/files/${currentJob.id}/${filename}`;
}

// Model information describes architecture tradeoffs, not validated accuracy rankings.
const MODEL_INFO = {
  ensemble: {
    name: '8:2 Ensemble',
    description: 'Combines SwinUNETR and the project’s residual-encoder U-Net (ResEnc) by averaging their probabilities with weights 80% and 20%.',
    advantages: 'Combines predictions from two architectures and supports inspecting disagreement between them.',
    limitations: 'Runs both networks and requires both trained checkpoints. Shared errors can remain; improved accuracy must be established on held-out cases.',
    use: 'Evaluate the combined prediction alongside the individual-model baselines.'
  },
  swinunetr: {
    name: 'SwinUNETR',
    description: 'A hierarchical 3D transformer with shifted-window attention and a U-shaped decoder, using the BraTS21 checkpoint.',
    advantages: 'Uses attention to combine spatial context across scales. Provides a standalone reference for the transformer component of the ensemble.',
    limitations: 'The 3D attention network can demand substantial memory. Predictions depend on matching the checkpoint’s channel order and preprocessing.',
    use: 'Inspect the transformer baseline or compare it with the ensemble.',
    source: 'https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BRATS21/README.md'
  },
  segresnet: {
    name: 'SegResNet',
    description: 'A 3D residual convolutional encoder-decoder loaded from the MONAI brain-tumor segmentation bundle. This is a separate baseline, not the ResEnc component of the ensemble.',
    advantages: 'Provides a convolutional baseline with residual connections and a single-network inference path.',
    limitations: 'Does not use Swin-style attention or combine a second model’s predictions. Its accuracy and runtime need comparison on the same cases and hardware.',
    use: 'Compare the MONAI convolutional baseline with SwinUNETR and the ensemble.',
    source: 'https://arxiv.org/abs/1810.11654'
  }
};

function setModel(modelId) {
  const info = MODEL_INFO[modelId];
  if (!info || isBusy) return;
  selectedModel = modelId;
  document.querySelectorAll('.model-btn').forEach(button => {
    const selected = button.dataset.model === modelId;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  $('header-model-name').textContent = info.name;
  const details = $('model-details');
  details.replaceChildren();
  const title = document.createElement('h4');
  title.textContent = `About ${info.name}`;
  details.append(title);
  for (const [label, text] of [
    ['', info.description], ['Advantages', info.advantages],
    ['Limitations', info.limitations], ['Use it for', info.use],
    ['Inputs / outputs', 'Four aligned MRI sequences: T1, T1Gd, T2 and FLAIR. Outputs: whole tumor, tumor core and enhancing tumor masks.']
  ]) {
    const paragraph = document.createElement('p');
    if (label) {
      const heading = document.createElement('strong');
      heading.textContent = `${label}: `;
      paragraph.append(heading);
    }
    paragraph.append(document.createTextNode(text));
    details.append(paragraph);
  }
  if (info.source) {
    const source = document.createElement('a');
    source.href = info.source;
    source.target = '_blank';
    source.rel = 'noopener noreferrer';
    source.textContent = 'Architecture reference ↗';
    details.append(source);
  }
}

document.querySelectorAll('.model-btn').forEach(button => {
  button.onclick = () => setModel(button.dataset.model);
});
setModel(selectedModel);

// Update Patient Case Selection
function setCaseActive(caseId) {
  selectedCase = caseId;
  document.querySelectorAll('.patient-case-btn').forEach(btn => {
    btn.classList.toggle('active-case', btn.dataset.case === caseId);
  });
}

// Refresh NiiVue Scene (Volumes & Meshes)
function refreshViewer(options = {}) {
  const token = ++viewerGeneration;
  $('gl-canvas').hidden = true;
  viewerQueue = viewerQueue.catch(() => {}).then(() => loadViewer(token, options.world));
  return viewerQueue;
}

async function loadViewer(token, requestedWorld) {
  if (!currentJob || !viewerReady) return;

  if (token !== viewerGeneration) return;
  const jobId = currentJob.id;
  const access = review?.access || denyAccess;
  const previousWorld = requestedWorld || (loadedJobId === jobId && nv.volumes.length ? Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) : null);
  const bgKey = $('select-background').value; // 't1c' or 'flair'
  const selectedOverlay = $('select-overlay').value;
  const overlayMode = selectedOverlay === 'expert' ? (access.expert ? 'expert' : 'none') : selectedOverlay === 'segmentation' && access.prediction ? 'segmentation' : 'none'; // 'segmentation', 'expert', or 'none'
  const opacity = parseFloat($('range-opacity').value);

  const volumes = [
    {
      url: getFileUrl(`${bgKey}.nii.gz`),
      colormap: 'gray',
      opacity: 1.0
    }
  ];

  if (overlayMode !== 'none') {
    const isExpert = overlayMode === 'expert';
    const isUncertainty = overlayMode === 'uncertainty';
    const hasUncertainty = Boolean(currentJob.result?.uncertainty?.available);
    if (isUncertainty && hasUncertainty) {
      volumes.push({
        url: getFileUrl('uncertainty.nii.gz'),
        opacity: opacity,
        colormap: 'warm',
        cal_min: 0.0,
        cal_max: 1.0
      });
    } else if (!isExpert || review.document.metadata.has_expert) {
      volumes.push({
        url: getFileUrl(`${overlayMode}.nii.gz`),
        opacity: opacity,
        cal_min: 0,
        cal_max: 4
      });
    }
  }

  await nv.loadVolumes(volumes);
  if (token !== viewerGeneration || currentJob?.id !== jobId) return;

  if (nv.volumes.length > 1 && overlayMode !== 'uncertainty') {
    // Custom clinical BraTS subregion palette
    // 0: Background, 1: Necrotic/Non-enhancing Core (Amber), 2: Edema (Teal), 3: Unused, 4: Enhancing Tumor (Rose)
    nv.volumes[1].setColormapLabel({
      R: [0, 251, 45, 0, 244],
      G: [0, 146, 212, 0, 114],
      B: [0, 60, 191, 0, 182],
      A: [0, 255, 255, 0, 255],
      I: [0, 1, 2, 3, 4],
      labels: ['Background', 'Necrotic Core', 'Peritumoral Edema', 'Unused', 'Enhancing Tumor']
    });
    nv.updateGLVolume();
  }

  // Load 3D Meshes
  const meshes = [];
  const availableMeshes = currentJob.result.meshes || [];

  if ($('chk-mesh-brain').checked && availableMeshes.includes('brain')) {
    meshes.push({
      url: getFileUrl('brain.obj'),
      rgba255: [56, 189, 248, 255],
      opacity: 0.10 // Translucent skull/brain shell
    });
  }

  if (access.prediction && overlayMode === 'segmentation') {
    if ($('chk-mesh-tumor').checked && availableMeshes.includes('tumor')) {
      meshes.push({
        url: getFileUrl('tumor.obj'),
        rgba255: [244, 63, 94, 255], // Enhancing / Whole tumor outer rim
        opacity: 0.85
      });
    }
    if ($('chk-mesh-core').checked && availableMeshes.includes('core')) {
      meshes.push({
        url: getFileUrl('core.obj'),
        rgba255: [251, 146, 60, 255], // Solid Core
        opacity: 0.95
      });
    }
  }

  await nv.loadMeshes(meshes);
  if (token !== viewerGeneration || currentJob?.id !== jobId) return;
  nv.setSliceType(currentView);
  if (previousWorld) nv.scene.crosshairPos = nv.mm2frac(previousWorld);
  if (loadedJobId !== jobId) nv.setRenderAzimuthElevation(120, 15);
  loadedJobId = jobId;
  nv.drawScene();
  nv.onLocationChange?.({mm:Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3)});
  $('gl-canvas').hidden = false;
  review?.render();
  $('empty-state').hidden = true;
}

// Display Complete Analysis Results
function renderResultPanels(job) {
  const res = job.result;

  // Header and Metadata
  $('active-case-title').textContent = res.case_name;
  $('active-model-tag').textContent = res.model || 'MONAI SegResNet';
  $('hud-spacing').textContent = `${res.spacing_mm.join(' × ')} mm`;

  // Volumetric Diagnostics
  const vols = res.volumes_ml;
  $('val-WT').textContent = vols.WT.toFixed(2);
  $('val-TC').textContent = vols.TC.toFixed(2);
  $('val-ET').textContent = vols.ET.toFixed(2);
  $('val-ED').textContent = vols.ED.toFixed(2);

  // Tumour composition chart: disjoint compartments, colours match the viewer legend
  const total = vols.WT;
  $('composition-total').textContent = `Whole tumour: ${total.toFixed(1)} mL`;
  const necrotic = Math.max(0, vols.TC - vols.ET);
  const share = v => (total > 0 ? (v / total) * 100 : 0);
  renderHBars($('volume-chart'), [
    { label: 'Edema (ED)', value: vols.ED, color: REGION_COLORS.edema, text: `${vols.ED.toFixed(1)} mL · ${share(vols.ED).toFixed(0)}%`,
      tip: `Peritumoral edema: ${vols.ED.toFixed(2)} mL (${share(vols.ED).toFixed(1)}% of whole tumour)` },
    { label: 'Necrotic core', value: necrotic, color: REGION_COLORS.necrotic, text: `${necrotic.toFixed(1)} mL · ${share(necrotic).toFixed(0)}%`,
      tip: `Necrotic / non-enhancing core: ${necrotic.toFixed(2)} mL (${share(necrotic).toFixed(1)}% of whole tumour)` },
    { label: 'Enhancing (ET)', value: vols.ET, color: REGION_COLORS.enhancing, text: `${vols.ET.toFixed(1)} mL · ${share(vols.ET).toFixed(0)}%`,
      tip: `Enhancing tumour: ${vols.ET.toFixed(2)} mL (${share(vols.ET).toFixed(1)}% of whole tumour)` },
  ], { max: Math.max(vols.ED, necrotic, vols.ET, 1e-6) });

  // Explainability numbers
  const unc = res.uncertainty || {};
  $('explain-entropy').textContent = unc.available && Number.isFinite(unc.mean_in_tumor) ? unc.mean_in_tumor.toFixed(3) : 'n/a';
  const dis = res.model_disagreement;
  $('explain-disagree-wrap').hidden = !dis;
  if (dis) $('explain-disagree').textContent = Number.isFinite(dis.tumor_region_mean) ? dis.tumor_region_mean.toFixed(3) : 'n/a';
  $('btn-show-uncertainty').disabled = !unc.available;

  // Dice vs expert chart (single series, one hue)
  const showDice = !!(res.metrics && review.access.expert);
  $('dice-chart').hidden = !showDice;
  if (showDice) {
    renderHBars($('dice-chart'), ['WT', 'TC', 'ET'].filter(k => res.metrics[k]).map(k => {
      const d = res.metrics[k].dice;
      return { label: k, value: d ?? 0, color: 'var(--primary)', text: d === null ? 'N/A' : d.toFixed(3),
        tip: `${({WT: 'Whole tumour', TC: 'Tumour core', ET: 'Enhancing tumour'})[k]} Dice: ${d === null ? 'N/A' : d.toFixed(4)} (1.0 = perfect overlap with the expert)` };
    }), { max: 1, ref: 0.85 });
  }

  // Measurements only: a single scan cannot establish tumor grade or RANO response.
  $('rano-measurable').textContent = `${vols.WT.toFixed(2)} mL`;
  $('rano-phenotype').textContent = `${vols.ET.toFixed(2)} mL`;
  $('rano-summary').textContent = res.volume_summary?.summary ||
    `Predicted whole-tumor region: ${vols.WT.toFixed(1)} mL; enhancing region: ${vols.ET.toFixed(1)} mL. These are segmentation measurements.`;

  // Telemetry Run Info
  $('run-telemetry').textContent = `Runtime: ${res.runtime_seconds.toFixed(2)}s · GPU Peak: ${(res.peak_gpu_mib / 1024).toFixed(2)} GB · Dimensions: ${res.shape.join('×')}`;

  // Ground Truth Benchmark Scores Table - safe DOM construction
  const tbody = $('scores-body');
  tbody.replaceChildren();
  if (res.metrics && review.access.expert) {
    for (const [key, score] of Object.entries(res.metrics)) {
      const tr = document.createElement('tr');
      const tdKey = document.createElement('td');
      const strong = document.createElement('strong');
      strong.textContent = key;
      tdKey.appendChild(strong);

      const dsc = score.dice === null ? 'N/A' : `${(score.dice * 100).toFixed(1)}%`;
      const sens = score.sensitivity === null ? 'N/A' : `${(score.sensitivity * 100).toFixed(1)}%`;

      const tdDsc = document.createElement('td');
      const badgeDsc = document.createElement('span');
      badgeDsc.className = 'score-badge';
      badgeDsc.textContent = dsc;
      tdDsc.appendChild(badgeDsc);

      const tdSens = document.createElement('td');
      const badgeSens = document.createElement('span');
      badgeSens.className = 'score-badge';
      badgeSens.textContent = sens;
      tdSens.appendChild(badgeSens);

      tr.append(tdKey, tdDsc, tdSens);
      tbody.appendChild(tr);
    }
  } else {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 3;
    td.className = 'muted-td';
    td.textContent = 'No expert reference provided. Run against BraTS demo to measure Dice/Sensitivity.';
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  // Overlay dropdown options availability
  const expertOption = $('select-overlay').querySelector('[value=expert]');
  if (expertOption) expertOption.disabled = !res.metrics;
  if (!res.metrics && $('select-overlay').value === 'expert') {
    $('select-overlay').value = 'segmentation';
  }

  const uncOption = $('opt-uncertainty');
  const hasUncertainty = Boolean(res.uncertainty?.available);
  if (uncOption) {
    uncOption.disabled = !hasUncertainty;
  }
  if (!hasUncertainty && $('select-overlay').value === 'uncertainty') {
    $('select-overlay').value = 'segmentation';
  }

  // Configure Download Links
  const maskLink = $('link-download-mask');
  setDownload('link-download-mask', 'segmentation.nii.gz');
  maskLink.classList.remove('disabled');

  const reportLink = $('link-download-report');
  setDownload('link-download-report', 'report.json');
  reportLink.classList.remove('disabled');

  const uncLink = $('link-download-uncertainty');
  if (uncLink) {
    setDownload('link-download-uncertainty', 'uncertainty.nii.gz');
    uncLink.classList.toggle('disabled', !hasUncertainty);
  }

  const meshLink = $('link-download-mesh');
  setDownload('link-download-mesh', 'tumor.obj');
  meshLink.classList.toggle('disabled', !(res.meshes || []).includes('tumor'));

  applyReviewGate();
}

function applyReviewGate() {
  const access = opening ? denyAccess : review?.access || denyAccess;
  const signature = `${access.prediction}:${access.expert}:${access.report}`;
  if (signature !== gateSignature) {
    gateSignature = signature;
    ++viewerGeneration;
    $('gl-canvas').hidden = true;
  }
  document.querySelectorAll('[data-answer]').forEach(element => { element.hidden = !access[element.dataset.answer]; });
  $('analysis-empty').hidden = !!access.prediction;
  const overlay = $('select-overlay');
  for (const option of overlay.options) {
    option.disabled = option.value === 'expert' ? !access.expert : option.value === 'segmentation' ? !access.prediction : false;
  }
  if (overlay.selectedOptions[0]?.disabled) overlay.value = 'none';
  for (const id of ['chk-mesh-tumor', 'chk-mesh-core']) $(id).disabled = !access.prediction;
  for (const [id, allowed, filename] of [
    ['link-download-mask', access.prediction, 'segmentation.nii.gz'],
    ['link-download-mesh', access.prediction && currentJob?.result.meshes?.includes('tumor'), 'tumor.obj'],
    ['link-download-report', access.report, 'report.json']
  ]) {
    if (allowed && currentJob) setDownload(id, filename);
    else $(id).removeAttribute('href');
  }
  if (opening || review?.loading) $('gl-canvas').hidden = true;
}

async function showResult(job) {
  requestedReviewJob = job;
  const generation = ++openingGeneration;
  ++viewerGeneration;
  opening = true;
  currentJob = null;
  $('gl-canvas').hidden = true;
  applyReviewGate();
  try {
    const blind = $('blind-next').checked || pendingBlindJob === job.id;
    await review.load(job.id, blind);
    if (generation !== openingGeneration) return;
    if (review.pending) throw Error('Blind attempt could not be saved. Retry the review save before continuing.');
    currentJob = job;
    try { sessionStorage.setItem('neuroscope-open-analysis', job.id); } catch {}
    pendingBlindJob = null;
    try { sessionStorage.removeItem('neuroscope-blind-job'); } catch {}
    const modalities = review.document.metadata.available_modalities;
    for (const option of $('select-background').options) {
      option.disabled = !modalities.includes(option.value);
      option.textContent = ({t1:'T1',t1c:'T1Gd',t2:'T2',flair:'FLAIR'})[option.value] + (option.disabled ? ' — unavailable' : '');
    }
    if (!modalities.includes($('select-background').value)) $('select-background').value = modalities[0] || '';
    opening = false;
    restoreOverlay();
    renderResultPanels(job);
    review.markExposure();
    await refreshViewer();
    if (generation !== openingGeneration) return;
    review.render();
    setStatus(!viewerReady ? 'Review notes ready. WebGL unavailable; spatial capture and navigation are disabled.' : review.attempt ? 'Review session ready.' : 'Analysis ready to inspect.');
  } catch (error) {
    if (generation === openingGeneration) {
      opening = true;
      applyReviewGate();
      review.error(error);
      setStatus(`Could not open review: ${error.message}`, true);
    }
  }
}

// Populate History
async function refreshHistory() {
  const jobs = await fetchApi('/api/jobs');
  const select = $('history-select');
  select.replaceChildren(new Option('Load prior analysis…', ''));

  const completed = jobs.filter(j => j.status === 'complete').reverse();
  for (const job of completed) {
    const oldSwin = job.result.model_name === 'swinunetr' && !job.result.input_channel_order;
    const label = `${oldSwin ? '[INVALID OLD RUN] ' : ''}${job.result.case_name} · ${job.id.slice(0, 6)}`;
    select.add(new Option(label, job.id));
  }
  return jobs;
}

// Poll Active Job
async function pollJob(jobId) {
  try {
    const job = await fetchApi(`/api/jobs/${jobId}`);
    setStatus(job.message);

    if (job.status === 'complete') {
      setBusy(false);
      await refreshHistory();
      $('history-select').value = jobId;
      await showResult(job);
    } else if (job.status === 'error') {
      setBusy(false);
      setStatus(job.message, true);
    } else {
      setTimeout(() => pollJob(jobId), 1000);
    }
  } catch (err) {
    setBusy(false);
    setStatus(err.message, true);
  }
}

// Start Analysis Job
async function launchAnalysis(endpoint, options = {}, onStart = null) {
  if (isBusy) return;
  if (review?.saving || review?.pending) { setStatus('Save or reload the pending review change before switching cases.', true); return; }
  setBusy(true);
  const gpuLabel = $('gpu-text')?.textContent || 'GPU';
  setStatus(`Running ${MODEL_INFO[selectedModel].name} segmentation on ${gpuLabel}…`);

  try {
    const blind = $('blind-next').checked;
    const res = await fetchApi(endpoint, options);
    if (onStart) onStart(res);
    if (blind) {
      pendingBlindJob = res.id;
      try { sessionStorage.setItem('neuroscope-blind-job', res.id); } catch {}
    }
    await pollJob(res.id);
  } catch (err) {
    setBusy(false);
    setStatus(err.message, true);
  }
}

// Unseen test patients: named zip download or one-click run
async function loadSamples() {
  const list = $('sample-list');
  if (!list) return;
  let samples = [];
  try { samples = await fetchApi('/api/samples'); } catch (err) { list.replaceChildren(); return; }
  list.replaceChildren();
  for (const sample of samples) {
    const item = document.createElement('li');
    item.className = 'sample-item';
    const name = document.createElement('span');
    name.className = 'sample-id';
    name.textContent = sample.id.replace('BraTS2021_', '#');
    name.title = sample.id;
    const actions = document.createElement('span');
    actions.className = 'sample-actions';
    const download = document.createElement('a');
    download.className = 'sample-btn';
    download.textContent = '⬇ Download';
    const run = document.createElement('button');
    run.type = 'button';
    run.className = 'sample-btn sample-run';
    run.textContent = '▶ Run';
    if (sample.available) {
      download.href = `/api/samples/${encodeURIComponent(sample.id)}/download`;
      download.download = `NeuroScope_TestCase_${sample.id}.zip`;
      run.onclick = () => launchAnalysis(`/api/samples/${encodeURIComponent(sample.id)}/run?model=${selectedModel}`, { method: 'POST' });
    } else {
      download.setAttribute('aria-disabled', 'true');
      run.disabled = true;
      item.title = 'Case files are not on this machine.';
    }
    actions.append(download, run);
    item.append(name, actions);
    list.append(item);
  }
}
loadSamples();

// Patient Case Click Handlers
['00000', '00002', '00003'].forEach(caseId => {
  const btn = $(`case-btn-${caseId}`);
  if (btn) {
    btn.onclick = () => {
      setCaseActive(caseId);
      launchAnalysis(`/api/demo?case=${caseId}&model=${selectedModel}`, { method: 'POST' });
    };
  }
});

// Zip upload: one file in, scan types auto-detected server-side
const ROLE_LABELS = { t1c: 'T1c', t1: 'T1', t2: 'T2', flair: 'FLAIR', expert: 'Expert mask' };
function uploadZip(file) {
  if (!file) return;
  const shown = $('zip-detected');
  if (!file.name.toLowerCase().endsWith('.zip')) {
    shown.textContent = 'Please choose a .zip file.';
    return;
  }
  if (isBusy) {
    shown.textContent = 'An analysis is still running — wait for it to finish, then drop the zip again.';
    return;
  }
  shown.textContent = `Uploading ${file.name}…`;
  const formData = new FormData();
  formData.set('archive', file);
  formData.set('model', selectedModel);
  launchAnalysis('/api/analyze-zip', { method: 'POST', body: formData }, res => {
    const parts = Object.entries(res.detected || {}).map(([role, name]) => `${ROLE_LABELS[role] || role} ← ${name}`);
    if (res.expert_et) parts.push(`ET label = ${res.expert_et}`);
    shown.textContent = `✓ Detected: ${parts.join(' · ')}`;
  }).finally(() => { if (shown.textContent.startsWith('Uploading')) shown.textContent = 'Upload failed — see the status card for the reason.'; });
  $('zip-input').value = '';
}
$('zip-input').addEventListener('change', e => uploadZip(e.target.files[0]));
const zipDrop = $('zip-drop');
['dragenter', 'dragover'].forEach(type => zipDrop.addEventListener(type, e => { e.preventDefault(); zipDrop.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(type => zipDrop.addEventListener(type, () => zipDrop.classList.remove('dragging')));
zipDrop.addEventListener('drop', e => { e.preventDefault(); uploadZip(e.dataTransfer.files[0]); });
// Dropping a zip anywhere on the page also works (the browser would otherwise open the file).
let dragDepth = 0;
window.addEventListener('dragenter', e => { if (e.dataTransfer?.types?.includes('Files')) { dragDepth++; document.body.classList.add('page-dragging'); } });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('page-dragging'); } });
window.addEventListener('dragover', e => { if (e.dataTransfer?.types?.includes('Files')) e.preventDefault(); });
window.addEventListener('drop', e => {
  dragDepth = 0; document.body.classList.remove('page-dragging');
  if (e.defaultPrevented || !e.dataTransfer?.files?.length) return;
  e.preventDefault();
  const zip = [...e.dataTransfer.files].find(f => f.name.toLowerCase().endsWith('.zip'));
  if (zip) uploadZip(zip); else setStatus('Drop a .zip of the patient scans, or use the individual file upload.', true);
});

$('upload-form').onsubmit = e => {
  e.preventDefault();
  const formData = new FormData(e.target);
  formData.set('model', selectedModel);
  launchAnalysis('/api/analyze', { method: 'POST', body: formData });
};

$('history-select').onchange = async () => {
  const id = $('history-select').value;
  if (!id) return;
  if (review?.saving || review?.pending) {
    $('history-select').value = review.jobId || '';
    setStatus('Save or reload the pending review change before switching cases.', true);
    return;
  }
  try {
    const job = await fetchApi(`/api/jobs/${id}`);
    await showResult(job);
  } catch (err) {
    setStatus(err.message, true);
  }
};

// View Tabs
$('view-tabs').onclick = e => {
  const btn = e.target.closest('[data-view]');
  if (!btn) return;
  currentView = parseInt(btn.dataset.view, 10);
  for (const el of $('view-tabs').children) {
    el.classList.toggle('active', el === btn);
  }
  if (viewerReady) nv.setSliceType(currentView);
};

// Controls Change
$('select-background').onchange = () => refreshViewer();
$('select-overlay').onchange = () => { preferredOverlay = $('select-overlay').value; refreshViewer(); };
$('btn-show-uncertainty').onclick = () => {
  const overlay = $('select-overlay');
  if ([...overlay.options].some(o => o.value === 'uncertainty' && !o.disabled)) {
    overlay.value = preferredOverlay = 'uncertainty';
    refreshViewer();
  }
};

$('range-opacity').oninput = e => {
  const val = e.target.value;
  $('opacity-val').textContent = `${Math.round(val * 100)}%`;
  if (viewerReady && nv.volumes.length > 1) {
    nv.setOpacity(1, parseFloat(val));
  }
};

$('chk-mesh-brain').onchange = () => refreshViewer();
$('chk-mesh-tumor').onchange = () => refreshViewer();
$('chk-mesh-core').onchange = () => refreshViewer();

$('btn-reset-view').onclick = () => {
  if (viewerReady) {
    nv.scene.crosshairPos = [0.5, 0.5, 0.5];
    nv.setRenderAzimuthElevation(120, 15);
    nv.drawScene();
  }
};

$('btn-snapshot').onclick = () => {
  if (viewerReady) {
    nv.saveScene('neuroscope_mri_snapshot.png');
  }
};

$('btn-print-report').onclick = () => {
  if (review?.access.report && !opening) window.print();
};

review = new ReviewController({
  fetch: fetchApi,
  canCapture: () => viewerReady && !!currentJob && loadedJobId === currentJob.id && !!nv.volumes.length && !opening && !$('gl-canvas').hidden,
  canNavigate: () => viewerReady && !!currentJob && !opening,
  capture: () => {
    if (!viewerReady || !currentJob || loadedJobId !== currentJob.id || !nv.volumes.length) throw Error('Load an MRI volume before saving a location.');
    return {world_mm:Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3), modality:$('select-background').value,
      view:currentView, overlay:review.attempt ? 'none' : $('select-overlay').value, opacity:Number($('range-opacity').value)};
  },
  navigate: async bookmark => {
    worldToVoxel(bookmark.world_mm, review.document.metadata.affine, review.document.metadata.shape);
    $('select-background').value = bookmark.modality;
    currentView = bookmark.view;
    $('view-tabs').querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', Number(b.dataset.view) === currentView));
    const allowed = review.access;
    $('select-overlay').value = bookmark.overlay === 'expert' && allowed.expert ? 'expert' : bookmark.overlay === 'segmentation' && allowed.prediction ? 'segmentation' : 'none';
    $('range-opacity').value = bookmark.opacity;
    $('opacity-val').textContent = `${Math.round(bookmark.opacity * 100)}%`;
    await refreshViewer({world:bookmark.world_mm});
  },
  refresh: async () => {
    if (opening || !currentJob) return;
    renderResultPanels(currentJob);
    restoreOverlay();
    review.markExposure();
    await refreshViewer();
  },
  reveal: async stage => {
    $('select-overlay').value = stage === 'expert' ? 'expert' : 'segmentation';
    renderResultPanels(currentJob);
    review.markExposure();
    await refreshViewer();
  },
  gate: () => applyReviewGate(),
  reopen: async () => {
    const id = requestedReviewJob?.id || review?.jobId;
    if (id) await showResult(await fetchApi(`/api/jobs/${id}`));
  }
});
applyReviewGate();

// File inputs change state update
document.querySelectorAll('#upload-form input[type="file"]').forEach(input => {
  input.onchange = () => {
    const label = input.parentElement.querySelector('.file-status');
    if (label) {
      label.textContent = input.files[0] ? input.files[0].name : 'Select file';
      label.style.color = input.files[0] ? 'var(--primary)' : 'var(--text-dim)';
    }
  };
});

// App Startup Initialization
try {
  const status = await fetchApi('/api/status');
  setModel(status.default_model || selectedModel);
  if (status.gpu) {
    $('gpu-text').textContent = status.gpu;
  }

  // Never offer the ensemble when its trained checkpoints are missing.
  const ensembleInfo = (status.models_available || []).find(m => m.id === 'ensemble');
  if ($('btn-ensemble') && ensembleInfo && !ensembleInfo.available) {
    $('btn-ensemble').disabled = true;
    $('btn-ensemble').title = 'Blocked: trained ResEnc checkpoint (models/resenc_brats/resenc_m_brats.pt) is missing.';
    $('btn-ensemble').querySelector('.model-desc').textContent = 'Blocked — trained ResEnc checkpoint missing. Not benchmarked.';
    if (selectedModel === 'ensemble') setModel('swinunetr');
  }

  const jobs = await refreshHistory();
  const running = jobs.find(j => ['queued', 'running'].includes(j.status));

  if (running) {
    setBusy(true);
    pollJob(running.id);
  } else {
    const requestedId = new URLSearchParams(window.location.search).get('analysis') || (() => { try { return sessionStorage.getItem('neuroscope-open-analysis'); } catch { return null; } })();
    const completed = jobs.filter(j => j.status === 'complete');
    const requested = completed.find(j => j.id === requestedId);
    const latest = requested;

    if (latest) {
      $('history-select').value = latest.id;
      await showResult(latest);
    }
  }
} catch (err) {
  setStatus(`Failed to connect to local backend: ${err.message}`, true);
}
