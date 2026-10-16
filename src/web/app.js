import { Niivue, SHOW_RENDER, MULTIPLANAR_TYPE } from '/static/vendor/niivue.js';

const $ = id => document.getElementById(id);

let currentJob = null;
let currentView = 3; // 3: Multiplanar Grid, 0: Axial, 1: Coronal, 2: Sagittal, 4: 3D Render
let isBusy = false;
let selectedModel = 'swinunetr';
let selectedCase = '00002';
let viewerReady = true;

// Initialize NiiVue WebGL2 Engine
const nv = new Niivue({
  backColor: [0.027, 0.039, 0.063, 1], // Deep obsidian match
  show3Dcrosshair: true,
  isColorbar: false,
  multiplanarShowRender: SHOW_RENDER.ALWAYS,
  multiplanarLayout: MULTIPLANAR_TYPE.GRID
});

try {
  await nv.attachTo('gl-canvas');
  nv.setSliceType(3);
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
      $('hud-crosshair').textContent = `X: ${mm[0].toFixed(1)} mm | Y: ${mm[1].toFixed(1)} mm | Z: ${mm[2].toFixed(1)} mm`;
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
  document.querySelectorAll('.patient-case-btn').forEach(b => b.disabled = value);
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

function getFileUrl(filename) {
  return `/api/files/${currentJob.id}/${filename}`;
}

// Update Model Switcher Buttons
function setModel(modelId) {
  selectedModel = modelId;
  const isSwin = modelId === 'swinunetr';
  const isEns = modelId === 'ensemble';
  const isRes = modelId === 'segresnet';
  $('btn-swinunetr').classList.toggle('active', isSwin);
  $('btn-segresnet').classList.toggle('active', isRes);
  if ($('btn-ensemble')) $('btn-ensemble').classList.toggle('active', isEns);
  $('header-model-name').textContent = isEns ? '8:2 Hybrid Ensemble' : (isSwin ? 'SwinUNETR (Transformer)' : 'SegResNet (CNN)');
}

$('btn-swinunetr').onclick = () => setModel('swinunetr');
$('btn-segresnet').onclick = () => setModel('segresnet');
if ($('btn-ensemble')) $('btn-ensemble').onclick = () => setModel('ensemble');

// Update Patient Case Selection
function setCaseActive(caseId) {
  selectedCase = caseId;
  document.querySelectorAll('.patient-case-btn').forEach(btn => {
    btn.classList.toggle('active-case', btn.dataset.case === caseId);
  });
}

// Refresh NiiVue Scene (Volumes & Meshes)
async function refreshViewer() {
  if (!currentJob || !viewerReady) return;

  const bgKey = $('select-background').value; // 't1c' or 'flair'
  const overlayMode = $('select-overlay').value; // 'segmentation', 'expert', or 'none'
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
    if (!isExpert || currentJob.result.metrics) {
      volumes.push({
        url: getFileUrl(`${overlayMode}.nii.gz`),
        opacity: opacity,
        cal_min: 0,
        cal_max: 4
      });
    }
  }

  await nv.loadVolumes(volumes);

  if (nv.volumes.length > 1) {
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

  if (overlayMode === 'segmentation') {
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
  nv.setSliceType(currentView);
  nv.setRenderAzimuthElevation(120, 15);
  $('empty-state').hidden = true;
}

// Display Complete Analysis Results
async function showResult(job) {
  currentJob = job;
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

  // Stacked Bar Calculation
  const total = vols.WT;
  $('composition-total').textContent = `Total Volume: ${total.toFixed(2)} mL`;
  if (total > 0) {
    const pET = (vols.ET / total) * 100;
    const pNC = ((vols.TC - vols.ET) / total) * 100;
    const pED = (vols.ED / total) * 100;

    $('bar-enhancing').style.width = `${pET.toFixed(1)}%`;
    $('bar-core').style.width = `${Math.max(0, pNC).toFixed(1)}%`;
    $('bar-edema').style.width = `${pED.toFixed(1)}%`;

    $('pct-enhancing').textContent = `${pET.toFixed(1)}%`;
    $('pct-core').textContent = `${Math.max(0, pNC).toFixed(1)}%`;
    $('pct-edema').textContent = `${pED.toFixed(1)}%`;
  } else {
    $('bar-enhancing').style.width = '0%';
    $('bar-core').style.width = '0%';
    $('bar-edema').style.width = '0%';
  }

  // Measurements only: a single scan cannot establish tumor grade or RANO response.
  $('rano-measurable').textContent = `${vols.WT.toFixed(2)} mL`;
  $('rano-phenotype').textContent = `${vols.ET.toFixed(2)} mL`;
  $('rano-summary').textContent = res.volume_summary?.summary ||
    `Predicted whole-tumor region: ${vols.WT.toFixed(1)} mL; enhancing region: ${vols.ET.toFixed(1)} mL. These are segmentation measurements.`;

  // Telemetry Run Info
  $('run-telemetry').textContent = `Runtime: ${res.runtime_seconds.toFixed(2)}s · GPU Peak: ${(res.peak_gpu_mib / 1024).toFixed(2)} GB · Dimensions: ${res.shape.join('×')}`;

  // Ground Truth Benchmark Scores Table
  const tbody = $('scores-body');
  tbody.replaceChildren();
  if (res.metrics) {
    for (const [key, score] of Object.entries(res.metrics)) {
      const tr = document.createElement('tr');
      const dsc = score.dice === null ? 'N/A' : `${(score.dice * 100).toFixed(1)}%`;
      const sens = score.sensitivity === null ? 'N/A' : `${(score.sensitivity * 100).toFixed(1)}%`;

      tr.innerHTML = `
        <td><strong>${key}</strong></td>
        <td><span class="score-badge">${dsc}</span></td>
        <td><span class="score-badge">${sens}</span></td>
      `;
      tbody.appendChild(tr);
    }
  } else {
    tbody.innerHTML = '<tr><td colspan="3" class="muted-td">No expert reference provided. Run against BraTS demo to measure Dice/Sensitivity.</td></tr>';
  }

  // Overlay dropdown options availability
  const expertOption = $('select-overlay').querySelector('[value=expert]');
  if (expertOption) expertOption.disabled = !res.metrics;
  if (!res.metrics && $('select-overlay').value === 'expert') {
    $('select-overlay').value = 'segmentation';
  }

  // Configure Download Links
  const maskLink = $('link-download-mask');
  maskLink.href = getFileUrl('segmentation.nii.gz');
  maskLink.classList.remove('disabled');

  const reportLink = $('link-download-report');
  reportLink.href = getFileUrl('report.json');
  reportLink.classList.remove('disabled');

  const meshLink = $('link-download-mesh');
  meshLink.href = getFileUrl('tumor.obj');
  meshLink.classList.toggle('disabled', !(res.meshes || []).includes('tumor'));

  try {
    await refreshViewer();
    if (res.model_name === 'swinunetr' && !res.input_channel_order) {
      setStatus('Old SwinUNETR run used the wrong MRI channel order. Rerun this case before using its scores.', true);
    } else {
      setStatus('Segmentation completed successfully. Ready to inspect.');
    }
  } catch (err) {
    setStatus(`Analysis completed, but 3D viewer encountered an error: ${err.message}`, true);
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
async function launchAnalysis(endpoint, options = {}) {
  if (isBusy) return;
  setBusy(true);
  setStatus('Running 3D segmentation on RTX 4060…');

  try {
    const res = await fetchApi(endpoint, options);
    await pollJob(res.id);
  } catch (err) {
    setBusy(false);
    setStatus(err.message, true);
  }
}

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

$('upload-form').onsubmit = e => {
  e.preventDefault();
  const formData = new FormData(e.target);
  formData.set('model', selectedModel);
  launchAnalysis('/api/analyze', { method: 'POST', body: formData });
};

$('history-select').onchange = async () => {
  const id = $('history-select').value;
  if (!id) return;
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
$('select-overlay').onchange = () => refreshViewer();

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
  window.print();
};

// File inputs change state update
document.querySelectorAll('input[type="file"]').forEach(input => {
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
    const requestedId = new URLSearchParams(window.location.search).get('analysis');
    const completed = jobs.filter(j => j.status === 'complete');
    const requested = completed.find(j => j.id === requestedId);
    const latest = requested || completed.filter(j => !(j.result.model_name === 'swinunetr' && !j.result.input_channel_order)).at(-1);

    if (latest) {
      $('history-select').value = latest.id;
      await showResult(latest);
    }
  }
} catch (err) {
  setStatus(`Failed to connect to local backend: ${err.message}`, true);
}
