/* ─── NeuroScope 3D — Upload Page Logic ─────────────────── */
/* Static UI & Client Validation — zero external telemetry */

(function () {
  'use strict';

  /* ─── Constants ──────────────────────────────────────── */
  const MAX_FILE_SIZE = 150 * 1024 * 1024; // 150 MB
  const VALID_NIFTI_EXT = /\.(nii|nii\.gz)$/i;
  const VALID_ZIP_EXT = /\.zip$/i;
  const REQUIRED_MODALITIES = ['t1', 't1c', 't2', 'flair'];

  /* ─── State ──────────────────────────────────────────── */
  let activeTab = 'zip'; // 'zip' | 'individual'
  let zipFile = null;
  const files = { t1: null, t1c: null, t2: null, flair: null, expert: null };
  let demoLoaded = null;
  let selectedModel = 'ensemble';
  let analysisMode = 'normal';
  let analysisRunning = false;

  /* ─── DOM References ─────────────────────────────────── */
  const runBtn = document.getElementById('run-analysis');
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');
  const modeDesc = document.getElementById('mode-desc');

  /* ─── Navbar mobile toggle ───────────────────────────── */
  const menuBtn = document.querySelector('.menu');
  const navEl = document.getElementById('navigation');
  if (menuBtn && navEl) {
    menuBtn.addEventListener('click', function () {
      const open = menuBtn.getAttribute('aria-expanded') !== 'true';
      menuBtn.setAttribute('aria-expanded', String(open));
      navEl.classList.toggle('open', open);
    });
    navEl.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        menuBtn.setAttribute('aria-expanded', 'false');
        navEl.classList.remove('open');
      });
    });
  }

  /* ─── Helper: Format Bytes ───────────────────────────── */
  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  /* ─── Update Run Button Status ───────────────────────── */
  function updateRunButton() {
    let hasValidInput = false;

    if (demoLoaded !== null) {
      hasValidInput = true;
    } else if (activeTab === 'zip') {
      hasValidInput = zipFile !== null;
    } else {
      hasValidInput = REQUIRED_MODALITIES.every(function (f) { return files[f] !== null; });
    }

    const canRun = hasValidInput && !analysisRunning;
    runBtn.disabled = !canRun;
    runBtn.setAttribute('aria-disabled', String(!canRun));

    if (canRun) {
      statusText.textContent = 'Ready — Press "Run Analysis" to execute ' + selectedModel.toUpperCase() + '.';
      statusDot.className = 'status-dot idle';
    } else if (!analysisRunning) {
      if (activeTab === 'zip' && !zipFile && !demoLoaded) {
        statusText.textContent = 'Idle — Upload study .zip archive or select a public demo.';
      } else if (!demoLoaded) {
        statusText.textContent = 'Idle — Select all 4 required MRI modalities or load a demo.';
      }
      statusDot.className = 'status-dot idle';
    }
  }

  /* ─── Clear Demo Selection ───────────────────────────── */
  function clearDemoState() {
    demoLoaded = null;
    document.querySelectorAll('.demo-chip').forEach(function (chip) {
      chip.classList.remove('loaded');
    });
  }

  /* ─── Tab Switching (ZIP vs Individual) ──────────────── */
  const tabBtnZip = document.getElementById('tab-btn-zip');
  const tabBtnInd = document.getElementById('tab-btn-individual');
  const paneZip = document.getElementById('pane-zip');
  const paneInd = document.getElementById('pane-individual');

  function switchTab(newTab) {
    activeTab = newTab;
    if (newTab === 'zip') {
      tabBtnZip.classList.add('active');
      tabBtnZip.setAttribute('aria-selected', 'true');
      tabBtnInd.classList.remove('active');
      tabBtnInd.setAttribute('aria-selected', 'false');
      paneZip.classList.add('active');
      paneZip.hidden = false;
      paneInd.classList.remove('active');
      paneInd.hidden = true;
    } else {
      tabBtnInd.classList.add('active');
      tabBtnInd.setAttribute('aria-selected', 'true');
      tabBtnZip.classList.remove('active');
      tabBtnZip.setAttribute('aria-selected', 'false');
      paneInd.classList.add('active');
      paneInd.hidden = false;
      paneZip.classList.remove('active');
      paneZip.hidden = true;
    }
    updateRunButton();
  }

  if (tabBtnZip && tabBtnInd) {
    tabBtnZip.addEventListener('click', function () { switchTab('zip'); });
    tabBtnInd.addEventListener('click', function () { switchTab('individual'); });
  }

  /* ─── Direct ZIP Upload Handling ─────────────────────── */
  const zipInput = document.getElementById('file-zip');
  const zipZone = document.getElementById('zone-zip');
  const zipStatus = document.getElementById('status-zip');
  const zipRemoveBtn = document.getElementById('remove-zip');

  if (zipInput) {
    zipInput.addEventListener('change', function () {
      const file = zipInput.files[0];
      if (!file) return;

      if (!VALID_ZIP_EXT.test(file.name)) {
        zipStatus.textContent = 'Invalid format. Please select a .zip study archive.';
        zipStatus.className = 'zone-status error-text';
        zipZone.classList.add('error');
        zipZone.classList.remove('has-file');
        zipInput.value = '';
        zipFile = null;
        zipRemoveBtn.hidden = true;
        updateRunButton();
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        zipStatus.textContent = 'Archive exceeds deployed limit (150 MB).';
        zipStatus.className = 'zone-status error-text';
        zipZone.classList.add('error');
        zipZone.classList.remove('has-file');
        zipInput.value = '';
        zipFile = null;
        zipRemoveBtn.hidden = true;
        updateRunButton();
        return;
      }

      zipFile = file;
      zipStatus.textContent = '✓ ' + file.name + ' (' + formatSize(file.size) + ') — Ready';
      zipStatus.className = 'zone-status success-text';
      zipZone.classList.remove('error');
      zipZone.classList.add('has-file');
      zipRemoveBtn.hidden = false;

      clearDemoState();
      updateRunButton();
    });

    zipRemoveBtn.addEventListener('click', function () {
      zipInput.value = '';
      zipFile = null;
      zipStatus.textContent = '';
      zipStatus.className = 'zone-status';
      zipZone.classList.remove('has-file', 'error');
      zipRemoveBtn.hidden = true;
      updateRunButton();
    });
  }

  /* ─── Individual Modalities Handling ─────────────────── */
  document.querySelectorAll('.upload-zone[data-field]').forEach(function (zone) {
    const field = zone.getAttribute('data-field');
    const input = zone.querySelector('input[type="file"]');
    const statusEl = zone.querySelector('.zone-status');
    const removeBtn = zone.querySelector('.zone-remove');

    if (!input) return;

    input.addEventListener('change', function () {
      const file = input.files[0];
      if (!file) return;

      if (!VALID_NIFTI_EXT.test(file.name)) {
        statusEl.textContent = 'Please choose a .nii or .nii.gz file.';
        statusEl.className = 'zone-status error-text';
        zone.classList.add('error');
        zone.classList.remove('has-file');
        input.value = '';
        files[field] = null;
        if (removeBtn) removeBtn.hidden = true;
        updateRunButton();
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        statusEl.textContent = 'Exceeds 150 MB limit.';
        statusEl.className = 'zone-status error-text';
        zone.classList.add('error');
        zone.classList.remove('has-file');
        input.value = '';
        files[field] = null;
        if (removeBtn) removeBtn.hidden = true;
        updateRunButton();
        return;
      }

      files[field] = file;
      statusEl.textContent = file.name + ' (' + formatSize(file.size) + ')';
      statusEl.className = 'zone-status success-text';
      zone.classList.remove('error');
      zone.classList.add('has-file');
      if (removeBtn) removeBtn.hidden = false;

      if (REQUIRED_MODALITIES.indexOf(field) !== -1) {
        clearDemoState();
      }

      updateRunButton();
    });

    if (removeBtn) {
      removeBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        input.value = '';
        files[field] = null;
        statusEl.textContent = '';
        statusEl.className = 'zone-status';
        zone.classList.remove('has-file', 'error');
        removeBtn.hidden = true;
        updateRunButton();
      });
    }
  });

  /* ─── ET Encoding Radio Toggle ───────────────────────── */
  const etRadios = document.querySelectorAll('input[name="et-encoding"]');
  const etCustomInput = document.getElementById('et-custom-value');
  etRadios.forEach(function (radio) {
    radio.addEventListener('change', function () {
      if (etCustomInput) {
        etCustomInput.disabled = radio.value !== 'custom';
      }
    });
  });

  /* ─── Compact Model Selector ─────────────────────────── */
  const modelRows = document.querySelectorAll('.model-row');

  function selectModel(row) {
    modelRows.forEach(function (r) {
      r.classList.remove('selected');
      r.setAttribute('aria-checked', 'false');
      r.setAttribute('tabindex', '-1');
    });
    row.classList.add('selected');
    row.setAttribute('aria-checked', 'true');
    row.setAttribute('tabindex', '0');
    selectedModel = row.getAttribute('data-model');
    updateRunButton();
  }

  modelRows.forEach(function (row) {
    row.addEventListener('click', function (e) {
      if (e.target.closest('.specs-toggle')) return;
      selectModel(row);
    });

    row.addEventListener('keydown', function (e) {
      const rows = Array.from(modelRows);
      const idx = rows.indexOf(row);
      let next;

      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        next = rows[(idx + 1) % rows.length];
        selectModel(next);
        next.focus();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        next = rows[(idx - 1 + rows.length) % rows.length];
        selectModel(next);
        next.focus();
      } else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        selectModel(row);
      }
    });

    // Technical Specs Accordion inside model row
    const specsToggle = row.querySelector('.specs-toggle');
    const drawer = row.querySelector('.model-drawer');
    if (specsToggle && drawer) {
      specsToggle.addEventListener('click', function (e) {
        e.stopPropagation();
        const expanded = specsToggle.getAttribute('aria-expanded') === 'true';
        specsToggle.setAttribute('aria-expanded', String(!expanded));
        drawer.hidden = expanded;
        specsToggle.textContent = expanded ? 'Specs' : 'Hide';
      });
    }
  });

  /* ─── Quick Demo Cases Strip ─────────────────────────── */
  document.querySelectorAll('.demo-chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      const demoId = chip.getAttribute('data-demo');

      if (demoLoaded === demoId) {
        clearDemoState();
        updateRunButton();
        return;
      }

      clearDemoState();
      demoLoaded = demoId;
      chip.classList.add('loaded');
      updateRunButton();
    });
  });

  /* ─── Mode Selector (Normal vs Blind) ────────────────── */
  const modeBtns = document.querySelectorAll('.mode-btn');
  const modeDescriptions = {
    normal: 'Standard analysis: overlays, physical volumes (mL), and metrics displayed immediately.',
    blind: 'Blind Teaching Mode: results hidden initially. Mark trainee observations first, then reveal predictions.'
  };

  modeBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      modeBtns.forEach(function (b) {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      analysisMode = btn.getAttribute('data-mode');
      if (modeDesc) {
        modeDesc.textContent = modeDescriptions[analysisMode] || '';
      }
    });
  });

  /* ─── Run Analysis: real backend (FastAPI on the same origin) ─── */
  function setPipelineStatus(state, msg) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = msg;
  }

  function etLabel() {
    const custom = document.querySelector('input[name="et-encoding"][value="custom"]');
    if (!custom || !custom.checked) return 4;
    const value = Number(document.getElementById('et-custom-value').value);
    if (value !== 3 && value !== 4) throw new Error('Enhancing-tumour label must be 3 (BraTS 2023+) or 4 (BraTS 2018–2021).');
    return value;
  }

  async function startJob() {
    let response;
    if (demoLoaded) {
      response = await fetch('/api/samples/' + encodeURIComponent(demoLoaded) + '/run?model=' + encodeURIComponent(selectedModel), { method: 'POST' });
    } else if (activeTab === 'zip' && zipFile) {
      const form = new FormData();
      form.set('archive', zipFile);
      form.set('model', selectedModel);
      const expertFile = document.getElementById('zip-expert')?.files?.[0];
      if (expertFile) form.set('expert', expertFile);
      response = await fetch('/api/analyze-zip', { method: 'POST', body: form });
    } else {
      const form = new FormData();
      for (const key of REQUIRED_MODALITIES) form.set(key, files[key]);
      if (files.expert) { form.set('expert', files.expert); form.set('expert_et', String(etLabel())); }
      form.set('model', selectedModel);
      response = await fetch('/api/analyze', { method: 'POST', body: form });
    }
    const data = await response.json().catch(function () { return {}; });
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The server rejected this upload (HTTP ' + response.status + ').');
    return data;
  }

  async function waitForJob(id) {
    for (;;) {
      const response = await fetch('/api/jobs/' + encodeURIComponent(id));
      const job = await response.json();
      if (!response.ok) throw new Error(job.detail || 'Lost track of the analysis.');
      if (job.status === 'complete') return job;
      if (job.status === 'error') throw new Error(job.message || 'The analysis failed.');
      setPipelineStatus(job.status === 'queued' ? 'queued' : 'running', job.message || 'Running…');
      await new Promise(function (resolve) { setTimeout(resolve, 1000); });
    }
  }

  if (runBtn) {
    runBtn.addEventListener('click', async function () {
      if (runBtn.disabled || analysisRunning) return;
      analysisRunning = true;
      runBtn.disabled = true;
      runBtn.setAttribute('aria-disabled', 'true');
      setPipelineStatus('queued', demoLoaded ? 'Starting sample patient…' : 'Uploading scans…');
      try {
        const started = await startJob();
        if (started.detected) {
          setPipelineStatus('queued', 'Detected ' + Object.keys(started.detected).length + ' files in the zip — validating…');
        }
        await waitForJob(started.id);
        setPipelineStatus('complete', 'Complete — opening the workspace…');
        window.location.href = '/workspace?analysis=' + encodeURIComponent(started.id);
      } catch (error) {
        analysisRunning = false;
        updateRunButton();                       // resets the button; status text set after so it is not overwritten
        setPipelineStatus('failed', 'Failed — ' + error.message);
      }
    });
  }

  // Initial update
  updateRunButton();

})();
