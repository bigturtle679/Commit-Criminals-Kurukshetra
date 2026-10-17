# NeuroScope dashboard redesign — handoff

Target: the approved reference image (dark charcoal dashboard: header with 6-step workflow,
left rail stepper + 2×2 MRI modality thumbnails, central hero 3D brain with 3 linked slice
previews on its right, bottom deck with view tabs + slice filmstrip + slice/opacity sliders,
right "Tumour Measurements" panel with WT/TC/ET rows, composition bar, uncertainty).

Rules: do NOT push to GitHub (user's instruction for this task). Work in
`/home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16`, branch `model-testing`, uncommitted.
Workspace page = `/workspace` (`src/web/index.html`); `/` is the teammate landing page (`frontend/`).
Restart server: kill the uvicorn python process, then
`.venv/bin/python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000`.
Browser QA: `work/uitest_venv/bin/python` has Playwright; launch
`/opt/microsoft/msedge/msedge` headless with `--use-gl=swiftshader --enable-unsafe-swiftshader`;
open `/workspace?analysis=<job id>` and wait for "analysis ready" in `.status-card`.

## Done by Astra (phase 1, uncommitted)
- `src/web/dashboard.js` (+ `tests/test_dashboard.mjs`): workflow stage buttons, display
  modes (Overlay / Original / Segmentation / Uncertainty), slice-axis + slice slider,
  centre-crosshair, fullscreen, collapsible side panels, `allowedOverlay`, `composition`.
- `src/web/dashboard.css`: design tokens, header workflow, rail, measurement styling.
- `app.js`: NiiVue COLUMN layout with heroImageFraction 0.72 (3D hero + 3 slices),
  label palette ED #47C7C5 / NC #E875B7 / ET #F5A623, uncertainty overlay fixed
  (previously mapped to 'none'), stale result cleared when a new analysis starts.

## Phase 2 (Claude) — status
See the checklist at the bottom; each item is marked when finished and browser-verified.

## Phase 2 done (Claude) — all browser-verified in headless Edge at 1600x900 and 1198x687
Uncommitted, not pushed (per user). Files touched:
- NEW `src/slices.py` — renders real MRI slices (RAS-canonical, windowed) as PNG with optional
  segmentation/uncertainty overlay; `geometry()` = RAS dims, brain volume (non-zero T1c voxels),
  tumour centre. NEW `tests/test_slices.py` (10 tests).
- `src/app.py` — `completed_run_dir()` guard (alnum id, complete, not expired, no traversal) and
  endpoints `GET /api/jobs/{id}/geometry`, `GET /api/jobs/{id}/slice.png?modality&axis&index&overlay&size`.
- `src/web/index.html` — viewer card rebuilt: canvas overlays (slice tile labels, region callout,
  split view), bottom deck (6 tabs incl. Expert mask + Split View, filmstrip, Slice Position,
  Mask Opacity, 4 icon buttons, "More options" with scan/overlay selects + 3D surface toggles);
  left-rail 2x2 MRI modality thumbnails; right panel WT/TC/ET rows with rings; inline favicon.
- `src/web/dashboard.js` — rewritten on top of Astra's exports (kept `allowedOverlay`,
  `composition`, `slicePosition`); adds `measureShares`, `filmstripIndices`, `LABELS`, geometry
  fetch, modality thumbnails, filmstrip, split view, region callout (reads the real label under
  the crosshair via `vol.mm2vox` + `vol.getValue`), tile labels, centre-on-tumour (also on load),
  clickable measurement rows. `tests/test_dashboard.mjs` extended (6 tests; fixed a float-equality bug).
- `src/web/app.js` — new hooks (getJob, getView, changeBackground, expert state), orient cube,
  hero zoom 1.4, short model tag (full checkpoint in tooltip), uncertainty colour range
  cal_min 0.05 / cal_max 0.8 (it painted the whole volume orange before).
- `src/web/dashboard.css` — phase-2 section + short-screen (max-height 820px) compact mode.

Verified: every tab switches the real overlay; Split View loads two 512px backend slices;
filmstrip click / arrows / slider / centre move the crosshair and all labels; modality thumbs
switch the background; rings: WT % of brain, TC and ET % of WT; no console errors; 0 px page
scroll at 1600x900 and 1198x687. Tests: 173 pytest, 6 node dashboard tests, review geometry test.

## Still open (next steps)
1. Leader line from the region callout to the 3D point (reference image) — callout is static top-right of the hero.
2. Translucent textured slice planes in the 3D view (NiiVue draws crosshair lines, not planes).
3. Orientation cube is NiiVue's coloured cube, bottom-left; reference shows a small wire cube top-left.
4. Upload / Validate stepper states are inferred from "result ready"; wire to real upload + validation progress.
5. Check < 1000 px (stacked layout) and mobile — not re-verified after phase 2.
6. Review findings still open from the earlier QA workflow: Model Evidence ensemble row should carry the
   training-overlap asterisk; "Unseen Test Patients" is unseen by ResEnc only; "Air-Gapped Offline"
   claim is false (Google Fonts); export buttons overflow at narrow widths.

## Round 3 (Claude) — user feedback, browser-verified
- Timeline now follows real job state: `stageFromMessage()` in dashboard.js maps backend status
  messages (queued/validating → Validate, standardising/executing/fusing/meshes → Segment);
  complete → steps 1-3 ✓ + Explore active; upload rejected or job error → failed stage shown red.
- Lag: brain-shell mesh (54k translucent triangles over the volume render) is off by default
  (toggle under "More options").
- Removed Demo Patients card. "Unseen Test Patients" → "Sample Patients" with visible help text
  (the old note used `.model-selection-note`, which dashboard.css hides); Run disabled while busy.
- Homepage (`frontend/index.html`) disclaimer: "Research and teaching prototype only … does not
  present model output as a diagnosis"; footer too.
- NEW `src/report.py` + endpoints `GET /api/jobs/{id}/quality` and `/report.html` (CSP-restricted,
  escaped): printable report with slices, volumes + formula, Dice/sensitivity, uncertainty,
  6 re-verified quality checks, SHA-256 provenance. Dashboard: "Quality checks" card +
  "Open research report" button. NEW `tests/test_report.py`.
- Tests: 176 pytest, 7 node dashboard tests.
