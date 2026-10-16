# Sync of `main` into `model-testing` — 2026-10-17

## Why `main` was broken
Commit `542001f` ("togglwe", merged as PR #2) committed unresolved conflict
markers (`<<<<<<< HEAD … >>>>>>> main`) into:

| File | Markers | Effect on `main` |
|---|---|---|
| `src/app.py` | 5 blocks | Python syntax error — server cannot start |
| `src/pipeline.py` | 2 blocks | Syntax error — no inference |
| `src/web/app.js` | 3 blocks | JS syntax error — UI script does not load |
| `src/web/index.html` | 2 blocks | Raw markers rendered on the page |

All were resolved here keeping **both** sides: the review workflow (bookmarks,
blind practice, model toggle) and the ensemble-consolidation refactor
(retention, path-traversal guards, uncertainty overlay, config loader).

## Other defects found and fixed during the merge
1. **SwinUNETR could not load on `main`.** The new `weights_only=True` loader
   rejected the checkpoint's NumPy scalar (`best_acc`). Fixed by allowlisting
   NumPy scalar types (still safe loading, no arbitrary code execution).
   Without this, SwinUNETR and the ensemble both fail in the app.
2. **Explicit checkpoint silently replaced.** Requesting a missing checkpoint
   path fell back to a different file. Now raises `CheckpointNotFoundError`.
3. **FP16 entropy fix lost.** Restored FP32 promotion and NaN rejection in
   `compute_voxel_entropy` (log(0) → −inf at half precision).
4. **`src.pipeline` import failed** outside the server (`from metrics import`).
5. **CLI crashed after inference** printing NumPy arrays in the report.
6. **`/api/status` lost per-model availability**, so the UI could offer the
   ensemble without its checkpoint. Restored.
7. Tests updated to the refactored APIs (`_build_channel_indices`,
   3-tuple `load_model`, `_gpu_name` stub).

## Verification on the merged code
- `pytest`: **139 passed** (all suites incl. review API/store, checkpoint
  loading, config loader, evaluation runner, retention, audit regressions).
- JS: `test_review_geometry.mjs` PASS; `node --check` on `app.js`, `review.js`.
- `tests/test_ensemble.py`: 8/8.
- All three checkpoints + ensemble load with safe loading.
- Real ensemble run through the restarted web server (BraTS2021_00003):
  complete in 10 s, WT 0.984 / TC 0.979 / ET 0.944.
- Ensemble CLI vs web: identical Dice and volumes; 1 / 8,928,000 voxels
  differ (threshold-boundary GPU float noise) — `pilot20_ensemble/ensemble_cli_web_parity.json`.
- 20-patient ensemble benchmark re-run on the merged code: 20/20 complete,
  mean Dice WT 0.9045 / TC 0.9232 / ET 0.8753, **identical** to the pre-merge
  run (max per-patient difference 0.0000). `benchmark_manifest.py` updated for
  the refactored ensemble bundle (hashes from `path_a`/`path_b`).
