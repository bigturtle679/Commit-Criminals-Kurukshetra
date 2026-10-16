# Spatial bookmarks and blind practice

On branch `toggle`, the model selector remains intact. The review workflow runs on completed analyses and does not run another model or change scoring masks.

## Use

1. Before selecting a patient case, uploading scans, or loading history, select **Open the next case in blind mode**. Alternatively, **Start blind attempt** arms the next case when no analysis is open. Cases whose answers have already been viewed are practice replays.
2. Move the linked crosshair, enter a title and optional observation, and select **Save current location**. Each bookmark records world millimetres, original-grid voxel coordinates, modality, view and overlay settings. Click its title to return to that location. Editing changes the note/title; delete and recapture to change the location.
3. Submit the observations, or explicitly submit **no areas bookmarked**. The submitted snapshot is locked. Reveal the model prediction, then the optional expert reference separately. Reflection notes remain editable.
4. **Saved attempts** restores previous attempts or ordinary case notes. **Return to normal viewing** intentionally exposes available results. **Start another blind attempt** creates a separate record.
5. Export review JSON to save notes and attempt history. This is separate from the unchanged inference artifacts. In a blind session, the model report stays hidden until all available references have been revealed.

This is self-guided practice, not an exam-security boundary. Answer APIs remain available. Exposure is recorded in this browser using a SHA-256 key of modality-ordered input hashes; clearing browser storage loses that history. Older analyses without hashes show **Prior exposure unknown**. A new tab without a requested analysis starts with no case open; the current tab restores its last case and saved review stage on reload.

New analyses export all four MRI modalities. Older analyses retain their available modalities; missing sequences are disabled. All bookmarks are tied to the original analysis grid. WebGL failure leaves existing notes readable/editable and exportable but disables spatial capture/navigation.

## Storage and API

- `GET /api/jobs/{uid}/review` returns `{schema_version: 1, revision, bookmarks, attempts, active_attempt_id, metadata}`. Metadata contains original `affine`, `shape`, available modalities, input hashes and expert-mask availability.
- `PUT` to the same URL accepts the review document with its current revision. The server recomputes voxel coordinates from world coordinates, validates bounds/settings/text, freezes submitted observations, and permits only sequential reveal transitions.
- Successful writes increment the revision and atomically replace `work/runs/{uid}/review.json`. A stale revision returns HTTP 409; invalid content returns 400. Reload discards unsaved changes; Retry repeats the pending save. Copy unsaved form text before discarding a conflict.
- Sidecars live inside the analysis directory and are removed whenever that analysis is deleted. This feature does not add a separate retention process or extend upload retention.
- The app serves `t1.nii.gz` and `t2.nii.gz` alongside the existing background files. Neither the original predicted label map nor an existing model report is rewritten by review actions.

## Checks

With NumPy, FastAPI, HTTPX, nibabel and python-multipart available:

```sh
python -m unittest discover -s tests -p 'test_review_*.py' -v
node tests/test_review_geometry.mjs
```

The Python tests exercise geometry, atomic persistence, conflicts, immutable submissions, reveal transitions and actual HTTP routes with inference stubbed. The Node checks cover full-affine round trips, reveal permissions and deterministic case keys. They do not load checkpoints or measure model accuracy.

For the full browser check, install Playwright's Python package and point `NEUROSCOPE_TEST_BROWSER` at a Chromium/Chrome executable if it is not in the default macOS Chrome location:

```sh
python tests/check_review_browser.py
```

This launches a temporary localhost server, generates small MRI fixtures under `work/`, and exercises the actual NiiVue renderer and review endpoints. It includes bookmark capture/replay, ordinary note CRUD/export, safe text rendering, staged reveal/reload/print behaviour, repeated-case exposure, missing expert masks, empty predictions, legacy modalities, failed-save retry and a simulated WebGL failure. It does not load a model or use patient data.
