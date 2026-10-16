# Antigravity assignment

Use shared branch model-testing. Read testing/README.md and protocols. Own acquisition,
patient manifests, sequential GPU runs and UI validation. Put acquisition helpers under
work/data_acquisition/, raw data under data/external/, logs/predictions under outputs/testing/.
Do not edit application/model code or switch branches. Coordinate Git commits through Codex
while both agents share this checkout to avoid index/commit races.

Inspect testing/data/sources.json and source inventories. Use official Kaggle tooling,
local authentication if required, pinned versions and bounded downloads. Build three-case
smoke data then a reproducible twenty-patient pilot. Validate modalities, labels, geometry,
duplicates and training overlap. Save a manifest following testing/data/manifest.schema.json.
Copy only compact manifests and aggregate reports into testing/ for Git publication.

Wait for Codex's explicit per-model readiness, then run the paired benchmark protocol.
If ResEnc weights remain missing, report ensemble BLOCKED and test ready baselines only.
Capture failures, timing, memory, repeatability and browser/CLI agreement. Publish actual
results and limitations in testing/ANALYSIS.md after coordinating that file with Codex.
Do not tune weights on evaluation cases or call exploratory scores clinical validation.
