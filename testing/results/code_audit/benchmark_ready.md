# Benchmark readiness — 2026-10-16

Code base: 5cbc5cc (plus tracked report-only updates). Run from the repository root.

- SwinUNETR: READY for exploratory labeled-case benchmarking after manifest validation.
- SegResNet: READY as additional exploratory baseline after manifest validation.
- Custom ResEnc: BLOCKED — compatible trained checkpoint and training semantics unavailable.
- 80:20 ensemble: BLOCKED by the missing ResEnc checkpoint. Do not use random weights.

The audit completed 18 regression tests, 6 ensemble software checks and GPU stability checks
(with the missing trained-ResEnc check explicitly skipped). Three real cases completed on
both available baselines. Two independent Swin runs on one case produced identical labels.
Synthetic CLI/web paths agree; this is not trained ensemble or browser visual validation.

## Commands

```bash
cd /home/ryie/Documents/Codex/DI-02-Hackathon-2026-10-16
.venv/bin/python scripts/benchmark_manifest.py --manifest testing/data/manifest.json --validate-only --output outputs/testing/unused
.venv/bin/python scripts/benchmark_manifest.py --manifest testing/data/manifest.json --models swinunetr segresnet --output outputs/testing/antigravity_pilot_01
```

The output directory must be new. Never run two GPU jobs together. This Codex audit has
finished its GPU inference; Qwen CPU review may still be active. Record that contention if
you benchmark runtime before Qwen completes. A model gate does not certify dataset labels
or training independence. Validate Antigravity's manifest and source label documentation.
Do not switch branches or edit application code during a run. Coordinate commits through
one agent because this is a shared checkout. Restart the old web server before browser
checks to load the code changes; do not restart it while a job is active.

HD95 is not implemented. Training overlap of local demo data is unknown. Do not call
these results clinical validation. See testing/ANALYSIS.md and testing/results/code_audit/.
