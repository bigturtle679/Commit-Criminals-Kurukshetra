# Model testing

This branch holds reproducible test protocols, manifests and analysis for NeuroScope.
Start with [ANALYSIS.md](ANALYSIS.md). Procedures are in `protocols/`;
agent assignments are in `handoff/`; dataset records belong in `data/`.

This is a testing setup, not a completed accuracy benchmark. Raw MRI files stay in
`data/external/` at the repository root, which is ignored. Generated predictions
and logs stay in `outputs/testing/`. Publish compact summaries and manifests here.
Never commit credentials, model weights, raw patient volumes or large archives.

Readiness requires compatible trained checkpoints, validated patient manifests,
and matching CLI/web evaluation semantics. See `protocols/readiness.md`.
