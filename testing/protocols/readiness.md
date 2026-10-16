# Readiness gate

Current status: NOT READY FOR FULL ENSEMBLE ACCURACY BENCHMARK.

Before marking a model ready, verify:
- Compatible trained checkpoint present, provenance/hash recorded, loading audited.
- Missing-weight fallback eliminated for inference.
- Input channel order and output semantics verified against training configuration.
- Validated case manifest, label meaning and reference geometry.
- Mask leakage excluded; training-overlap status recorded.
- Effective preprocessing, CLI/web agreement and metric semantics checked.
- Focused regression tests passed, with captured logs.

Codex writes outputs/testing/benchmark_ready.md with ready/blocked status PER MODEL,
exact runnable commands and evidence. Antigravity checks that document rather than its
mere existence. A ready baseline does not unblock an ensemble with missing weights.
This file is a gate definition; it is not an authorization to label pending checks passed.
